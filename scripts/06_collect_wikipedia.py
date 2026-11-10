"""Resume-safe, category-balanced Wikipedia corpus collection.

Run after 05_build_category_catalog.py. Uses 50 areas and catalog subtopics.
By default the database lives in LOCALAPPDATA on Windows; set ADM_DATA_ROOT
outside OneDrive for downloaded documents too.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import re
from collections import Counter
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from bs4 import BeautifulSoup
import requests

from wikimedia_common import (API, REST, CATALOG_PATH, DATA_DIR, DB_PATH,
                              PROJECT_DIR, StopCrawl, WikiClient, write_json_atomic)

RAW_DIR = DATA_DIR / 'raw'
QUARANTINE_DIR = DATA_DIR / 'quarantine'
PROCESSED_DIR = DATA_DIR / 'processed'
BAD_HEADINGS = {'references', 'notes', 'footnotes', 'bibliography', 'sources',
                'further reading', 'external links', 'see also', 'citations'}
BAD_CLASSES = re.compile(r'(navbox|vertical-navbox|sidebar|infobox|metadata|ambox|'
                         r'mbox|hatnote|reflist|reference|mw-editsection|toc|'
                         r'sistersitebox|shortdescription)', re.I)
BAD_TITLES = re.compile(r'^(List of |Lists of |Index of |Outline of |Timeline of )', re.I)


def utc_now():
    return datetime.now(timezone.utc).isoformat()



class MeasuredWikiClient(WikiClient):
    """Count physical HTTP requests by purpose, including failed attempts."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.request_breakdown = Counter()

    def get(self, url, params=None):
        params = params or {}
        if params.get('list') == 'categorymembers':
            purpose = 'category_discovery'
        elif params.get('prop') == 'info|pageprops':
            purpose = 'legacy_validation' if getattr(self, 'legacy_mode', False) else 'metadata_validation'
        elif '/with_html' in url:
            purpose = 'html_download'
        else:
            purpose = 'other'
        before = self.requests_made
        try:
            return super().get(url, params=params)
        finally:
            delta = self.requests_made - before
            self.request_breakdown[purpose] += delta
            self.request_breakdown['retry_requests'] += max(0, delta - 1)


def rejection_reason(error):
    if not error:
        return 'Unspecified'
    if error.startswith('Too short:'):
        return 'Too short'
    if error.startswith('Page ID mismatch'):
        return 'Page ID mismatch / redirect'
    if error.startswith('Missing, non-article'):
        return 'Missing / namespace / redirect'
    if error.startswith('Disambiguation'):
        return 'Disambiguation'
    if error.startswith('List/outline'):
        return 'List / outline'
    return error.split(':', 1)[0][:90]


def persist_request_stats(conn, client):
    """Persist counted requests, without double counting a running session."""
    for key, value in client.request_breakdown.items():
        previous = client._persisted_breakdown.get(key, 0)
        increment = value - previous
        if increment <= 0:
            continue
        row = conn.execute('SELECT value FROM stats WHERE key=?',
                           ('http_total_' + key,)).fetchone()
        current = int(row[0]) if row else 0
        conn.execute('INSERT OR REPLACE INTO stats(key,value) VALUES (?,?)',
                     ('http_total_' + key, str(current + increment)))
        client._persisted_breakdown[key] = value
    conn.commit()


def record_legacy_review(conn, pageid, status, reason):
    conn.execute("INSERT OR REPLACE INTO legacy_review (pageid,status,reason,reviewed_at) VALUES (?,?,?,?)",
        (pageid,status,reason,utc_now()))


def quarantine_legacy(conn, pageid, reason):
    """Remove an invalid legacy document from the active corpus, never its raw source."""
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    active = PROCESSED_DIR / f'{pageid}.json'
    if active.exists():
        shutil.move(str(active), str(QUARANTINE_DIR / active.name))
    conn.execute('DELETE FROM documents WHERE pageid=? AND area=?',
                 (pageid, 'Legacy'))
    record_legacy_review(conn, pageid, 'rejected', reason)


def revalidate_legacy_documents(conn, client, min_words):
    """Recheck imported legacy records once; use original HTML when available.

    Metadata is verified using a batched Wikipedia query. If the API is
    unavailable, leave rows pending_review rather than making assumptions.
    """
    records = conn.execute("SELECT pageid,title FROM documents WHERE area='Legacy' AND pageid NOT IN (SELECT pageid FROM legacy_review WHERE status IN ('accepted','rejected')) ORDER BY pageid").fetchall()
    if not records:
        return
    print(f'Revalidating legacy documents: {len(records)}', flush=True)
    client.legacy_mode = True
    for start in range(0, len(records), 50):
        batch = records[start:start+50]
        ids = '|'.join(str(pid) for pid, _ in batch)
        try:
            metadata = client.query(prop='info|pageprops', pageids=ids)
            pages = {int(page['pageid']): page for page in
                     metadata.get('query', {}).get('pages', [])
                     if 'pageid' in page}
        except (requests.exceptions.RequestException, ValueError) as exc:
            for pid, _ in batch:
                record_legacy_review(conn, pid, 'pending_review',
                                     f'Metadata unavailable: {str(exc)[:120]}')
            conn.commit()
            continue
        for pid, title in batch:
            page = pages.get(pid, {})
            reason = None
            if not page or 'missing' in page or page.get('ns') != 0 or 'redirect' in page:
                reason = 'Missing / namespace / redirect'
            elif 'disambiguation' in page.get('pageprops', {}):
                reason = 'Disambiguation'
            elif BAD_TITLES.search(page.get('title', title)):
                reason = 'List / outline'
            if reason:
                quarantine_legacy(conn, pid, reason)
                continue
            active = PROCESSED_DIR / f'{pid}.json'
            raw_path = RAW_DIR / f'{pid}.json'
            try:
                with active.open(encoding='utf-8') as handle:
                    processed = json.load(handle)
                if raw_path.exists():
                    with raw_path.open(encoding='utf-8') as handle:
                        raw = json.load(handle)
                    clean, paragraphs, sections = clean_html(raw['html'])
                    processed['text'] = clean
                    processed['paragraphs'] = paragraphs
                    processed['sections'] = sections
                    processed['word_count'] = len(clean.split())
                    processed['revision_id'] = raw.get('latest', {}).get('id')
                    processed['revision_timestamp'] = raw.get('latest', {}).get('timestamp')
                else:
                    processed['word_count'] = len(processed['text'].split())
                words = processed['word_count']
                if words < min_words:
                    quarantine_legacy(conn, pid, f'Too short: {words} words')
                    continue
                write_json_atomic(active, processed)
                conn.execute('UPDATE documents SET word_count=?, title=? WHERE pageid=?',
                             (words, page.get('title', title), pid))
                record_legacy_review(conn, pid, 'accepted',
                    'Reprocessed from original HTML' if raw_path.exists()
                    else 'Validated from existing text (raw unavailable)')
            except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
                record_legacy_review(conn, pid, 'pending_review',
                                     f'Unable to validate local content: {str(exc)[:120]}')
        conn.commit()
    client.legacy_mode = False
    counts = dict(conn.execute(
        'SELECT status, COUNT(*) FROM legacy_review GROUP BY status').fetchall())
    print('Legacy review:', counts, flush=True)


def initialize_db(conn):
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS candidates (
            pageid INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            area TEXT NOT NULL,
            topic TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            added_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_candidates_area_status
            ON candidates(area,status,pageid);
        CREATE TABLE IF NOT EXISTS documents (
            pageid INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            area TEXT NOT NULL,
            topic TEXT NOT NULL,
            word_count INTEGER NOT NULL,
            saved_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS topics (
            category TEXT NOT NULL,
            area TEXT NOT NULL,
            next_cmcontinue TEXT,
            done INTEGER NOT NULL DEFAULT 0,
            pages_fetched INTEGER NOT NULL DEFAULT 0,
            priority INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (area, category)
        );
        CREATE TABLE IF NOT EXISTS legacy_review (
            pageid INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            reason TEXT,
            reviewed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS stats (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    ''')
    conn.commit()


def current_count(conn):
    return conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]


def area_counts(conn):
    return dict(conn.execute('SELECT area,COUNT(*) FROM documents GROUP BY area').fetchall())


def area_backlogs(conn):
    return dict(conn.execute("SELECT area,COUNT(*) FROM candidates WHERE status IN ('pending','retry') GROUP BY area").fetchall())


def import_old_documents(conn):
    """Import existing processed metadata; never touch or delete old files."""
    old_dir = PROJECT_DIR / 'data' / 'processed'
    directories = {old_dir, PROCESSED_DIR}
    imported = 0
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.glob('*.json'):
            try:
                with path.open(encoding='utf-8') as f:
                    obj = json.load(f)
                pageid = int(obj['document_id'])
                blocked = conn.execute(
                    "SELECT 1 FROM legacy_review WHERE pageid=? AND status='rejected'",
                    (pageid,)).fetchone()
                if blocked:
                    continue
                if not isinstance(obj.get('text'), str):
                    continue
                title = str(obj['title'])
                words = int(obj.get('word_count', len(obj['text'].split())))
                # Legacy documents count globally, but not against any new area's quota.
                if directory != PROCESSED_DIR:
                    dst=PROCESSED_DIR / path.name
                    dst.parent.mkdir(parents=True,exist_ok=True)
                    if not dst.exists():
                        shutil.copy2(path,dst)
                    raw_src=PROJECT_DIR / 'data' / 'raw' / path.name
                    raw_dst=RAW_DIR / path.name
                    if raw_src.exists() and not raw_dst.exists():
                        raw_dst.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(raw_src,raw_dst)
                row = conn.execute('INSERT OR IGNORE INTO documents '
                    '(pageid,title,area,topic,word_count,saved_at) VALUES (?,?,?,?,?,?)',
                    (pageid,title,'Legacy','Legacy',words,utc_now()))
                imported += row.rowcount
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
    conn.commit()
    return imported


def load_catalog(conn):
    if not CATALOG_PATH.exists():
        raise SystemExit(f'Missing catalog: {CATALOG_PATH}. Run 05_build_category_catalog.py first.')
    with CATALOG_PATH.open(encoding='utf-8') as f:
        catalog = json.load(f)
    if len(catalog.get('areas', {})) != 50:
        raise SystemExit('Catalog incomplete: run 05_build_category_catalog.py until all 50 areas are present.')
    for area, entry in catalog['areas'].items():
        for priority,topic in enumerate(entry['topics']):
            conn.execute('INSERT OR IGNORE INTO topics(category,area,priority) VALUES (?,?,?)',
                         (topic, area, priority))
    conn.commit()
    return catalog


def discover_batch(client, conn, area, max_new=100, max_pages_per_topic=2):
    """Use categorymembers; page IDs form the global deduplication key.

    Prioritizes unfinished topics with fewer batches; each call visits one
    category page, allowing the caller to interleave areas fairly.
    """
    topics = conn.execute('''
        SELECT category,next_cmcontinue,pages_fetched FROM topics
        WHERE area=? AND done=0 AND pages_fetched < ?
        ORDER BY pages_fetched ASC, priority ASC LIMIT 1
    ''', (area,max_pages_per_topic)).fetchone()
    if topics is None:
        return 0, False
    category, cmcontinue, pages_fetched = topics
    params = {'list': 'categorymembers', 'cmtitle': category, 'cmtype': 'page',
              'cmnamespace': 0, 'cmlimit': min(max_new,500),
              'cmprop': 'ids|title'}
    if cmcontinue:
        params['cmcontinue'] = cmcontinue
    result = client.query(**params)
    members = result.get('query',{}).get('categorymembers',[])
    new = 0
    for member in members:
        if member.get('ns') != 0 or 'pageid' not in member:
            continue
        pid = int(member['pageid'])
        if BAD_TITLES.search(member['title']):
            continue
        cursor = conn.execute('''INSERT OR IGNORE INTO candidates
            (pageid,title,area,topic,added_at) VALUES (?,?,?,?,?)''',
            (pid,member['title'],area,category,utc_now()))
        new += cursor.rowcount
    next_continue = result.get('continue',{}).get('cmcontinue')
    new_count = pages_fetched + 1
    done = (next_continue is None)
    conn.execute('''UPDATE topics SET next_cmcontinue=?,done=?,pages_fetched=?
                    WHERE area=? AND category=?''',
                 (next_continue, int(done), new_count,area,category))
    conn.commit()
    return new, True


def clean_html(html):
    """Extract paragraphs and headings, excluding Wikipedia navigational noise."""
    soup = BeautifulSoup(html,'html.parser')
    for tag in soup(['script','style','noscript','sup','table','nav','footer','aside']):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        if tag.parent is None:
            continue
        classes = ' '.join(tag.get('class',[]))
        if BAD_CLASSES.search(classes) or tag.get('role') == 'navigation':
            tag.decompose()
    body = soup.body if soup.body else soup
    paragraphs=[]
    sections=[]
    excluded=False
    # Only text-bearing blocks, not a blanket .get_text() over everything.
    for block in body.find_all(['h1','h2','h3','h4','p']):
        if block.find_parent(['table','nav','aside','footer']):
            continue
        content = ' '.join(block.get_text(' ',strip=True).split())
        if not content:
            continue
        if block.name in ('h2','h3','h4'):
            excluded = content.casefold().rstrip(':') in BAD_HEADINGS
            if not excluded:
                sections.append({'heading': content,'paragraphs': []})
        elif block.name == 'h1':
            continue
        elif not excluded and len(content.split()) >= 8:
            paragraphs.append(content)
            if sections:
                sections[-1]['paragraphs'].append(content)
    text='\n\n'.join(paragraphs)
    return text, paragraphs, sections


def fetch_article(client, pageid, expected_title, area, topic, min_words):
    """Validate namespace/pageprops by pageid before requesting REST HTML."""
    check=client.query(prop='info|pageprops',pageids=str(pageid), redirects='')
    pages=check.get('query',{}).get('pages',[])
    if len(pages) != 1:
        return 'rejected', None, 'Missing page metadata'
    page=pages[0]
    if 'missing' in page or page.get('ns') != 0 or 'redirect' in page:
        return 'rejected', None, 'Missing, non-article namespace or redirect'
    if 'disambiguation' in page.get('pageprops',{}):
        return 'rejected',None,'Disambiguation page'
    actual_title=page.get('title',expected_title)
    if BAD_TITLES.search(actual_title):
        return 'rejected',None,'List/outline page'
    encoded=quote(actual_title.replace(' ','_'),safe='')
    response=client.get(f'{REST}/{encoded}/with_html')
    article=response.json()
    actual_id=int(article['id'])
    if actual_id != pageid:
        return 'rejected',None,f'Page ID mismatch or redirect: {actual_id}'
    text, paragraphs, sections=clean_html(article['html'])
    words=len(text.split())
    if words < min_words:
        return 'rejected',None,f'Too short: {words} words'
    latest=article.get('latest',{})
    processed={'document_id':pageid,'title':article.get('title',actual_title),
               'text':text,'paragraphs':paragraphs,'sections':sections,
               'word_count':words,'source_category':area,'source_topic':topic,
               'source_url':f'https://en.wikipedia.org/wiki/{encoded}',
               'revision_id':latest.get('id'),
               'revision_timestamp':latest.get('timestamp'),
               'retrieved_at':utc_now()}
    return 'completed',(article,processed),None


def download_one(client,conn,area,min_words,max_retries):
    row=conn.execute('''SELECT pageid,title,topic,attempts FROM candidates
        WHERE area=? AND status IN ('pending','retry') AND attempts<?
        ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END,
                 ((pageid * 2654435761) % 2147483647)
        LIMIT 1''',(area,max_retries)).fetchone()
    if row is None:
        return False
    pid,title,topic,attempts=row
    if conn.execute('SELECT 1 FROM documents WHERE pageid=?',(pid,)).fetchone():
        conn.execute("UPDATE candidates SET status='completed' WHERE pageid=?",(pid,))
        conn.commit()
        return True
    try:
        status,data,error=fetch_article(client,pid,title,area,topic,min_words)
        if status=='completed':
            raw,processed=data
            # Write both files before changing DB status; safe after interruption.
            write_json_atomic(RAW_DIR / f'{pid}.json',raw)
            write_json_atomic(PROCESSED_DIR / f'{pid}.json',processed)
            conn.execute('''INSERT OR IGNORE INTO documents
                (pageid,title,area,topic,word_count,saved_at) VALUES (?,?,?,?,?,?)''',
                (pid,processed['title'],area,topic,processed['word_count'],utc_now()))
        conn.execute('''UPDATE candidates SET status=?,attempts=attempts+1,
                        last_error=? WHERE pageid=?''',(status,error,pid))
    except StopCrawl:
        raise
    except (requests.exceptions.RequestException, ValueError, KeyError,
            TypeError, json.JSONDecodeError) as exc:
        status='retry' if attempts+1<max_retries else 'failed'
        conn.execute('''UPDATE candidates SET status=?,attempts=attempts+1,
                        last_error=? WHERE pageid=?''',
                     (status,str(exc)[:350],pid))
    conn.commit()
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--hours',type=float,default=9)
    parser.add_argument('--target',type=int,default=30000)
    parser.add_argument('--min-words',type=int,default=300)
    parser.add_argument('--delay',type=float,default=1.0)
    parser.add_argument('--max-retries',type=int,default=3)
    parser.add_argument('--max-backlog',type=int,default=500)
    parser.add_argument('--category-pages',type=int,default=3)
    parser.add_argument('--report-every',type=int,default=100)
    args=parser.parse_args()
    if args.hours<=0 or args.target<=0 or args.max_backlog<=0:
        parser.error('hours, target, max-backlog must be positive')
    RAW_DIR.mkdir(parents=True,exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True,exist_ok=True)
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(DB_PATH)
    conn.execute('PRAGMA busy_timeout=5000')
    initialize_db(conn)
    imported=import_old_documents(conn)
    catalog=load_catalog(conn)
    areas=list(catalog['areas'])
    started=time.monotonic()
    client=MeasuredWikiClient(delay=args.delay,deadline=started+args.hours*3600)
    client._persisted_breakdown = {}
    print(f'DB: {DB_PATH}\nData: {DATA_DIR}\nImported existing docs: {imported}',flush=True)
    try:
        revalidate_legacy_documents(conn, client, args.min_words)
    except StopCrawl:
        print('Time budget reached during legacy validation; can resume next run.')
    persist_request_stats(conn, client)
    print(f'Total unique docs: {current_count(conn)} / {args.target}',flush=True)
    last_report=current_count(conn)
    session_initial=current_count(conn)
    try:
        while current_count(conn)<args.target:
            did_something=False
            counts=area_counts(conn)
            backlog=area_backlogs(conn)
            # Lower counts first. This is a balanced category sampler, not BFS.
            areas.sort(key=lambda a:(counts.get(a,0),backlog.get(a,0),a))
            for area in areas:
                if time.monotonic()>=started+args.hours*3600-45:
                    raise StopCrawl('Time budget reached')
                if current_count(conn)>=args.target:
                    break
                quota=(args.target+len(areas)-1)//len(areas)
                if counts.get(area,0)>=quota:
                    continue
                if backlog.get(area,0)<args.max_backlog:
                    new,attempted=discover_batch(client,conn,area,
                        max_new=min(200,args.max_backlog),
                        max_pages_per_topic=args.category_pages)
                    did_something |= attempted
                    backlog[area]=backlog.get(area,0)+new
                processed=download_one(client,conn,area,args.min_words,args.max_retries)
                did_something |= processed
                if processed:
                    counts=area_counts(conn)
                    backlog=area_backlogs(conn)
                total=current_count(conn)
                if total-last_report>=args.report_every:
                    elapsed=max((time.monotonic()-started)/3600,1e-6)
                    print(f'[{utc_now()}] total={total}/{args.target}; '
                          f'session={elapsed:.2f}h; API calls={client.requests_made}; '
                          f'session rate={(total-session_initial)/elapsed:.0f} docs/h',flush=True)
                    last_report=total
            if not did_something:
                print('No more eligible candidates. Review category coverage; stopping safely.')
                break
    except (StopCrawl, KeyboardInterrupt) as exc:
        print(f'Paused: {exc}',flush=True)
    finally:
        counts=area_counts(conn)
        print(f'Unique documents: {current_count(conn)}',flush=True)
        print(f'API requests this run: {client.requests_made}',flush=True)
        print('By area:',counts,flush=True)
        print('API requests by purpose:', dict(client.request_breakdown), flush=True)
        print('Candidate rejection reasons:', dict(conn.execute(
            "SELECT COALESCE(last_error,'Unspecified'),COUNT(*) FROM candidates "
            "WHERE status='rejected' GROUP BY COALESCE(last_error,'Unspecified')"
        ).fetchall()), flush=True)
        persist_request_stats(conn, client)
        conn.close()

if __name__=='__main__':
    main()
