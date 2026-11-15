"""AMDM local retrieval baseline evaluation. No API calls; all outputs PRIVATE.

Evaluate accepted script-11 queries against the complete clean corpus, without
using ground truth to select retrieval candidates. Title-only retrieval is a
private leakage diagnostic, not an allowed student baseline.

Dependencies: numpy (already a pandas dependency). No external downloads.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import unicodedata
from datetime import datetime, timezone

import numpy as np

VERSION = '1.0'
DEFAULT_ROOT = Path(os.environ.get('ADM_DATA_ROOT', r'C:\adm-hackathon-data'))
TOKEN_RE = re.compile(r'[^\W_]+', re.UNICODE)
# Deliberately small, fixed, transparent English function-word list. No stemming.
STOP = frozenset('a an and are as at be been being but by can could did do does for from had has have he her hers him his how i if in into is it its may me might my no nor not of on or our ours she should so some than that the their theirs them then there these they this those through to too under up us was we were what when where which while who whom why will with would you your yours'.split())
METHODS = ('boolean_overlap', 'tfidf_cosine', 'bm25', 'title_tfidf')
GROUP_KEYS = ('document_length_group', 'query_length', 'evidence_structure', 'paraphrasing_intensity', 'source_category')


def tokens(text, keep_stopwords=False):
    out = TOKEN_RE.findall(unicodedata.normalize('NFKC', text).casefold())
    return out if keep_stopwords else [t for t in out if t not in STOP]


def digest(text):
    return hashlib.sha256(' '.join(text.split()).encode('utf-8')).hexdigest()


def read_jsonl(path):
    with path.open(encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_no}: invalid JSON') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_no}: expected object')
            yield row


def unique_rows(rows, key):
    result = {}
    for row in rows:
        k = row.get(key)
        if not isinstance(k, str) or not k or k in result:
            raise ValueError(f'Missing or duplicate {key}: {k!r}')
        result[k] = row
    return result


def load_queries(run_dir):
    candidates = unique_rows(read_jsonl(run_dir / 'candidate_queries.jsonl'), 'query_id')
    truth = unique_rows(read_jsonl(run_dir / 'private_ground_truth.jsonl'), 'query_id')
    if not candidates or set(candidates) != set(truth):
        raise ValueError('Candidate and ground-truth query IDs must match and be nonempty.')
    report = json.loads((run_dir / 'generation_report.json').read_text(encoding='utf-8-sig'))
    if report.get('accepted') != len(candidates):
        raise ValueError('Generation report accepted count does not match candidates.')
    seen_docs = set()
    for qid, q in candidates.items():
        t = truth[qid]
        if not isinstance(q.get('query'), str) or not q['query'].strip() or q['query'] != t.get('query'):
            raise ValueError(f'Query text mismatch: {qid}')
        if t.get('generation_status') != 'mechanically_accepted_pending_semantic_validation':
            raise ValueError(f'Unexpected generation status: {qid}')
        doc_id = t.get('doc_id')
        if not isinstance(doc_id, str) or doc_id in seen_docs:
            raise ValueError(f'Invalid or repeated source document: {qid}')
        seen_docs.add(doc_id)
    return list(truth.values()), report


def corpus_factory(path):
    def factory():
        for row in read_jsonl(path):
            yield row['doc_id'], row['text']
    return factory


class Index:
    """Two-pass query-restricted postings, with full-corpus DF and document norms.

    Restricting postings to query terms is an exact computational optimization:
    omitted terms have zero query weight, but still contribute to document
    lengths, DF statistics and TF-IDF normalization. No ground truth is used.
    """
    def __init__(self, factory, query_terms, keep_stopwords=False, expected_ids=None,
                 source_hashes=None, label='body'):
        self.keep_stopwords = keep_stopwords
        self.ids = []
        self.df = Counter()
        self.postings = {}
        raw_postings = {}
        lengths = []
        seen = set()
        source_hashes = source_hashes or {}
        checked_hashes = set()
        self.text_hash = hashlib.sha256()
        for i, (doc_id, text) in enumerate(factory()):
            if not isinstance(doc_id, str) or not doc_id or doc_id in seen:
                raise ValueError(f'Invalid/duplicate document ID: {doc_id!r}')
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'Empty or invalid text: {doc_id}')
            seen.add(doc_id)
            self.ids.append(doc_id)
            h = digest(text)
            self.text_hash.update((doc_id + ':' + h + '\n').encode('utf-8'))
            if doc_id in source_hashes:
                if h != source_hashes[doc_id]:
                    raise ValueError(f'Source text changed since evidence selection: {doc_id}')
                checked_hashes.add(doc_id)
            counts = Counter(tokens(text, keep_stopwords))
            lengths.append(sum(counts.values()))
            self.df.update(counts.keys())
            for term, tf in counts.items():
                if term in query_terms:
                    if term not in raw_postings:
                        raw_postings[term] = (array('I'), array('I'))
                    raw_postings[term][0].append(i)
                    raw_postings[term][1].append(tf)
            if (i + 1) % 5000 == 0:
                print(f'{label}: indexed {i+1:,} documents', flush=True)
        if not self.ids or (expected_ids is not None and seen != expected_ids):
            raise ValueError(f'{label}: corpus/mapping ID mismatch or empty corpus.')
        if checked_hashes != set(source_hashes):
            raise ValueError('Some ground-truth source documents are missing from corpus.')
        self.n = len(self.ids)
        self.pos = {d: i for i, d in enumerate(self.ids)}
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.avgdl = float(self.lengths.mean())
        if self.avgdl == 0:
            raise ValueError(f'{label}: all documents empty after tokenization.')
        self.tfidf_idf = {t: math.log((1+self.n)/(1+df))+1 for t, df in self.df.items()}
        self.norms = np.zeros(self.n, dtype=np.float64)
        # Verify the same corpus and order in pass 2, including normalized text.
        second_hash = hashlib.sha256()
        count2 = 0
        for i, (doc_id, text) in enumerate(factory()):
            if i >= self.n or doc_id != self.ids[i]:
                raise ValueError('Corpus changed between indexing passes.')
            second_hash.update((doc_id + ':' + digest(text) + '\n').encode('utf-8'))
            c = Counter(tokens(text, keep_stopwords))
            self.norms[i] = math.sqrt(sum(((1+math.log(tf))*self.tfidf_idf[t])**2 for t,tf in c.items()))
            count2 += 1
            if count2 % 5000 == 0:
                print(f'{label}: normalized {count2:,} documents', flush=True)
        if count2 != self.n or second_hash.hexdigest() != self.text_hash.hexdigest():
            raise ValueError('Corpus changed between indexing passes.')
        self.corpus_digest = self.text_hash.hexdigest()
        for t, (idx, tf) in raw_postings.items():
            self.postings[t] = (np.asarray(idx, dtype=np.int64), np.asarray(tf, dtype=np.float64))
        # All deterministic ties broken by randomized competition doc_id, not
        # source file order. Report optimistic/pessimistic ranks as well.
        self.lex_order = np.argsort(np.asarray(self.ids), kind='stable')
        self.tie_key = np.empty(self.n, dtype=np.int64)
        self.tie_key[self.lex_order] = np.arange(self.n)
        print(f'{label}: ready; {self.n:,} documents, {len(self.df):,} terms, '
              f'{len(self.postings):,} query-term postings lists', flush=True)

    def score(self, text, k1=1.5, b=0.75, title_only=False):
        q = Counter(tokens(text, self.keep_stopwords))
        if not q:
            raise ValueError('Empty tokenized query.')
        cosine = np.zeros(self.n, dtype=np.float64)
        boolean = np.zeros(self.n, dtype=np.float64)
        bm25 = np.zeros(self.n, dtype=np.float64)
        # OOV query terms ignored, as in a fixed-vocabulary vectorizer.
        qnorm2 = 0.0
        for t, qtf in q.items():
            if t not in self.tfidf_idf:
                continue
            idf = self.tfidf_idf[t]
            qw = (1+math.log(qtf))*idf
            qnorm2 += qw*qw
            idx, tf = self.postings[t]
            cosine[idx] += qw*(1+np.log(tf))*idf
            if not title_only:
                boolean[idx] += 1
                bm_idf = math.log1p((self.n-self.df[t]+0.5)/(self.df[t]+0.5))
                denom = tf + k1*(1-b+b*self.lengths[idx]/self.avgdl)
                bm25[idx] += bm_idf*tf*(k1+1)/denom
        if qnorm2:
            denom = self.norms*math.sqrt(qnorm2)
            np.divide(cosine, denom, out=cosine, where=denom>0)
        if title_only:
            return {'title_tfidf': cosine}, q
        return {'boolean_overlap': boolean, 'tfidf_cosine': cosine, 'bm25': bm25}, q


def rank_result(index, scores, target, titles, top_k):
    # Round to 12 decimal places for numerical tie stability. Rankings and
    # tie statistics use exactly this same policy.
    scores = np.round(scores, 12)
    order = np.lexsort((index.tie_key, -scores))
    rank = int(np.flatnonzero(order == target)[0])+1
    target_score = float(scores[target])
    better = int(np.count_nonzero(scores > target_score))
    equal = int(np.count_nonzero(scores == target_score))
    best_wrong = int(order[1] if order[0] == target else order[0]) if index.n>1 else None
    wrong_score = float(scores[best_wrong]) if best_wrong is not None else None
    margin = target_score-wrong_score if wrong_score is not None else None
    def item(i, r):
        d=index.ids[int(i)]
        return {'rank': r, 'doc_id': d, 'title': titles[d], 'score': float(scores[i]), 'is_source': int(i)==target}
    result = {'rank': rank, 'optimistic_rank': better+1, 'pessimistic_rank': better+equal,
              'tie_count': equal, 'source_score': target_score,
              'best_wrong_doc_id': index.ids[best_wrong] if best_wrong is not None else None,
              'best_wrong_title': titles[index.ids[best_wrong]] if best_wrong is not None else None,
              'best_wrong_score': wrong_score, 'score_margin': margin,
              'relative_margin': margin/max(abs(target_score), abs(wrong_score), 1e-12) if margin is not None else None,
              'positive_score_documents': int(np.count_nonzero(scores>0)),
              'top_results': [item(i,r) for r,i in enumerate(order[:top_k],1)]}
    for k in (1,5,10):
        result[f'recall_at_{k}'] = int(rank<=k)
        result[f'expected_recall_at_{k}_random_ties'] = max(0,min(equal,k-better))/equal
    result['reciprocal_rank'] = 1/rank
    result['expected_rr_random_ties'] = math.fsum(1/r for r in range(better+1,better+equal+1))/equal
    return result


def contains_phrase(haystack, needle):
    return bool(needle) and any(haystack[i:i+len(needle)]==needle for i in range(len(haystack)-len(needle)+1))


def summary(rows):
    if not rows:
        return {}
    n = len(rows)
    avg = lambda k: statistics.mean(r[k] for r in rows)
    margins = [r['relative_margin'] for r in rows if r['relative_margin'] is not None]
    return {'queries': n, **{f'recall_at_{k}': avg(f'recall_at_{k}') for k in (1,5,10)},
            'mrr': avg('reciprocal_rank'),
            **{f'expected_recall_at_{k}_random_ties': avg(f'expected_recall_at_{k}_random_ties') for k in (1,5,10)},
            'expected_mrr_random_ties': avg('expected_rr_random_ties'),
            'correct_at_1': sum(r['recall_at_1'] for r in rows),
            'top_score_ties': sum(r['optimistic_rank']==1 and r['tie_count']>1 for r in rows),
            'zero_source_scores': sum(r['source_score']==0 for r in rows),
            'median_source_rank': statistics.median(r['rank'] for r in rows),
            'median_relative_margin': statistics.median(margins) if margins else None}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--run-dir', type=Path, required=True, help='Script-11 experiment directory')
    p.add_argument('--output-dir', type=Path, help='Default: private/baseline_evaluation/<run-id>')
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--k1', type=float, default=1.5)
    p.add_argument('--b', type=float, default=0.75)
    p.add_argument('--keep-stopwords', action='store_true')
    a = p.parse_args()
    if not (1<=a.top_k<=100) or not (math.isfinite(a.k1) and a.k1>0 and math.isfinite(a.b) and 0<=a.b<=1):
        p.error('Require 1<=top-k<=100, k1>0, and 0<=b<=1.')
    root = a.data_root
    truth, gen_report = load_queries(a.run_dir)
    mapping = unique_rows(read_jsonl(root/'private'/'document_mapping.jsonl'), 'doc_id')
    titles = {d: r['title'] for d,r in mapping.items()}
    for d,t in titles.items():
        if not isinstance(t,str) or not t.strip():
            raise ValueError(f'Invalid title in private mapping: {d}')
    source_hashes = {}
    for t in truth:
        d=t['doc_id']
        if d not in mapping or t.get('source_title')!=titles[d] or str(t.get('wikipedia_id'))!=str(mapping[d].get('wikipedia_id')):
            raise ValueError(f'Ground truth/mapping mismatch: {t["query_id"]}')
        if not isinstance(t.get('source_text_sha256'),str) or not re.fullmatch(r'[a-f0-9]{64}',t['source_text_sha256']):
            raise ValueError(f'Invalid source hash: {d}')
        source_hashes[d]=t['source_text_sha256']
    query_terms = {w for t in truth for w in tokens(t['query'],a.keep_stopwords)}
    run_id=gen_report['run_id']
    out=a.output_dir or root/'private'/'baseline_evaluation'/f'{a.run_dir.name}_{"allwords" if a.keep_stopwords else "stopwords"}_k1{a.k1:g}_b{a.b:g}'
    if out.resolve()==a.run_dir.resolve():
        raise ValueError('Output must not overwrite the generation directory.')
    print(f'Run: {run_id}; accepted candidates: {len(truth)}; planned target: {gen_report.get("target_prefix")}',flush=True)
    print(f'Corpus: {root / "clean" / "corpus.jsonl"}; output: {out}',flush=True)
    index=Index(corpus_factory(root/'clean'/'corpus.jsonl'),query_terms,a.keep_stopwords,set(mapping),source_hashes,label='corpus')
    title_index=Index(lambda: ((d,titles[d]) for d in index.ids),query_terms,a.keep_stopwords,set(mapping),label='titles')
    per_query=[]; top_records=[]; title_diagnostics=[]
    for i,t in enumerate(truth,1):
        qid,d,q=t['query_id'],t['doc_id'],t['query']
        scores,qcounts=index.score(q,a.k1,a.b)
        ts,_=title_index.score(q,title_only=True)
        scores.update(ts)
        raw_q=tokens(q,True); raw_title=tokens(titles[d],True)
        full_title_match=contains_phrase(raw_q,raw_title)
        title_diagnostics.append({'query_id':qid,'doc_id':d,'source_title':titles[d],
                                  'full_title_token_phrase':full_title_match,
                                  'source_title_token_coverage':len(set(raw_q)&set(raw_title))/len(set(raw_title)) if raw_title else 0.0})
        base={'query_id':qid,'source_doc_id':d,'source_title':titles[d],**{k:t.get(k,'unknown') for k in GROUP_KEYS},
              'query_words':len(q.split()),'query_unique_tokens':len(qcounts),
              'query_oov_terms':sum(w not in index.df for w in qcounts),
              'boolean_and_matching_documents':int(np.count_nonzero(scores['boolean_overlap']==len(qcounts))),
              'full_title_token_phrase':full_title_match,
              'unigram_containment':t.get('metrics',{}).get('containment_1'),
              'trigram_containment':t.get('metrics',{}).get('containment_3')}
        for m in METHODS:
            idx=title_index if m=='title_tfidf' else index
            r=rank_result(idx,scores[m],idx.pos[d],titles,a.top_k)
            top=r.pop('top_results')
            per_query.append({**base,'method':m,**r})
            top_records.append({'query_id':qid,'query':q,'source_doc_id':d,'source_title':titles[d],
                                'method':m,'source_rank':r['rank'],'best_wrong_doc_id':r['best_wrong_doc_id'],
                                'best_wrong_title':r['best_wrong_title'],'results':top})
        if i%10==0 or i==len(truth):
            print(f'Evaluated {i}/{len(truth)} queries',flush=True)
    overall={m:summary([r for r in per_query if r['method']==m]) for m in METHODS}
    grouped=[]
    for k in (*GROUP_KEYS,'factorial_cell'):
        groups=defaultdict(list)
        for r in per_query:
            v=' / '.join(str(r[x]) for x in GROUP_KEYS[:4]) if k=='factorial_cell' else str(r[k])
            groups[(r['method'],v)].append(r)
        for (m,v),rows in sorted(groups.items()):
            grouped.append({'group_variable':k,'group_value':v,'method':m,**summary(rows)})
    report={'version':VERSION,'created_at':datetime.now(timezone.utc).isoformat(),'run_id':run_id,
            'generation_config':gen_report['config'],'generation_target':gen_report.get('target_prefix'),
            'accepted_queries_evaluated':len(truth),'generation_failed_or_pending':gen_report.get('target_failed_or_pending'),
            'corpus_documents':index.n,'corpus_digest':index.corpus_digest,'title_digest':title_index.corpus_digest,
            'avg_tokenized_document_length':index.avgdl,
            'config':{'tokenization':'Unicode NFKC + casefold + alphanumeric tokens; no stemming',
                      'stopwords':[] if a.keep_stopwords else sorted(STOP), 'k1':a.k1,'b':a.b,'top_k':a.top_k,
                      'tie_break':'score rounded to 12 decimals, then doc_id ascending',
                      'boolean_overlap':'OR retrieval ranked by number of distinct matched query terms; AND count diagnostic only',
                      'tfidf_cosine':'sublinear TF (1+ln tf), smoothed IDF ln((N+1)/(df+1))+1, L2 cosine; OOV ignored',
                      'bm25':'Robertson-style positive IDF ln(1+(N-df+0.5)/(df+0.5)); distinct query terms, no qtf factor',
                      'title_tfidf':'same TF-IDF on private article titles only; diagnostic, not student baseline'},
            'overall':overall,'title_diagnostics':{'full_title_phrase_count':sum(r['full_title_token_phrase'] for r in title_diagnostics)},
            'limitations':['Preliminary evaluation on internal clean extracted text including introductions, not final sanitized HTML.',
                           'Single source label per query; mechanical acceptance does not establish factual support or uniqueness.',
                           'Tiny sample; subgroup results descriptive only. No grading thresholds inferred.',
                           'Title-only uses private mapping; not a student-facing baseline. No external search.',
                           'Boolean ranking is a specific overlap baseline, not all Boolean retrieval systems.',
                           'Unmatched and zero-score documents are ranked for diagnostics; ties may dominate weak methods.']}
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'baseline_report.json',report)
    write_csv(out/'per_query_results.csv',per_query)
    write_csv(out/'metrics_by_group.csv',grouped)
    write_csv(out/'title_diagnostics.csv',title_diagnostics)
    with (out/'top_results.jsonl').open('w',encoding='utf-8') as f:
        for r in top_records: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    # Concise markdown with worst-ranked BM25 candidates for human inspection.
    md=['# Retrieval baseline results — PRIVATE','',f'{len(truth)} mechanically accepted queries; {index.n:,} searchable documents.','',
        '| Method | R@1 | R@5 | R@10 | MRR | Correct / total |','|---|---:|---:|---:|---:|---:|']
    print('\nBASELINE RESULTS (PRIVATE)')
    print(f'{"Method":<20} {"R@1":>8} {"R@5":>8} {"R@10":>8} {"MRR":>8} {"Correct":>10}')
    for m,s in overall.items():
        vals=[s[f'recall_at_{k}'] for k in (1,5,10)]
        md.append(f'| {m} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {s["mrr"]:.4f} | {s["correct_at_1"]}/{s["queries"]} |')
        print(f'{m:<20} {vals[0]:8.4f} {vals[1]:8.4f} {vals[2]:8.4f} {s["mrr"]:8.4f} {s["correct_at_1"]:>4}/{s["queries"]:<5}')
    md.extend(['','## BM25: most competitive queries','', '| Source | Rank | Best incorrect title | Relative score margin |','|---|---:|---|---:|'])
    def esc(x): return str(x).replace('|','\\|').replace('\n',' ')
    for r in sorted((x for x in per_query if x['method']=='bm25'),key=lambda x:(-x['rank'],x['relative_margin'] if x['relative_margin'] is not None else 0))[:20]:
        margin=r['relative_margin']; sm=f'{margin:.4f}' if margin is not None else 'N/A'
        md.append(f'| {esc(r["source_title"])} | {r["rank"]} | {esc(r["best_wrong_title"])} | {sm} |')
    md.extend(['','Title-only retrieval uses private titles and is a shortcut diagnostic only.','', 'All results are preliminary; no semantic validation or final grading calibration.'])
    (out/'baseline_summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(f'\nTitle phrase matches: {report["title_diagnostics"]["full_title_phrase_count"]}/{len(truth)}')
    print(f'Report: {out / "baseline_report.json"}')
    print('API calls: 0. Corpus and generation outputs unchanged. All results PRIVATE.')


if __name__=='__main__':
    main()
