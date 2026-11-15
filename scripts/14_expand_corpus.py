"""Private pilot corpus expansion. Reuses 06's Wikipedia client and HTML cleaner.

No query text is sent to Wikipedia. No original corpus, mapping, raw files or
crawler database is modified. Default is a local dry run. --execute discovers
one-hop related candidates, caches them and builds a separate evaluator root.
Only one process should use an experiment directory at a time.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import time

DEFAULT_ROOT = Path(os.environ.get('ADM_DATA_ROOT', r'C:\adm-hackathon-data'))
VERSION = 'v1'


def digest(text):
    return hashlib.sha256(' '.join(text.split()).encode('utf-8')).hexdigest()


def read_jsonl(path):
    with path.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def json_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def stable_seed(seed, pid, method):
    return int(hashlib.sha256(f'{seed}:{pid}:{method}'.encode()).hexdigest()[:16], 16)


def discover(client, pid, title, seed, pool):
    # morelike is a greedy keyword; use a separate request from graph discovery.
    specs = {
        'morelike': dict(list='search', srsearch='morelike:' + title.replace(' ', '_'),
                         srnamespace=0, srlimit=pool),
        'backlinks': dict(list='backlinks', blpageid=str(pid), blnamespace=0,
                          blfilterredir='nonredirects', bllimit=pool),
        'outgoing': dict(generator='links', pageids=str(pid), gplnamespace=0,
                         gpllimit=pool, prop='info|pageprops'),
    }
    result = {}
    for method, params in specs.items():
        data = client.query(**params)
        if data.get('error'):
            raise RuntimeError(f'Wikipedia {method} error: {data["error"]}')
        q = data.get('query', {})
        raw = q.get({'morelike':'search','backlinks':'backlinks','outgoing':'pages'}[method], [])
        if not isinstance(raw, list):
            raise ValueError('Expected formatversion=2 API list; check wikimedia_common.')
        seen, items = set(), []
        for row in raw:
            if 'pageid' not in row or row.get('ns') != 0 or 'missing' in row or 'redirect' in row or 'disambiguation' in row.get('pageprops', {}):
                continue
            n = int(row['pageid'])
            if n == pid or n in seen or n <= 0:
                continue
            seen.add(n)
            items.append({'pageid':n, 'title':row['title']})
        # Preserve search relevance; graph API order is not relevance-ranked.
        if method != 'morelike':
            items.sort(key=lambda r:r['pageid'])
            random.Random(stable_seed(seed,pid,method)).shuffle(items)
        result[method] = items
    return result


def candidate_order(pools, target):
    # For target=8, try six morelike, one backlink and one outgoing; continue
    # cycling to fill gaps due to original-corpus overlap or rejection.
    # The first candidate from each graph channel is tried before extra MLT.
    methods = ['morelike','backlinks','outgoing'] + ['morelike'] * max(1, target-3)
    indices = Counter(); seen = set()
    while any(indices[k]<len(pools[k]) for k in pools):
        for k in methods:
            while indices[k]<len(pools[k]):
                row = pools[k][indices[k]]; indices[k]+=1
                if row['pageid'] in seen:
                    continue
                seen.add(row['pageid'])
                yield k,row
                break


def load_inputs(root, run):
    report = load_json(run/'generation_report.json')
    truth = list(read_jsonl(run/'private_ground_truth.jsonl'))
    if report.get('accepted') != len(truth) or not truth:
        raise ValueError('Generation report/ground truth mismatch.')
    if report.get('config',{}).get('prompt_version') != 'v2-hard-v1':
        raise ValueError('Expected hard pilot run, not the standard experiment.')
    mapping_path = root/'private'/'document_mapping.jsonl'
    corpus_path = root/'clean'/'corpus.jsonl'
    mapping = {}; wiki_ids = set()
    for row in read_jsonl(mapping_path):
        doc = row['doc_id']; pid = int(row['wikipedia_id'])
        if doc in mapping or pid in wiki_ids:
            raise ValueError('Duplicate mapping ID.')
        mapping[doc]=row; wiki_ids.add(pid)
    seeds=[]; seen=set()
    for row in truth:
        d=row['doc_id']; m=mapping[d]; pid=int(m['wikipedia_id'])
        if d in seen or pid!=int(row['wikipedia_id']) or m['title']!=row['source_title']:
            raise ValueError('Invalid/repeated source mapping.')
        seen.add(d)
        seeds.append({'doc_id':d,'pageid':pid,'title':m['title'],
                      'source_text_sha256':row['source_text_sha256']})
    hashes=set(); by_doc={}; ch=hashlib.sha256()
    for row in read_jsonl(corpus_path):
        d=row['doc_id']; h=digest(row['text'])
        if d in by_doc: raise ValueError('Duplicate corpus ID.')
        by_doc[d]=h; hashes.add(h); ch.update(f'{d}:{h}\n'.encode())
    if set(by_doc)!=set(mapping): raise ValueError('Corpus/mapping ID mismatch.')
    if any(by_doc[s['doc_id']]!=s['source_text_sha256'] for s in seeds):
        raise ValueError('Source changed since generation.')
    corpus_digest=ch.hexdigest()
    expected=report['config'].get('hard_experiment',{}).get('corpus_digest')
    if expected and expected!=corpus_digest: raise ValueError('Original corpus digest differs from hard experiment.')
    return report,seeds,mapping,wiki_ids,hashes,corpus_digest


def collect(client, collector, seeds, original_ids, original_hashes, out, target, pool, min_words, seed,
            *, phases=None, schedule_target=None, on_phase=None):
    accepted={}; hash_owner={}; provenance=[]; counts=Counter(); seed_reports=[]
    phases = phases or [target]
    schedule_target = schedule_target or target
    # Finish the moderate round for EVERY seed before adding any strong-round articles.
    # Replaying the cached first round preserves both membership and ID ordering.
    state = {}
    for phase in phases:
      seed_reports=[]
      for i,s in enumerate(seeds,1):
        pid=s['pageid']; dp=out/'cache'/'discovery'/f'{pid}.json'
        if dp.exists(): pools=load_json(dp)
        else:
            pools=discover(client,pid,s['title'],seed,pool); json_write(dp,pools)
        if pid not in state:
            state[pid] = ([], Counter(), iter(candidate_order(pools,schedule_target)))
        selected, per_method, candidates = state[pid]
        while len(selected)<phase:
            try: method,row = next(candidates)
            except StopIteration: break
            n=row['pageid']
            if n in original_ids: counts['already_in_original']+=1; continue
            if collector.BAD_TITLES.search(row['title']): counts['excluded_title']+=1; continue
            fp=out/'cache'/'articles'/f'{n}.json'
            if fp.exists(): obj=load_json(fp)
            else:
                status,payload,reason=collector.fetch_article(client,n,row['title'],'Expansion','pilot_neighbor',min_words)
                if status=='completed':
                    raw,processed=payload
                    obj={'status':status,'processed':processed,'raw':raw}
                elif status=='rejected': obj={'status':status,'reason':reason}
                else: raise RuntimeError(f'Unexpected status for {n}: {status}')
                json_write(fp,obj)
            if obj['status']!='completed': counts['rejected']+=1; continue
            proc=obj['processed']; text=' '.join(proc['text'].split()); h=digest(text)
            if not text or int(proc['document_id'])!=n or not proc.get('revision_id'):
                counts['invalid_article']+=1; continue
            if h in original_hashes or (h in hash_owner and hash_owner[h]!=n):
                counts['exact_duplicate']+=1; continue
            if n not in accepted:
                # Experimental IDs intentionally separated from main corpus IDs.
                # Never release these IDs; final corpus must be rebuilt uniformly.
                doc='doc_'+hashlib.sha256(f'pilot-expansion:{seed}:{n}'.encode()).hexdigest()[:20]
                accepted[n]={'doc_id':doc,'text':text,'title':proc['title'],
                    'wikipedia_id':str(n),'revision_id':proc['revision_id'],
                    'revision_timestamp':proc.get('revision_timestamp'),
                    'source_url':proc.get('source_url'),'text_sha256':h}
                hash_owner[h]=n
            selected.append(n); per_method[method]+=1
            provenance.append({'seed_doc_id':s['doc_id'],'seed_title':s['title'],
                'candidate_doc_id':accepted[n]['doc_id'],'candidate_wikipedia_id':n,
                'candidate_title':accepted[n]['title'],'discovery_method':method,'selection_rank':len(selected)})
        seed_reports.append({'source_title':s['title'],'seed_doc_id':s['doc_id'],'selected':len(selected),'methods':dict(per_method),
                             'pool_sizes':{k:len(v) for k,v in pools.items()}})
        print(f'Round {phase}; seed {i}/{len(seeds)}: {s["title"]} — {len(selected)} related; {len(accepted)} unique additions',flush=True)
      if on_phase:
          on_phase(phase, accepted, provenance, counts, seed_reports)
    return accepted,provenance,counts,seed_reports


def export(root,out,mapping,accepted,provenance,config,counts,seed_reports):
    new_ids={r['doc_id'] for r in accepted.values()}
    if len(new_ids)!=len(accepted) or new_ids.intersection(mapping): raise ValueError('Expansion ID collision.')
    for rel in ['clean/corpus.jsonl','private/document_mapping.jsonl']:
        dest=out/rel; dest.parent.mkdir(parents=True,exist_ok=True); temp=dest.with_suffix('.tmp')
        with (root/rel).open('rb') as orig,temp.open('wb') as f:
            shutil.copyfileobj(orig,f)
            if f.tell(): f.write(b'\n')
            for r in accepted.values():
                row=({'doc_id':r['doc_id'],'text':r['text']} if rel.startswith('clean')
                     else {k:v for k,v in r.items() if k!='text'})
                f.write((json.dumps(row,ensure_ascii=False)+'\n').encode('utf-8'))
        temp.replace(dest)
    json_write(out/'private'/'neighbor_provenance.json',provenance)
    summary={**config,'created_at':datetime.now(timezone.utc).isoformat(),
             'original_documents':len(mapping),'new_documents':len(accepted),
             'expanded_documents':len(mapping)+len(accepted),'candidate_filter_counts':dict(counts),
             'seed_reports':seed_reports,'warnings':['Pilot source-targeted expansion; NOT for release.',
             'Live Wikipedia revisions may differ in date from original corpus.',
             'Related candidates are not certified negatives; inspect factual overlap.',
             'Exact normalized duplicates removed; near duplicates intentionally retained.',
             'Graph pools capped, not exhaustive. No generated query text used for discovery.']}
    json_write(out/'expansion_report.json',summary)
    print(f'Expanded corpus: {len(mapping):,} + {len(accepted):,} = {len(mapping)+len(accepted):,}',flush=True)
    print(f'Evaluator data root: {out}\nReport: {out / "expansion_report.json"}',flush=True)


def legacy_main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--per-seed',type=int,default=8)
    p.add_argument('--pool',type=int,default=40)
    p.add_argument('--min-words',type=int,default=300)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--hours',type=float,default=0.5)
    p.add_argument('--delay',type=float,default=1.0)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not (3<=a.per_seed<=30 and a.per_seed<=a.pool<=50 and a.min_words>=1 and a.hours>0 and a.delay>=1):
        p.error('Require 3<=per-seed<=30, per-seed<=pool<=50, min-words>0, hours>0, delay>=1.')
    report,seeds,mapping,original_ids,hashes,ch=load_inputs(a.data_root,a.run_dir)
    config={'version':VERSION,'parent_run_id':report['run_id'],'original_corpus_digest':ch,
            'per_seed':a.per_seed,'pool':a.pool,'min_words':a.min_words,'seed':a.seed,
            'seed_doc_ids':[s['doc_id'] for s in seeds], 'discovery':'morelike + backlinks + outgoing links'}
    tag=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    out=a.data_root/'private'/'corpus_expansion'/f'{report["run_id"]}_{tag}'
    print(f'Seeds: {len(seeds)}; original: {len(mapping):,}; at most {len(seeds)*a.per_seed} new articles')
    print(f'Output: {out}',flush=True)
    if not a.execute:
        print('Dry run: no network, no writes. Add --execute to collect.'); return
    collector=importlib.import_module('06_collect_wikipedia')
    common=importlib.import_module('wikimedia_common')
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'config.json'
    if manifest.exists() and load_json(manifest)!=config: raise ValueError('Configuration mismatch.')
    json_write(manifest,config)
    client=common.WikiClient(delay=a.delay,deadline=time.monotonic()+a.hours*3600)
    try:
        result=collect(client,collector,seeds,original_ids,hashes,out,a.per_seed,a.pool,a.min_words,a.seed)
    except common.StopCrawl:
        print('Collection time limit reached. Cached progress retained; rerun same command to resume.',flush=True)
        raise SystemExit(2)
    export(a.data_root,out,mapping,*result[:2],config,*result[2:])
    print('All outputs PRIVATE. Original corpus unchanged. No LLM API calls.')




def load_plan_inputs(root, parent, first, last):
    """Use the exact standard generator's ordering, not accepted query files."""
    generator = importlib.import_module('11_generate_queries')
    report = load_json(parent/'generation_report.json')
    cfg = report['config']
    if cfg['prompt_version'] != 'v2' or cfg['stage'] != 'pilot':
        raise ValueError('Expected standard V2 pilot parent.')
    if report['run_id'] != generator.sha(generator.canonical(cfg).encode())[:20]:
        raise ValueError('Parent run ID does not match configuration.')
    plan = root/'private'/'evidence_selection'/'pilot'/'generation_plan.jsonl'
    rows, plan_hash = generator.read_plan(plan)
    if plan_hash != cfg['plan_sha256']:
        raise ValueError('Plan hash differs from parent report.')
    if not 1 <= first <= last <= report['target_prefix'] <= len(rows):
        raise ValueError('Requested positions outside parent attempted prefix.')
    chosen = generator.ordered_plan(rows, cfg['seed'])[first-1:last]
    mapping={}; wiki_ids=set()
    for row in read_jsonl(root/'private'/'document_mapping.jsonl'):
        d=row['doc_id']; pid=int(row['wikipedia_id'])
        if d in mapping or pid in wiki_ids: raise ValueError('Duplicate mapping ID.')
        mapping[d]=row; wiki_ids.add(pid)
    seeds=[]
    for pos,row in enumerate(chosen,first):
        d=row['doc_id']; m=mapping[d]; pid=int(m['wikipedia_id'])
        if pid != int(row['wikipedia_id']) or m['title'] != row['source_title']:
            raise ValueError(f'Source mapping mismatch: {d}')
        seeds.append({'doc_id':d,'query_id':row['query_id'],'pageid':pid,'title':m['title'],
                      'source_text_sha256':row['source_text_sha256'],'plan_position':pos})
    hashes=set(); by_doc={}; ch=hashlib.sha256()
    for row in read_jsonl(root/'clean'/'corpus.jsonl'):
        d=row['doc_id']; h=digest(row['text'])
        if d in by_doc: raise ValueError('Duplicate corpus ID.')
        by_doc[d]=h; hashes.add(h); ch.update(f'{d}:{h}\n'.encode())
    if set(by_doc)!=set(mapping): raise ValueError('Corpus/mapping ID mismatch.')
    if any(by_doc[s['doc_id']]!=s['source_text_sha256'] for s in seeds):
        raise ValueError('Source changed since evidence selection.')
    return report,seeds,mapping,wiki_ids,hashes,ch.hexdigest()


def plan_main():
    p=argparse.ArgumentParser(description='Private nested corpus expansion from frozen plan positions.')
    p.add_argument('--data-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--parent-run-dir',type=Path,required=True)
    p.add_argument('--first',type=int,default=25)
    p.add_argument('--last',type=int,default=96)
    p.add_argument('--level',choices=('moderate','strong'),default='moderate')
    p.add_argument('--pool',type=int,default=40)
    p.add_argument('--min-words',type=int,default=300)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--hours',type=float,default=1.0)
    p.add_argument('--delay',type=float,default=1.0)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not (16<=a.pool<=50 and a.min_words>=1 and a.hours>0 and a.delay>=1):
        p.error('Require 16<=pool<=50, min-words>0, hours>0, delay>=1.')
    report,seeds,mapping,original_ids,hashes,ch=load_plan_inputs(a.data_root,a.parent_run_dir,a.first,a.last)
    config={'version':'v2-plan-nested','parent_run_id':report['run_id'],
            'plan_sha256':report['config']['plan_sha256'],'plan_seed':report['config']['seed'],
            'first':a.first,'last':a.last,'original_corpus_digest':ch,'levels':[8,16],
            'pool':a.pool,'min_words':a.min_words,'seed':a.seed,
            'seed_doc_ids':[s['doc_id'] for s in seeds],
            'discovery':'morelike + backlinks + outgoing; fixed 8-candidate cycle; query blind'}
    tag=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    out=a.data_root/'private'/'corpus_expansion'/f'confirmation_{a.first}_{a.last}_{tag}'
    print(f'Plan positions {a.first}–{a.last}: {len(seeds)} seeds; original: {len(mapping):,}')
    for name,n in [('moderate',8),('strong',16)]:
        print(f'{name.title()} evaluator data root: {out/name}; max additions: {len(seeds)*n}')
    print(f'Collection level: {a.level}; cache: {out / "cache"}',flush=True)
    if not a.execute:
        print('Dry run: no network, no writes. Add --execute to collect.'); return
    collector=importlib.import_module('06_collect_wikipedia')
    common=importlib.import_module('wikimedia_common')
    out.mkdir(parents=True,exist_ok=True)
    manifest=out/'config.json'
    if manifest.exists() and load_json(manifest)!=config: raise ValueError('Configuration mismatch.')
    json_write(manifest,config)
    json_write(out/'private'/'planned_seeds.json',seeds)
    client=common.WikiClient(delay=a.delay,deadline=time.monotonic()+a.hours*3600)
    def save_level(phase, accepted, provenance, counts, seed_reports):
        name='moderate' if phase==8 else 'strong'
        levelroot=out/name
        # If a moderate corpus has already been exported, refuse silent drift on replay.
        expected_ids={r['doc_id'] for r in accepted.values()}
        previous=levelroot/'expansion_report.json'
        if previous.exists():
            old=load_json(previous)
            current={r['doc_id']:digest(r['text']) for r in read_jsonl(levelroot/'clean'/'corpus.jsonl')}
            expected={r['doc_id']:r['text_sha256'] for r in accepted.values()}
            if (old['new_documents']!=len(expected_ids) or set(current)-set(mapping)!=expected_ids
                    or any(current[d]!=h for d,h in expected.items())):
                raise ValueError('Existing level differs from cached replay; refusing overwrite.')
            print(f'{name.title()} already exported and verified: {levelroot}',flush=True)
            return
        export(a.data_root,levelroot,mapping,accepted,provenance,{**config,'level':name,'per_seed':phase},counts,seed_reports)
    try:
        phases=[8] if a.level=='moderate' else [8,16]
        collect(client,collector,seeds,original_ids,hashes,out,phases[-1],a.pool,a.min_words,a.seed,
                phases=phases,schedule_target=8,on_phase=save_level)
    except common.StopCrawl:
        print('Collection time limit reached. Cached progress retained; rerun same command to resume.',flush=True)
        raise SystemExit(2)
    print('All outputs PRIVATE. Original corpus unchanged. No LLM API calls.')


if __name__=='__main__':
    import sys
    if '--parent-run-dir' in sys.argv:
        plan_main()
    else:
        legacy_main()
