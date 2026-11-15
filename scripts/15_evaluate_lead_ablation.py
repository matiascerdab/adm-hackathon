"""Private lead ablation: same documents, queries, and script-12 scoring.
Run from repository scripts/ beside 09_analyze_corpus.py and 12_evaluate_baselines.py.
No API calls. Never modifies corpus, mapping, or generation artifacts.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import tempfile


def load_local(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = load_local('12_evaluate_baselines.py', 'baseline12')
audit = load_local('09_analyze_corpus.py', 'audit09')
METHODS = ('boolean_overlap', 'tfidf_cosine', 'bm25')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--standard-run-dir', type=Path, required=True)
    p.add_argument('--hard-run-dir', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=base.DEFAULT_ROOT,
                   help='Corpus to evaluate (original by default; optionally moderate/strong)')
    p.add_argument('--original-root', type=Path, default=base.DEFAULT_ROOT,
                   help='Root containing original processed/ records')
    p.add_argument('--output-dir', type=Path)
    a = p.parse_args()
    root = a.data_root.resolve()
    original = a.original_root.resolve()
    expanded = root != original
    cache = root.parent / 'cache' / 'articles' if expanded else None
    out = a.output_dir or root / 'private' / 'lead_ablation'
    mapping = base.unique_rows(base.read_jsonl(root/'private'/'document_mapping.jsonl'), 'doc_id')
    original_mapping = (base.unique_rows(base.read_jsonl(original/'private'/'document_mapping.jsonl'), 'doc_id')
                        if expanded else mapping)
    titles = {d: r['title'] for d, r in mapping.items()}
    if len(set(str(int(r['wikipedia_id'])) for r in mapping.values())) != len(mapping):
        raise ValueError('Duplicate Wikipedia IDs in mapping')
    runs = {}
    source_hashes = {}
    run_ids = {}
    for name, path in [('standard', a.standard_run_dir), ('hard', a.hard_run_dir)]:
        truth, rep = base.load_queries(path)
        runs[name] = truth
        run_ids[name] = rep['run_id']
        for t in truth:
            d = t['doc_id']
            m = mapping[d]
            if t['source_title'] != m['title'] or str(t['wikipedia_id']) != str(m['wikipedia_id']):
                raise ValueError(f'Ground truth/mapping mismatch: {d}')
            h = t['source_text_sha256']
            if d in source_hashes and source_hashes[d] != h:
                raise ValueError(f'Inconsistent original source hashes: {d}')
            source_hashes[d] = h
    query_terms = {w for rows in runs.values() for t in rows for w in base.tokens(t['query'])}
    print(f'Input corpus: {root}\nQueries: ' + ', '.join(f'{k}={len(v)}' for k,v in runs.items()), flush=True)
    # Both views are materialized temporarily, in identical original order.
    with tempfile.TemporaryDirectory(prefix='adm_lead_ablation_') as tmp:
        full_path = Path(tmp)/'full.jsonl'
        body_path = Path(tmp)/'body.jsonl'
        seen, eligible = set(), set()
        excluded = Counter()
        excluded_docs = []
        full_hashes, body_hashes = {}, {}
        words_full = words_body = lead_nonempty = 0
        with full_path.open('w', encoding='utf-8') as f, body_path.open('w', encoding='utf-8') as b:
            for i, row in enumerate(base.read_jsonl(root/'clean'/'corpus.jsonl'), 1):
                d, text = row['doc_id'], row['text']
                if d in seen or d not in mapping or not isinstance(text, str) or not text.strip():
                    raise ValueError(f'Invalid corpus row: {d}')
                seen.add(d)
                h = base.digest(text)
                if d in source_hashes and h != source_hashes[d]:
                    raise ValueError(f'Original source text changed: {d}')
                pid = str(int(mapping[d]['wikipedia_id']))
                if d in original_mapping:
                    path = original/'processed'/f'{pid}.json'
                    obj = json.loads(path.read_text(encoding='utf-8-sig'))
                else:
                    if cache is None:
                        raise ValueError(f'Unexpected expansion document {d}')
                    path = cache/f'{pid}.json'
                    cached = json.loads(path.read_text(encoding='utf-8-sig'))
                    if cached['status'] != 'completed':
                        raise ValueError(f'Invalid cache record: {path}')
                    obj = cached['processed']
                if str(int(obj['document_id'])) != pid or base.digest(obj['text']) != h:
                    raise ValueError(f'Processed/clean mismatch: {d} ({path})')
                status, lead, body = audit.get_body_sections(obj)
                if status != 'ok':
                    # Only genuine unsectioned articles are excluded. Structural
                    # errors must not silently change the experimental sample.
                    if status != 'no_identifiable_body':
                        raise ValueError(f'Unexpected structure {status}: {d}')
                    excluded[status] += 1
                    excluded_docs.append({'doc_id': d, 'reason': status})
                    continue
                body_text = ' '.join(p for _, _, ps in body for p in ps)
                if not body_text.strip():
                    raise ValueError(f'Empty verified body: {d}')
                eligible.add(d)
                words_full += len(text.split())
                words_body += len(body_text.split())
                lead_nonempty += bool(lead)
                if d in source_hashes:
                    full_hashes[d] = h
                    body_hashes[d] = base.digest(body_text)
                for handle, value in [(f, text), (b, body_text)]:
                    handle.write(json.dumps({'doc_id':d, 'text':value}, ensure_ascii=False)+'\n')
                if i % 5000 == 0:
                    print(f'Checked {i:,} documents', flush=True)
        if seen != set(mapping):
            raise ValueError('Corpus/mapping ID mismatch')
        if not set(source_hashes) <= eligible:
            raise ValueError(f'Ground-truth sources without verified body: {set(source_hashes)-eligible}')
        print(f'Paired corpus: {len(eligible):,}/{len(seen):,}; excluded {sum(excluded.values()):,}; nonempty lead {lead_nonempty:,}', flush=True)
        print(f'Words removed: {words_full-words_body:,} ({(words_full-words_body)/words_full:.2%})', flush=True)
        results = []
        digests = {}
        avgdl = {}
        for view, path, hashes in [('full_matched',full_path,full_hashes), ('body_only',body_path,body_hashes)]:
            index = base.Index(base.corpus_factory(path), query_terms, False, eligible, hashes, label=view)
            digests[view] = index.corpus_digest
            avgdl[view] = index.avgdl
            for mode, truth in runs.items():
                for t in truth:
                    d = t['doc_id']
                    scores, _ = index.score(t['query'], 1.5, .75)
                    for method in METHODS:
                        r = base.rank_result(index, scores[method], index.pos[d], titles, 10)
                        r.pop('top_results')
                        results.append({'view':view,'mode':mode,'query_id':t['query_id'],
                                        'source_doc_id':d,'source_title':titles[d],
                                        **{k:t.get(k,'unknown') for k in base.GROUP_KEYS},
                                        'method':method, **r})
                print(f'{view}: evaluated {mode} {len(truth)} queries', flush=True)
            del index
    groups = defaultdict(list)
    for r in results:
        groups[(r['view'],r['mode'],r['method'])].append(r)
    overall = {v:{m:{method:base.summary(groups[(v,m,method)]) for method in METHODS}
                  for m in runs} for v in ('full_matched','body_only')}
    report = {'corpus_root':str(root),'input_documents':len(seen),'matched_documents':len(eligible),
              'excluded':dict(excluded),'nonempty_leads_removed':lead_nonempty,
              'full_words':words_full,'body_words':words_body,'run_ids':run_ids,
              'corpus_digests':digests,'avg_tokenized_length':avgdl,'overall':overall,
              'config':{'k1':1.5,'b':.75,'keep_stopwords':False,'methods':list(METHODS)},
              'limitations':['Unsectioned articles excluded from BOTH views; compare matched full vs body, not full original vs body.',
                             'Lead defined by crawler paragraph/section invariant, not manual annotation.',
                             'Original source hashes verified before removing lead; both indexes verify derived view hashes.',
                             'Same queries; no regeneration or semantic validation. Title-only omitted as unaffected.']}
    out.mkdir(parents=True,exist_ok=True)
    base.write_json(out/'ablation_report.json',report)
    base.write_csv(out/'ablation_per_query.csv',results)
    base.write_csv(out/'excluded_documents.csv',excluded_docs)
    print('\nLEAD ABLATION (PRIVATE; MATCHED CORPUS)')
    print(f'{"Mode":<10} {"Method":<18} {"Full R@1":>9} {"Body R@1":>9} {"Delta pp":>9} {"Full MRR":>9} {"Body MRR":>9} {"Body R@10":>10}')
    for mode in runs:
        for method in METHODS:
            f = overall['full_matched'][mode][method]
            b = overall['body_only'][mode][method]
            print(f'{mode:<10} {method:<18} {f["recall_at_1"]:9.4f} {b["recall_at_1"]:9.4f} '
                  f'{100*(b["recall_at_1"]-f["recall_at_1"]):+9.2f} {f["mrr"]:9.4f} {b["mrr"]:9.4f} {b["recall_at_10"]:10.4f}')
    print(f'Report: {out/"ablation_report.json"}\nAPI calls: 0. Original data unchanged.',flush=True)


if __name__ == '__main__':
    main()
