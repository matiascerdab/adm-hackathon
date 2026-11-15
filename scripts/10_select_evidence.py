"""Select private, reproducible LLM evidence plans for the AMDM hackathon.

Run after 09_analyze_corpus.py. No API, no third-party dependencies, no changes
to the corpus. All outputs are private. Strict Strategy C: 20/40/40 document
length groups, balanced binary factors, one query per source document.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import re
import shutil
import tempfile

VERSION = '1.0'
AUDIT_PATH = Path(__file__).resolve().with_name('09_analyze_corpus.py')
spec = importlib.util.spec_from_file_location('corpus_analysis_09', AUDIT_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f'Cannot import {AUDIT_PATH}')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
GROUPS = ('short', 'medium', 'long')
STRUCTURES = ('single', 'multi')
INTENSITIES = ('low', 'high')
LENGTHS = ('short', 'long')
QUERY_RANGES = {'short': (40, 70, 55), 'long': (150, 250, 200)}
EXPECTED_CONFIG = {'evidence_min': 350, 'evidence_target': 400, 'evidence_max': 500,
                   'multi_fragment_min': 175, 'multi_fragment_max': 250}


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def stable_rng(*parts):
    payload = json.dumps(parts, ensure_ascii=False, separators=(',', ':'))
    return random.Random(int(hashlib.sha256(payload.encode('utf-8')).hexdigest(), 16))


def boolean(value):
    if value not in ('True', 'False'):
        raise ValueError(f'Invalid boolean in document_profiles.csv: {value!r}')
    return value == 'True'


def read_profiles(analysis_dir, mapping, clean):
    report_path = analysis_dir / 'corpus_analysis_report.json'
    profile_path = analysis_dir / 'document_profiles.csv'
    report = json.loads(report_path.read_text(encoding='utf-8-sig'))
    if report['version'] != audit.VERSION:
        raise ValueError('Script 09 version differs from audit report; rerun script 09.')
    if any(report['config'].get(k) != v for k, v in EXPECTED_CONFIG.items()):
        raise ValueError('Audit must use strict 350/400/500 evidence configuration. Rerun script 09 with defaults.')
    if report['coverage']['missing_or_mismatched'] or report['coverage']['matched_clean_documents'] != len(clean):
        raise ValueError('Audit does not cover the current complete clean corpus.')
    q1, q2 = (report['length_group_thresholds'][k] for k in ('q_1_3', 'q_2_3'))
    profiles, wiki_ids = [], set()
    with profile_path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            wiki_id = str(int(row['wikipedia_id']))
            doc_id = row['doc_id']
            if wiki_id in wiki_ids or mapping.get(wiki_id) != doc_id or doc_id not in clean:
                raise ValueError(f'Duplicate or stale profile: {wiki_id}')
            wiki_ids.add(wiki_id)
            words = int(row['document_words'])
            group = 'short' if words <= q1 else 'medium' if words <= q2 else 'long'
            if row['length_group'] != group:
                raise ValueError(f'Length-group mismatch for {doc_id}')
            single, multi = (boolean(row[k]) for k in ('strict_single', 'strict_multi'))
            both = boolean(row['strict_both'])
            if (single != bool(int(row['single_windows'])) or multi != bool(int(row['multi_pairs']))
                    or both != (single and multi)
                    or ((single or multi) and row['body_status'] != 'ok')):
                raise ValueError(f'Inconsistent evidence eligibility for {doc_id}')
            profiles.append({**row, 'wikipedia_id': wiki_id, 'document_words': words,
                             'strict_single': single, 'strict_multi': multi, 'strict_both': both,
                             'raw_available': boolean(row['raw_available']),
                             'single_windows': int(row['single_windows']), 'multi_pairs': int(row['multi_pairs'])})
    if wiki_ids != set(mapping) or len(profiles) != len(clean):
        raise ValueError('Profiles do not match current clean corpus/mapping. Rerun script 09.')
    for field in ('strict_single', 'strict_multi', 'strict_both'):
        if sum(r[field] for r in profiles) != report['overall_eligibility'][field]:
            raise ValueError(f'Profiles and report disagree: {field}')
    if sum(r['strict_single'] or r['strict_multi'] for r in profiles) != report['overall_eligibility']['strict_either']:
        raise ValueError('Profiles and report disagree on the union of eligible sources.')
    return profiles, report


def factorial_quotas(n):
    """Exactly balance the 3 binary margins in every length group.

    The requested count is a multiple of 10, so the 20/40/40 counts are even.
    Allocate the remainder in complementary pairs; no rounding drift.
    """
    if n < 10 or n % 10:
        raise ValueError('num-queries must be a positive multiple of 10 (exact Strategy C margins).')
    group_counts = dict(zip(GROUPS, (n//5, 2*n//5, 2*n//5)))
    cells = list(itertools.product(STRUCTURES, INTENSITIES, LENGTHS))
    pairs = [
        (('single', 'low', 'short'), ('multi', 'high', 'long')),
        (('single', 'high', 'long'), ('multi', 'low', 'short')),
        (('single', 'low', 'long'), ('multi', 'high', 'short')),
        (('single', 'high', 'short'), ('multi', 'low', 'long')),
    ]
    quotas = {}
    for group, count in group_counts.items():
        base, rem = divmod(count, 8)
        q = {cell: base for cell in cells}
        for a, b in pairs[:rem//2]:
            q[a] += 1
            q[b] += 1
        quotas.update({(group, *cell): v for cell, v in q.items()})
        assert sum(q.values()) == count
        for dim, values in enumerate((STRUCTURES, INTENSITIES, LENGTHS)):
            assert all(sum(v for c, v in q.items() if c[dim] == x) == count//2 for x in values)
    return quotas


def kind_of(row):
    if row['strict_both']:
        return 'both'
    return 'single' if row['strict_single'] else 'multi'


def feasible(counts, remaining):
    return (counts['single'] + counts['both'] >= remaining['single'] and
            counts['multi'] + counts['both'] >= remaining['multi'] and
            sum(counts.values()) >= sum(remaining.values()))


def choose_documents(profiles, quotas, excluded, rng):
    """Randomized, category-diverse allocation with exact structure feasibility.

    Known subject areas are soft-balanced globally, not hard quotas. Unknown
    (legacy records) stays eligible but receives 1/5 the balancing weight.
    Protects single/multi capacities and their union at every selection.
    """
    pools = {g: defaultdict(list) for g in GROUPS}
    capacities = {g: Counter() for g in GROUPS}
    remaining = {g: Counter() for g in GROUPS}
    for (group, structure, _, _), quota in quotas.items():
        remaining[group][structure] += quota
    for row in sorted(profiles, key=lambda r: r['doc_id']):
        if (row['doc_id'] in excluded or row['body_status'] != 'ok' or not row['raw_available']
                or not (row['strict_single'] or row['strict_multi'])):
            continue
        g, kind = row['length_group'], kind_of(row)
        pools[g][(row['category'], kind)].append(row)
        capacities[g][kind] += 1
    initial = {g: dict(capacities[g]) for g in GROUPS}
    slots = []
    for g in GROUPS:
        if not feasible(capacities[g], remaining[g]):
            raise ValueError(f'Not enough unused eligible documents for {g}: '
                             f'capacities={dict(capacities[g])}, quotas={dict(remaining[g])}')
        for key in sorted(pools[g]):
            rng.shuffle(pools[g][key])
        for s in STRUCTURES:
            slots.extend([(g, s)] * remaining[g][s])
    rng.shuffle(slots)
    counts, local_counts = Counter(), Counter()
    chosen = defaultdict(list)
    for g, structure in slots:
        remaining[g][structure] -= 1
        allowed_kinds = []
        for kind in (structure, 'both'):
            if capacities[g][kind] <= 0:
                continue
            capacities[g][kind] -= 1
            if feasible(capacities[g], remaining[g]):
                allowed_kinds.append(kind)
            capacities[g][kind] += 1
        categories = sorted({c for (c, k), rows in pools[g].items() if rows and k in allowed_kinds})
        if not categories:
            raise RuntimeError(f'Allocation unexpectedly failed for {g}/{structure}')
        # Integer score to avoid floating ties. Rare categories may exhaust;
        # exact structural quotas always take priority over category balance.
        def score(c):
            weight_inverse = 5 if c == 'Unknown' else 1
            return ((counts[c]+1)*weight_inverse, local_counts[g, c], rng.random())
        category = min(categories, key=score)
        keys = [(category, k) for k in allowed_kinds if pools[g][category, k]]
        key = rng.choices(keys, weights=[len(pools[g][k]) for k in keys], k=1)[0]
        row = pools[g][key].pop()
        capacities[g][key[1]] -= 1
        counts[category] += 1
        local_counts[g, category] += 1
        chosen[g, structure].append(row)
    assigned = []
    for g in GROUPS:
        for s in STRUCTURES:
            cells = [(p, q) for p in INTENSITIES for q in LENGTHS
                     for _ in range(quotas[g, s, p, q])]
            rng.shuffle(cells)
            rows = chosen[g, s]
            rng.shuffle(rows)
            assert len(cells) == len(rows)
            assigned.extend((row, s, p, q) for row, (p, q) in zip(rows, cells))
    rng.shuffle(assigned)
    assert len({r['doc_id'] for r, *_ in assigned}) == len(assigned)
    return assigned, initial


def section_units(body, lead_count):
    """Exact same sentence heuristic as 09, with paragraph provenance."""
    sections = []
    global_paragraph = lead_count
    for idx, heading, paragraphs in body:
        units, origins, prefix = [], [], [0]
        for pi, paragraph in enumerate(paragraphs):
            for si, unit in enumerate(audit.sentence_units(paragraph)):
                units.append(unit)
                origins.append({'paragraph_index': global_paragraph,
                                'section_paragraph_index': pi, 'sentence_index': si})
                prefix.append(prefix[-1] + audit.word_count(unit))
            global_paragraph += 1
        sections.append({'index': idx, 'heading': heading, 'units': units,
                         'origins': origins, 'prefix': prefix})
    return sections


def windows(section, minimum, maximum):
    prefix = section['prefix']
    result = []
    for start in range(len(prefix)-1):
        lo = bisect_left(prefix, prefix[start]+minimum, start+1)
        hi = bisect_right(prefix, prefix[start]+maximum, lo)
        result.extend((start, end, prefix[end]-prefix[start]) for end in range(lo, hi))
    return result


def pick_window(section, eligible, target, rng):
    # Soft preference for the target, never a hard threshold. Every eligible
    # window has positive probability. No retrieval score is used.
    start, end, words = rng.choices(eligible,
        weights=[1/(1+abs(w-target)/50) for _, _, w in eligible], k=1)[0]
    text = ' '.join(section['units'][start:end])
    origins = section['origins'][start:end]
    assert audit.word_count(text) == words
    return {'section_index': section['index'], 'section_heading': section['heading'],
            'sentence_unit_start': start, 'sentence_unit_end_exclusive': end,
            'paragraph_indices': sorted({o['paragraph_index'] for o in origins}),
            'start_position': origins[0], 'end_position_inclusive': origins[-1],
            'word_count': words, 'text': text, 'sha256': audit.digest(text)}


def select_passages(sections, structure, config, rng):
    if structure == 'single':
        lo, hi, target = (config[k] for k in ('evidence_min', 'evidence_max', 'evidence_target'))
        candidates = [(s, windows(s, lo, hi)) for s in sections]
        candidates = [(s, ws) for s, ws in candidates if ws]
        if not candidates:
            raise ValueError('No strict single-passage window')
        # Section-uniform sampling avoids automatically favoring long sections.
        s, ws = rng.choice(candidates)
        return [pick_window(s, ws, target, rng)]
    lo, hi = config['multi_fragment_min'], config['multi_fragment_max']
    candidates = [(s, windows(s, lo, hi)) for s in sections]
    candidates = [(s, ws) for s, ws in candidates if ws]
    if len(candidates) < 2:
        raise ValueError('No strict cross-section pair')
    selected = rng.sample(candidates, 2)
    selected.sort(key=lambda x: x[0]['index'])
    return [pick_window(s, ws, config['evidence_target']/2, rng) for s, ws in selected]


def normalized_tokens(text):
    return re.findall(r'\w+', text.casefold(), flags=re.UNICODE)


def title_occurs(title, text):
    title_tokens, words = normalized_tokens(title), normalized_tokens(text)
    if not title_tokens:
        return False
    n = len(title_tokens)
    return any(words[i:i+n] == title_tokens for i in range(len(words)-n+1))


def build_record(root, row, structure, intensity, qlength, clean, config, seed, stage, query_id):
    wiki_id, doc_id = row['wikipedia_id'], row['doc_id']
    path = root/'processed'/f'{wiki_id}.json'
    obj = json.loads(path.read_text(encoding='utf-8-sig'))
    if (str(int(obj['document_id'])) != wiki_id or audit.digest(obj['text']) != clean[doc_id]
            or audit.word_count(obj['text']) != row['document_words']
            or str(obj.get('source_category') or 'Unknown') != row['category']
            or not (root/'raw'/f'{wiki_id}.json').is_file()):
        raise ValueError(f'Selected source no longer matches audit: {doc_id}. Rerun 09.')
    status, lead, body = audit.get_body_sections(obj)
    if status != 'ok':
        raise ValueError(f'Body is not verified for {doc_id}: {status}')
    stats = audit.eligibility(body, config['evidence_min'], config['evidence_max'], 150)
    if any(stats[k] != row[k] for k in ('single_windows', 'multi_pairs')):
        raise ValueError(f'Passage counts changed for {doc_id}. Rerun 09.')
    sections = section_units(body, len(lead))
    rng = stable_rng(seed, stage, doc_id, 'evidence')
    passages = select_passages(sections, structure, config, rng)
    count = sum(p['word_count'] for p in passages)
    assert config['evidence_min'] <= count <= config['evidence_max']
    if structure == 'multi':
        assert len({p['section_index'] for p in passages}) == 2
        assert all(config['multi_fragment_min'] <= p['word_count'] <= config['multi_fragment_max'] for p in passages)
    qmin, qmax, qtarget = QUERY_RANGES[qlength]
    title = str(obj.get('title') or '')
    text = '\n\n'.join(p['text'] for p in passages)
    return {'query_id': query_id, 'doc_id': doc_id, 'wikipedia_id': wiki_id,
            'source_title': title, 'source_category': row['category'],
            'source_revision_id': obj.get('revision_id'), 'document_words': row['document_words'],
            'body_words': sum(audit.word_count(p) for _, _, ps in body for p in ps),
            'lead_words': sum(map(audit.word_count, lead)), 'document_length_group': row['length_group'],
            'eligible_single': row['strict_single'], 'eligible_multi': row['strict_multi'],
            'evidence_structure': structure, 'paraphrasing_intensity': intensity,
            'query_length': qlength, 'query_min_words': qmin, 'query_max_words': qmax,
            'query_target_words': qtarget, 'evidence_word_count': count,
            'evidence_contains_title': any(title_occurs(title, p['text']) for p in passages),
            'source_text_sha256': clean[doc_id], 'evidence_sha256': audit.digest(text),
            'passages': passages, 'evidence_text': text}


def load_exclusions(paths, clean):
    excluded, sources = set(), []
    for path in paths:
        ids = set()
        for row in audit.read_jsonl(path):
            doc_id = row['doc_id']
            if doc_id not in clean or doc_id in ids:
                raise ValueError(f'Invalid or duplicate source in exclusion plan {path}: {doc_id}')
            ids.add(doc_id)
        if not ids:
            raise ValueError(f'Empty exclusion plan: {path}')
        excluded.update(ids)
        sources.append({'path': str(path.resolve()), 'documents': len(ids), 'sha256': file_hash(path)})
    return excluded, sources


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')


def write_preview(path, records):
    # One actual example per nonempty factorial cell; all details PRIVATE.
    seen = set()
    lines = ['# PRIVATE — Evidence review', '', 'Not for students or publication.', '',
             'One example per nonempty factorial cell. Inspect factual richness, coherence and title clues. ',
             'No queries have been generated yet.', '']
    for r in sorted(records, key=lambda x: (GROUPS.index(x['document_length_group']), x['evidence_structure'],
                                             x['paraphrasing_intensity'], x['query_length'], x['query_id'])):
        cell = tuple(r[k] for k in ('document_length_group', 'evidence_structure', 'paraphrasing_intensity', 'query_length'))
        if cell in seen:
            continue
        seen.add(cell)
        lines.extend(['## ' + ' / '.join(cell), '', f"Source: {r['source_title']} ({r['doc_id']})",
                      f"Category: {r['source_category']} | Evidence: {r['evidence_word_count']} words | "
                      f"Title occurrence: {r['evidence_contains_title']}", ''])
        for p in r['passages']:
            lines.extend([f"### Section {p['section_index']}: {p['section_heading']} ({p['word_count']} words)",
                          '', p['text'], ''])
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return len(seen)


def save_outputs(out, records, report):
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}. Use a new --output-dir; do not overwrite a private plan.')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.selection-', dir=out.parent))
    try:
        write_jsonl(tmp/'generation_plan.jsonl', records)
        audit.write_csv(tmp/'selected_documents.csv', [
            {k: v for k, v in r.items() if k not in ('passages', 'evidence_text')} for r in records])
        report['preview_examples'] = write_preview(tmp/'evidence_preview.md', records)
        report['generation_plan_sha256'] = file_hash(tmp/'generation_plan.jsonl')
        audit.write_json(tmp/'selection_report.json', report)
        if out.exists():
            raise FileExistsError(f'Output already exists: {out}')
        tmp.rename(out)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=audit.DEFAULT_ROOT)
    parser.add_argument('--analysis-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--stage', choices=('pilot', 'final'), default='pilot')
    parser.add_argument('--num-queries', type=int, help='Default: 150 pilot, 1500 final. Multiple of 10.')
    parser.add_argument('--seed', type=int, help='Default: 42 pilot, 123 final.')
    parser.add_argument('--exclude-plan', action='append', type=Path, default=[],
                        help='Private generation_plan.jsonl to exclude; repeat for additional experiments.')
    args = parser.parse_args()
    root = args.data_root.resolve()
    private = (root/'private').resolve()
    analysis_dir = (args.analysis_dir or private/'corpus_analysis').resolve()
    out = (args.output_dir or private/'evidence_selection'/args.stage).resolve()
    n = args.num_queries if args.num_queries is not None else (150 if args.stage == 'pilot' else 1500)
    seed = args.seed if args.seed is not None else (42 if args.stage == 'pilot' else 123)
    if not out.is_relative_to(private) or out == private:
        parser.error('Output must be a new directory inside data-root/private.')
    if out.exists():
        parser.error(f'Output directory already exists: {out}. Use a new --output-dir to avoid overwriting.')
    try:
        quotas = factorial_quotas(n)
        mapping, clean = audit.load_inputs(root)
        profiles, audit_report = read_profiles(analysis_dir, mapping, clean)
        exclude_paths = [p.resolve() for p in args.exclude_plan]
        if args.stage == 'final':
            default_pilot = private/'evidence_selection'/'pilot'/'generation_plan.jsonl'
            if default_pilot.is_file():
                exclude_paths.append(default_pilot.resolve())
            elif not exclude_paths:
                raise ValueError('Final stage requires the pilot plan or an explicit --exclude-plan.')
        exclude_paths = sorted(set(exclude_paths))
        excluded, exclusion_sources = load_exclusions(exclude_paths, clean)
        rng = stable_rng(seed, args.stage, 'document_selection')
        assigned, initial = choose_documents(profiles, quotas, excluded, rng)
        print(f'Data root: {root}\nStage: {args.stage}; seed: {seed}; planned queries: {n:,}')
        print(f'Corpus: {len(clean):,}; excluded prior source documents: {len(excluded):,}')
        print(f'Output: {out}', flush=True)
        records = []
        # Namespace IDs by the complete assignment, so a different plan with
        # the same seed does not recycle query IDs.
        assignment_fingerprint = hashlib.sha256(json.dumps(
            [(r['doc_id'], s, p, q) for r, s, p, q in assigned],
            separators=(',', ':')).encode('utf-8')).hexdigest()
        qid_rng = stable_rng(seed, args.stage, assignment_fingerprint, 'query_ids')
        for i, (row, structure, intensity, qlength) in enumerate(assigned, 1):
            query_id = f'q_{qid_rng.getrandbits(128):032x}'
            records.append(build_record(root, row, structure, intensity, qlength, clean,
                                        audit_report['config'], seed, args.stage, query_id))
            if i % 100 == 0 or i == n:
                print(f'Selected evidence: {i:,}/{n:,}', flush=True)
        actual = Counter((r['document_length_group'], r['evidence_structure'], r['paraphrasing_intensity'],
                          r['query_length']) for r in records)
        if any(actual[k] != q for k, q in quotas.items()) or len(records) != n:
            raise RuntimeError('Factorial quota verification failed.')
        if (len({r['doc_id'] for r in records}) != n or len({r['query_id'] for r in records}) != n
                or any(r['doc_id'] in excluded for r in records)):
            raise RuntimeError('Duplicate or excluded document/query ID detected.')
        counts = Counter(r['source_category'] for r in records)
        report = {'version': VERSION, 'generated_at': datetime.now(timezone.utc).isoformat(),
                  'stage': args.stage, 'seed': seed, 'num_queries': n, 'api_calls': 0,
                  'searchable_corpus_documents': len(clean), 'unique_sources': n,
                  'excluded_prior_sources': len(excluded), 'exclusion_plans': exclusion_sources,
                  'audit_report_sha256': file_hash(analysis_dir/'corpus_analysis_report.json'),
                  'profiles_sha256': file_hash(analysis_dir/'document_profiles.csv'),
                  'script_09_sha256': file_hash(AUDIT_PATH), 'script_10_sha256': file_hash(Path(__file__)),
                  'evidence_config': EXPECTED_CONFIG, 'length_group_thresholds': audit_report['length_group_thresholds'],
                  'available_by_group_and_kind_before_sampling': initial,
                  'by_document_length': dict(Counter(r['document_length_group'] for r in records)),
                  'by_query_length': dict(Counter(r['query_length'] for r in records)),
                  'by_evidence_structure': dict(Counter(r['evidence_structure'] for r in records)),
                  'by_paraphrasing': dict(Counter(r['paraphrasing_intensity'] for r in records)),
                  'by_category': dict(sorted(counts.items())),
                  'known_categories_represented': len(set(counts)-{'Unknown'}),
                  'known_categories_not_represented': sorted({r['category'] for r in profiles}-set(counts)-{'Unknown'}),
                  'factorial_cells': [dict(document_length_group=g, evidence_structure=s,
                      paraphrasing_intensity=p, query_length=q, target=v, selected=actual[g,s,p,q])
                      for (g,s,p,q),v in quotas.items()],
                  'evidence_words': audit.describe([r['evidence_word_count'] for r in records]),
                  'title_occurrence_flags': sum(r['evidence_contains_title'] for r in records),
                  'selection_policy': {'subject_areas': 'Soft balance across known categories; Unknown weight 0.2. No hard topic quotas.',
                      'evidence': 'Uniform eligible section or section pair; randomized windows weighted softly toward 400 total words.',
                      'intro': 'Excluded using exact script 09 verified suffix alignment.',
                      'sentence_indices': 'Zero-based, end exclusive for section units, end inclusive for last paragraph/sentence position.'},
                  'limitations': ['Structural eligibility does not guarantee semantic quality or uniqueness.',
                      'Short documents eligible for both structures are rare; structure comparisons are observational.',
                      'Title occurrences in evidence are flagged, not removed. Do not send source title, IDs or section headings to the generator.',
                      'Final sanitized HTML must be checked for evidence availability.',
                      'One source per plan; final excludes supplied prior plans. Pass every additional experiment plan explicitly.']}
        save_outputs(out, records, report)
        print('\nSELECTION SUMMARY')
        print(f'{"Group":<10}{"Single":>10}{"Multi":>10}{"Short q":>10}{"Long q":>10}{"Total":>10}')
        for g in GROUPS:
            rr = [r for r in records if r['document_length_group'] == g]
            print(f'{g:<10}{sum(r["evidence_structure"]=="single" for r in rr):>10,}'
                  f'{sum(r["evidence_structure"]=="multi" for r in rr):>10,}'
                  f'{sum(r["query_length"]=="short" for r in rr):>10,}'
                  f'{sum(r["query_length"]=="long" for r in rr):>10,}{len(rr):>10,}')
        print(f'Known categories represented: {report["known_categories_represented"]}; '
              f'title occurrence flags: {report["title_occurrence_flags"]}; '
              f'mean evidence words: {report["evidence_words"]["mean"]}')
        print(f'Plan: {out/"generation_plan.jsonl"}\nReport: {out/"selection_report.json"}')
        print('API calls: 0. All outputs PRIVATE. Corpus unchanged.')
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise SystemExit(f'Evidence selection failed: {exc}') from exc


if __name__ == '__main__':
    main()
