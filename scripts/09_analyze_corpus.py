"""Local, read-only corpus audit for AMDM query generation (no API calls).

Reads processed/, clean/corpus.jsonl and private/document_mapping.jsonl.
Writes only private/corpus_analysis/ (or --output-dir). Standard library only.
The evidence rules are provisional and intentionally shared as functions so
script 10 can reuse exactly the same eligibility logic.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics

DEFAULT_ROOT = Path(os.environ.get('ADM_DATA_ROOT', r'C:\adm-hackathon-data'))
VERSION = '1.0'
# Conservative heuristic, not a linguistic sentence tokenizer. Paragraph
# boundaries are always respected; common abbreviations are not split.
SENTENCE_END = re.compile(r'''[.!?]+["'”’»\)\]]*\s+(?=["'“‘(]*[A-Z0-9])''')
ABBREVIATION = re.compile(r'(?i)(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|fig|no|vol|pp|approx|e\.g|i\.e|et al)\.|\b[A-Z]\.|(?:\b[A-Za-z]\.){2,})["\'”’»\)\]]*$')


def normalize(text):
    return ' '.join(text.split())


def digest(text):
    return hashlib.sha256(normalize(text).encode('utf-8')).hexdigest()


def word_count(text):
    return len(text.split())


def sentence_units(paragraph):
    """Return consecutive, nonempty text units with heuristic sentence ends."""
    paragraph = normalize(paragraph)
    units, start = [], 0
    for match in SENTENCE_END.finditer(paragraph):
        end = match.end()
        if ABBREVIATION.search(paragraph[start:end].rstrip()):
            continue
        piece = paragraph[start:end].strip()
        if piece:
            units.append(piece)
        start = end
    if paragraph[start:].strip():
        units.append(paragraph[start:].strip())
    return units


def get_body_sections(obj):
    """Recover lead and body from crawler's exact ordered suffix invariant.

    In 06_collect_wikipedia.py, paragraphs before the first included heading
    are NOT in sections. All subsequent accepted paragraphs are appended to
    sections[-1]. Flattened section paragraphs must therefore equal the
    suffix of the full paragraph list. No guessing on legacy/malformed data.
    """
    paragraphs, sections = obj.get('paragraphs'), obj.get('sections')
    if not isinstance(paragraphs, list) or not all(isinstance(p, str) and p.strip() for p in paragraphs):
        return 'invalid_paragraphs', [], []
    if normalize(' '.join(paragraphs)) != normalize(obj['text']):
        return 'paragraph_text_mismatch', [], []
    if not isinstance(sections, list):
        return 'missing_sections', [], []
    body = []
    for i, section in enumerate(sections):
        if not isinstance(section, dict) or not isinstance(section.get('heading'), str):
            return 'invalid_sections', [], []
        ps = section.get('paragraphs')
        if not isinstance(ps, list) or not all(isinstance(p, str) and p.strip() for p in ps):
            return 'invalid_sections', [], []
        if ps:
            body.append((i, section['heading'], [normalize(p) for p in ps]))
    if not body:
        return 'no_identifiable_body', [], []
    flat = [p for _, _, ps in body for p in ps]
    full = [normalize(p) for p in paragraphs]
    if len(flat) > len(full) or full[-len(flat):] != flat:
        return 'section_alignment_mismatch', [], []
    return 'ok', full[:-len(flat)], body


def prefix_words(paragraphs):
    """Prefix sums of sentence-unit word counts in one section."""
    prefix = [0]
    for paragraph in paragraphs:
        for unit in sentence_units(paragraph):
            prefix.append(prefix[-1] + word_count(unit))
    return prefix


def count_windows(prefix, minimum, maximum):
    """Count all contiguous sentence-unit windows in the inclusive word range.

    O(n log n), no enumeration of text spans or pairs. A future sampler can
    reuse the same prefix sums and endpoint ranges to select actual windows.
    """
    if minimum > maximum:
        return 0
    count = 0
    for start in range(len(prefix) - 1):
        lo = bisect_left(prefix, prefix[start] + minimum, start + 1)
        hi = bisect_right(prefix, prefix[start] + maximum, lo)
        count += hi - lo
    return count


def count_cross_section_pairs(counts):
    """Unordered pairs of eligible windows from distinct section entries."""
    total = seen = 0
    for count in counts:
        total += seen * count
        seen += count
    return total


def eligibility(body, evidence_min, evidence_max, short_min):
    prefixes = [prefix_words(ps) for _, _, ps in body]
    # Require each of two fragments to contribute roughly half the words.
    half_min, half_max = math.ceil(evidence_min / 2), evidence_max // 2
    single = sum(count_windows(p, evidence_min, evidence_max) for p in prefixes)
    multi_counts = [count_windows(p, half_min, half_max) for p in prefixes]
    multi = count_cross_section_pairs(multi_counts)
    # Diagnostic only: smaller windows for short queries if strict coverage
    # is inadequate. Do not mix these into the balanced strict design.
    relaxed_max = evidence_min - 1
    relaxed_single = sum(count_windows(p, short_min, relaxed_max) for p in prefixes)
    relaxed_multi = count_cross_section_pairs([
        count_windows(p, math.ceil(short_min / 2), relaxed_max // 2)
        for p in prefixes
    ])
    return dict(single_windows=single, multi_pairs=multi,
                relaxed_single_windows=relaxed_single, relaxed_multi_pairs=relaxed_multi,
                strict_single=bool(single), strict_multi=bool(multi),
                strict_both=bool(single and multi),
                short_single_possible=bool(single or relaxed_single),
                short_multi_possible=bool(multi or relaxed_multi),
                max_section_words=max((p[-1] for p in prefixes), default=0),
                sentence_units=sum(len(p)-1 for p in prefixes))


def read_jsonl(path):
    with path.open(encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError('expected object')
                yield obj
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{path}:{n}: {exc}') from exc


def load_inputs(root):
    mapping, doc_ids = {}, set()
    for row in read_jsonl(root / 'private' / 'document_mapping.jsonl'):
        wiki_id = str(int(row['wikipedia_id']))
        doc_id = row['doc_id']
        if not isinstance(doc_id, str) or not doc_id or wiki_id in mapping or doc_id in doc_ids:
            raise ValueError(f'Invalid or duplicate mapping for Wikipedia ID {wiki_id}')
        mapping[wiki_id] = doc_id
        doc_ids.add(doc_id)
    clean = {}
    for row in read_jsonl(root / 'clean' / 'corpus.jsonl'):
        doc_id = row['doc_id']
        if not isinstance(doc_id, str) or not doc_id or doc_id in clean:
            raise ValueError(f'Invalid or duplicate clean doc_id: {doc_id}')
        if not isinstance(row['text'], str) or not row['text'].strip():
            raise ValueError(f'Empty or invalid clean text: {doc_id}')
        clean[doc_id] = digest(row['text'])
    if not mapping or not clean:
        raise ValueError('Mapping or clean corpus is empty.')
    if doc_ids != set(clean):
        raise ValueError(f'Mapping/clean ID mismatch: mapping-only={len(doc_ids-set(clean))}, '
                         f'clean-only={len(set(clean)-doc_ids)}. Check script 08 outputs.')
    return mapping, clean


def percentile(values, p):
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs)-1)*p
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi]-xs[lo])*(pos-lo)


def describe(values):
    if not values:
        return {'n': 0}
    return {'n': len(values), 'min': min(values), 'mean': round(statistics.mean(values), 2),
            'std': round(statistics.pstdev(values), 2),
            **{f'p{p}': round(percentile(values,p/100),2) for p in (10,25,50,75,90,95,99)},
            'max': max(values)}


def aggregate(rows):
    verified = [r for r in rows if r['body_status'] == 'ok']
    return {'documents': len(rows), 'verified_body': len(verified),
            'raw_available': sum(r['raw_available'] for r in rows),
            'strict_single': sum(r['strict_single'] for r in rows),
            'strict_multi': sum(r['strict_multi'] for r in rows),
            'strict_both': sum(r['strict_both'] for r in rows),
            'strict_either': sum(r['strict_single'] or r['strict_multi'] for r in rows),
            'strict_single_only': sum(r['strict_single'] and not r['strict_multi'] for r in rows),
            'strict_multi_only': sum(r['strict_multi'] and not r['strict_single'] for r in rows),
            'short_single_possible': sum(r['short_single_possible'] for r in rows),
            'short_multi_possible': sum(r['short_multi_possible'] for r in rows),
            'short_single_relaxed_only': sum(r['short_single_possible'] and not r['strict_single'] for r in rows),
            'short_multi_relaxed_only': sum(r['short_multi_possible'] and not r['strict_multi'] for r in rows)}


def write_csv(path, rows):
    if not rows:
        return
    # UTF-8 BOM makes CSV easy to inspect in Windows Excel.
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--evidence-target', type=int, default=400,
                        help='Record preferred selection target for script 10; does not affect eligibility.')
    parser.add_argument('--evidence-min', type=int, default=350)
    parser.add_argument('--evidence-max', type=int, default=500)
    parser.add_argument('--short-evidence-min', type=int, default=150)
    parser.add_argument('--report-every', type=int, default=2000)
    args = parser.parse_args()
    if not 0 < args.short_evidence_min < args.evidence_min <= args.evidence_target <= args.evidence_max:
        parser.error('Require 0 < short-evidence-min < evidence-min <= evidence-target <= evidence-max.')
    if args.report_every < 0:
        parser.error('report-every cannot be negative.')
    root = args.data_root
    out = args.output_dir or root / 'private' / 'corpus_analysis'
    if not (root / 'processed').is_dir():
        parser.error(f'Missing processed directory: {root / "processed"}')
    try:
        mapping, clean = load_inputs(root)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise SystemExit(f'Cannot load corpus inputs: {exc}') from exc
    print(f'Data root: {root}\nClean corpus: {len(clean):,} documents\nOutput: {out}', flush=True)
    rows, issues, seen = [], [], set()
    paragraph_lengths, section_lengths = [], []
    paths = sorted((root / 'processed').glob('*.json'))
    for n, path in enumerate(paths, 1):
        try:
            obj = json.loads(path.read_text(encoding='utf-8-sig'))
            wiki_id = str(int(obj['document_id']))
            doc_id = mapping.get(wiki_id)
            if doc_id is None:
                issues.append({'file': path.name, 'reason': 'not_in_clean_mapping'})
                continue
            if wiki_id in seen:
                issues.append({'file': path.name, 'reason': 'duplicate_processed_id'})
                continue
            text = obj['text']
            if not isinstance(text, str) or not text.strip():
                raise ValueError('invalid text')
            if digest(text) != clean[doc_id]:
                issues.append({'file': path.name, 'reason': 'clean_text_mismatch'})
                continue
            words = word_count(text)
            status, lead, body = get_body_sections(obj)
            if status != 'ok':
                issues.append({'file': path.name, 'reason': status})
            stats = eligibility(body, args.evidence_min, args.evidence_max, args.short_evidence_min)
            body_words = sum(word_count(p) for _, _, ps in body for p in ps) if status == 'ok' else None
            lead_words = sum(map(word_count, lead)) if status == 'ok' else None
            if status == 'ok':
                paragraph_lengths.extend(word_count(p) for p in lead)
                paragraph_lengths.extend(word_count(p) for _, _, ps in body for p in ps)
                section_lengths.extend(sum(map(word_count, ps)) for _, _, ps in body)
            reported_words = obj.get('word_count')
            if reported_words != words:
                issues.append({'file': path.name, 'reason': 'reported_word_count_mismatch'})
            rows.append({'doc_id': doc_id, 'wikipedia_id': wiki_id,
                         'category': str(obj.get('source_category') or 'Unknown'),
                         'document_words': words, 'reported_word_count': reported_words,
                         'lead_words': lead_words, 'body_words': body_words,
                         'lead_paragraphs': len(lead) if status == 'ok' else None,
                         'body_paragraphs': sum(len(ps) for _, _, ps in body) if status == 'ok' else None,
                         'nonempty_body_sections': len(body) if status == 'ok' else None,
                         'body_status': status,
                         'raw_available': (root / 'raw' / f'{wiki_id}.json').is_file(), **stats})
            seen.add(wiki_id)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            issues.append({'file': path.name, 'reason': 'invalid_processed_record', 'detail': str(exc)})
        finally:
            if args.report_every and n % args.report_every == 0:
                print(f'Analyzed {n:,}/{len(paths):,} processed files...', flush=True)
    if not rows:
        raise SystemExit('No processed documents matched the clean corpus. No report written.')
    # Quantiles use ALL matching docs, not only eligible ones. Equal word
    # counts remain in the same bin; therefore group sizes may differ.
    lengths = [r['document_words'] for r in rows]
    q1, q2 = percentile(lengths, 1/3), percentile(lengths, 2/3)
    for r in rows:
        r['length_group'] = ('short' if r['document_words'] <= q1 else
                             'medium' if r['document_words'] <= q2 else 'long')
    by_category, by_length, by_joint = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        by_category[r['category']].append(r)
        by_length[r['length_group']].append(r)
        by_joint[r['category'],r['length_group']].append(r)
    categories = [{'category': k, **aggregate(v)} for k,v in sorted(by_category.items())]
    groups = [{'length_group': k, **aggregate(by_length[k])} for k in ('short','medium','long')]
    joint = [{'category': k[0], 'length_group': k[1], **aggregate(v)} for k,v in sorted(by_joint.items())]
    missing = sorted(set(mapping)-seen, key=int)
    issues.extend({'wikipedia_id': k, 'doc_id': mapping[k], 'reason': 'no_verified_processed_record'} for k in missing)
    body_rows = [r for r in rows if r['body_status']=='ok']
    report = {'version': VERSION, 'generated_at': datetime.now(timezone.utc).isoformat(),
              'config': {'evidence_min': args.evidence_min, 'evidence_max': args.evidence_max,
                         'evidence_target': args.evidence_target,
                         'multi_fragment_min': math.ceil(args.evidence_min/2),
                         'multi_fragment_max': args.evidence_max//2,
                         'short_relaxed_min': args.short_evidence_min,
                         'short_relaxed_max': args.evidence_min-1,
                         'sentence_segmentation': 'heuristic punctuation + paragraph boundaries'},
              'coverage': {'clean_documents': len(clean), 'processed_files': len(paths),
                           'matched_clean_documents': len(rows), 'missing_or_mismatched': len(missing),
                           'categories': len(categories), 'issue_counts': dict(Counter(x['reason'] for x in issues)),
                           'body_status_counts': dict(Counter(r['body_status'] for r in rows))},
              'length_group_thresholds': {'q_1_3': q1, 'q_2_3': q2},
              'statistics': {'document_words': describe(lengths),
                             'lead_words': describe([r['lead_words'] for r in body_rows]),
                             'body_words': describe([r['body_words'] for r in body_rows]),
                             'body_paragraphs': describe([r['body_paragraphs'] for r in body_rows]),
                             'nonempty_body_sections': describe([r['nonempty_body_sections'] for r in body_rows]),
                             'paragraph_words': describe(paragraph_lengths), 'section_words': describe(section_lengths),
                             'max_section_words': describe([r['max_section_words'] for r in body_rows])},
              'overall_eligibility': aggregate(rows), 'by_length_group': groups, 'by_category': categories,
              'pilot_diagnostic': {'balanced_per_length_group': 50,
                  'final_per_length_group': 500,
                  'note': 'For each length group, strict_single and strict_multi pools overlap. '
                          'For 25+25 pilot (250+250 final), check both pools and strict_either union. '
                          'This does not guarantee topic quotas, semantic quality, or final HTML coverage.'},
              'limitations': ['Eligibility is structural, not a measure of factual richness or uniqueness.',
                              'Unverified lead/body structures are excluded from eligibility, not from the clean corpus.',
                              'Section entries may be subsections of the same higher-level heading.',
                              'Relaxed short-query evidence is diagnostic only; not used in strict counts.',
                              'Raw availability is a file-existence check, not an HTML integrity or evidence check.']}
    # Include exact feasibility of disjoint allocations of single/multi within each length group.
    for g in groups:
        for name, quota in [('pilot',25),('final',250)]:
            g[f'{name}_structure_quota_feasible'] = (g['strict_single'] >= quota and g['strict_multi'] >= quota
                                                      and g['strict_either'] >= 2*quota)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'document_profiles.csv',rows)
    write_csv(out/'eligibility_by_category.csv',categories)
    write_csv(out/'eligibility_by_length.csv',groups)
    write_csv(out/'eligibility_by_category_and_length.csv',joint)
    write_json(out/'audit_issues.json',issues)
    write_json(out/'corpus_analysis_report.json',report)
    print('\nCORPUS ANALYSIS')
    print(f'Matched: {len(rows):,}/{len(clean):,}; verified body: {len(body_rows):,}; issues: {len(issues):,}')
    print(f'Length tertiles: {q1:.1f}, {q2:.1f} words')
    print(f'{"Length":<10}{"Docs":>8}{"Single":>10}{"Multi":>10}{"Both":>10}{"Either":>10}{"Pilot":>9}{"Final":>9}')
    for g in groups:
        print(f'{g["length_group"]:<10}{g["documents"]:>8,}{g["strict_single"]:>10,}{g["strict_multi"]:>10,}'
              f'{g["strict_both"]:>10,}{g["strict_either"]:>10,}'
              f'{str(g["pilot_structure_quota_feasible"]):>9}{str(g["final_structure_quota_feasible"]):>9}')
    print(f'\nReport: {out / "corpus_analysis_report.json"}')
    print('API calls: 0. Source files unchanged. Reports are PRIVATE.')
    if missing:
        print('WARNING: clean corpus coverage incomplete; inspect audit_issues.json before sampling.')


if __name__ == '__main__':
    main()
