# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 Charles Durham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Summarize completed paired outcomes without treating timeouts as successes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from .common import save, load


def summarize(root):
    root = Path(root)
    rows = []
    for path in sorted((root / 'hosts').glob('*/results.jsonl')):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue  # a live snapshot may end in an incomplete line
            case = root / 'cases' / row['case_id']
            if (case / 'generation.json').exists():
                row['generation'] = load(case / 'generation.json')
            if (case / 'public.json').exists():
                cfg = load(case / 'public.json')['grammar']
                row['state_count'] = cfg['states']
                row['grammar_depth'] = cfg['grammar']['max_depth']
            rows.append(dict(row, lane=path.parent.name))
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row['lane'], row['group_id'])][row['method']] = row
    complete_pairs = [p for p in pairs.values() if set(p) == {'native_lm', 'curvature'}]
    paired = []
    for p in complete_pairs:
        lm, curvature = p['native_lm'], p['curvature']
        both = lm.get('verified') and curvature.get('verified')
        paired.append(dict(case_id=lm['case_id'], lane=lm['lane'], kind=lm['kind'], settings=lm['settings'],
            states=lm.get('state_count'), parameters=lm.get('fitted_parameters'),
            native_verified=lm.get('verified', False), curvature_verified=curvature.get('verified', False),
            native_complete=lm.get('work_complete'), curvature_complete=curvature.get('work_complete'),
            same_bank=lm.get('bank_sha256')==curvature.get('bank_sha256') if lm['kind']=='fit' else None,
            curvature_over_native_time=curvature.get('seconds', 0)/lm['seconds'] if both and lm.get('seconds') else None))
    save(root / 'paired-results.json', paired)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['phase'], row['kind'], row['method'], row['lane'])].append(row)
    summary = []
    for key, items in sorted(groups.items()):
        summary.append(dict(zip(('phase', 'kind', 'method', 'lane'), key), trials=len(items),
            verified=sum(r.get('verified', False) for r in items),
            unsupported=sum(r['status'] == 'unsupported_resource' for r in items),
            errors=sum(r['status'] in ('error', 'interrupted') for r in items),
            median_seconds_all=statistics.median(r['total_seconds'] for r in items),
            configurations=sum(r.get('configurations', 0) for r in items)))
    result = dict(summary=summary, completed_pairs=len(complete_pairs), incomplete_pairs=len(pairs)-len(complete_pairs),
                  results=rows, interpretation='Prepared fitting and blind recovery are separate strata. Same-system repeats are correlated; unsupported cases remain in the denominator. No PySR speed claim.')
    save(root / 'summary.json', result)
    lines = ['# LM tuning results', '', '| Phase | Work | Method | Lane | Verified | Unsupported | Median seconds, all |',
             '|---|---|---|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['phase']} | {r['kind']} | {r['method']} | {r['lane']} | {r['verified']}/{r['trials']} | {r['unsupported']} | {r['median_seconds_all']:.2f} |")
    lines.extend(['', result['interpretation'], ''])
    (root / 'SUMMARY.md').write_text('\n'.join(lines))
    import csv
    columns = ['case_id','lane','kind','phase','method','status','verified','state_count','grammar_depth','fitted_parameters',
               'seconds','total_seconds','requested_fits','work_complete','configurations','heldout_mse',
               'starts','candidates','iterations','toggle_width','initial_damping','damping_attempts','max_step',
               'batch_size','initial','banks','parents','wave','generation_limit']
    with (root / 'tuning.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, **row['settings']))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    summarize(args.root)
