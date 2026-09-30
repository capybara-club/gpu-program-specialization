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
import copy
import json
from pathlib import Path
import tempfile
import unittest
from benchmarks.search_study.prepare import select, variants
from benchmarks.search_study.report import paired, rows_from

class StudyTests(unittest.TestCase):
    def test_selection_keeps_matching_ancestry_and_distinct_biology(self):
        manifests={k:{'cases':[]} for k in ('mdbench','odebench-v2','biological-v2')}
        for n in (1,2,3,4):
            for corpus,prefix in [('mdbench','mdbench'),('odebench-v2','odebench')]:
                manifests[corpus]['cases'].append(dict(case_id=f'{prefix}-{n:03d}-clean-rhs-00',state_count=n,
                    unknown_rhs='x0',status='prepared',noise='none'))
        for n in (3,5):
            for i in range(3):
                for rhs in range(2):
                    manifests['biological-v2']['cases'].append(dict(case_id=f'bio-N{n}-model{i}-rhs-{rhs:02d}',
                        state_count=n,unknown_rhs=f'x{rhs}',status='prepared',noise='none'))
        original=copy.deepcopy(manifests);chosen=select(manifests)
        self.assertEqual(len(chosen),12);self.assertEqual(original,manifests)
        for i in (0,2,4,6):self.assertEqual(chosen[i][1]['case_id'].split('-')[1],chosen[i+1][1]['case_id'].split('-')[1])
        self.assertEqual(len({c['case_id'].rsplit('-rhs-',1)[0] for _,c in chosen[8:]}),4)

    def test_variants_preserve_budget_and_change_only_declared_policy(self):
        first=variants(0);rotated=variants(1)
        self.assertEqual(first[1:]+first[:1],rotated)
        self.assertEqual(first[0]['settings'],first[1]['settings'])
        a=dict(first[0]['settings']);b=dict(first[2]['settings']);a.pop('observations');b.pop('observations')
        self.assertEqual(a,b);self.assertEqual(a['target_mse'],1e-6)

    def test_paired_report_counts_late_verification_as_budget_miss(self):
        def row(case,policy,verified,seconds):
            return dict(lane='a',case_id=case,trial_id=policy,verified=verified,total_seconds=seconds,settings={'seconds':10})
        rows=[row('1','base',True,3),row('1','new',True,12),row('2','base',False,4),row('2','new',True,5),row('3','base',True,2)]
        result,=paired(rows,'base');self.assertEqual(result['paired_cases'],2)
        self.assertEqual((result['gained'],result['lost']),(1,1))
        self.assertEqual(result['baseline_mean_penalty'],6.5);self.assertEqual(result['policy_mean_penalty'],7.5)

    def test_duplicate_results_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'hosts/lane/results.jsonl';p.parent.mkdir(parents=True)
            row=json.dumps(dict(group_id='g',trial_id='t'));p.write_text(row+'\n'+row+'\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'):rows_from(root)

if __name__=='__main__':unittest.main()
