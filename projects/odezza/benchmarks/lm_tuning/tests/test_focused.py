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
from benchmarks.lm_tuning.batching import batches
from benchmarks.lm_tuning.focused_design import search_variants, fit_variants, specifications, group, reserved_seconds, calibration_subset
from benchmarks.lm_tuning.focused import summarize
from benchmarks.lm_tuning.common import save
from benchmarks.lm_tuning.work_report import collect


class FocusedTests(unittest.TestCase):
    def test_calibration_always_contains_unlabelled_planted_structure(self):
        payload=dict(candidates=[dict(id=str(i),program_hex=str(i)) for i in range(64)])
        selected=calibration_subset(payload,'63')
        self.assertEqual(len(selected['candidates']),8)
        self.assertIn(dict(id='63',program_hex='63'),selected['candidates'])
        self.assertEqual(len(payload['candidates']),64)
        self.assertTrue(all(set(c)=={'id','program_hex'} for c in selected['candidates']))
    def test_bank_and_queue_partition_preserves_work(self):
        packs=[dict(id=i,permutation_count=v) for i,v in enumerate([1,4,2,4,1])]
        for limit in (1,2,16):
            bs=list(batches(packs,64,dict(submission_packs=limit,buffer_fit_limit=320)))
            self.assertEqual([p for b in bs for p in b],packs)
            self.assertTrue(all(len(b)<=limit for b in bs))
            self.assertTrue(all(sum(64*p['permutation_count'] for p in b)<=320 for b in bs))
        # Legacy two-pack submissions must not be narrowed by a new soft cap.
        big=[dict(permutation_count=4)]*2
        self.assertEqual(len(list(batches(big,65536,{}))),1)
        with self.assertRaisesRegex(ValueError,'million'):
            list(batches(big,1048576,{}))

    def test_search_contrasts_preserve_initial_work(self):
        policies={p['trial_id']:p for p in search_variants()}
        for name in ('lm_baseline','more_structures','more_coefficients','fewer_bindings'):
            s=policies[name]['settings']
            self.assertEqual(s['initial']*s['banks']*s['leaf_toggle_width'],268435456)
        for name in ('lm_baseline','fit_more_candidates','fit_fewer_candidates'):
            s=policies[name]['settings'];self.assertEqual(s['candidates']*s['starts'],256)
        self.assertEqual(len({p['trial_id'] for p in policies.values()}),8)

    def test_plan_cells_and_budgets(self):
        specs=specifications();self.assertEqual(len(specs),36)
        self.assertEqual(len({(s['states'],s['parameters']) for s in specs}),9)
        self.assertTrue(all(s['dependencies']+s['parameters']<=2**s['depth'] for s in specs))
        self.assertEqual(len(fit_variants()),24)
        worst=9*reserved_seconds(group('x',{},'fit',fit_variants(),0,1,'calibration'))+9*reserved_seconds(group('x',{},'search',search_variants(),0,1,'recovery'))
        self.assertLess(worst,9*3600-120)

    def test_policy_groups_rotate_without_seed_or_identity_collisions(self):
        vs=search_variants();a=group('x',{'public':'only'},'search',vs,0,31,'recovery')
        b=group('x',{'public':'only'},'search',vs,1,31,'recovery')
        self.assertNotEqual(a['trials'][0]['trial_id'],b['trials'][0]['trial_id'])
        self.assertTrue(all(t['settings']['seed']==31 for t in a['trials']))
        self.assertTrue(all('seed' not in t['settings'] for t in vs))
        self.assertEqual(len({t['trial_id'] for t in a['trials']}),8)

    def test_summary_keeps_same_backend_policies_and_failed_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);g=group('x',{},'search',search_variants(),0,1,'recovery')
            save(root/'groups'/(g['group_id']+'.json'),g)
            save(root/'plan.json',dict(schedule=[dict(lane='ada_0',group_id=g['group_id'])]))
            save(root/'cases/x/origin.json',dict(specification=dict(states=6,parameters=3,view='dense')))
            rs=[]
            for t in g['trials'][:2]:
                rs.append(dict(case_id='x',group_id=g['group_id'],kind='search',trial_id=t['trial_id'],settings=t['settings'],
                    status='budget_exhausted',verified=False,total_seconds=240.1))
            out=root/'hosts/ada_0/results.jsonl';out.parent.mkdir(parents=True);out.write_text(''.join(json.dumps(r)+'\n' for r in rs))
            result=summarize(root)
            self.assertEqual(result['completed_trials'],2);self.assertEqual(result['unreported_trials'],6)
            self.assertEqual(len(result['summary']),2)
            self.assertTrue(all(r['mean_budget_penalized_seconds']==240 for r in result['summary']))
            out.write_text(out.read_text()+json.dumps(rs[0])+'\n')
            with self.assertRaisesRegex(ValueError,'duplicate'):summarize(root)

    def test_search_work_report_uses_completed_commands_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save(root/'store/searches/x/wave-0000-scoring.json',dict(deadline_discarded=[{}],commands=[
                dict(counts=dict(coefficient_trials=400,rk4_steps=3200,scored_model_occurrences=1))]))
            save(root/'store/campaigns/c/commands/b/report.json',dict(lm_profile=[dict(fit_count=8,compiled_parameter_count=3)]))
            r=collect(root)
            self.assertEqual(r['screen']['coefficient_trials'],400)
            self.assertEqual(r['screen']['deadline_discarded_commands'],1)
            self.assertEqual(len(r['native_lm_profiles']),1)


if __name__=='__main__':unittest.main()
