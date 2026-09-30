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
import importlib.util
import json
from pathlib import Path
import random
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import ode_game as game
import ode_game_probe as probe


class ProbeTests(unittest.TestCase):
    def config(self): return json.loads((ROOT/'examples/grammar_game/depth3.json').read_text())

    def test_partition_preserves_grammar_and_all_depths(self):
        config=self.config(); original=copy.deepcopy(config)
        request,coverage=probe.prepare(config)
        full,_=game.compile_search(config,768)
        self.assertEqual(config,original)
        self.assertEqual(request['grammar_search']['grammar'],full['grammar_search']['grammar'])
        self.assertEqual(request['grammar_search']['rng'],full['grammar_search']['rng'])
        self.assertEqual(request['fit']['bounds'],{p:[-1.,1.] for p in request['parameters']})
        self.assertEqual(sum(int(f['population']) for f in coverage['partitions']),2164156924)
        self.assertEqual(coverage['requested_asts'],65536)
        self.assertEqual(coverage['requested_configurations'],50331648)
        self.assertEqual(request['workflow']['screen']['system_capacity'],256)
        self.assertLessEqual(len(coverage['partitions']),64)
        rng=random.Random(18)
        def nodes(tree):return 1 if tree[0] in ('state','constant') else 1+sum(map(nodes,tree[1:]))
        for rank in [0,3,23259,23260,2164156923]+[rng.randrange(2164156924) for _ in range(1000)]:
            tree=game.unrank(config,rank);n=nodes(tree)
            parent='root-'+('x'+str(tree[1]) if tree[0]=='state' else 'p0' if tree[0]=='constant' else tree[0])
            matched=[p for p in coverage['partitions'] if p['parent_family']==parent and p['node_range'][0]<=n<=p['node_range'][1]]
            self.assertEqual(len(matched),1)
        # The previous solve's balanced 13-node quotient is eligible immediately.
        self.assertTrue(any(p['parent_family']=='root-div' and p['node_range']==[13,13] and p['requested']>0
                            for p in coverage['partitions']))

    def test_counts_and_allocation_small_language(self):
        c=self.config();c['states']=1;c['blinded_rhs']=['x0']
        c['grammar'].update(unary=['sin'],binary=['add'])
        counts={}
        def size(t):return 1 if t[0] in ('state','constant') else 1+sum(map(size,t[1:]))
        for rank in range(game.counts(c)[-1]):
            n=size(game.unrank(c,rank));counts[n]=counts.get(n,0)+1
        self.assertEqual(counts,probe.size_counts(c)[-1])
        for budget in [32,100,10000]:
            q,coverage=probe.prepare(c,ast_budget=budget)
            self.assertEqual(coverage['requested_asts'],min(budget,5552))
            self.assertTrue(all(0<p['requested']<=int(p['population']) for p in coverage['partitions']))
            probe.validate_budget(q)

    def test_scope_grouping_preserves_complete_best_record(self):
        def row(ex,mse,objective,work):
            return dict(expression=ex,mse=mse,objective_id=objective,work_id=work,
                        parameter_values=[.1,.2],families=['div'],tags={'bank_row':work})
        a=row('x1/(p0+x0)',.1,'short','old')
        b=row('x1/(p2+x0)',.01,'short','new')
        c=row('x1/(p0+x0)',.001,'long','different-objective')
        result=probe.summarize(dict(status='awaiting_revision',frontiers={'refine':[a,b,c],'screen':[a]}),['x0','x1','x2'])
        groups=result['groups_by_scope']['refine'];self.assertEqual(len(groups),2)
        best=next(g for g in groups if g['objective_id']=='short')
        self.assertEqual(best['records'],2);self.assertEqual(best['best_record'],b)
        self.assertEqual(best['child_depths'],[0,1])
        self.assertNotEqual(probe.shape('p0+p0',[])['signature'],probe.shape('p0+p1',[])['signature'])

    def test_budget_and_incomplete_work_diagnostics(self):
        q,c=probe.prepare(self.config());bad=copy.deepcopy(q)
        bad['workflow']['screen']['wall_seconds']=bad['budget']['wall_seconds']
        with self.assertRaises(ValueError):probe.validate_budget(bad)
        state=dict(status='budget_exhausted',phase_seconds={'screen':2},accounting_complete=True,
                   grammar_coverage={p['name']:{'evaluated_model_occurrences':p['requested']} for p in c['partitions']})
        rec=probe.recommend(state,q,c)
        self.assertEqual(rec['suggested_ast_budget'],163840)
        state['phase_seconds']['screen']=10
        self.assertEqual(probe.recommend(state,q,c)['action'],'review_groups_for_refinement')
        state['phase_seconds']['screen']=2
        state['accounting_complete']=False
        self.assertEqual(probe.recommend(state,q,c)['action'],'inspect_incomplete_work')
        state['status']='verified'
        self.assertEqual(probe.recommend(state,q,c)['action'],'stop_verified')


if __name__=='__main__':unittest.main()
