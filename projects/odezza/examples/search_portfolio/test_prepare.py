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
"""Contract checks for the prepared client, without a GPU or service."""
from copy import deepcopy
import unittest

from prepare import build, tree_count
from run import feedback


def problem(n=3):
    states=[f'x{i}' for i in range(n)]
    trajectories=[dict(initial=[1.0]*n,times=[0,1],values=[[1.0]*n,[.5]*n]) for _ in range(4)]
    return dict(states=states,known_rhs={s:'0' for s in states[:-1]},trajectories=trajectories)


class PreparedTests(unittest.TestCase):
    def test_feedback_group_counts_partition_each_dimension(self):
        q,m=build(problem(),dt=.1)
        for i,f in enumerate(m['families']):
            f.update(argument_class='simple' if i%2 else 'compound',background='linear')
        families=[dict(id=f['id'],generation_stop=1,pruned=0,configuration_limited_derivations=0,
            completed_configurations=3,reserved_configurations=3,valid=2,invalid=1) for f in q['grammar']['families']]
        r=dict(status='complete',counts={},timing={},families=families,leaderboards={'families':{}},candidates={})
        summary=feedback(q,r,m)
        for groups in summary['groups'].values():
            self.assertEqual(sum(g['completed_configurations'] for g in groups.values()),60)
            self.assertEqual(sum(g['valid'] for g in groups.values()),40)
        self.assertEqual(len(summary['groups']['argument_class']),2)

    def test_finite_language_size(self):
        self.assertEqual(tree_count(4,5,4,2),31420)
        self.assertEqual(tree_count(7,5,4,2),227773)
        for n,expected in [(3,62840),(6,455546)]:
            q,m=build(problem(n),dt=.1,rows=128)
            self.assertEqual(sum(f['ordered_labelled_trees'] for f in m['families']),expected)
            self.assertEqual(len(q['grammar']['families']),20)

    def test_disjoint_toggle_coverage(self):
        for n in range(1,11):
            q,_=build(problem(n),dt=.1,depth=1,rows=1)
            g=q['grammar'];covered=[]
            for leaf in g['rules']['PortfolioStateroot']:
                if leaf.startswith('leaf.'):
                    item=g['leaves'][leaf[5:]]
                    self.assertIn(item['arity'],(2,4))
                    self.assertEqual(item['groups'],[item['states']])
                    covered.extend(item['states'])
                elif leaf!='1':covered.append(leaf)
            self.assertEqual(covered,q['problem']['states'])

    def test_joint_numeric_axis(self):
        q,_=build(problem(),dt=.1,rows=512)
        self.assertEqual({r['axis'] for r in q['grammar']['rng'].values()},{'trial'})
        self.assertEqual(len({r['stream'] for r in q['grammar']['rng'].values()}),len(q['grammar']['rng']))
        self.assertEqual(q['grammar']['rng_banks']['draws']['count'],512)

    def test_subset_preserves_entire_trajectory_and_source(self):
        p=problem();before=deepcopy(p)
        q,m=build(p,dt=.1,trajectory_indices=[3,1])
        self.assertEqual(p,before)
        self.assertEqual(q['problem']['trajectories'],[p['trajectories'][3],p['trajectories'][1]])
        self.assertEqual(m['trajectory_selection'],dict(original_count=4,indices=[3,1]))
        for indices in ([],[1,1],[-1],[4],[1.0]):
            with self.assertRaises(ValueError):build(p,dt=.1,trajectory_indices=indices)

    def test_reject_impractical_full_depth_three(self):
        with self.assertRaisesRegex(ValueError,'exceeding local ceiling'):
            build(problem(),dt=.1,depth=3)
        q,m=build(problem(),dt=.1,depth=3,sample_asts=16,rows=64)
        self.assertEqual(q['grammar']['expansion']['strategy'],'sample')
        self.assertEqual({f['coverage'] for f in m['families']},{'sampled'})

    def test_invalid_inputs(self):
        for kwargs in (dict(dt=0),dict(dt=float('nan')),dict(dt=.1,rows=1.5),
                       dict(dt=.1,seed=-1),dict(dt=.1,seed=2**64),dict(dt=.1,scale=float('inf'))):
            with self.assertRaises(ValueError):build(problem(),**kwargs)
        p=problem();p['known_rhs']={}
        with self.assertRaisesRegex(ValueError,'exactly one'):build(p,dt=.1)

    def test_feedback_does_not_overclaim_coverage(self):
        q,m=build(problem(),dt=.1)
        families=[dict(id=f['id'],generation_stop=1,pruned=0,configuration_limited_derivations=0,
            completed_configurations=1,reserved_configurations=1) for f in q['grammar']['families']]
        r=dict(status='complete',counts={},timing={},families=families,leaderboards={'families':{}},candidates={})
        self.assertTrue(feedback(q,r,m)['all_families_exhausted'])
        q['grammar']['expansion']['strategy']='sample'
        self.assertFalse(feedback(q,r)['all_families_exhausted'])
        q['grammar']['expansion']['strategy']='enumerate'
        for changed in (dict(status='cancelled'),dict(families=families[:-1])):
            self.assertFalse(feedback(q,dict(r,**changed),m)['all_families_exhausted'])
        families[0]['completed_configurations']=0
        self.assertFalse(feedback(q,r,m)['all_families_exhausted'])


if __name__=='__main__':unittest.main()
