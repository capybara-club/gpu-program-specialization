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
import ast
from copy import deepcopy
import itertools
import json
import math
import re
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from followup import narrow
from prepare import build
from sparse import load_profile


def problem():
    return dict(states=['x','y','z'], known_rhs={'x':'-0.1*x','y':'-0.2*y'},
                trajectories=[dict(initial=[1,1,1],times=[0,1],values=[[1,1,1],[.9,.8,.6]])])


def expand_leaves(expression, grammar):
    match = re.search(r'leaf\.([A-Za-z_0-9]+)', expression)
    if not match:
        return [expression]
    return [item for state in grammar['leaves'][match[1]]['states']
            for item in expand_leaves(expression[:match.start()] + state + expression[match.end():], grammar)]


def supports(expression):
    """Independent polynomial support, ignoring coefficient names/values."""
    def visit(n):
        if isinstance(n, ast.Name):
            return {(n.id,)}
        if isinstance(n, (ast.Attribute, ast.Constant)):
            return {()}
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            return visit(n.left) | visit(n.right)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            return {tuple(sorted(a+b)) for a in visit(n.left) for b in visit(n.right)}
        raise AssertionError(ast.dump(n))
    return frozenset(visit(ast.parse(expression,mode='eval').body))


class SparseTests(unittest.TestCase):
    def test_complete_distinct_quadratic_support(self):
        q,m = build(problem(), rows=1)
        monomials = [()] + [(s,) for s in 'xyz'] + list(itertools.combinations_with_replacement('xyz',2))
        for role in ('A','B'):
            for kind,length in [('Simple',1),('Compound',2)]:
                expressions = [e for rule in q['grammar']['rules'][f'Argument{role}{kind}']
                               for e in expand_leaves(rule,q['grammar'])]
                actual = [supports(e) for e in expressions]
                expected = {frozenset(pair) for pair in itertools.combinations(monomials,length)}
                self.assertEqual(set(actual),expected)
                self.assertEqual(len(actual),len(expected))
        self.assertIn(frozenset({('x','z'),('y',)}),set(actual))
        self.assertIn(frozenset({(),('x','y')}),set(actual))
        self.assertIn(frozenset({('x','x'),('z','z')}),set(actual))

    def test_separate_budgets_and_coefficient_streams(self):
        q,m = build(problem())
        self.assertEqual(len(m['families']),58)
        self.assertEqual(m['expected_configurations'],367202304)
        self.assertEqual(sum(m['category_configurations'].values()),m['expected_configurations'])
        for family,allocation in zip(q['grammar']['families'],m['families']):
            counts={q['grammar']['rng_banks'][s['bank']]['count'] for s in family['rng'].values()}
            self.assertEqual(counts,{allocation['coefficient_rows']})
            self.assertEqual({s['axis'] for s in family['rng'].values()},{'trial'})
            self.assertEqual(len({s['stream'] for s in family['rng'].values()}),len(family['rng']))
        profile=load_profile();profile['coefficient_rows']['double_compound']=1
        q2,m2=build(problem(),profile=profile)
        for before,after in zip(m['families'],m2['families']):
            if before['argument_class']!='double_compound':self.assertEqual(before,after)

    def test_input_and_unrelated_profile_are_preserved(self):
        p=problem();before=deepcopy(p);profile=load_profile();original=deepcopy(profile)
        q,m=build(p,profile=profile,rows=1,hint='cos(2.4*x)')
        self.assertEqual(p,before);self.assertEqual(profile,original)
        self.assertEqual(len(m['families']),20)
        self.assertEqual(q['problem']['trajectories'],p['trajectories'])
        self.assertEqual(m['public_motif'],'cos(2.4*x)')

    def test_invalid_and_oversized_requests(self):
        for kwargs in (dict(seed=-1),dict(rows=1.5),dict(dt=math.nan),dict(indices=[]),dict(indices=[0,0])):
            with self.assertRaises(ValueError):build(problem(),**kwargs)
        profile=load_profile();profile['max_configurations']=1
        with self.assertRaisesRegex(ValueError,'reserves'):build(problem(),profile=profile)
        profile=load_profile();del profile['coefficient_rows']['compound']
        with self.assertRaisesRegex(ValueError,'each argument class'):build(problem(),profile=profile)

    def test_four_way_toggles_and_sparse_state_support(self):
        p=problem();p['states']=['a','b','c','d'];p['known_rhs']={'a':'0','b':'0','c':'0'}
        # Payload shape belongs to native validation; only grammar construction is under test here.
        q,m=build(p,rows=1)
        self.assertTrue(any(l['arity']==4 for l in q['grammar']['leaves'].values()))
        expressions=[e for rule in q['grammar']['rules']['ArgumentASimple'] for e in expand_leaves(rule,q['grammar'])]
        self.assertIn(frozenset({('c',)}),{supports(e) for e in expressions})

    def test_followup_changes_only_background_ranges(self):
        q,m=build(problem(),rows=1)
        source=dict(origin={'ast_index':'7'},values=[.2,-.3],slots=[dict(name='rng.linearx'),dict(name='rng.quadratic')])
        fitted=dict(candidate_id='winner',source_origin=source['origin'],source_values=source['values'],
                    best=dict(mse=.01,parameters=[.1,.25]))
        report={'candidates':{'winner':source}};fit={'results':[fitted]}
        updated,plan=narrow(q,m,report,fit)
        self.assertEqual(updated['problem'],q['problem'])
        for key in ('rules','leaves','rng_banks','retain','expansion'):
            self.assertEqual(updated['grammar'][key],q['grammar'][key])
        for old,new in zip(q['grammar']['families'],updated['grammar']['families']):
            self.assertEqual(old['rhs'],new['rhs']);self.assertEqual(old['limits'],new['limits'])
            for name in old['rng']:
                if name not in ('linearx','quadratic'):self.assertEqual(old['rng'][name],new['rng'][name])
        self.assertEqual(plan['expected_configurations'],m['expected_configurations'])
        self.assertFalse(plan['coefficient_guidance']['structure_language_changed'])
        bad=deepcopy(fit);bad['results'][0]['source_origin']={}
        with self.assertRaisesRegex(ValueError,'provenance'):narrow(q,m,report,bad)
        bad=deepcopy(fit);bad['results'][0]['best']['parameters']=[1]
        with self.assertRaisesRegex(ValueError,'count'):narrow(q,m,report,bad)

    def test_cli_rejects_unbound_artifacts_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            (p/'request.json').write_text('{}')
            (p/'report.json').write_text('{}')
            (p/'fit.json').write_text(json.dumps({'source_report_sha256':'wrong'}))
            (p/'submitted.json').write_text(json.dumps({'request_sha256':hashlib.sha256(b'{}').hexdigest()}))
            command=[sys.executable,str(Path(__file__).with_name('followup.py')),
                     '--request',str(p/'request.json'),'--report',str(p/'report.json'),
                     '--fit',str(p/'fit.json'),'--output',str(p/'next.json')]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('exact report SHA',result.stderr)
            self.assertFalse((p/'next.json').exists())
            (p/'submitted.json').write_text(json.dumps({'request_sha256':'wrong'}))
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertIn('different request',result.stderr)
            self.assertFalse((p/'next.json').exists())


if __name__=='__main__':unittest.main()
