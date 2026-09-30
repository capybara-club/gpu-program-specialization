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
"""Read retained reports after a search; no device work or search decisions."""
from collections import Counter
import json


def collect(root):
    screen=Counter(); native=[];setup=[]
    for p in root.glob('store/searches/*/wave-*-scoring.json'):
        r=json.loads(p.read_text())
        screen['deadline_discarded_commands']+=len(r['deadline_discarded'])
        for c in r['commands']:
            screen['commands']+=1
            for key in ('coefficient_trials','rk4_steps','scored_model_occurrences',
                        'binding_variant_occurrences','padding_configurations','invalid_configurations'):
                screen[key]+=c['counts'].get(key,0)
    for p in root.glob('store/campaigns/*/commands/*/report.json'):
        r=json.loads(p.read_text())
        for item in r.get('lm_profile',[]):
            native.append(dict(item,report=str(p.relative_to(root))))
    for pattern in ('store/searches/*/commands/*/report.json','store/campaigns/*/commands/**/report.json'):
        for p in root.glob(pattern):
            r=json.loads(p.read_text())
            if r.get('setup'):setup.append(dict(r['setup'],report=str(p.relative_to(root))))
    return dict(screen=dict(screen),native_lm_profiles=native,scoring_setup=setup,
                semantics='Scheduled rollout work may exit early; native event and load spans overlap. AST occurrences are not global distinct structures.')
