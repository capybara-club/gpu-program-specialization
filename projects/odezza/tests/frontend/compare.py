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
"""Compare native frontend output with the supplied Python compiler oracle."""
import json
from pathlib import Path
import subprocess
import sys
from odegrammar.compiler import Compiler
from odezza.grammar.lowering import lower_system

binary = sys.argv[1]
root = Path(__file__).resolve().parents[2]
for name in ['adaptation_search', 'adaptation_toggles', 'gene_circuit', 'gene_circuit_constant_grid', 'enzyme_pathway', 'constant_rng_product', 'toggles_all', 'million']:
    path = root/'examples'/'grammar'/f'{name}.json'
    request = json.loads(path.read_text())
    compiler = Compiler(request)
    skeletons = {}
    expected = []
    family = compiler.families[0]['id']
    for record in compiler.records():
        if record['type'] == 'skeleton':
            skeletons[record['id']] = record
        if record['type'] == 'variant' and record['family_id'] == family:
            layout, programs = lower_system(skeletons[record['skeleton_id']], record, request['states'])
            expected.append((len(layout.slots), layout.toggle_bits, layout.numeric_count, [p.hex() for p in programs]))
            if len(expected) == 32:
                break
    output = subprocess.run([binary, str(path), '0', str(len(expected))], text=True, capture_output=True, check=True)
    actual = []
    for line in output.stdout.splitlines():
        fields = line.split()
        actual.append((int(fields[1]), int(fields[2]), int(fields[3]), fields[4:]))
    assert actual == expected, (name, next(((a,b) for a,b in zip(actual, expected) if a!=b), (len(actual),len(expected))))
    print(f'{name}: {len(actual)} native programs match')
