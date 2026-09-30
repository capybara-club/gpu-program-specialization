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
"""Exercise SQLite artifact identity and coalesced cold compilation on CUDA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
    root = Path.cwd(); executable = root/'build/frontend/gpu_test'
    env = dict(os.environ, CUDA_MODULE_LOADING='EAGER'); rows = []
    cache = Path(tempfile.mkdtemp(prefix='odezza-cache-gpu-'))
    command = [str(executable), 'examples/grammar/submit.json', str(cache)]

    def collect(process, label):
        output, _ = process.communicate(timeout=120)
        assert process.returncode == 0, output
        passes = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
        assert len(passes) == 2 and all(row['mse'] < 1e-10 for row in passes)
        assert passes[1]['cache_misses'] == 0 and passes[1]['nvrtc_seconds'] == 0
        rows.append(dict(name=label, passes=passes))
        return passes

    def start():
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

    # Both processes open the same initially empty DB and request both shapes.
    processes = [start(), start()]
    concurrent = [collect(proc, 'concurrent_cold') for proc in processes]
    assert sum(passes[0]['cache_misses'] for passes in concurrent) == 2, concurrent
    collect(start(), 'fully_warm')
    db = sqlite3.connect(cache/'templates.sqlite3')
    for kind in ('generator_identity', 'checksum'):
        key, blob = db.execute('SELECT key,artifact FROM artifacts ORDER BY key LIMIT 1').fetchone()
        damaged = bytearray(blob); damaged[40] ^= 1
        if kind == 'generator_identity':
            # Outer digest remains valid; the core must still reject old source identity.
            db.execute('UPDATE artifacts SET artifact=?,sha256=? WHERE key=?',
                       (damaged, hashlib.sha256(damaged).digest(), key))
        else:
            db.execute('UPDATE artifacts SET artifact=? WHERE key=?', (damaged, key))
        db.commit()
        result = collect(start(), kind)
        assert result[0]['cache_misses'] == 1 and result[0]['nvrtc_seconds'] > 0
    db.close()
    Path(a.output).write_text(json.dumps(dict(passed=True, runs=rows, cache=str(cache)), indent=2)+'\n')
    print('PASS concurrent cold compilation, warm reuse, generator identity, checksum repair')

if __name__ == '__main__':
    main()
