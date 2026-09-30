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
"""Real CUDA lifetime regressions. Run in the isolated trial checkout on a GPU host.

--failure requires the existing one-shot LD_PRELOAD injection shim.
"""
import argparse
import ctypes as C
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--failure', action='store_true')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    lib = C.CDLL(str(root / 'build/runtime/libodezza_runtime.so'))
    pointer = C.c_void_p
    lib.odz_runtime_create.argtypes = [C.c_uint, C.c_char_p, C.POINTER(pointer)]
    lib.odz_runtime_attempt_inputs.argtypes = [pointer]
    lib.odz_job_create.argtypes = [C.c_char_p, C.c_size_t, C.POINTER(pointer)]
    lib.odz_job_run.argtypes = [pointer, pointer]
    lib.odz_job_report.argtypes = [pointer, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
    lib.odz_job_destroy.argtypes = [pointer]
    lib.odz_runtime_destroy.argtypes = [pointer]
    base = json.loads((root / 'examples/grammar/submit.json').read_text())
    base['grammar']['retain'] = {'global': {'k': 1, 'unit': 'numeric_candidate'}}
    runs = []

    def run(runtime, values):
        base['grammar']['constants']['k']['values'] = values
        raw = json.dumps(base).encode()
        job = pointer()
        assert lib.odz_job_create(raw, len(raw), C.byref(job)) == 0
        lib.odz_job_run(runtime, job)
        required = C.c_size_t()
        assert lib.odz_job_report(job, None, 0, C.byref(required)) == 0
        buf = C.create_string_buffer(required.value)
        assert lib.odz_job_report(job, buf, len(buf), C.byref(required)) == 0
        result = json.loads(buf.value)
        lib.odz_job_destroy(job)
        return result

    runtime = pointer()
    assert lib.odz_runtime_create(0, b'/tmp/odezza-service-trial-cache', C.byref(runtime)) == 0
    assert lib.odz_runtime_attempt_inputs(runtime) == 0
    values = [.5 + i/64 for i in range(65)]
    if a.failure:
        first = run(runtime, values)
        second = run(runtime, values)
        assert first['status'] == second['status'] == 'failed'
        assert first['runtime_quarantined'] and second['runtime_quarantined']
        assert second['counts']['completed_configurations'] == 0
        runs += [first, second]
    else:
        for values in ([.5, 1, 2], [3, 4, 5], [.5, 1, 2], [.5+i/64 for i in range(65)]):
            result = run(runtime, values)
            assert result['status'] == 'complete', result['error']
            winners = list(result['candidates'].values())
            assert winners and all(w['values'][0] in values for w in winners)
            if 1 in values:
                assert min(w['mse'] for w in winners) < 1e-12
            runs.append(result)
    lib.odz_runtime_destroy(runtime)
    fresh = pointer()
    assert lib.odz_runtime_create(0, b'/tmp/odezza-service-trial-cache', C.byref(fresh)) == 0
    assert lib.odz_runtime_attempt_inputs(fresh) == 0
    control = run(fresh, [.5 + i/64 for i in range(65)])
    assert control['status'] == 'complete'
    assert min(w['mse'] for w in control['candidates'].values()) < 1e-12
    lib.odz_runtime_destroy(fresh)
    Path(a.output).write_text(json.dumps(dict(fault_injected=a.failure, runs=runs, fresh_control=control), indent=2)+'\n')
    print(json.dumps(dict(passed=True, fault_injected=a.failure, runs=len(runs)+1)))


if __name__ == '__main__':
    main()
