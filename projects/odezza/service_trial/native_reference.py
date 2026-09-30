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
"""Direct C reference/timing harness; run on a GPU host, without NATS."""
import argparse
import ctypes as C
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('requests', nargs='+')
    p.add_argument('--output', required=True)
    p.add_argument('--attempt-inputs', action='store_true')
    p.add_argument('--repeats', type=int, default=1)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    lib = C.CDLL(str(root/'build/runtime/libodezza_runtime.so'))
    ptr = C.c_void_p
    lib.odz_runtime_create.argtypes = [C.c_uint, C.c_char_p, C.POINTER(ptr)]
    lib.odz_runtime_attempt_inputs.argtypes = [ptr]
    lib.odz_job_create.argtypes = [C.c_char_p, C.c_size_t, C.POINTER(ptr)]
    lib.odz_job_run.argtypes = [ptr, ptr]
    lib.odz_job_report.argtypes = [ptr, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
    lib.odz_job_destroy.argtypes = lib.odz_runtime_destroy.argtypes = [ptr]
    runtime = ptr()
    assert not lib.odz_runtime_create(0, b'/tmp/odezza-service-trial-cache', C.byref(runtime))
    if a.attempt_inputs:
        assert not lib.odz_runtime_attempt_inputs(runtime)
    runs = []
    for path in a.requests:
        raw = Path(path).read_bytes()
        for repeat in range(a.repeats):
            job = ptr()
            start = time.perf_counter()
            assert not lib.odz_job_create(raw, len(raw), C.byref(job))
            lib.odz_job_run(runtime, job)
            native_elapsed = time.perf_counter()-start
            required = C.c_size_t()
            assert not lib.odz_job_report(job, None, 0, C.byref(required))
            output = C.create_string_buffer(required.value)
            assert not lib.odz_job_report(job, output, len(output), C.byref(required))
            report = json.loads(output.value)
            assert report['status'] == 'complete', report.get('error')
            runs.append(dict(name=Path(path).stem, repeat=repeat, elapsed=native_elapsed, report=report))
            lib.odz_job_destroy(job)
    lib.odz_runtime_destroy(runtime)
    Path(a.output).write_text(json.dumps(runs, indent=2)+'\n')
    print(json.dumps([dict(name=x['name'], repeat=x['repeat'], seconds=x['elapsed'], configs=x['report']['counts']['completed_configurations']) for x in runs]))


if __name__ == '__main__':
    main()
