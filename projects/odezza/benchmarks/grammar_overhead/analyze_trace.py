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
"""Export inspectable aggregate timings from an existing Nsight SQLite capture."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def union_ns(intervals):
    end, total = None, 0
    for begin, finish in sorted(intervals):
        if end is None or begin >= end:
            total += finish-begin
        elif finish > end:
            total += finish-end
        end = max(finish, end) if end is not None else finish
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('database', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    db = sqlite3.connect(f'file:{args.database.resolve()}?mode=ro', uri=True)
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    result = dict(database_sha256=hashlib.sha256(args.database.read_bytes()).hexdigest(),
                  semantics='API/kernel totals are sums and may overlap; kernel_union_seconds merges concurrent intervals per GPU.')
    for table, field, key in [('CUPTI_ACTIVITY_KIND_KERNEL', 'shortName', 'kernels'),
                               ('CUPTI_ACTIVITY_KIND_RUNTIME', 'nameId', 'cuda_apis'),
                               ('OSRT_API', 'nameId', 'os_apis')]:
        if table in tables:
            result[key] = [dict(name=name, calls=count, seconds=seconds) for name, count, seconds in db.execute(
                f'SELECT s.value,COUNT(*),SUM(a.end-a.start)/1e9 FROM {table} a JOIN StringIds s '
                f'ON a.{field}=s.id GROUP BY s.value ORDER BY 3 DESC')]
    result['gpu_activity'] = []
    for device, in db.execute('SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL'):
        intervals = db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=?', (device,)).fetchall()
        row = dict(device=device, kernel_sum_seconds=sum(b-a for a, b in intervals)/1e9,
                   kernel_union_seconds=union_ns(intervals)/1e9,
                   first_kernel_ns=min(a for a, b in intervals), last_kernel_ns=max(b for a, b in intervals))
        result['gpu_activity'].append(row)
    result['transfers'] = [dict(copy_kind=kind, calls=n, bytes=size, seconds=seconds)
        for kind, n, size, seconds in db.execute('SELECT copyKind,COUNT(*),SUM(bytes),SUM(end-start)/1e9 '
                                                'FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind')]
    result['scoring_shapes'] = [dict(grid=[gx, gy, gz], block=[bx, by, bz], registers=registers,
        local_memory_per_thread=local, calls=count, seconds=seconds) for gx, gy, gz, bx, by, bz, registers, local, count, seconds in db.execute(
        "SELECT gridX,gridY,gridZ,blockX,blockY,blockZ,registersPerThread,localMemoryPerThread,COUNT(*),SUM(a.end-a.start)/1e9 "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL a JOIN StringIds s ON a.shortName=s.id WHERE s.value='odezza_scoring' "
        'GROUP BY gridX,gridY,gridZ,blockX,blockY,blockZ,registersPerThread,localMemoryPerThread')]
    result['devices'] = [dict(id=i, name=name, sms=sms, max_warps_per_sm=warps) for i, name, sms, warps
                         in db.execute('SELECT id,name,smCount,maxWarpsPerSm FROM TARGET_INFO_GPU')]
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(result['gpu_activity']))


if __name__ == '__main__':
    main()
