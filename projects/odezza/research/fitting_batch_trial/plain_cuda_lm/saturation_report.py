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
"""Tables of measured throughput, resource occupancy and explicit unknowns."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline_runner.occupancy import plateau,scaling_slope,launch_metrics

def build(root):
    rows=json.loads((root/'results.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    case_order={name:i for i,name in enumerate(manifest['cases'])}
    summaries=[]
    for case,width in dict.fromkeys((r['case'],r['width']) for r in rows):
        group=sorted((r for r in rows if r['case']==case and r['width']==width),key=lambda r:r['count'])
        last=group[-1];evidence=plateau(group);slope=scaling_slope(group)
        summaries.append(dict(case=case,width=width,count=last['count'],resources=last['resources'],
                              geometry=launch_metrics(last['count'],width,last['resources'],last['median_seconds']),plateau=evidence,scaling_slope=slope,equivalent=last['equivalent_to_one_lane'],
                              output_valid=all(r['exact_tiled_outputs'] for r in group),last_seconds=last['median_seconds']))
    winners=[]
    for case in dict.fromkeys(s['case'] for s in summaries):
        group=[s for s in summaries if s['case']==case]
        eligible=[s for s in group if s['plateau']['confirmed'] and s['equivalent'] and s['output_valid']]
        best=max(eligible,key=lambda s:s['geometry']['fits_per_second']) if eligible else None
        unresolved=[s['width'] for s in group if s['equivalent'] and not s['plateau']['confirmed']]
        unresolved+=sorted(set(manifest['widths'])-{s['width'] for s in group})
        compatible=[s['width'] for s in group if s['equivalent']]
        common=set.intersection(*[{r['count'] for r in rows if r['case']==case and r['width']==w} for w in compatible]) if compatible else set()
        common_count=max(common) if common else None
        measured=[r for r in rows if r['case']==case and r['count']==common_count and r['width'] in compatible]
        matched_best=min(measured,key=lambda r:r['median_seconds']) if measured else None
        winners.append(dict(case=case,selected_width=best['width'] if best and not unresolved else None,
                            best_confirmed_width=best and best['width'],
                            fits_per_second=best and best['geometry']['fits_per_second'],
                            largest_common_count=common_count,matched_best_width=matched_best and matched_best['width'],
                            matched_best_seconds=matched_best and matched_best['median_seconds'],
                            unresolved_widths=unresolved,all_equivalent_widths_confirmed=not unresolved,
                            status='confirmed among measured equivalent shapes' if best and not unresolved else 'provisional; unconfirmed widths remain' if best else 'no confirmed selection'))
    out=dict(summaries=summaries,winners=winners,achieved_occupancy=None)
    (root/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    lines=['# Prepared LM kernels: occupancy and throughput scaling','',
           'rack1 RTX 5080, device 0; one borrowed stream, 32 threads per CTA. Kernels, equations, '
           'trajectories, integration steps and fitting settings are unchanged from the earlier 48-cell grid. '
           'Each larger population tiles the original 1,024-start bank. These are repeated LM fits, '
           'not new unique ASTs or coefficient samples. All outputs and counters must equal that repeated bank bit for bit.','',
           '## Definitions','',
           '- **Theoretical occupancy:** CUDA-reported resident CTAs/SM × warps/CTA ÷ hardware maximum warps/SM. '
           'It includes the compiled register/shared-memory/block constraints; more input work cannot raise this ceiling.',
           '- **Resident waves:** launched CTAs ÷ (SM count × theoretical resident CTAs/SM). '
           'One wave supplies enough CTAs to fill every predicted slot once; it does not prove achieved residency.',
           '- **SM µs/fit:** kernel seconds × SM count × 1,000,000 ÷ fits. It charges all SMs, including idle ones. '
           'Fits/s/SM is its reciprocal with units converted. This is measured device throughput normalized by SM count.',
           '- **CTA/s:** total completed CTAs ÷ kernel seconds. Its reciprocal is amortized completion spacing, '
           'not individual CTA latency; blocks overlap and contain different numbers of fits at different widths.',
           '- **Plateau:** the last three increasing populations have a throughput range ≤5%, each population’s '
           'three timing samples span ≤5%, and the largest grid supplies at least four resident waves. '
           'This is empirical evidence of throughput saturation, not proof of peak hardware utilization.','',
           'Achieved occupancy and issue utilization are **unmeasured**: the installed Nsight Compute reports '
           '`ERR_NVGPUCTRPERM`. No administrative setting was changed. Theoretical capacity is never labelled achieved occupancy. '
           'See [NVIDIA’s occupancy definition](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) '
           'and [CUDA occupancy APIs](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html).','',
           '## Throughput choices','',
           'Selections permit spills and require exact equivalent work plus plateau evidence. '
           'An unresolved width prevents a final all-shape winner claim. Compare finite batch latency separately.','',
           '| System | Plateau-qualified lanes | Best confirmed fits/s | Status | Unconfirmed lanes |',
           '|:---|---:|---:|:---|:---|']
    for w in winners:
        rate=f'{w["fits_per_second"]:,.0f}' if w['fits_per_second'] is not None else '—'
        lines.append(f'| {w["case"]} | {w["selected_width"] or "—"} | {rate} | {w["status"]} | {w["unresolved_widths"] or "none"} |')
    lines+=['','## Matched finite-batch choices','',
            'Largest population measured for every comparable width in each case. These are measured latency winners, '
            'not claims of peak throughput. The entire rate curve and plateau/slope checks follow.','',
            '| System | Common fits | Best measured lanes | Kernel ms |',
            '|:---|---:|---:|---:|']
    for w in winners:
        if w['matched_best_width'] is not None:
            lines.append(f'| {w["case"]} | {w["largest_common_count"]:,} | {w["matched_best_width"]} | {w["matched_best_seconds"]*1000:.2f} |')
    lines+=['','## Largest measured population per shape','',
            '| System | Lanes | Fits | CTAs | Resident CTAs/SM | Theoretical occupancy | Waves | Kernel ms | Fits/s | SM µs/fit | CTA/s | Spills | Plateau | Equivalent |',
            '|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|:---|']
    for s in sorted(summaries,key=lambda s:(case_order[s['case']],s['width'])):
        g=s['geometry'];r=s['resources']
        lines.append(f'| {s["case"]} | {s["width"]} | {s["count"]:,} | {g["ctas"]:,} | {r["active_blocks_per_sm"]} | {100*r["theoretical_occupancy"]:.1f}% | {g["resident_waves"]:.2f} | {1000*s["last_seconds"]:.2f} | {g["fits_per_second"]:,.0f} | {g["sm_microseconds_per_fit"]:.2f} | {g["ctas_per_second"]:,.0f} | {r["spills"]} | {s["plateau"]["confirmed"]} | {s["equivalent"]} |')
    lines+=['','## Incremental cost from population scaling','',
            'Fits `T(N) = fixed_time + N × fit_cost` to the largest three populations using mean event times. '
            'A usable estimate requires positive incremental costs, adjacent slopes agreeing within 5%, '
            'residuals and repeat spreads within 5%, and at least one full predicted resident wave at each size. '
            'These are local estimates; they do not replace hardware-counter measurements. A stable slope may '
            'exist before raw throughput plateaus, because fixed cost still matters.','',
            '| System | Lanes | Usable model | Equivalent work | Estimated SM µs/fit | Estimated fits/s/SM | Estimated CTA/s/SM | Fixed time ms |',
            '|:---|---:|:---|:---|---:|---:|---:|---:|']
    for row in sorted(summaries,key=lambda s:(case_order[s['case']],s['width'])):
        s=row['scaling_slope']
        if 'estimated_sm_microseconds_per_fit' not in s:continue
        lines.append(f'| {row["case"]} | {row["width"]} | {s["usable"]} | {row["equivalent"]} | {s["estimated_sm_microseconds_per_fit"]:.2f} | {s["estimated_fits_per_sm_second"]:,.0f} | {s["estimated_ctas_per_sm_second"]:,.0f} | {1000*s["intercept_seconds"]:.2f} |')
    lines+=['','## All populations','',
            '| System | Lanes | Fits | Kernel ms | Fits/s | Waves | Maximum SM coverage |',
            '|:---|---:|---:|---:|---:|---:|---:|']
    for row in sorted(rows,key=lambda r:(case_order[r['case']],r['width'],r['count'])):
        g=row['geometry']
        lines.append(f'| {row["case"]} | {row["width"]} | {row["count"]:,} | {row["median_seconds"]*1000:.2f} | {g["fits_per_second"]:,.0f} | {g["resident_waves"]:.2f} | {g["sm_coverage_upper_bound"]*100:.1f}% |')
    lines+=['','## Limits','',
            'No division by theoretical occupancy is used to invent a 100%-occupancy performance number. '
            'The 32-thread CTA remains fixed; this sweep does not test larger blocks, which could change '
            'the resource ceiling. Clocks are not locked; telemetry before and after each cell is retained. '
            'Idle resident service processes remain on the GPU; no concurrent work was observed at launch. '
            'Timing excludes preparation, module loading and transfers. The original 3-state/6-coefficient '
            'width-2 numerical-path difference remains excluded from equivalent-work selection. '
            'Correct structures are deliberately present, so this measures fitting throughput rather than recovery.','',
            'Raw evidence: [results.json](results.json), [analysis.json](analysis.json), [manifest.json](manifest.json).','']
    (root/'OCCUPANCY.md').write_text('\n'.join(lines));return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=build(a.root)
    print(json.dumps(dict(shapes=len(r['summaries']),plateaus=sum(s['plateau']['confirmed'] for s in r['summaries']))))
