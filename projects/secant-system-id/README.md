<!--
SPDX-FileCopyrightText: 2026 Charles Durham
SPDX-License-Identifier: MIT

MIT License

Copyright (c) 2026 Charles Durham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# Secant System ID

> **Collection category:** Reference consumers. See the [project map](../../docs/PROJECTS.md) for entry points and status, and [research coverage](../../docs/RESEARCH_COVERAGE.md) for limitations.

This is a new, standalone repository for experimenting with native GPU kernel
shapes for ODE system identification. It does not link to or import Secant.
Its serialized post-order instruction values intentionally match Secant's C99
AST ABI so the prototype Python specializer can later be replaced by the
production C99 writer without changing programs or search data.

## First kernel

The first experiment is deliberately score-only rather than LM:

- one CUDA thread owns one setting and all three fed-batch trajectories;
- the complete fixed-step RK4 integration and trajectory score are fused;
- the known mass-balance equations remain ordinary CUDA;
- two missing rate-law ASTs are specialized into one SASS site;
- each AST has eight input slots;
- every setting maps each input slot to either one of four current state values
  or one of eight per-thread constants;
- state and constant values occupy one shared-memory bank, and the per-setting
  index performs the same dynamic load for both kinds of leaf; and
- the planted setting reproduces the published dual-substrate astaxanthin model.

The number of missing sites is not fixed. A `SystemModel` contains a dynamic
tuple of named missing sites, each with its own leaf count. The CUDA marker site
returns one value per missing site.

## Encoding known equations

Known equations use a small typed Python expression model and are emitted as
ordinary CUDA/C++ around the specialization site. For example:

```python
from secant_system_id import MissingSite, SystemModel, missing, state

x = state(0)
u = missing(0)
model = SystemModel(
    name="controlled_decay",
    state_names=("x",),
    missing_sites=(MissingSite("forcing", 4),),
    derivatives=(u - 0.2 * x,),
    observation_interval=0.5,
)
shape = model.make_shape(
    constant_count=4,
    trajectory_count=8,
    observation_count=20,
)
```

The generated CUDA contains the equivalent known derivative:

```cuda
const float derivative0 = (missing_output0 - (0.2f * stage_state[0]));
```

Raw CUDA snippets are not accepted as model expressions. The whitelist keeps
generation inspectable and makes state and missing-site references verifiable.

The source generator and ELF/CUBIN inspection remain Python. Postorder SASS
assembly and specialization now have byte-identical Python and C99
implementations. The C99 runner uses several CPU specialization workers, one
writable CUBIN copy per worker, one CUDA context, eager module loading, event-
tracked execution, and safe unloading. A generated CUBIN may hold multiple
independently specializable packed kernels, all loaded as one module.

The first complete search loop is now also C99. It owns two preallocated
system-genome population arenas, evolves one AST per missing equation, and
feeds the eager runtime without returning to Python between generations.
Search settings are generated statelessly on the GPU around each genome's
incumbent constants and leaf bindings, so the host uploads only compact
incumbent state and downloads one reduced winner per genome.

Search modules can optionally replace the full per-setting MSE surface with an
on-device winner path. Each scoring CTA reduces its settings to one local
`(score, setting_index)` pair, then the fixed `ssid_reduce_winners` kernel emits
one pair per genome. Distinct in-flight tickets may use distinct output slices,
so the host can recover the winning constants and leaf bindings from the
resident population without copying every score.

Controlled recovery runs can instead retain the full MSE surface and select
winners in C. This path is also the correctness fallback for architectures on
which the fused reducer has not passed replay validation. Runs durably write a
raw checkpoint trace and update `progress.json` after every seed; checkpoint
analysis and final materialized GPU replay remain outside the timed search.

Experimental trajectory-LM promotions specialize their primal and derivative
sites in Python, then batch the resulting modules through a persistent C99
loader. The LM reference and output storage remain allocated for the campaign;
module load, launch, event retirement, and unload no longer create a fresh
Python CUDA context for every promoted genome. The optional `novelty-random`
selector compares the best newly observed AST families with reproducibly random
controls from the same objective-ranked pool. The family hash retains operators,
tree order, fixed literals, and repeated-leaf aliasing while ignoring arbitrary
dynamic-leaf slot numbers, incumbent constants, and incumbent bindings.
Per-boundary Venn counts, eligibility failures, and LM improvement rates for
both arms are written to the recovery report.

## Repository layout

- `ast.py`: Secant-compatible post-order ABI and expression builder.
- `bindings.py`: typed state/constant leaf sources encoded into one shared bank.
- `template.py`: configurable CUDA template and marker-site generator.
- `packed_template.py`: compact multi-genome dispatch and multi-kernel module generation.
- `model.py`: typed known equations and dynamic missing-site descriptions.
- `genome.py`: one site-local post-order program per missing equation.
- `cubin.py`: dependency-free ELF/CUBIN inspection.
- `packed_cubin.py`: per-entry-point target-table inspection and module plans.
- `sass.py`: initial Python SASS writer for `sm_89`, `sm_90`, and `sm_120`.
- `packed_sass.py`: variable-length genome packing across every module arena.
- `manifest.py`: self-verifying source metadata and CUBIN template identity.
- `inspection.py`: verified, serializable direct/packed inspection records.
- `fed_batch.py`: planted AST pair, settings, bindings, and CPU reference.
- `runtime.py`: one-load CUDA Driver API execution for direct, packed, and multi-kernel modules.
- `gpu_gradient_check.py`: randomized production-SASS primal/gradient differential gate.
- `trajectory_lm.py`: thread-owned trajectory-LM population and launch adapter for experimental GP promotion.
- `c_runtime.py`: flat postorder buffers, `ctypes` ABI, worker benchmarks, and C99 pipeline access.
- `cli.py`: generation, compilation, inspection, specialization, and run path.
- `csrc/`: C99 GP loop, SASS specializer, fixed-size worker buffers, queues, and single-context CUDA runner.

## Run

No package installation is required when running directly from the repository.
The generator is stdout-oriented so it can be inspected or piped:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m secant_system_id.generate > generated/template.cu
PYTHONPATH=src python3 -m secant_system_id.generate \
  | PYTHONPATH=src python3 -m secant_system_id.compile_stdin --arch sm_120 \
  > generated/template.cubin
PYTHONPATH=src python3 -m secant_system_id.inspect \
  --source generated/template.cu generated/template.cubin \
  > generated/template.inspection.json
PYTHONPATH=src python3 -m secant_system_id.cli fedbatch-run --arch sm_120
PYTHONPATH=src python3 -m secant_system_id.cli fedbatch-packed-run --arch sm_120
PYTHONPATH=src python3 -m secant_system_id.cli packed-module-generate --kernels 2 \
  -o generated/fedbatch_module.cu
PYTHONPATH=src python3 -m secant_system_id.cli fedbatch-module-run \
  --arch sm_120 --kernels 2 --genomes 16
PYTHONPATH=src python3 -m secant_system_id.cli fedbatch-module-run \
  --arch sm_120 --kernels 2 --genomes 16 --cpu-check-all-settings
make c99
PYTHONPATH=src python3 -m secant_system_id.cli fedbatch-c99-pipeline-run \
  --arch sm_120 --kernels 2 --genomes 16 --settings 2048 \
  --worker-counts 1,2,4,8 --workers 4 --loaded-modules 8
PYTHONPATH=src python3 -m secant_system_id.cli fedbatch-c99-unique-sweep \
  --arch sm_120 --genomes 128 --genome-capacity 128 --genomes-per-cta 1 \
  --settings-counts 256,512,1024 --loaded-module-depths 1,2,3 \
  --threads 256 --workers 1 --submissions 512 --winner-output
PYTHONPATH=src python3 -m secant_system_id.cli fedbatch-c99-gp-run \
  --arch sm_120 --population 1024 --settings 512 --generations 500 \
  --kernels 1 --genome-capacity 128 --genomes-per-cta 1 \
  --workers 1 --loaded-modules 2
PYTHONPATH=src python3 -m secant_system_id.cli fedbatch-c99-recovery-run \
  --arch sm_120 --training-design product_paired16 --trajectories 16 \
  --population 1024 --settings 512 --generations 2000 \
  --checkpoint-stride 10 --loss relative --seeds 7,17,29
PYTHONPATH=src python3 -m secant_system_id.cli fedbatch-c99-recovery-run \
  --arch sm_120 --training-design product_paired16 --trajectories 16 \
  --population 1024 --settings 512 --generations 500 \
  --lm-promotion-interval 100 --lm-promotion-count 4 \
  --lm-selection-mode random --lm-selection-pool 32 \
  --lm-random-trigger-probability 0.14 \
  --lm-settings 8192 --lm-starts 4 --seeds 7
```

The default compiler adapter calls the installed NVRTC library directly from
Python. `--nvcc /path/to/nvcc` remains available when a compatible command-line
toolchain is preferred. Neither path downloads or installs tooling.

Packed runs normally CPU-check setting zero for each genome. The optional
`--cpu-check-all-settings` mode reconstructs and scores every genome-major
constant and leaf-binding configuration, reports valid and invalid agreement,
and summarizes the worst absolute and relative discrepancies. This is intended
as an exhaustive correctness mode rather than part of GPU throughput timing.

See [`GP_DESIGN.md`](GP_DESIGN.md) for the split-AST system genome, settings
layout, mutation/crossover boundaries, and the conditions under which a shared
multi-output DAG would become worthwhile.

See [`KERNEL_TOPOLOGY.md`](KERNEL_TOPOLOGY.md) for the packed branch mapping,
the eight-genomes-per-CTA example, and the current dense aligned trajectory-data
contract.

See [`MODULE_FORMAT.md`](MODULE_FORMAT.md) for the generated CUDA manifest,
source/CUBIN identity binding, inspection JSON schema, and pipe-oriented
commands.

See [`C99_RUNTIME.md`](C99_RUNTIME.md) for the flat genome/AST byte ABI, one-
CUBIN-copy-per-worker ownership, eager module lifecycle, and GP integration
boundary.

See [`C99_GP.md`](C99_GP.md) for the implemented generation lifecycle, compact
GPU settings, correctness checks, and RTX 5090 end-to-end measurements.

See [`RECOVERY.md`](RECOVERY.md) for the controlled recovery protocol, metrics,
pilot results, product-paired training design, checkpoint contract, and current
architecture-specific runner choices.

See [`BENCHMARKS.md`](BENCHMARKS.md) for the external model-discovery strategy,
the MDBench adapter contract, the separate PEtab parameter-estimation lane, and
the steps required to remove the remaining fed-batch assumptions.

Use `sm_89` for an RTX 4090. Generated source, CUBINs, and the JSON run report
are placed in `generated/`, which is ignored by Git.

Use `sm_90` for Hopper GPUs such as the GH200. The Python and C99 writers use
the same postorder instruction encoding on `sm_90` and `sm_120`; the compact
dispatch inspection already accounts for Hopper's zero-`UMOV` form. The GH200
path is covered by byte-equality tests and was validated by executing the
fed-batch reference on hardware.

## Planned kernel sequence

1. Measure the current dynamic shared-bank shape over settings, CTA sizes,
   integration step counts, and AST sizes.
2. Compare register-select leaves against shared-memory indexed leaves for the
   four-state case.
3. Measure eager C99 specialization, module loading, execution, and unloading over module shapes and worker counts.
4. Implement the C99 GP producer against the flat genome/AST byte ABI. **Done.**
5. Run controlled, multi-seed recovery of the blinded fed-batch rate laws and
   measure time to numerical and structural recovery. **Implemented; larger
   campaign in progress.**
6. Add selective thread-owned resident LM promotion for promising survivors,
   including forward-mode AST and state sensitivities. **Experimental path
   implemented behind `--lm-promotion-interval`; default search is unchanged.**

LM is a promotion stage rather than the first search kernel: it should receive
promising structures and bindings, fit a modest number of constants, and keep
its module resident across iterations.
