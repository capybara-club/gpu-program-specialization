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

# Secant System ID: hardware, memory, settings, and job queue

**Decision date:** 2026-08-26  
**Scope:** Machines intended primarily for Secant System ID trajectory search.

## Current recommendation

Do not buy another RTX 5090 at the current standalone street price. The best
next purchase is one ordinary RTX 5080, installed in the existing Ryzen 9
7900X machine, followed by the exact Secant direct, packed, LM, and module-load
benchmarks. If it sustains at least **72 million direct settings/s** and its
module-loading rate is healthy, use that as the repeatable fleet node.

For a larger purchase, the likely cost/density sweet spot is **two RTX 5080s
per full-size AM5 host**, after the first card passes the Secant acceptance
benchmarks. Two independent devices should provide roughly one RTX 5090's
aggregate execution resources at substantially lower current GPU cost, while
sharing one CPU, motherboard, memory kit, SSD, and network connection. Use two
such hosts for four cards rather than either four commodity hosts or one dense
four-GPU workstation. The existing RTX 5090 and RTX 4090 remain the fast and
known-good nodes in the same queue.

This revises the earlier preference for four single-GPU nodes. The expected
cash saving relative to two separate 5080 hosts is modest because a dual-GPU
host needs a more expensive x8/x8 motherboard, 1,500-1,600 W power supply,
full-size chassis, and careful cooling. The stronger reason for two cards per
host is density and simpler administration. Prefer single-GPU nodes when
reusing an existing host, when only wide three- or four-slot cards are
available, or when failure isolation matters more than density.

Before scaling the fleet, port the LM derivative specializer from Python to
C99 and put it in the existing per-GPU eager worker pipeline. The current
Python implementation is a reference/oracle and has been adequate for sparse
LM promotions, but it is an unacceptable production hot-path boundary. Keep
the completed campaigns as valid end-to-end measurements of the current path;
do not project CPU requirements for multiple GPUs until the C99 port has exact
byte/correctness parity, pressure-fallback coverage, and a worker-count sweep.

The reason this recommendation is possible is that Secant System ID is not
VRAM-capacity constrained. The live jobs use approximately **1.0 GiB total on
the RTX 5090** and **0.9 GiB total on the RTX 4090**, while the explicit
trajectory/search arrays account for only about 12 MiB on the efficient path.
Sixteen gigabytes is already ample for the current kernel regime.

## GPU market decision

Prices below are observed US asking prices on 2026-08-26 and can change
quickly. The RTX 5080 has recently been available around $1,400-$1,700, while
in-stock RTX 5090 cards have been around $4,500-$4,830. NVIDIA lists 10,752
CUDA cores, 16 GB, and 360 W for the RTX 5080 versus 21,760 cores, 32 GB, and
575 W for the RTX 5090.

| GPU | Observed price | VRAM | What is known for Secant | Decision |
|---|---:|---:|---|---|
| RTX 5090 | $4,500-$4,830 | 32 GB | Measured 145.5M direct settings/s and 115.2M packed configurations/s | Keep Rohini; do not buy at this premium |
| RTX 4090 | Roughly $2,300-$2,850 on the current secondary market | 24 GB | Measured 111.0M direct settings/s and 94.63M packed configurations/s | Known-good fallback if a safe card is at or below about $2,500 |
| RTX 5080 | $1,400-$1,700 | 16 GB | Not measured yet; same Blackwell compute capability as the 5090 and enough VRAM | **Best first node to test and likely fleet target** |
| RTX 5070 Ti | Roughly $1,170-$1,350 | 16 GB | Not measured; good theoretical value but only a modest saving over the cheapest 5080 | Consider only when substantially below a 5080 |
| RTX 5070 | Roughly $640-$800 | 12 GB | Not measured; potentially excellent GPU-only throughput per dollar, but lower density makes host and management costs more important | Interesting later fleet ablation, not the first purchase |
| RTX PRO 6000 Blackwell | $12,500 or more | 96 GB ECC | Most of its memory and professional premium would be unused | Reject for this workload |

Our measured RTX 5090 advantage over the RTX 4090 is only **31.1%** in the
direct benchmark and **21.7%** in the packed benchmark. If an RTX 4090 is worth
$2,500 for this workload, the corresponding performance-proportional RTX 5090
price is only about **$3,040-$3,280**. A $4,500 card is well outside that range.

At a $1,600 purchase price, an RTX 5080 needs about **52M settings/s** to beat
a $4,500 RTX 5090 on GPU purchase price per unit of direct throughput, and
about **71M settings/s** to beat a $2,500 RTX 4090. The revised 72M acceptance
gate therefore includes a small safety margin against the known 4090. This is
an inference from our measured cards and NVIDIA's specifications, not a
substitute for the actual Secant benchmark.

### Dual-RTX-5080 cost optimization

As of 2026-08-27, an ordinary in-stock RTX 5080 is approximately $1,600-$1,660
at B&H, while the only listed in-stock RTX 5090 is $4,829.99. Two ordinary
5080s therefore cost about $3,200-$3,320, roughly $1,500-$1,630 less than that
5090 before considering the host.

The RTX 5080 has 10,752 CUDA cores and the RTX 5090 has 21,760. Two 5080s have
almost the same aggregate core count as one 5090, and both use `sm_120`, so
they can use the same specialized CUBIN family. That does not prove identical
Secant throughput: clocks, SM scheduling, power, and per-device launch/module
overheads still matter. A reasonable provisional range is 65-80M direct
settings/s per 5080, or 130-160M across two independent jobs. Compare this
with the measured 145.5M settings/s on Rohini's 5090.

Planning ranges before tax are:

| Configuration | GPU cost | Complete-system cost | Provisional aggregate direct throughput | Comment |
|---|---:|---:|---:|---|
| One RTX 5080 in an existing suitable host | $1,600-$1,660 | $1,600-$1,660 incremental | 65-80M settings/s | Cheapest and necessary first measurement |
| One new RTX 5080 node | $1,600-$1,660 | $2,700-$3,100 | 65-80M settings/s | Simplest cooling and best isolation |
| Two separate new RTX 5080 nodes | $3,200-$3,320 | $5,400-$6,200 | 130-160M settings/s | Maximum isolation and CPU capacity |
| Two RTX 5080s in one AM5 host | $3,200-$3,320 | **$4,900-$5,500** | **130-160M settings/s** | Recommended repeatable density node if thermals validate |
| One new RTX 5090 node at current in-stock price | $4,830 | $6,000-$6,500 | 145.5M settings/s measured on Rohini | Poor value unless one-job latency or 32 GB matters |

The dual host saves only about $500-$700 versus two separate new nodes under
these assumptions, around 9-12%. It can save roughly $1,100-$1,600 versus a
new 5090 node while offering comparable aggregate throughput. It does not
halve the cost because the dual host's motherboard, power supply, case, and
cooling are all premium parts.

Use each 5080 as an independent queue target running its own seed, dataset, or
problem. Do not make one search synchronize across both devices. Secant has no
need for NVLink or peer-to-peer traffic in this regime.

The host should use a Ryzen 9 9950X or similar 16-core CPU, 32-64 GB DDR5, a
board explicitly wired for CPU-direct PCIe 5.0 x8/x8, a 1,500-1,600 W ATX 3.1
power supply with two native GPU cables, and a high-airflow eight-slot or
larger chassis. PCIe 5.0 x8 per card is ample for the compact hashed path;
nevertheless, measure x16 versus x8 with Secant before replicating the build.
AMD documents x8/x8 support at the X870E platform level, but it must also be
present in the exact board layout. The ASRock X870E Taichi and ASUS ProArt
X870E-Creator are examples; the latter can reduce the second GPU to x4 when a
shared M.2 slot is populated.

Physical fit is the gating risk. NVIDIA's reference 5080 is nominally two
slots, but current add-in-board cards vary widely and usually require extra
air space. Select the exact two card models, slot spacing, power-connector
clearance, and chassis together. If two cards cannot run a sustained Secant
load without throttling, the modest host saving is not worth it.

### Recommended node

The existing Ryzen 9 7900X machine is still the correct first RTX 5080 test host,
assuming its case, power supply, and cooling fit the particular card. A new
single-GPU validation node should use approximately:

- RTX 5080 bought near the bottom of the available range, without paying for a
  premium factory overclock;
- Ryzen 7 9700X or Ryzen 9 9900X, with preference for the 9900X when the price
  difference is small;
- 32 GB DDR5 is sufficient; use 64 GB when the small incremental price is
  worthwhile for concurrent compilation, development tools, and future jobs;
- 2 TB NVMe storage;
- a reputable 1,000 W ATX 3.1 power supply with a native GPU power cable;
- wired 2.5 GbE or faster networking; and
- a large airflow-oriented case rather than a compact presentation build.

After validating one card, use two GPUs per carefully designed AM5 host when
buying a repeatable fleet. Keep one GPU per host when reusing ordinary existing
machines or when the available card coolers cannot be spaced safely. A dual
5080 system can draw roughly 850-1,000 W at the wall under sustained Secant
load; verify this directly and distribute multiple hosts sensibly across
circuits.

Before duplicating the node, measure:

1. direct settings/s;
2. packed configurations/s for the current fed-batch kernel;
3. LM fits/s and search quality for a fixed generation budget;
4. AST specialization and eager module-load throughput; and
5. wall power during the actual kernel, not a gaming benchmark.

### Host RAM requirement

The 2026-08-27 live campaigns show that 64 GB is not required by the present
search. Rohini's complete host used 3.0 GiB and its campaign process had a
302,652 KiB RSS. Ada's complete host used 3.8 GiB and its campaign process had
a 299,180 KiB RSS. Neither machine used swap. Their multi-gigabyte virtual
address sizes include mappings and reservations and must not be mistaken for
resident RAM.

Therefore:

- 32 GB is ample for one or two current Secant System ID GPU workers;
- 32 GB should probably also execute four current workers, although that
  topology has not been validated end to end;
- 64 GB is a convenience/future-proofing choice for simultaneous template
  compilation, development sessions, large diagnostic outputs, and future
  datasets—not a throughput requirement; and
- 96 GB has no demonstrated value for the current ODE search.

If the price difference is small, two 32 GB DIMMs are the cleanest long-lived
build. For a strict value node, two 16 GB DIMMs are sufficient and should not
reduce Secant throughput.

### Market/specification references

- [Current B&H RTX 5090 listings](https://www.bhphotovideo.com/c/buy/rtx-5090/ci/60217)
- [Current B&H RTX 5080 listings](https://www.bhphotovideo.com/c/buy/rtx-5080/ci/60168)
- [Current B&H RTX 5070 Ti listings](https://www.bhphotovideo.com/c/buy/rtx-5070-ti/ci/60216)
- [Current B&H RTX 5070 listings](https://www.bhphotovideo.com/c/buy/rtx-5070/ci/60215)
- [NVIDIA GeForce specification comparison](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/?section=compare-specs)
- [AMD AM5 chipset PCIe lane modes](https://www.amd.com/en/products/processors/chipsets/am5.html)
- [ASRock X870E Taichi x8/x8 specification](https://www.asrock.com/mb/AMD/X870E%20Taichi/)
- [ASUS ProArt X870E-Creator expansion-slot specification](https://www.asus.com/sg/motherboards-components/motherboards/proart/proart-x870e-creator-wifi/techspec/)
- [Current RTX 4090 market tracking](https://gpupoet.com/gpu/learn/price/august-2026/nvidia-geforce-rtx-4090)
- [B&H RTX PRO 6000 Blackwell listing](https://www.bhphotovideo.com/c/product/1895402-REG/nvidia_900_5g144_2200_000_rtx_pro_6000_blackwell.html)

## What currently occupies VRAM

The current dense `product_paired16` experiment has four states, sixteen
trajectories, twelve observation points, eight constants, and sixteen dynamic
leaf slots. Times and the known equation structure are compiled into the
kernel; the global reference table contains initial states and observed target
states.

### Efficient RTX 5090 path

The current Rohini run uses population 1,024, 512 GP settings, compact GPU
winner reduction, four LM candidates, 8,192 LM binding settings, and four
starts per binding.

| Allocation | Formula | Size |
|---|---:|---:|
| Initial states | `4 states * 16 trajectories` floats | 256 B |
| Target trajectory points | `4 * 16 * 12` floats | 3,072 B |
| Complete reference table | `4 * 16 * (12 + 1)` floats | 3,328 B (3.25 KiB) |
| GP incumbent constants | `1,024 * 8` floats | 32 KiB |
| GP incumbent leaf bindings | `1,024 * 16` 32-bit indices | 64 KiB |
| Compact CTA and genome winner buffers | Two setting tiles per genome, plus final score/index | 24 KiB |
| Full GP MSE table | Disabled; would be `1,024 * 512` floats | 0 B now; 2 MiB if enabled |
| Persistent LM queue | Four candidates, 8,192 bindings, four starts, constants and solver outputs | 10.5 MiB |
| Current GP and LM CUBIN file images | At most two 450,720 B GP modules plus one 67,224 B LM module | Under 1 MiB before driver alignment |

The identifiable persistent data and output buffers therefore total only about
**11-12 MiB**. `nvidia-smi` reports approximately **1,010 MiB** for the live
process. Most of the difference is CUDA context, driver, executable-module,
relocation, and internal allocation overhead rather than trajectories,
settings, or MSE values.

The kernel copies the 3.25 KiB reference table from VRAM into shared memory for
each CTA. With 256 threads, the per-thread twelve-slot state/constant bank and
the reduction scratch bring the ordinary GP CTA to about **17.25 KiB of shared
memory**. This is on-chip SRAM allocated per resident CTA, not persistent VRAM.

### Current RTX 4090 fallback path

Ada is deliberately running the validation fallback with population 2,048,
512 settings, `--materialized-gp`, and `--full-mse-gp`.

| Allocation | Size |
|---|---:|
| Materialized constant settings | 32 MiB |
| Materialized binding settings | 64 MiB |
| Full MSE table | 4 MiB |
| Persistent LM queue | 10.5 MiB |
| Reference table and other explicit outputs | Well under 1 MiB |

This path has roughly **111 MiB** of identifiable problem/search allocations,
while the live process reports approximately **886 MiB**. It physically
generates about 96 MiB of setting inputs on the CPU and uploads them every
generation, then downloads 4 MiB of MSE values. That fallback is a major
performance defect, not a reason to buy more VRAM. Fixing the compact hashed
winner path on `sm_89` should remove almost all of that repeated work.

### Capacity conclusion

VRAM does not currently select the GPU. Twelve or sixteen gigabytes is enough
for this regime with a large safety margin. The important hardware resources
are FP32/branch throughput, register and shared-memory capacity per SM, the
number of SMs, power, and the host's specialization/module-loading rate.

VRAM becomes important only if the design changes to retain many populations,
full trajectories, full MSE histories, large irregular observation tables, or
multiple simultaneous jobs on one device. Even then, score histories should
normally be streamed to host storage instead of accumulated indefinitely in
VRAM.

## How settings work now

There are three different paths, which should not be described as one policy.

### 1. Hashed GP settings: the intended path on Rohini

Each stored genome contains a concrete postorder structure, eight incumbent
constants, and sixteen incumbent leaf bindings. Setting zero is the incumbent.
For every other setting, the GPU deterministically derives the constant
perturbations and binding substitutions from the trial seed, generation,
global genome index, slot, and setting ID.

The full setting table is **not built or uploaded**. Only the 32 KiB constant
centers and 64 KiB incumbent bindings are uploaded each generation. After the
GPU returns the winning setting ID, the C99 host code repeats the same hash for
that one setting and writes the concrete constants and bindings into the
genome. Crossover and mutation therefore see the actual winning model rather
than a hidden setting.

The logical exploratory settings change each generation because generation is
part of their identity, but they are generated on demand inside the kernel.

### 2. Materialized GP settings: Ada's temporary fallback

Ada currently constructs every constant and binding setting on the CPU for
every generation, uploads the 96 MiB table, writes every MSE value, downloads
the 4 MiB score table, and selects winners on the CPU. This exists because the
compact fused reducer needed a correctness fallback on `sm_89`. It should stay
available for validation and diagnostics, but it should not be the normal
search path.

### 3. LM settings

At each LM promotion event, Python currently creates a fresh candidate-local
population around the selected incumbent:

- 8,192 binding settings per candidate;
- four constant starts per binding, for 32,768 thread-owned fits;
- start zero retains the incumbent;
- the other three starts perturb the six optimized constants; and
- each leaf usually keeps its incumbent binding and otherwise samples one of
  the twelve shared-memory bank slots: four current states or eight constants.

These arrays are rebuilt only when LM is triggered for a new candidate batch,
not on every LM iteration. The device allocations and LM module remain loaded
and are reused throughout the solve schedule.

## Settings policy to borrow from Secant-SR

The current hashed path is excellent for memory traffic, but its sampling
policy is mostly independent random mutation and all non-incumbent settings
rotate every generation. Secant-SR's useful lesson is to treat a setting ID as
a **recipe in a virtual setting bank**, with stable and rotating tranches, and
to record which tranches actually produce winners.

Recommended System ID policy:

1. Keep setting zero as the exact incumbent.
2. Reserve a stable tranche for reproducible masks, one-leaf changes,
   state-versus-constant density coverage, deliberate repeated-state choices,
   and deterministic multi-scale constant offsets.
3. Reserve a rotating tranche for fresh Philox/hash exploration.
4. Describe recipes by virtual IDs and generate them in CUDA from the genome's
   incumbents; do not return to full materialized tables.
5. Reproduce the same recipe in C99 only for the winner, materialize it into the
   genome, and replay its score before promotion.
6. Use the same binding-recipe family for LM. Keep a few deterministic
   multi-scale starts per binding rather than spending lanes on many redundant
   random starts.
7. Report exposure, wins, accepted promotions, and quality improvement by
   tranche so the allocation can be tuned from evidence.

The settings policy should be an explicit command-line/configuration choice,
as it is in Secant-SR: fixed legacy, rotating legacy, structured fixed, and
structured rotating. That makes persistence and distribution separately
testable. The compact hashed implementation should remain the production
storage mechanism.

## Multi-server job queue

JSON is the right job boundary, but a directory of Bash scripts is too weak as
the durable source of truth. Bash is acceptable as the remote launch wrapper;
it is poor at atomic claims, leases, retries, heartbeats, validation, and
reconstructing why a scientific result was produced.

The smallest robust design is a **Python-standard-library coordinator on
mac1 backed by SQLite**, plus one small worker process on each GPU server. It
requires no cluster package, database server, containers, administrator access,
Slurm, Kubernetes, Ray, or Celery.

### Control plane

```text
secantq submit job.json
        |
        v
mac1: HTTP/Unix service + SQLite WAL + artifact index
        |
        +---- worker on rohini ---- one exclusive GPU lease ---- Secant CLI
        |
        +---- worker on ada ------- one exclusive GPU lease ---- Secant CLI
        |
        +---- future worker ------- capability/benchmark record - Secant CLI
```

The coordinator should:

- validate the JSON against a versioned schema;
- assign a content-derived job ID;
- atomically lease a compatible idle GPU to one worker;
- store pending, running, completed, failed, and cancelled state in SQLite;
- receive heartbeats and progress/checkpoint summaries;
- expire abandoned leases and retry only according to an explicit policy;
- predict completion from that machine's measured throughput for the job
  class; and
- send Telegram start, seed-complete, failure, and completion notifications.

Each worker should:

- advertise hostname, GPU model, compute capability, VRAM, CPU count, current
  code commit, and per-job-class benchmark rates;
- claim one compatible job at a time per GPU;
- stage the problem bundle into a job-specific scratch directory;
- translate the validated fields into an allowlisted Secant CLI invocation;
- launch it under `tmux` initially, or a user-level service later;
- report progress without giving the job arbitrary shell access; and
- copy compact results, checkpoints, logs, and diagnostics back to mac1.

Running work should survive loss of mac1. The worker keeps the local job alive,
buffers status, and reconnects. `tmux` remains useful for observation but is not
the database or scheduler.

### Job schema

A job should normally target Secant System ID, but the envelope should contain
an engine name rather than baking that assumption into every queue primitive.
Separate the scientific problem from the run request so the same problem can
be searched under several budgets and policies.

```json
{
  "$schema": "secant.job.v1",
  "engine": "secant-system-id",
  "problem": {
    "id": "astaxanthin-fedbatch-v1",
    "spec": "problems/astaxanthin-fedbatch-v1.json",
    "data_sha256": "..."
  },
  "search": {
    "population": 1024,
    "settings": 512,
    "generations": 50000,
    "max_depth": 8,
    "setting_policy": "virtual-bank"
  },
  "lm": {
    "trigger_probability": 0.14,
    "promotion_count": 4,
    "binding_settings": 8192,
    "starts_per_binding": 4
  },
  "replicates": {
    "seeds": [2101, 2153, 2203, 2309]
  },
  "resources": {
    "gpu_count": 1,
    "minimum_compute_capability": "8.9",
    "exclusive_gpu": true
  },
  "outputs": {
    "checkpoint_stride": 25,
    "full_mse": false,
    "retain_modules": true
  },
  "provenance": {
    "git_commit": "...",
    "problem_sha256": "...",
    "template_sha256": "..."
  },
  "notifications": ["start", "seed_complete", "failure", "complete"]
}
```

The problem specification should hold states, known equations, missing sites,
observation times and masks, trajectory data references, loss definition,
operator grammar, parameter bounds, and train/validation/test policy. JSON is
preferable as the canonical machine form because it is unambiguous and easy to
hash. A human-facing YAML file can be accepted later and normalized into this
canonical JSON before submission.

Do not put arbitrary shell commands, Telegram tokens, SSH credentials, or
machine-local absolute data paths in a job. The coordinator stores secrets;
the worker constructs a supported invocation from validated fields.

### Implementation sequence

1. Define `secant.problem.v1` and `secant.job.v1` JSON Schemas.
2. Implement local `submit`, `list`, `show`, `cancel`, and `retry` commands on
   mac1 with SQLite.
3. Implement one polling worker and the lease/heartbeat protocol; run it on
   rohini first.
4. Connect the existing `progress.json`, LM diagnostics, checkpoints, and
   Telegram notifier to job status.
5. Add ada and use measured per-job-class rates for earliest-finish scheduling.
6. Add result/artifact synchronization and completed-seed restart semantics.
7. Only then consider a web interface, cloud provisioning, or a public API.

This is small enough to build and understand, while retaining the pieces that
matter for reproducible long-running searches. Slurm or a general cloud
orchestrator would add more operational surface than this two-to-ten-node
system currently needs.
