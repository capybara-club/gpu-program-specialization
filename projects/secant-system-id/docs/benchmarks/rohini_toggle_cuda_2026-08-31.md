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

# Register-resident bit-toggle SSE prototype on Rohini

Date: 2026-08-31
GPU: NVIDIA GeForce RTX 5090 (`sm_120`)
Model: four-state astaxanthin fed-batch system, two missing RHS ASTs, three trajectories, twelve observations, sixteen RK4 steps per observation

## Outcome

The CUDA-only prototype validates the proposed bit-toggle layout. At the recommended provisional shape of 16 packed systems, five toggle bits, 256 unique constant banks per system, and 512 threads per CTA, it evaluates 131,072 distinct system configurations in 0.606 ms:

| Metric | Register-toggle CUDA | Materialized settings | Ratio |
|---|---:|---:|---:|
| Configurations/s | 216.44M | 81.05M | 2.67x |
| ns/configuration | 4.62 | 12.34 | 0.37x |
| Trajectories/s | 649.33M | 243.14M | 2.67x |
| Registers/thread | 61 | 73 | 0.84x |
| Local stack/spill | 0 B | 0 B | equal |
| Requested shared memory/CTA | 624 B | 25,200 B | 0.025x |
| Configuration input bytes | 128 KiB | 12 MiB | 0.010x |

The two GPU paths produced bit-for-bit identical FP32 MSE for all 131,072 configurations, including identical invalid classifications and the same winner for all 16 systems. The planted system and configuration also match the independent CPU reference.

This establishes a useful kernel-level result: implicit toggle bits plus register-resident constants are substantially cheaper than materializing every leaf binding and staging a per-thread state/constant bank in shared memory.

## Configuration mapping

Each AST leaf is encoded as one of:

- a fixed state or constant source;
- a one-bit choice between two state/constant sources; or
- a two-bit choice between four state/constant sources.

The alternatives may be state/state, state/constant, or constant/constant; the encoded leaf-source values determine the combination without a separate runtime kind tag.

The kernel maps work as follows:

```text
system_index  = blockIdx.y
configuration = blockIdx.x * blockDim.x + threadIdx.x
toggle_bits   = configuration & ((1 << toggle_bit_count) - 1)
constant_bank = configuration >> toggle_bit_count
```

With five toggle bits, every consecutive warp covers all 32 leaf permutations. With a 512-thread CTA, its sixteen warps evaluate sixteen constant banks for the same system. Every thread loads its system/bank's eight constants into local scalar variables once, before trajectory integration. Toggle leaves select directly between the RK4 `stage_state` registers and those constant registers.

The per-thread shared state/constant bank and materialized binding array are absent. Shared memory is retained only for the common read-only trajectory reference.

## Packed-system crossover

The fixed-work sweep held total work at 131,072 distinct configurations, used 512-thread CTAs, and changed the number of systems compiled into the uniform `blockIdx.y` switch:

| Packed systems/kernel | Constant banks/system | Registers/thread | Configurations/s | ns/configuration |
|---:|---:|---:|---:|---:|
| 1 | 4,096 | 36 | 280.85M | 3.56 |
| 4 | 1,024 | 40 | 264.76M | 3.78 |
| 8 | 512 | 47 | 232.21M | 4.31 |
| 16 | 256 | 61 | 216.27M | 4.62 |
| 32 | 128 | 95 | 167.41M | 5.97 |
| 64 | 64 | 157 | 512-thread launch rejected |
| 128 | 32 | 255 plus 112 B stack | 512-thread launch rejected |

The toggle operations are not inherently register-heavy. Register pressure comes from asking CUDA/PTXAS to allocate across a large switch containing many separately compiled systems. Sixteen systems/kernel is the current balance between packing, register use, and throughput. Thirty-two remains runnable but is slower; 64 and 128 require smaller CTAs and are not recommended for this source shape.

## Arithmetic semantics

The first CUDA version used ordinary arithmetic expressions under fast math. CUDA legally reassociated and fused visible AST operations, whereas the existing SASS specializer preserves postorder instruction order. Winning configurations still agreed, but some unstable losing trajectories crossed finite/invalid boundaries differently.

The final renderer uses explicit FP32 arithmetic intrinsics for AST add, subtract, multiply, and divide nodes. This preserves the encoded postorder tree. After that change, the toggle and materialized GPU outputs match exactly across the entire batch. It raises the 16-system kernel to 61 registers, but that is still below the materialized kernel's 73 registers.

The sampled Python reference uses float64. One sensitive, non-winning rollout produced MSE 31.6 in Python versus 22.8 in both identical FP32 GPU paths. This discrepancy is retained in the JSON report and is not counted as proof of CPU/GPU numerical identity.

## Specialization allocation fix

`ssid_specialize_kernel` previously allocated and freed its packed-body offset table once per kernel specialization. It now uses a 128-entry, 512-byte caller-stack array and validates the count against that capacity. This matches the established maximum packed-genome capacity and preserves thread isolation without a heap call in the AST specialization function.

Long-lived lifecycle allocations remain in GP creation, pipeline creation, ticket submission, and benchmark setup. They were not removed because they are not part of the per-kernel AST assembly path addressed here.

## Artifacts

- Final report: `generated/toggle_cuda_2026-08-31/final_strict_unique_16x256x512/report.json`
- Generated CUDA: `generated/toggle_cuda_2026-08-31/final_strict_unique_16x256x512/toggle_packed.cu`
- Toggle CUBIN: `generated/toggle_cuda_2026-08-31/final_strict_unique_16x256x512/toggle_packed.cubin`
- Exact materialized baseline CUBIN: `generated/toggle_cuda_2026-08-31/final_strict_unique_16x256x512/materialized_specialized.cubin`

All 98 local unit tests pass, including toggle encoding, permutation materialization, source generation, and existing C99/runtime coverage.

## Material deviations

- This measures resident kernel execution. Compilation, module loading, function lookup, output transfer, and module unloading are reported or exercised separately but are not included in configurations/s. It is not a full eager module-pipeline throughput result.
- The prototype directly compiles packed systems into a CUDA `switch`; it does not yet add these toggle instructions to the C99/SASS specializer or GP mutation/crossover path.
- The prototype currently implements only the dense, aligned fed-batch score shape and writes every MSE. Sparse/irregular observations, multi-horizon archives, and winner-only reduction are not implemented in this path.
- Shared memory is not eliminated entirely: the common trajectory reference still uses 624 requested bytes per CTA. Only the per-thread dynamic leaf bank was removed.
- The constant banks are deterministic, distinct, and bounded around the planted constants for a controlled benchmark. This tests the intended memory and execution behavior, but it is not evidence that this particular constant distribution is optimal for GP search.
- CUDA FP32 trajectory results can diverge from the Python float64 reference on sensitive losing candidates. The exact all-score comparison against the existing validated GPU executor is the authoritative layout-equivalence check here.
