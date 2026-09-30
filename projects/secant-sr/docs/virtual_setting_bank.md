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

# Virtual Dynamic-Leaf Setting Bank

Secant-SR exposes setting distribution and persistence as independent policy
axes:

```text
legacy             Legacy 50/50 distribution, fixed for the trial.
legacy-rotating    Legacy 50/50 distribution, regenerated every generation.
virtual-bank-fixed Structured distribution, fixed at generation zero.
virtual-bank       Structured distribution with rotating tranches.
```

`legacy` remains the default. The four modes form the paired ablation:

| Distribution | Fixed | Rotating |
|---|---|---|
| Legacy 50/50 | `legacy` | `legacy-rotating` |
| Structured virtual | `virtual-bank-fixed` | `virtual-bank` |

Select the rotating structured policy with:

```text
--leaf-settings 8192 --leaf-setting-policy virtual-bank
```

The virtual setting ID selects a recipe. Philox4x32-10 supplies deterministic
random words for recipes that require sampling. The trial seed, generation,
cohort, setting ID, and sampling stream form the random identity. Stable
tranches deliberately omit generation and cohort from that identity.

## Tranches

| Setting IDs | Stability | Implemented behavior |
|---|---|---|
| `0-255` | Stable | Explicit mask coverage and deterministic column/constant anchors. Eight holes enumerate all 256 masks. Four holes enumerate 16 masks with 16 assignments each. |
| `256-1023` | Stable | Column count is uniform over `0..D`; each mask has exactly that popcount. |
| `1024-2047` | Stable | Original independent 50/50 column-or-constant policy as a control group. |
| `2048-3071` | Rotating | One ordered hole pair is selected by generation and cohort. Up to 32 columns receive explicit ordered-pair coverage, including repeated columns. |
| `3072-6143` | Rotating | Exploratory fallback with uniform column density and a mixture of pool and bounded continuous constants. This is reserved for learned sampling but is not represented as learned yet. |
| `6144+` | Rotating | Fresh exploration with uniform column density, bounded exploratory constants, and deliberate repeated-column settings. |

The pair tranche is generated separately for full-dynamic and mixed cohorts.
This matters because a four-hole mixed AST requires four-hole mask-density
coverage; interpreting an eight-hole table through only its first four slots
does not preserve that distribution.

## Search Integration

At evaluator creation, Secant-SR generates generation zero and uploads it with
the other evaluator inputs. Fixed policies leave that device storage unchanged.
At each later active dynamic generation, rotating policies:

1. Regenerates the full-dynamic bank on the CPU.
2. Copies it into the existing device allocations.
3. Regenerates and copies the mixed bank when the maturity policy is active.
4. Runs the already-created evaluators and CUBIN modules.

No evaluator, runner, CUBIN module, or device allocation is recreated. Runs are
blocking already, so replacement happens between completed generations and
does not add a device-wide synchronization.

For 8,192 settings, eight allocated dynamic slots, and both full and mixed
banks, each generation replaces about 576 KiB of device input. A Release smoke
run on the RTX 5090 measured approximately 1.01 ms per generation for CPU
generation plus both host-to-device copies. This fixed cost was about 12% of an
intentionally tiny 64-AST test generation and should be measured again on
representative populations. The legacy policy performs no per-generation
setting generation or copy.

## Current Boundary

The `3072-6143` range is not learned yet. In `virtual-bank`, it intentionally uses fresh,
structured exploration until exposure-normalized telemetry and a trustworthy
cohort-level learned descriptor exist. This avoids silently turning early
search noise into a self-reinforcing sampling policy.

`virtual-bank-fixed` freezes the complete generation-zero realization,
including ranges described as rotating above. It exists to distinguish the
quality of the structured distribution from the effect of losing exact-setting
continuity. Likewise, `legacy-rotating` isolates continuity while retaining the
legacy distribution. Neither ablation policy changes the default search path.

Run the paired shortlist with per-generation SSE trajectories using:

```text
CUDA_MODULE_LOADING=EAGER python3 python/benchmark_leaf_setting_ablation.py \
    --output-dir scratch/leaf_setting_ablation_2x2
```

The output directory contains per-policy trial and generation CSVs plus
`summary.csv`, `paired.csv`, and `trajectories.csv`.

The first completed experiment and its interpretation are recorded in
[`leaf_setting_ablation_20260810.md`](leaf_setting_ablation_20260810.md).
