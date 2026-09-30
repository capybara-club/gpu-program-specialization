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

# Codex handoff: connect the ODE grammar compiler to the execution engine

Prepared September 11, 2026. This archive is self-contained. No earlier chat,
download, account, or network access is needed to inspect and test the compiler.
The user's GPU engine source is not included; inspect it in the destination
workspace before selecting an ABI, kernel profile, or MCP implementation.

## What is ready

- A Python 3.10+ standard-library grammar compiler, under
  `ode_grammar_compiler/`. This is the **only compiler source copy** in this
  archive. Its implementation is unchanged from the validated prototype.
- Safe infix expressions, bounded structural generation, binary/quad state
  toggles, constant Cartesian products, RNG-bank/prelude plans, shared named
  shapes, family/tag provenance, compact JSONL, and separate LM metadata.
- The complete 83-test compiler suite, core examples, biological requests,
  reproducible example builders, and structural/semantic verification evidence.
- A historical MCP/product design, explicitly separated from implemented syntax.

No GPU adapter, production Philox kernel, prepared-observation store, trajectory
solver, score reduction, top-k service, LM optimizer, or MCP server is supplied.
The task for the next Codex instance is to connect this compiler to the existing
engine and implement the missing orchestration, not recreate the grammar.

## Read in this order

1. `INTEGRATION_HANDOFF.md`: engine context, actual output contract, integration
   steps, acceptance criteria, and unresolved backend decisions.
2. `ode_grammar_compiler/README.md`: implemented request syntax and examples.
3. `ode_grammar_compiler/IMPLEMENTATION_CONTRACT.md`: implementation details;
   handoff copy corrected to use the real output fields.
4. `ode_grammar_use_cases/README.md`: executable biology examples and proof of
   their expansion counts.
5. `DOCUMENT_MAP.md`: status of all documents and differences from the earlier
   design. Read `design/ode_search_mcp_design.md` for architecture rationale,
   not as an executable schema.

`CODEX_TASK.md` is a pasteable task brief to accompany this archive.

## Check the bundle

From the archive root:

```bash
python verify_handoff.py
```

This runs the compiler tests, small semantic checks, sample-stream checks, and
the stored request-hash checks. It does not integrate trajectories or contact a
GPU. Reproduce all five larger use-case expansions with:

```bash
python verify_handoff.py --full
```

Python 3.10+ is enough; installation and third-party packages are not required.
`pyproject.toml` is included for later packaging, but installing its build
dependencies is unnecessary for these commands.

## Start integrating with a small fixture

```bash
python ode_grammar_compiler/compile_grammar.py compile ode_grammar_compiler/examples/constant_rng_product.json -o product.jsonl
python ode_grammar_compiler/compile_grammar.py compile ode_grammar_use_cases/requests/adaptation_toggles.json --compact -o toggles.jsonl
```

Use `configuration_indices`, `evaluate_prelude`, and `evaluate_postorder` as
reference helpers for a tiny candidate's inputs and RHS values. Then connect
actual engine integration, masked scoring, top-k retention, and separate LM.

## Verified example scale

| Request | States | Distinct opcode-only shapes | Configuration visits | Active fit scalars |
|---|---:|---:|---:|---:|
| Four-gene circuit | 8 | 10,000 | 1,000,000 | 8 |
| Same circuit with constant grid | 8 | 10,000 | 1,000,000 | 8 |
| Six-flux enzyme pathway | 4 | 15,625 | 1,000,000 | 6 |
| Adaptation families | 3 | 8,192 | 1,048,576 | 6 |
| Binary-source adaptation example | 3 | 64 | 2,048 | 6 |

These five requests all exhaust their bounded grammar. Configuration counts
describe pools, not completed solves. `evidence/` under the use-case directory
contains their exact request hashes, selected output records, and verification
results. Parameter ranges are illustrative; no biological fit was performed.

## Essential constraints from the user

- Each GPU thread integrates its own ODE candidate; threads within a warp use
  the same executable arithmetic skeleton.
- State leaves can vary across configurations through exactly 2-way or 4-way
  toggles. They read evolving states throughout the solve.
- RHS structure preparation is very cheap in the user's engine. Avoid turning
  every numeric candidate into a separately compiled program.
- Knowns are prepared before mutable RHS batches. Planned observation support
  includes regular/irregular times, entire unobserved states, and sparse masks.
  Check the actual engine for what is already implemented.
- Only RK4 is currently reported as supported. The scoring state limit is
  unspecified here; do not assume it is 8 or 16.
- The reported LM kernel supports up to 8 states and 8 fitted constants, with
  additional non-fitted constants. Scoring and LM are separate requests.
- Return top-k globally and per family/tag, with replayable candidate identity.

The user's reported peak rates (about 1M distinct skeleton preparations/s and
up to 768M short-trajectory configurations/s) are workload-specific engine
claims, not measurements of this Python compiler. Keep host expansion, engine
preparation, integration, reduction, and transport timings separate.
