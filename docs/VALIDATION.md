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

# Handoff validation — 2026-09-30

## Clean-checkout packaging audit

The first GitHub Actions run (`36762377659`, root commit `cb09c499`) stopped at
the license audit: `million-2048-request.json.license` existed locally but a
component ignore rule kept it out of Git. The local audit had accepted that
ignored sidecar, so its earlier pass did not prove complete license packaging.
The CPU checks had not started in that run; kernel behavior was unaffected.

The sidecar is now included explicitly, and the audit only accepts sidecars
present in its exported-file inventory. The corrected audit reproduced the
failure locally before the sidecar was included, then passed with it included.
GitHub Actions runs the source audit from a clean checkout. CPU tests are manual;
they are not run automatically on push or pull requests.

## Current snapshot

The source audit and full selected host runner passed for the 15-component
snapshot. The project map, source manifest and export tooling describe the same
component inventory. MM Kermac is documented as a secondary PTX-injection
example. No GPU execution was rerun for this packaging task.

A documentation consistency check confirmed agreement between the README's
component count, project-map links, manifest entries, export lists and actual
project directories. The source audit was rerun after the documentation cleanup;
execution code was unchanged.

## Selected research checks

Additional checks against the copied research source passed:

- AST tools: C99 enumeration/evaluation checks and all 10 Python parity,
  serialization, toggle and sampling tests.
- Native Rosenbrock/RK4 CPU reference: 7/7 tests, including FP32 arithmetic,
  stiff masked integration, Jacobians and failure behavior.
- Secant-SINDy: 4/4 selected standalone CPU, STLSQ edge, generated-source and
  C++ header checks. Legacy Secant integration and GPU checks were not run.

The retained RFM, GPU Rosenbrock, CUDA LM controls, cuSR and historical benchmark
paths are preserved references; these checks do not requalify their GPU
behavior or performance. Their scope is described in
[RESEARCH_COVERAGE.md](RESEARCH_COVERAGE.md).

## Initial engine snapshot

Validation ran against the **copied source**, on macOS/AppleClang 21, using the
available CMake, Make and Python tools. No package installation was needed.
The entire host runner also passed from a fresh checkout of only the staged
source, with no inherited build directory. This checks that the selected tests
do not depend on untracked files left behind in the working snapshot.

| Check | Outcome |
|---|---|
| Secant host CMake build + CTest | 6/6 passed, including fake-driver failure/lifetime tests |
| Secant-SR host CMake build + CTest | 11/11 passed |
| Odezza arena/frontend suite | Work limits, trajectories/static parsing, grammar cursor, contracts and mutation tests passed |
| Odezza frontend allocation audit | No allocator, file/database or CUDA symbols in the CPU archive |
| Odezza native/Python grammar parity | Eight supplied grammar examples matched, up to 32 native programs per example |
| Odezza SQLite cache | Cold/warm, corruption, eviction, oversize, fresh/concurrent initialization and schema tests passed |
| PTX toolkit selected host tests | Five C tests passed; C++ AST test passed under C++20 |

The frontend mutation test performed 18,000 measurement calls; it reported 817
successfully measured and 798 built arenas. Rejected malformed inputs are part
of that test, not missing successful work.

## C++ language-standard finding

The first standalone AST PTX C++ compile used C++17 and failed on a constexpr
union-member assignment. The same existing test passes with C++20. The new host
runner explicitly selects C++20; no diagnostic or test was suppressed, and the
original generated helper was not altered. This limits the host validation claim
to that standard; other historical consumers may need their build settings
updated.

The macOS linker/compiler also emitted existing incbin/App Store bitcode and
duplicate `-lm` warnings. These did not affect host test completion. This handoff
is not an App Store build configuration.
Historical whitespace is retained, including in third-party files. A whole-import
`git diff --check` reports inherited trailing whitespace; the new handoff docs
and tools are checked separately. This is not a wholesale formatting rewrite.

## What was not tested

- No GPU benchmark or GPU numerical suite was rerun for this packaging task.
- Optional Python extension builds were not run; no missing tools were installed.
- Archived remote compiler workers, service deployment, Kermac and FusedSINDy
  integration environments were not reconstructed.
- Historical benchmark numbers are retained reports, not measurements of this
  new repository layout.

`tools/audit.py` additionally checks export hashes, owned license coverage,
byte-preserved third-party sources, file size/type, common credential patterns,
Python/JSON syntax and links in the new handoff documentation. It is a targeted
check, not a universal security or licensing guarantee.
