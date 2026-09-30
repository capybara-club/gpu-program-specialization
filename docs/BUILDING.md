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

# Building and checking the collection

## Host checks: the recommended first step

Prerequisites: Python 3.9+, C99 and C++20 compilers, CMake 3.20+, Make, and CTest.
The host checks do not download dependencies, invoke package managers or need
CUDA. Use an existing supported toolchain; the runner reports missing tools.

From the repository root:

```sh
python3 tools/check_host.py
python3 tools/audit.py
```

Host tests are run manually. GitHub Actions runs the source/licensing audit only.

Select a component with `--only secant`, `--only secant-sr`, `--only odezza` or
`--only mm-ptx`. Build products are ignored. The checks cover:

- Secant CPU semantics, header contracts, AST/toggle behavior and fake-driver
  resource-failure handling.
- Secant-SR's host consumer/model/variation/refinement/archive and Python contracts.
- Odezza's arena-based frontend, mutation and work-limit checks, native/Python
  grammar parity, allocator-symbol audit, and compiled-template SQLite cache.
- Six existing PTX toolkit CPU tests, compiled directly without its CUDA-centric
  top-level CMake project.

The AST PTX C++ helper uses constexpr union-member changes accepted under C++20.
AppleClang rejected its existing code under C++17 during packaging; the host
runner explicitly uses C++20. No assertion or compiler diagnostic was disabled.
This is not a claim that every historical C++ consumer has been migrated.

## CUDA builds

Install NVIDIA's supported driver and CUDA development toolkit for the target
platform using its documented installation process. macOS host checks do not
provide CUDA execution. Match the toolkit/GPU to a component's recorded support
and rerun device correctness tests before relying on throughput.

For current Secant and Secant-SR, the sibling layout is intentional:

```sh
cmake -S projects/secant -B build/secant-cuda -DSECANT_ENABLE_CUDA_INTEGRATION=ON
cmake --build build/secant-cuda --parallel
ctest --test-dir build/secant-cuda --output-on-failure

cmake -S projects/secant-sr -B build/secant-sr-cuda -DSECANT_SR_ENABLE_CUDA=ON
cmake --build build/secant-sr-cuda --parallel
ctest --test-dir build/secant-sr-cuda --output-on-failure
```

Check the configure output: Secant can omit the CUDA runner when the toolkit is
not found. A successful host build is not evidence that GPU targets were built.
Do not enable the legacy Secant benchmark/Python options expecting the old
settings API to work with the current toggle API; CMake rejects that combination.

For Odezza, follow [core README](../projects/odezza/core/README.md) and
[core validation](../projects/odezza/core/VALIDATION.md). Its Makefile includes
`test-c`, `test-cuda`, `test-lm-core`, `test-lm-failures`, `test-public-api`, and
frontend/runtime-specific checks. Read the selected target before running it:
some broad historical targets reference omitted scratch experiments. Start with
the native core and the prepared frontend rather than deploying the service.

## Dependency and prototype boundaries

| Component | Additional requirements / caveats |
|---|---|
| MM PTX full CMake tests/examples | CUDA toolkit, CMake 3.22+, Python; CUTLASS for relevant examples |
| Stack PTX Emit | Sibling MM PTX; its original build also enables CUDA |
| CUBIN Function Patch | CUDA, nvPTXCompiler/nvJitLink; test fixtures are compiled during GPU testing |
| FusedSINDy | CUDA and its documented MathDx/cuSOLVERDx/Boost/Python integration dependencies; inspect CMake's dependency provisioning before configuring |
| Compiler workers | Parent `mm_ptx_headers` target, CUDA, Threads, optional OpenMP, NNG; set `STACK_PTX_COMPILER_FETCH_NNG=OFF` to prohibit its historical FetchContent path |
| MM PTX Python | Declared scikit-build-core/nanobind workflow; versioned embedded header copy |
| MM Kermac | Prototype parent-build assumptions, CUDA/cuTENSOR, GLFW/Vulkan and related dependencies |

CUTLASS was a submodule in MM PTX, pinned to
`e6e2cc29f5e7611dfc6af0ed6409209df0068cf2`. It is not vendored here.
Its upstream is [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass). Original
submodule declarations are omitted to avoid pretending this fresh monorepo has
their Git links. Acquire optional dependencies through your chosen supported
dependency process before attempting the corresponding full builds.
For the full MM PTX CMake build, explicitly configure
`-DCMAKE_CXX_STANDARD=20` for the existing AST PTX C++ helpers.

There is deliberately no root "build everything" command that silently fetches
dependencies or deploys services. The source collection includes independently
developed prototypes. The main host checks are a bounded, reproducible entry point.

## Selected research checks

The selected research sources are independent references. These commands use
existing host tools and do not install dependencies:

```sh
make -C projects/odezza/research/ast_tools test
python3 -B projects/odezza/research/rosenbrock_cpu_trial/test_cpu.py
```

AST tools cover C99 enumeration and Python parity. The second command compiles
small native CPU solver fixtures and checks seven numerical/contract cases.
The GPU Rosenbrock/RFM/plain-CUDA LM trials have additional CUDA or scientific
Python requirements; their historical results are not rerun by these commands.

For Secant-SINDy's independent numerical reference, configure its CMake project
and build the named `secant_sindy_cpu_test`, `secant_sindy_stlsq_edge_test`,
`secant_sindy_source_test` and `secant_sindy_cpp_header_test` targets. Those four
checks do not establish compatibility of its older Secant integration adapter.
The standalone Secant Benchmark and cuSR trees are historical references, not
additional backends for the current Secant API.

Maintainers reconstructing the snapshot from local source checkouts can run
`tools/assemble_snapshot.py` followed by `tools/include_research_snapshot.py`.
The latter adds the curated older repositories and research sources; neither
tool downloads source or imports an old Git history.

## Repository and contributions

The public repository is
[capybara-club/gpu-program-specialization](https://github.com/capybara-club/gpu-program-specialization).
Its history starts from the curated source snapshot. Original development
histories, datasets and private operational artifacts are not imported.

Run the host checks and source audit before contributing changes; see
[CONTRIBUTING.md](../CONTRIBUTING.md). Original source provenance remains in
the manifest, and subsequent changes are recorded in ordinary Git commits.
