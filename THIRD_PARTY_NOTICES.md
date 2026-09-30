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

# Third-party notices

The root MIT license applies to Charles Durham's material, including his Meta
Machines / MetaMachines projects. It does not replace the licenses below.

| Material | Location | License / provenance |
|---|---|---|
| Boost.Preprocessor subset | MM PTX `thirdparty/boost_subset/`; MM PTX Python `src/mm_ptx/include/boost/` | Boost Software License 1.0; original headers retained; [license text](LICENSES/BSL-1.0.txt) |
| SQLite amalgamation | Odezza `third_party/sqlite/sqlite3.c`, `sqlite3.h`, `sqlite3ext.h` | Public domain dedication in the files; original bytes retained; [upstream licensing](https://sqlite.org/copyright.html) |
| incbin by Dale Weiler | Vendored `incbin.h` copies identified in the file manifest, including the added historical/statistics components; CUBIN Function Patch's `thirdparty/incbin/` sources and documentation | Unlicense/public domain; original files retained; [license text](LICENSES/Unlicense.txt), [upstream](https://github.com/graphitemaster/incbin/blob/master/UNLICENSE) |

SQLite's source archive identity is recorded in its retained
[dependency manifest](projects/odezza/third_party/sqlite/manifest.json).
The collection's [file manifest](docs/source-manifest.json) identifies these
exceptions individually and preserves original hashes.

CUDA, CUB/CCCL, nvPTXCompiler, nvJitLink, cuTENSOR, MathDx/cuSOLVERDx, CUTLASS,
NNG, nanobind, Python packages, benchmark tools and datasets are separate
dependencies under their respective terms. They are not relicensed by this
handoff. Some optional historical build files can acquire dependencies; read
docs/BUILDING.md before using them.

The embedded PTX Inject, Stack PTX, AST PTX and CUBIN Function Patch headers are
Charles Durham's code even when located in a directory called `thirdparty`.
They have been updated under the owner's explicit MIT authorization. Their
historical versions are retained for consumer compatibility.
