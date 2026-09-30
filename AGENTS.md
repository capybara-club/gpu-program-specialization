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

# Guidance for an LLM working with this handoff

Read README.md and docs/ARCHITECTURE.md before selecting components. This is an
execution-technique collection, not a request to build or tune a symbolic
regression product. Secant-SR and historical search tools are reference consumers.
Use docs/PROJECTS.md to select a component and docs/RESEARCH_COVERAGE.md to check
whether a path is integrated, experimental or omitted. Avoid loading vendored
dependencies and the entire provenance manifest as initial context.

- Read the actual public header for the selected component; do not conflate its
  AST encoding, stream ownership or LM path with another project's API.
- Preserve the core boundary: instruction generation, inspection, specialization
  and resource lifetime belong in the core; search/service policy does not.
- Do not infer performance from peak counts without matching workload and timing
  boundaries. Read docs/BENCHMARKS.md and disclose fallbacks or reduced coverage.
- Run tools/check_host.py and tools/audit.py for applicable changes. GPU changes
  additionally require real GPU validation; host/fake-driver tests do not prove
  device correctness. See docs/BUILDING.md.
- Keep binary outputs, datasets, logs, credentials and private machine operations
  out of source control. Do not automatically install missing dependencies.
- Preserve third-party licensing. Owned additions use Charles Durham MIT notices;
  use .license sidecars for formats that cannot accept comments.
- The source manifest describes this handoff's provenance. Do not rewrite its
  original source hashes to disguise subsequent implementation changes.
- Nested component guidance may further constrain a subsystem. Personal machine
  names in historical reports are context, not required deployment targets.
