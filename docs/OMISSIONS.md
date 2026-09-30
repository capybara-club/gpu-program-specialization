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

# Snapshot scope and omissions

The goal is reusable source and explanations, not an archive of every experiment
or the original machines. The per-file inventory and exclusion reasons are in
[source-manifest.json](source-manifest.json).

## Deliberately omitted

- All original `.git` directories and histories, nested worktrees, local virtual
  environments, compiled binaries/modules and build outputs.
- Bulk scratch/run directories, raw benchmark output, logs, large result banks,
  trajectories/datasets, copied research PDFs and generated plots.
- Private notification/deployment helpers and credentials. Historical reports
  can mention machine labels; they are not live configuration instructions.
- External source submodules such as CUTLASS. Selected small vendored source
  dependencies needed for the host paths remain, under their own licenses.
- Separate unrelated applications and research projects not needed to explain
  this program-specialization lineage.

Small grammar specifications, synthetic requests and test program corpora are
retained when they describe an input/API rather than record a bulk campaign.
Compact benchmark tables and human-readable reports remain as evidence.
Selected research sources live under `projects/odezza/research/`; see
[RESEARCH_COVERAGE.md](RESEARCH_COVERAGE.md).

## Historical references

Component READMEs and reports retain context from their original repositories.
Some references to raw output, plots, machine paths, scratch scripts or sibling
applications do not resolve in this source-only snapshot. They are historical
references, not required steps in the main handoff's host validation. Start with
the top-level README and docs/BUILDING.md for supported entry points.

The audit checks local links in the **new top-level documentation**. It does not
claim all historical report links resolve. This distinction prevents silently
discarding useful reports or pretending omitted experimental data is present.

## Intentional export edits

Owned files received Charles Durham MIT notices; existing Meta Machines and
Charlie Durham notices were updated. JSON, text fixtures and other formats that
cannot safely take comments use `.license` sidecars. Third-party source is
preserved byte-for-byte and audited against its recorded original hash.

Odezza's copied root AGENTS.md is adapted for public handoff rather than requiring
a private scratch TODO and personal conversational reminders. Other subsystem
boundaries remain. Citation names and owned author metadata use Charles Durham.
No kernel/search algorithm was changed as part of this export. The host test
entry point uses C++20 for the existing AST PTX constexpr helpers.

The source manifest records hashes before and after export. The original working
repositories were left intact. This is a snapshot, not a synchronization mechanism
between the new collection and those development checkouts.
