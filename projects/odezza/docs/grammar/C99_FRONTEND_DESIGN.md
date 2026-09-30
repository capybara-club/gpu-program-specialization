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

# C99 request frontend and family scheduling

Recorded 2026-09-11. The phased C99 frontend, bounded family cursors and C-managed
scoring-template bridge are now implemented in [frontend/](../../frontend/README.md).
This document retains the broader design, including scheduler/service work that is
not yet implemented. See the [implementation validation](C99_FRONTEND_VALIDATION.md)
for the exact current boundary.
The user requires no database and no internal allocation: callers supply workspace
for parsing and AST generation. Keep this frontend separate from the hardened
kernel/core API. The existing Python compiler remains the compatibility reference.

## Parse into an immutable grammar plan

Read the complete request once into compact tables: state/symbol names, fixed RHS
programs, grammar template nodes, rule alternatives, formal arguments, shared-shape
references, family roots/overrides/tags, coefficient/toggle axes and RNG/prelude
descriptors. Trajectories become typed arrays with offsets, masks and complete ICs.
This represents the generator, not an expanded population. Numeric axes remain
descriptors rather than one AST copy per configuration.

An explicit expansion cursor references that plan. Each family keeps its own
choice/recursion stack, local-binding scope and sampling coordinates. Emit bounded
batches directly as native postorder AST bytes plus configuration descriptors.
Rule holes choose independently; repeated shared-shape references preserve their
shared choice. Known RHS programs are prepared once and referenced by candidates.

Separate policy from mechanics: parser produces the plan; expansion advances a
family cursor; scheduler chooses which cursor and work range to advance; the core
consumes the prepared batch. No MSE-dependent policy belongs in the parser/core.

## Caller-owned memory contract

- All frontend-owned dynamic state resides in supplied buffers, including parser
  stacks, diagnostics, symbol tables, cursors, duplicate tables and AST output.
  No malloc/calloc/realloc, database, or hidden disk fallback.
- Allow one arena partitioned into persistent plan/data, mutable cursor/scratch,
  and reusable output regions; separate buffers are useful for overlapping batches.
- Provide a sizing/counting operation for parse requirements and capacity-based
  runtime requirements. Size the parsed grammar, never enumerate its full language
  merely to measure parse memory. Check integer overflow and report alignment.
- Persistent strings/metadata are copied into the plan arena so the input JSON
  can be released after successful parsing. Plan/data live while cursors use them;
  a batch buffer stays valid until the pipeline finishes consuming it.
- Workspace exhaustion is explicit, with phase/capacity diagnostics; no silent
  dedup eviction, family dropping or alternate execution. Parsing failure produces
  no usable plan. A full output buffer returns a completed partial batch and keeps
  the next item pending; it must not consume or lose that item.
- Exact population-wide dedup requires memory proportional to remembered entries.
  Bound and size it explicitly. Hash matches need exact identity comparison; a
  hash table alone is not proof of equality. Unlimited exact dedup in fixed memory
  is not promised. Retain only the replay information required for final winners.
- Allocation-free applies to this frontend. NVRTC/CUDA and the existing native
  runtime have their own allocation/lifetime contracts.

## Confirmed current family starvation

`Compiler.records()` visits families sequentially under shared global limits.
Family entries currently have no independent structure/configuration budget.
`toggle_sampling.count` controls a different dimension and is not a family quota.
The entire configuration product of a variant must fit; otherwise generation
stops globally instead of trying later families or emitting a partial range.

Two CPU reproductions against the unchanged compiler on 2026-09-11:

| Request | Actual result |
|---|---|
| Family A offers `x`, `x*x`; B offers `sin(x)`; max_skeletons=1 | A emits `x`; B is never reached; stop=max_skeletons |
| Family A offers `k*x` with k=[1,2,3]; B offers `sin(x)`; max_configurations=2 | No variants emitted; B is never reached; stop=max_configurations |

This affects capped requests in the current imported v1 compiler. Earlier fully
exhausted multi-family fixtures remain valid; no retrospective claim is made that
all prior jobs starved. Repair the allocation/scheduling semantics before comparing
capped multi-family coverage or starting new campaigns under a fairness claim.

## Proposed family allocation and interleaving

Updated after the user's supplied allocation recommendation: use explicit reserved
per-family limits and derive request totals by checked addition. A second user
global limit is optional; server admission ceilings still apply. If an explicit
global limit conflicts with the family allocations, reject before GPU admission,
not by truncating later families. Receipt of a queued job handle can remain
immediate; admission/preparation success is a separate observable state.

Unused allocations stay unused by default (`redistribute_unused: false`). This
supersedes the earlier automatic-redistribution suggestion. Redistribution would
be an explicit later policy with original/revised allocations reported. A
request-wide-only budget can later be normalized to equal or weighted integer
family allocations before generation; record exact allocations and deterministic
remainder handling. Explicit allocations alone are sufficient for the first C99
implementation. Do not silently invent allocations for ambiguous mixed modes.

Interleave bounded chunks from active family cursors independently of allocation.
Optional per-skeleton evaluation allowances prevent one giant configuration bank
from consuming a whole family budget before other structures are evaluated.
Limits are maxima, not promises that filtered/exhausted families can fill them.
One multi-family submission and several single-family submissions can use the
same scheduling units; transport boundaries need not dictate scheduling.

Keep three accounting dimensions separate:

1. Accepted distinct structures per family, excluding filtered/duplicate proposals
   within that family. Report new globally unique structures separately because
   families may overlap; attach all discovered family/tag memberships.
2. Actual configuration evaluations, including coefficient rows and toggle
   assignments. One accepted structure can imply millions of evaluations. Invalid
   evaluated configurations consume work budget; rejected pre-execution proposals
   consume generation effort instead. Report allocated, dispatched, completed,
   finite/invalid, unused counts and explicit stop reasons separately.
3. Generation effort (attempts/expansion steps), including rejected and duplicate
   proposals, plus time limits. A family producing no acceptable structures must
   yield control rather than starving the others.

Use both expansion-step and configuration-work quanta. A configuration product
can be resumed as explicit index ranges; do not force a giant first bank to run
before other families get coverage. The configured policy must say whether ranges
are prefixes or sampled; interleaving alone is not uniform sampling of structures
or coefficients. Retain original coordinates and report partial products explicitly.

The present native scoring launch covers `bank_count * 2^active_toggle_count`,
with toggle bits fastest in native indexing. The grammar adapter's public index
is `permutation * numeric_count + bank`. Arbitrary public start/count slices are
therefore not automatically legal native slices. First support resumable bank
ranges across the full toggle set, which match existing kernel shapes. If a
remaining allocation cannot fit one such group, report the unused remainder and
granularity reason and continue eligible work; never overspend or claim full
coverage. Arbitrary toggle subsets require explicit frontend mapping or a separately
reviewed native capability. Validate the selection/replay mapping before advertising
the proposal's general `configuration_start/configuration_count` interface.

Configuration selection order is also a search policy: a prefix of public indices
can spend its whole budget on one toggle assignment. Native bank-range batches
cover all toggles for their chosen coefficient rows. Record which policy actually
ran, preserving pool coordinates and RNG values under chunking.

Logical fairness does not require one GPU launch per family: pack compatible work
from different families into the same native batch with provenance intact. Choose
quantum sizes by throughput and feedback-latency measurements, not an assumed magic
batch size. Per-family PRNG/cursor state must be independent of physical launch order.

## Compatibility and acceptance

Balanced scheduling changes which structures survive a cap. Introduce an explicit
versioned policy; retain sequential behavior for reference comparisons and do not
silently reinterpret v1 requests. Likewise preserve/version seeded sampling and
addressing rather than claiming that a new C PRNG matches Python's random module.

Validate parse/expansion parity, shared bindings, local scope, exact dedup, global
and family budgets, configuration slicing, rejected-branch fairness, buffer limits,
allocation absence and index replay. Use fully exhausted finite grammars for set
parity and explicit sequential mode for ordered/capped parity. Benchmark the C99
implementation separately from changes to allocation policy, then validate the
new balanced policy's per-family coverage and recovery behavior.
