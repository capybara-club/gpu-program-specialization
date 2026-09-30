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

# Prepared binary job protocol — deferred draft

Status: deferred. The initial selected route forwards JSON and parses it in C on
the GPU host. This sketch is only for a later, measured binary-format extension;
no codec or GPU execution entry point exists yet.
Keep this layer independent of NATS so the exact same payload can be validated
locally, transported inline, fetched by descriptor or supplied through a C API.

## Representation

Use an explicitly versioned byte format, not native struct layout. Define
fixed-width integer fields, little-endian encoding, IEEE-754 FP32 bit patterns,
lengths and offsets relative to the start of the message. Never send host/device
pointers, `size_t`, compiler padding, CUDA handles or callbacks.

The logical header contains:

- Magic, protocol major/minor, header bytes and total payload bytes.
- Stable job ID and partition ID; attempt identity belongs to dispatch metadata.
- Format/features required by the producer and worker, including numeric and
  grammar sampling profiles. Reject unknown required capabilities.
- Section count, section table offset and digest of immutable payload contents.
- Requested runtime budgets and declared application workspace requirements.
  The receiving worker independently checks requirements before reserving pools.

Each section has a type/version, offset, byte length, element count and required
alignment. Sections cannot overlap or refer outside the payload. Validate every
offset/count/product before addition, multiplication, pointer formation or copy.
The receiver rejects unsupported versions, duplicate mandatory sections, malformed
instructions and invalid cross-references before GPU submission.

## Sections

| Section | Prepared data |
|---|---|
| States | Ordered state identifiers and counts |
| Trajectories | FP32 times/values, offsets, complete IC layout, observation masks/metadata |
| Fixed RHS | State indices and native postorder instruction bytes |
| Search work | Either typed grammar/rule tables or native AST batch descriptors; distinct formats |
| Constants | Explicit FP32 grids and fixed values, with indexed bindings |
| RNG | Seed, stream identity, pool count, distribution, scope and transform/dependency descriptors |
| Execution | RK4 layout, kernel capacities, configuration/toggle ranges and deadline budget |
| Retention | Typed global/family/tag policies; no embedded retention JSON requiring worker parsing |
| Provenance | Family/AST/bank/permutation identities, tags and requested coverage |

Unpack into caller-provided CPU workspace. Metadata can borrow immutable payload
bytes only if alignment/endian requirements and payload lifetime are satisfied.
Pinned host input buffers remain owned by the execution slot until their H2D
completion event. Device buffers remain owned until their last GPU consumer
finishes. Queue acknowledgements do not establish either memory lifetime.

## Queue descriptor and attempts

The queue may carry a small job inline or a descriptor containing the job/partition
ID, immutable payload digest and length, capability class and bounded fetch token.
mac3 retains descriptor-backed bytes until the accepted result or terminal failure
is recorded. The worker fetches the complete buffer into reserved input storage;
failure to fetch cannot be treated as successful job completion.

Execution is potentially at least once. A disconnect after successful GPU work
but before acknowledgement may cause a second attempt. Results must identify the
job, partition, attempt, worker generation, source digest and compiler/numeric
profile. mac3 accepts one terminal result per partition and never adds duplicate
attempts to logical coverage or customer usage. Physical repeated work is reported
separately. A broker acknowledgement alone does not supply exactly-once results.

Terminal result kinds distinguish success, invalid input, resource rejection,
cancelled, deadline, worker loss and CUDA failure. Report completed/valid/invalid
configuration counts, retention completeness and compact winner provenance.
Permanent input errors do not retry. Retryable worker loss is bounded by attempt
and total-budget policy; cancellation and expired deadlines stop new attempts.

## Grammar expansion placement

Prepared-grammar messages require an explicit serializer for immutable rule,
expression and binding tables, followed by arena-based reference reconstruction.
Native pointers in the current `OdrGrammar` cannot be transmitted directly.

Ready-to-score messages require mac3-owned generation/provenance and bounded
binary AST batches. Preserve source partition IDs and numeric stream identities
across batches/retries. Measure network bytes per AST: a central AST generator
can become bandwidth-limited despite fast JSON parsing.

No implementation should silently select a different mode to avoid an unsupported
grammar feature. Reject unsupported payload features before submission.
