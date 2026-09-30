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

# Generated module metadata and inspection format

## Source manifest

Every generated CUDA source begins with a versioned JSON manifest delimited by
these exact lines:

```text
/* SECANT_SYSTEM_ID_MANIFEST_BEGIN
{ ... }
SECANT_SYSTEM_ID_MANIFEST_END */
```

The manifest describes information known before compilation. Version 2 binds
one direct or packed kernel. Version 3 makes the module the primary artifact
and records:

- every kernel name, its global genome base, and packed dispatch dimensions;
- total kernel count and module genome capacity;
- model name, state names, missing sites, leaf counts, and leaf offsets;
- constant, trajectory, observation, input, and output counts;
- dense aligned trajectory layout and observation interval;
- per-genome patch capacity and packed dispatch dimensions; and
- marker, post-order, generator, and manifest ABI versions.

The same generator object produces both the CUDA body and this manifest. The
manifest is not used as evidence for physical facts such as registers or file
offsets; those are recovered from the CUBIN.

## Source/CUBIN identity

The generator normalizes the template-ID declaration to zero and hashes the
canonical semantic manifest together with the exact normalized CUDA payload.
The complete 256-bit SHA-256 digest becomes `template_id`, and the manifest
records `template_id_algorithm` as `sha256`. The actual source payload contains:

```cuda
extern "C" __device__ __constant__ unsigned int ssid_template_id[8] = {...};
```

NVCC/NVRTC preserves this as a 32-byte `ssid_template_id` object in the
CUBIN's `.nv.constant3` section. Inspection verifies all of the following:

1. the source payload SHA-256 matches the manifest;
2. the normalized source and semantic manifest reproduce `template_id`;
3. the CUBIN contains exactly one 32-byte template-ID symbol;
4. its value equals the source template ID; and
5. the declared dimensions agree with marker and target-table inspection.

The template ID is intentionally architecture-independent: the identical CUDA
source may produce valid `sm_89`, `sm_90`, and `sm_120` CUBINs. The inspection
record stores a separate SHA-256 for the exact CUBIN.

## Inspection document

Successful multi-kernel inspection emits version 2 JSON with this top-level shape:

```json
{
  "schema": "secant-system-id.module-inspection",
  "schema_version": 2,
  "status": "verified",
  "identity": {
    "template_id": "...",
    "source_sha256": "...",
    "cubin_sha256": "..."
  },
  "declared": {},
  "observed": {},
  "verification": {}
}
```

`declared` is the complete source manifest. `observed` contains the CUBIN
architecture, byte size, template-ID symbol, function location, register
count, register-count metadata offsets, and the physical specialization plan.
For a direct site this includes marker inputs, outputs, available registers,
wait masks, and patch boundaries. For a packed site it additionally contains
dispatch instruction words and offsets, compact-arena boundaries, cleanup
locations, and target-table entries. A version-3 source manifest produces an
observed `packed_module` record containing one complete packed plan per kernel
entry point.

`plan_from_inspection(document, cubin)` verifies the stored CUBIN hash and size,
then reconstructs `CubinPlan`, `PackedCubinPlan`, or `PackedModulePlan`. This
makes the JSON a usable intermediate artifact rather than only a human-readable
report, while preventing a stored plan from being applied to the wrong binary.

The C99 hot path does not parse this JSON. Python converts a verified packed
module plan into fixed-width `ssid_template_desc` structures once, and
`ssid_template_create` validates and deep-copies those structures. Both Python
through `ctypes` and the future C99 GP loop therefore submit the same flat ABI.
A compact binary sidecar may later persist these structures for a standalone C
process without changing the specializer or module pipeline.

## Multiple kernels in one module

`packed-module-generate` emits entry points named `ssid_score_packed_0`,
`ssid_score_packed_1`, and so on. Each has an independent branch-target table,
compact SASS arena, register contract, and global genome range. The module has
one SHA-256 identity and compiles to one CUBIN.

Module-wide inspection recovers every arena. Module-wide specialization patches
all populated ranges into one output CUBIN. The runtime performs exactly one
`cuModuleLoadData`, resolves each populated function with
`cuModuleGetFunction`, and launches the entry points over shared settings,
bindings, reference data, and output storage.

## Pipe-oriented commands

Generate, compile, and inspect using files:

```sh
PYTHONPATH=src python3 -m secant_system_id.generate > generated/template.cu
PYTHONPATH=src python3 -m secant_system_id.compile_stdin --arch sm_120 \
  < generated/template.cu > generated/template.cubin
PYTHONPATH=src python3 -m secant_system_id.inspect \
  --source generated/template.cu generated/template.cubin \
  > generated/template.inspection.json
```

The source or CUBIN may independently use standard input:

```sh
cat generated/template.cu | PYTHONPATH=src python3 -m secant_system_id.inspect \
  --source - generated/template.cubin

cat generated/template.cubin | PYTHONPATH=src python3 -m secant_system_id.inspect \
  --source generated/template.cu -
```

They cannot both use standard input in the same invocation. The older
CUBIN-only command remains available when `--source` is omitted, but it cannot
produce a source-bound verified module record.

Generate a two-entry packed module directly:

```sh
PYTHONPATH=src python3 -m secant_system_id.cli packed-module-generate \
  --kernels 2 --genome-capacity 8 --genomes-per-cta 8 \
  -o generated/fedbatch_module.cu
```
