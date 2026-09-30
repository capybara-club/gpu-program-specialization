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

# stack_ptx_compiler

> **Collection category:** Bindings and execution infrastructure. See the [project map](../../docs/PROJECTS.md) for entry points and status, and [research coverage](../../docs/RESEARCH_COVERAGE.md) for limitations.

## Overview

`stack_ptx_compiler` compiles stack-PTX programs (arrays of `StackPtxInstruction`) into CUDA cubins. It is meant to be embedded as a C library and can compile locally (single-core/OpenMP) or via a remote NNG server. It depends on `mm-ptx` for the `stack_ptx`/`ptx_inject` APIs.

## CMake integration

`mm-ptx` must be added before this project so the `mm_ptx_headers` target and codegen helpers are available:

```cmake
add_subdirectory(thirdparty/mm-ptx)
add_subdirectory(thirdparty/stack_ptx_compiler)
```

Notes:
- Requires CUDA Toolkit with `nvptxcompiler` (CMake checks for `CUDA::nvptxcompiler_static` or `CUDA::nvptxcompiler`).
- NNG is required for the NNG backend; it can be fetched when `STACK_PTX_COMPILER_FETCH_NNG=ON` (network required).

## Targets

- `stack_ptx_compiler`: local backend only.
- `stack_ptx_compiler_nng`: adds the NNG client backend (requires `nng`).
- Apps: `stack_ptx_nng_server`, `stack_ptx_nng_client` (controlled by `STACK_PTX_COMPILER_BUILD_APPS`).

## Scripts

- `scripts/build_stack_ptx_nng_server_sbsa.sh`: builds a Linux/arm64 `stack_ptx_nng_server` in a Podman container, extracting `nvPTXCompiler` from a CUDA runfile.
- `scripts/deploy_stack_ptx_nng_server_sbsa.sh`: copies a prebuilt arm64 server to a remote macOS host and runs it in a Colima container.
- `scripts/deploy_stack_ptx_nng_server_sbsa_vz.sh`: same deployment flow but forces the macOS VZ backend (with stricter OpenMP/runtime checks).
- `scripts/deploy_compile_server.sh`: deploys a separate OpenMP compile-server example to a macOS host via Colima (expects the `mm-ptx-remote` style repo layout).
- `scripts/kill_compile_server.sh`: stops the remote compile-server container/process.

The sections below go into implementation details for reference.

## Directory layout

- `stack_ptx_compiler.h`: Public API (create/destroy/submit/poll).
- `stack_ptx_compiler.c`: Dispatcher: holds a vtable + backend impl pointer.
- `stack_ptx_compiler_backend_local.c`: Local compiler backend (worker thread, optional OpenMP).
- `stack_ptx_compiler_backend_nng.c`: NNG **client** backend (serializes requests, talks to a server).
- `stack_ptx_nng_wire.h` / `stack_ptx_nng_wire.c`: The NNG wire framing + message parsing helpers.

## Public API and job model

The API is intentionally minimal:

- `stack_ptx_compiler_create()`: Creates a compiler handle for a specific annotated PTX kernel template and metadata.
- `stack_ptx_compiler_submit()`: Enqueue one compile job.
- `stack_ptx_compiler_poll()`: Poll for completion (non-blocking); returns `out_has_result=1` when a result is available.
- `stack_ptx_compiler_destroy()`: Destroys the handle.

Work and output:

- `StackPtxCompilerWork`
  - `population`: Flat array of `StackPtxInstruction` containing many “genes” back-to-back.
  - `population_instructions`: Total instruction count in `population`.
  - `gene_length`: Fixed instruction count per individual (all individuals must be the same length).
  - `module_idx`: Which module in the population to compile.
  - `job_id`: User tag echoed in the output.
- `StackPtxCompilerOutput`
  - `job_id`, `module_idx`: Echoed from the job.
  - `status`: `STACK_PTX_COMPILER_SUCCESS` on success.
  - `cubin` / `cubin_size`: Owned by the caller; free with `free()`.
  - `compile_ms`: Wall time measured inside the backend.

`stack_ptx_compiler_create()` also reports:

- `out_capabilities`: Roughly “how many compiles can run concurrently” (threads).
- `out_queue_slots`: How many jobs can be queued before `STACK_PTX_COMPILER_ERROR_QUEUE_FULL`.

## Backend selection (composability)

`StackPtxCompilerHandleConfig.backend` selects the backend:

- `STACK_PTX_COMPILER_BACKEND_LOCAL`: Local compilation (single-core or OpenMP, depending on build).
- `STACK_PTX_COMPILER_BACKEND_NNG`: Remote compilation via NNG (requires `stack_ptx_compiler_nng` target).

The dispatcher (`stack_ptx_compiler.c`) creates the appropriate backend and stores a backend vtable:

- `destroy(impl)`
- `submit(impl, work)`
- `poll(impl, out, has)`

This makes it easy to add more backends later (e.g. an NNG pool/load-balancer backend) without changing call sites.

## Local backend (single-core / OpenMP)

Implementation: `stack_ptx_compiler_backend_local.c`.

High-level design:

- On create:
  - Parses the annotated PTX template via `ptx_inject` and validates injection sites.
  - Creates per-thread scratch:
    - One `PtxInjectHandle` per worker thread.
    - One reusable workspace buffer per worker thread (size `config->workspace_bytes`).
  - Starts one pthread worker thread which drains a lock-protected work queue.
- On submit:
  - Copies a `StackPtxCompilerWork` into the queue.
- Worker thread:
  - If built with OpenMP and `capabilities > 1`, uses an OpenMP region and `#pragma omp task` to compile jobs concurrently.
  - Otherwise runs a single-threaded loop.
- On poll:
  - Pops a completed `StackPtxCompilerOutput` from a lock-protected results queue (if available).

Key performance property:

- `ptx_inject_create()` and injection-site analysis happen **once** at create time.
- Each compile still calls `ptx_inject_render_ptx()` (to splice the newly-generated PTX stubs into the annotated PTX) and then runs `nvPTXCompiler` to produce a cubin.

Thread selection:

- `config->requested_capabilities` requests a thread count (clamped to build-time limits).
- If `requested_capabilities == 0`, the backend chooses a default (OpenMP max threads when enabled; otherwise 1).

Device capability selection:

- Preferred: `config->device_capability_{major,minor}` if set.
- Else: environment fallback via `STACK_PTX_COMPILER_SM` / `STACK_PTX_NNG_SM`, then default `sm_80`.

## NNG backend (client)

Implementation: `stack_ptx_compiler_backend_nng.c`.

It uses NNG `REQ0` to talk to a server using `REP0`:

- Multiple in-flight requests per handle:
  - The backend opens multiple `REQ0` sockets (possibly across multiple servers).
  - Each socket has at most one in-flight request.
- `submit()` picks an idle socket and sends a request message.
- `poll()` uses `NNG_FLAG_NONBLOCK` to check for responses.

On create, the client precomputes a serialized “compiler state” blob that describes the annotated PTX and how to interpret instruction stubs:

- The blob is produced by `stack_ptx_inject_compiler_state_serialize()` from `stack_ptx_inject_serialize.h` (mm-ptx).
- The client sends that blob **once** in an `INIT` request and receives a `session_id`.
- Subsequent compile requests only send `{session_id, job_id, module_idx, instructions_wire}`.

### What is “compiler state”?

The serialized compiler state includes (conceptually):

- Annotated PTX template (`annotated_ptx`).
- `StackPtxCompilerInfo` + `StackPtxStackInfo`.
- `StackPtxExtraInfo`:
  - device SM (major/minor)
  - execution limit
- Register bindings (`StackPtxRegister[]`).
- Request stubs (`request_stubs[]` / `request_stub_sizes[]`), describing what outputs are requested per injection site.

This is sufficient for a server to build an equivalent local compiler instance.

## NNG wire format (framing)

Wire helpers live in `stack_ptx_nng_wire.h` / `stack_ptx_nng_wire.c`.

Important: the framing currently writes integers/doubles as **native-endian raw bytes** (no byte swapping).  
On x86_64 this is little-endian; cross-endian interoperability is not guaranteed.

All messages begin with:

- `u32 magic` = `STACK_PTX_NNG_WIRE_MAGIC` (`0x58545053`, ASCII `"SPTX"`)
- `u16 ver_major` = `2`
- `u16 ver_minor` = `0`
- `u32 msg_type`

Message types:

- `1`: `COMPILE_REQUEST`
- `2`: `COMPILE_RESPONSE`
- `3`: `INIT_REQUEST`
- `4`: `INIT_RESPONSE`
- `5`: `EVICT_REQUEST`
- `6`: `EVICT_RESPONSE`
- `7`: `STATUS_REQUEST`
- `8`: `STATUS_RESPONSE`

### Init request (`msg_type = 3`)

Layout (in order):

1. Header
2. `u16 compiler_version_major`
3. `u16 compiler_version_minor`
4. `u16 compiler_version_patch`
5. `u64 compiler_state_wire_size`
6. `u8 compiler_state_wire[compiler_state_wire_size]`

Where:

- `compiler_state_wire` is the output of `stack_ptx_inject_compiler_state_serialize(...)`.

### Init response (`msg_type = 4`)

Layout (in order):

1. Header
2. `u64 session_id`
3. `i32 status` (`StackPtxCompilerResult` cast to int32)
4. `u16 server_version_major`
5. `u16 server_version_minor`
6. `u16 server_version_patch`
7. `u64 server_capabilities`
8. `u64 server_queue_slots`
9. `u64 num_injects`

Notes:

- `session_id` is assigned by the server and must be included in all subsequent compile requests.
- `server_capabilities`/`server_queue_slots` come from the server’s local backend (`stack_ptx_compiler_create`).
- `num_injects` is the number of instruction stubs expected per compile request for this session.

### Compile request (`msg_type = 1`)

Layout (in order):

1. Header
2. `u64 session_id`
3. `u64 job_id`
4. `u64 module_idx`
5. `u64 instructions_wire_size`
6. `u8 instructions_wire[instructions_wire_size]`

Where:

- `instructions_wire` is the output of `stack_ptx_instructions_serialize(...)` over the instruction stub pointer list.

Instruction stub ordering:

- The `stack_ptx_instructions_serialize(...)` list is in **population index order**: stub `i` corresponds to `func_i` (the numeric suffix after `inject_prefix`).
- This is intentional: it avoids depending on `ptx_inject_inject_info_by_index(...)` enumeration order being identical across processes/machines.

### Compile response (`msg_type = 2`)

Layout (in order):

1. Header
2. `u64 session_id`
3. `u64 job_id`
4. `u64 module_idx`
5. `i32 status` (`StackPtxCompilerResult` cast to int32)
6. `double compile_ms`
7. `double total_ms`
8. `u64 cubin_size`
9. `u8 cubin[cubin_size]`

Notes:

- `compile_ms` is the backend’s measured compile time.
- `total_ms` is server-side wall time from submit → completion (includes queuing + compile).

### Evict request (`msg_type = 5`)

Layout (in order):

1. Header
2. `u64 session_id`

### Evict response (`msg_type = 6`)

Layout (in order):

1. Header
2. `u64 session_id`
3. `i32 status` (`StackPtxCompilerResult` cast to int32)

### Status request (`msg_type = 7`)

Layout (in order):

1. Header

### Status response (`msg_type = 8`)

Layout (in order):

1. Header
2. `i32 status` (`StackPtxCompilerResult` cast to int32)
3. `u64 server_requested_capabilities`
4. `u64 num_sessions`
5. `u64 total_compiles`
6. `double uptime_ms`
7. `u64 total_in_flight`
8. `i32 last_error` (`StackPtxCompilerResult` cast to int32)
9. `double last_error_uptime_ms`

## Server side

The reference server lives at:

- `app/stack_ptx_nng_server/stack_ptx_nng_server.c`

It deserializes the request, ensures a local `stack_ptx_compiler` handle exists, then submits work via the **same API** the client uses.

Performance behavior today:

- `INIT` establishes a server-side session by sending `compiler_state_wire` once.
- Compile requests for that session only send instruction stubs (fast path).
- The server can handle up to `--threads` concurrent requests (uses NNG contexts + worker threads).
- The server supports TTL/LRU eviction to avoid orphaned sessions leaking indefinitely:
  - `--session-ttl-ms N` (default 5 minutes; `0` disables)
  - `--max-sessions N` (default 1024; `0` disables)
  - `--janitor-interval-ms N` (default 1000)
- The server defaults to `--threads = all online CPUs` (override with `--threads N`).

If the state changes (e.g. different number of stubs / different annotated PTX), clients should create a new session via `INIT`.

## Roadmap / next protocol steps (recommended)

Now that the protocol is session-based, the next incremental steps are:

1. UI status:
   - Display per-server `STATUS` (up/down, uptime, total_in_flight, last_error).
   - Track RTT and show connection health.
2. Smarter scheduling:
   - Add a dedicated multi-server dispatcher backend (separate from the raw NNG backend) that manages many servers, retries, and backoff.
3. Faster server hot path:
   - Reduce per-request allocations/copies for instruction stubs (keep a scratch arena per worker).

## Building

Main targets:

- `stack_ptx_compiler` (local-only)
- `stack_ptx_compiler_nng` (local + NNG backend + wire format)

Key CMake options:

- NNG backend is built unconditionally; install nng or enable fetching it.
- `STACK_PTX_COMPILER_FETCH_NNG` (FetchContent nng if not installed; requires network)
- `STACK_PTX_COMPILER_ENABLE_OPENMP`
- `STACK_PTX_COMPILER_OPENMP_MAX_NUM_CPUS`

NNG apps:

- `app/stack_ptx_nng_server/stack_ptx_nng_server`
- `app/stack_ptx_nng_client/stack_ptx_nng_client`

### Headless build (server/client only)

If you only want the NNG server/client (and do not want to install GLFW/Vulkan), configure with:

```sh
cmake -S . -B build-nng \
  -DMM_NLE_BUILD_ELITE_NLE=OFF
```

Then build only what you need:

```sh
cmake --build build-nng --target stack_ptx_nng_server
cmake --build build-nng --target stack_ptx_nng_client
```

## Elite NLE integration notes (UI direction)

The intended minimal integration path for `elite_nle`:

- Add a UI toggle:
  - Local: `cfg.backend = STACK_PTX_COMPILER_BACKEND_LOCAL`
  - Server: `cfg.backend = STACK_PTX_COMPILER_BACKEND_NNG` and set `cfg.nng_addr`
- Add a text field for a comma/space-separated list of server addresses (the backend accepts a list in `cfg.nng_addr` / `STACK_PTX_NNG_ADDR`).
- Add background “status” probes (`STATUS_REQUEST`) to show: connected/disconnected, last RTT, last error.
