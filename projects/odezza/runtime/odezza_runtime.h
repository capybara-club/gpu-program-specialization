/*
 * SPDX-FileCopyrightText: 2026 Charles Durham
 * SPDX-License-Identifier: MIT
 *
 * MIT License
 *
 * Copyright (c) 2026 Charles Durham
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef ODEZZA_RUNTIME_H
#define ODEZZA_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
/* Native request safety ceilings; smaller integration.max_steps is honored.
 * Tile work per configuration is max(RK4 steps, trajectory points).
 * Each RK4 step evaluates the complete RHS four times.
 * These are not wall-time guarantees. Core scoring APIs have no service policy. */
#define ODZ_MAX_REPORT_BYTES UINT64_C(8388608)
#define ODZ_MAX_POINTS_PER_CONFIGURATION 65536u
#define ODZ_MAX_STEPS_PER_OBSERVATION 4096u
#define ODZ_MAX_STEPS_PER_CONFIGURATION UINT64_C(65536)
/* Effective per-device tile configurations are max(base/work_per_config,
 * next_power_of_two(2 * SMs * max_threads_per_SM)). The parallel target is
 * bounded by ODZ_MAX_PARALLEL_CONFIGURATIONS; explicit request and memory
 * ceilings can further reduce it. */
#define ODZ_BASE_TILE_WORK_UNITS UINT64_C(67108864)
#define ODZ_MAX_PARALLEL_CONFIGURATIONS UINT64_C(4194304)
#ifdef __cplusplus
extern "C" {
#endif
/* Service layer, separate from the allocation-free frontend and native core.
 * Runtime creation/run/destruction must occur on one worker thread. Jobs own
 * their request and result memory. status/report/cancel are safe from other threads.
 * Customer requests/results are volatile. Only the C-owned compiled-template
 * cache uses SQLite; it contains no customer numeric inputs. */
typedef struct OdzRuntime OdzRuntime;
typedef struct OdzJob OdzJob;
int odz_runtime_create(unsigned device, const char *cache_directory, OdzRuntime **out);
/* One parser/producer, independent context-owning workers; 1..8 distinct devices.
 * The caller remains the owner thread for the first device. */
int odz_runtime_create_devices(const unsigned *devices, size_t count,
    const char *cache_directory, OdzRuntime **out);
/* Select before the first run, on the runtime owner thread. When enabled,
 * numeric allocations may be reused but their contents/identities are valid
 * only within one attempt. Explicit grids are uploaded anew on the next job.
 * Applies to every device in the group; no kernel/search policy change. */
int odz_runtime_attempt_inputs(OdzRuntime *runtime);
/* Operator ceiling for the shared per-job host reservation, all GPU workers
 * combined. Set before any job, including a rejected job. Zero is invalid.
 * Requests above it fail rather than being silently clamped. Opaque CUDA/NVRTC
 * allocations require separate process supervision; this is not an RSS cap. */
int odz_runtime_host_limit(OdzRuntime *runtime, size_t bytes);
/* Before the first run: reserve per-device trajectory and scoring/reduction
 * slabs plus CPU reduction scratch. Capacities are fixed; tiles shrink to fit,
 * oversized trajectories fail explicitly. No fallback allocation for these
 * buffers. Handles, compilation, numeric-bank storage and retained records are
 * outside this pool contract. A failed reservation requires runtime destruction.
 * Device capacities include alignment padding. Applies to every device. */
int odz_runtime_reserve_buffers(OdzRuntime *runtime, size_t trajectory_device_bytes,
    size_t tile_device_bytes, size_t tile_host_bytes);
/* Lightweight published snapshot; independent of preparation/result locks.
 * Same measure/short-buffer convention as report. */
int odz_job_status(OdzJob *job, char *output, size_t capacity, size_t *required);
const char *odz_runtime_error(const OdzRuntime *runtime);
void odz_runtime_destroy(OdzRuntime *runtime);
int odz_job_create(const char *json, size_t bytes, OdzJob **out);
/* Borrow immutable JSON and a writable, malloc-aligned trajectory arena until
 * job destruction. Only the small job handle is allocated here. The trajectory
 * phase measures its requirement and fails if the arena is short; it never
 * allocates a replacement. Caller must not reuse either buffer before destroy. */
int odz_job_create_borrowed(const char *json, size_t bytes, void *trajectory_arena,
    size_t trajectory_capacity, OdzJob **out);
int odz_job_run(OdzRuntime *runtime, OdzJob *job);
void odz_job_cancel(OdzJob *job);
/* NULL output measures including terminating NUL; 1 means short buffer. */
int odz_job_report(OdzJob *job, char *output, size_t capacity, size_t *required);
void odz_job_destroy(OdzJob *job); /* only after run returns */
#ifdef __cplusplus
}
#endif
#endif
