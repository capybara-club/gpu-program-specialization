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
#ifndef ODZ_INTERNAL_H
#define ODZ_INTERNAL_H
#define _POSIX_C_SOURCE 200809L
#include "odezza_runtime.h"
#include "odz_sizing.h"
#include "odezza_scoring_request.h"
#include "odr_grammar_internal.h"
#include "o_sha256.h"
#include <pthread.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#define ODZ_DEVICES 8
#define ODZ_PAGES (2*ODZ_DEVICES)
#define ODZ_TAGS 64
#define ODZ_SESSIONS 16
#define ODZ_POOLS 4096
#define ODZ_K 256
#define ODZ_RETAIN_BUCKETS 4096
#define ODZ_RNG_PROFILE "odezza.native.philox4x32-10.stream-v1"
typedef struct DeviceBuffer { CUdeviceptr ptr; size_t size; int borrowed; } DeviceBuffer;
typedef struct NumericSlot { uint64_t data,divisor,count; uint32_t kind; int32_t a,b; float v0,v1; } NumericSlot;
typedef struct Pool { unsigned char key[32]; DeviceBuffer data; uint64_t used,host_job; const float *host_source; int valid; } Pool;
typedef struct Session { OdrScoringSession *handle; uint32_t slots,module; uint64_t used; } Session;
typedef struct Options {
    uint32_t batch,module,patch,workers,slots;
    uint64_t chunk,device_bytes,pool_bytes,host_bytes,dedup_bytes,report_bytes;
    double seconds, tile_seconds;
    int auto_chunk,auto_batch,auto_module,profile_timing;
} Options;
typedef struct Provenance {
    uint32_t family; uint64_t ast,tags;
    size_t refs,count; const char **names;
    int initialized,pinned; struct Provenance *next;
} Provenance;
typedef struct KeptShared {
    size_t refs,structure_size; uint32_t family,permutation,nslots; int inline_storage; uint64_t ast;
    OdrAxis *slots; unsigned char *structure; Provenance *provenance;
    struct KeptShared *next;
} KeptShared;
typedef struct Kept {
    size_t refs,bytes;
    uint32_t family,permutation,nslots;
    uint64_t ast,derivation,variant,bank;
    float mse;
    float *values;
    OdrAxis *slots;
    unsigned char *numeric,*structure;
    size_t numeric_size,structure_size;
    uint64_t tags;
    KeptShared *shared;
    struct Kept *archive_next, *row_archive_next;
} Kept;
typedef struct Ranking { uint32_t k,unit,count; Kept **rows; } Ranking;
typedef struct Page { void *arena; size_t capacity; OdrBatch batch; uint64_t *rows; int state; } Page;
typedef struct FamilyRun {
    OdrProducer *producer; void *arena; size_t bytes;
    Page pages[ODZ_PAGES]; int done;
    uint32_t batch_size;
    Kept **archive, **row_archive; void *alias_user;
    uint64_t generated,scored,valid,invalid,chunks;
    OdrBatch last;
    Ranking ranking;
} FamilyRun;
typedef struct Tag {const char *name;Ranking ranking;} Tag;
typedef struct Snapshot {
    int state, quarantined, retention_incomplete;
    uint64_t configurations,valid,invalid,chunks,generated,host_bytes,retention_bytes;
    uint64_t charged_host_bytes,peak_host_bytes,host_limit_bytes,denied_reservations;
    uint64_t rss_bytes,peak_sampled_rss_bytes,process_peak_rss_bytes;
    int rss_valid;
    double started,finished;
    char error[512];
} Snapshot;
typedef struct DeviceStats {
    unsigned device;
    OdzSizing sizing;
    uint64_t configurations,chunks,max_tile;
    double busy_seconds,wait_seconds;
    OdezzaScoringRunReport pipeline;
    uint64_t trajectory_h2d_bytes,numeric_h2d_bytes,result_d2h_bytes;
    double upload_seconds,result_copy_seconds,worker_started,worker_finished;
    uint32_t module_capacity,minimum_module_capacity,maximum_patch_capacity;
    uint64_t allocated_bytes,peak_allocated_bytes,numeric_pool_bytes,peak_numeric_pool_bytes;
    uint64_t device_total_bytes,device_used_bytes,peak_device_used_bytes;
    int device_sample_valid;
} DeviceStats;
enum { ODZ_MEM_REQUEST, ODZ_MEM_ARENA, ODZ_MEM_RETAINED, ODZ_MEM_TILE,
       ODZ_MEM_PIPELINE, ODZ_MEM_CLASSES };
typedef struct HostMemory {
    uint64_t live[ODZ_MEM_CLASSES],peak[ODZ_MEM_CLASSES],total,peak_total,denied;
} HostMemory;
typedef struct OdzWorker OdzWorker;
struct OdzJob {
    pthread_mutex_t mutex, snapshot_mutex, memory_mutex; pthread_cond_t condition;
    HostMemory memory;
    uint64_t host_ceiling;
    uint64_t rss_bytes,peak_sampled_rss_bytes,process_peak_rss_bytes;
    int rss_valid;
    Snapshot snapshot;
    size_t page_count,turn,device_count;
    DeviceStats devices[ODZ_DEVICES];
    char *json;size_t json_size;char error[512];
    int state,cancelled,cancel_signal,generator_done,generator_started,quarantined,retention_incomplete,attempt_inputs;
    pthread_t generator;
    void *ta,*sa,*ga;
    int borrowed_inputs,pooled_buffers;
    size_t trajectory_capacity,trajectory_device_capacity,tile_device_capacity,tile_host_capacity;
    const OdrTrajectories *trajectories;const OdrStatic *fixed;const OdrGrammar *grammar;
    OdrRk4Layout rk4; Options options;
    FamilyRun *families;size_t family_count;
    Ranking global;Tag tags[ODZ_TAGS];size_t tag_count;
    uint32_t local_k,row_k,tag_local_k,tag_row_k;
    int provenance_possible;
    Provenance *provenance[ODZ_RETAIN_BUCKETS];
    KeptShared *shared[ODZ_RETAIN_BUCKETS];
    Kept **report_seen;size_t report_seen_capacity;
    uint64_t reduction_passes,reduction_continuations;
    double reduction_kernel_seconds;
    uint64_t report_bound;int report_rejected;
    uint64_t configurations,valid,invalid,chunks;
    uint64_t cache_hits,cache_misses,growths,retries,pool_evictions;
    uint64_t template_evictions,template_invalidations,template_oversized;
    double started,finished,parse_seconds,generation_seconds,pipeline_seconds;
    double execution_started,execution_finished;
    double prelude_seconds,rng_seconds,reduction_seconds,retention_seconds,template_seconds,nvrtc_seconds;
    double auxiliary_nvrtc_seconds,reducer_setup_seconds;
};
struct OdzRuntime {
    CUdevice device; CUcontext context; uint32_t sm; char cache[1024],error[512];
    uint32_t multiprocessors,threads_per_sm,rate_family,rate_slots,rate_bits;
    CUmodule numeric_module;CUfunction rng,prelude,consume;CUstream stream;
    CUevent ready,start,end,uploaded;
    OdezzaScoreReducer *reducer;
    Pool pools[ODZ_POOLS];uint64_t epoch,pool_bytes,clock,job_epoch;
    Session sessions[ODZ_SESSIONS];
    DeviceBuffer descriptors,coefficients,scores,workspace,winners,counts,gather;
    DeviceBuffer offsets,times,reference;
    DeviceBuffer trajectory_slab,tile_slab;
    void *result_slab; size_t result_capacity;
    int pooled_buffers,reservation_failed;
    double auxiliary_nvrtc_seconds,reducer_setup_seconds;
    int fatal,attempt_inputs;
    int submitted;
    uint64_t host_ceiling;
    size_t device_count,worker_index;
    OdzWorker *workers;
    double configs_per_second;
    uint64_t allocated_bytes,peak_allocated_bytes,peak_pool_bytes;
    double last_memory_sample;

};
double odz_now(void);
void odz_publish(OdzJob *j); /* caller holds job mutex */
int odz_stopped(OdzJob *j); /* caller holds job mutex */
int odz_device_run(OdzRuntime *r,OdzJob *j);
int odz_workers_start(OdzRuntime *r,OdzJob *j);
void odz_workers_wait(OdzRuntime *r,OdzJob *j);
void odz_workers_close(OdzRuntime *r);
int odz_fail(OdzJob *j,const char *fmt,...);
void *odz_alloc(OdzJob *j,size_t bytes);
void *odz_host_alloc(OdzJob *j,size_t bytes,unsigned kind);
void *odz_host_resize(OdzJob *j,void *ptr,size_t bytes,unsigned kind);
void odz_host_free(OdzJob *j,void *ptr);
int odz_host_claim(OdzJob *j,size_t bytes,unsigned kind);
void odz_host_release(OdzJob *j,size_t bytes,unsigned kind);
HostMemory odz_host_snapshot(OdzJob *j);
void odz_memory_sample(OdzRuntime *r,OdzJob *j,int force);
int odz_gpu_free(OdzRuntime *r,DeviceBuffer *b);
int odz_options(OdzJob *j);
int odz_retention_parse(OdzJob *j);
int odz_report_preflight(OdzJob *j);
int odz_gpu_init(OdzRuntime *r);
int odz_gpu_job_begin(OdzRuntime *r,OdzJob *j);
void odz_gpu_close(OdzRuntime *r);
int odz_gpu_buffer(OdzRuntime *r,DeviceBuffer *b,size_t bytes);
int odz_reserve_device_buffers(OdzRuntime *r,size_t trajectories,size_t device,size_t host);
int odz_device_views(DeviceBuffer *slab,DeviceBuffer **views,const size_t *sizes,size_t count);
size_t odz_aligned_bytes(size_t size);
int odz_gpu_bind(OdzRuntime *r,OdzJob *j,const OdrCandidate **c,size_t count,uint32_t slots);
int odz_gpu_prepare(OdzRuntime *r,OdzJob *j,uint32_t slots,uint32_t count,uint64_t start,uint32_t banks);
void odz_bank_address(const OdrCandidate *c,const OdrAxis *a,uint64_t *seed,uint64_t *stream);
int odz_score_tile(OdzRuntime *r,OdzJob *j,Page *page);
int odz_offer(OdzJob *j,const OdrCandidate *c,uint32_t permutation,uint64_t bank,float mse,const float *values,Kept **out);
void odz_keep_unref(OdzJob *j,Kept *k);
int odz_rank_order(const Kept *a,const Kept *b);
void odz_rank_offer(OdzJob *j,Ranking *ranking,Kept *k);
void odz_alias(void *user,uint64_t index,uint64_t derivation,uint64_t variant,const char *const *tags,size_t count);
int odz_same_numeric(const Kept *a,const Kept *b);
void odz_aliases_close(OdzJob *j,FamilyRun *f);
void odz_retention_close(OdzJob *j);
#endif
