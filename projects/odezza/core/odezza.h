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
#ifndef ODEZZA_H
#define ODEZZA_H

#include <cuda.h>

#include <stddef.h>
#include <stdint.h>

#define ODEZZA_MAX_TOGGLE_BITS 32u

#ifdef __cplusplus
extern "C" {
#endif

typedef enum OdezzaResult {
    ODEZZA_SUCCESS = 0,
    ODEZZA_ERROR_INVALID_ARGUMENT = 1,
    ODEZZA_ERROR_INSUFFICIENT_BUFFER = 2,
    ODEZZA_ERROR_OVERFLOW = 3,
    ODEZZA_ERROR_FORMAT = 4,
    ODEZZA_ERROR_ALLOCATION = 5,
    ODEZZA_ERROR_IO = 6,
    ODEZZA_ERROR_UNSUPPORTED = 7,
    ODEZZA_ERROR_COMPILATION = 8,
    ODEZZA_ERROR_COMPILER = 9,
    ODEZZA_ERROR_AST = 10,
    ODEZZA_ERROR_SPECIALIZATION_CAPACITY = 11,
    ODEZZA_ERROR_REGISTER_PRESSURE = 12,
    ODEZZA_ERROR_STATE_LAYOUT = 13,
    ODEZZA_ERROR_QUEUE_FULL = 14,
    ODEZZA_ERROR_CUDA = 15,
    ODEZZA_ERROR_THREAD = 16,
    ODEZZA_ERROR_BUSY = 17,
    /* CUDA could not fence submitted work. Retain handle and buffers until
     * the application has recovered or destroyed its CUDA context. */
    ODEZZA_ERROR_CUDA_UNFENCED = 18,
    ODEZZA_RESULT_COUNT
} OdezzaResult;

/*
 * Variable-width FP32 postorder encoding (odezza-postorder-f32-v2).
 *
 * STATE_F32 and CONSTANT_F32 carry one unsigned-byte index. LITERAL_F32
 * carries four little-endian IEEE-754 FP32 bytes. TOGGLE2_F32 consumes two
 * preceding direct leaves and carries one toggle-bit index. TOGGLE4_F32
 * consumes four preceding direct leaves and carries two distinct toggle-bit
 * indices. Ordinary operators carry no payload. RETURN_F32 terminates a
 * complete RHS.
 */
typedef enum OdezzaAstOpcode {
    ODEZZA_AST_RETURN_F32 = 0x80,
    ODEZZA_AST_STATE_F32 = 0x81,
    ODEZZA_AST_CONSTANT_F32 = 0x82,
    ODEZZA_AST_LITERAL_F32 = 0x83,
    ODEZZA_AST_TOGGLE2_F32 = 0x84,
    ODEZZA_AST_TOGGLE4_F32 = 0x85,
    ODEZZA_AST_ADD_F32 = 0x90,
    ODEZZA_AST_SUB_F32 = 0x91,
    ODEZZA_AST_MUL_F32 = 0x92,
    ODEZZA_AST_DIV_F32 = 0x93,
    ODEZZA_AST_NEG_F32 = 0x94,
    ODEZZA_AST_SQRT_F32 = 0x95,
    ODEZZA_AST_RCP_F32 = 0x96,
    ODEZZA_AST_ABS_F32 = 0x97,
    ODEZZA_AST_MIN_F32 = 0x98,
    ODEZZA_AST_MAX_F32 = 0x99,
    ODEZZA_AST_FMA_F32 = 0x9a,
    ODEZZA_AST_SIN_F32 = 0x9b,
    ODEZZA_AST_COS_F32 = 0x9c,
    ODEZZA_AST_EX2_F32 = 0x9d,
    ODEZZA_AST_LG2_F32 = 0x9e,
    ODEZZA_AST_RSQRT_F32 = 0x9f,
    ODEZZA_AST_TANH_F32 = 0xa0,
    ODEZZA_AST_EXP_F32 = 0xa1,
    ODEZZA_AST_LOG_F32 = 0xa2
} OdezzaAstOpcode;

typedef struct OdezzaAstProgram {
    const uint8_t *bytes;
    size_t byte_count;
} OdezzaAstProgram;

/* Associates one postorder program with the derivative it writes. */
typedef struct OdezzaScoringRhs {
    uint8_t state_index;
    OdezzaAstProgram program;
} OdezzaScoringRhs;

/* One candidate system supplies every derivative not fixed at handle creation. */
typedef struct OdezzaScoringSystem {
    const OdezzaScoringRhs *rhs;
    size_t rhs_count;
} OdezzaScoringSystem;

typedef struct OdezzaScoringPipeline OdezzaScoringPipeline;

/* Immutable, host-only compiled scoring template. No CUDA context or device
 * resources. Share across pipelines/devices of the same architecture; caller
 * owns its lifetime. Pipelines copy what they need during creation. */
typedef struct OdezzaScoringTemplate OdezzaScoringTemplate;
typedef struct OdezzaScoringTemplateInfo {
    uint32_t sm_version, state_capacity, constant_capacity, system_capacity;
    uint32_t shared_patch_capacity, system_patch_capacity;
} OdezzaScoringTemplateInfo;
OdezzaResult odezza_scoring_template_create(const OdezzaScoringTemplateInfo *info,
                                            OdezzaScoringTemplate **out);
/* Opaque, checksummed artifact. NULL buffer measures. Storage, eviction and
 * atomic publication are caller policy. Read checks shape, generator identity,
 * artifact integrity and inspection; incompatible artifacts fail explicitly.
 * Artifacts are local build products, not an interchange ABI for other compilers. */
OdezzaResult odezza_scoring_template_write(const OdezzaScoringTemplate *value,
    void *buffer, size_t capacity, size_t *size_ret);
OdezzaResult odezza_scoring_template_read(const OdezzaScoringTemplateInfo *info,
    const void *buffer, size_t size, OdezzaScoringTemplate **out);
/* Time inside nvrtcCompileProgram for this creation; zero for artifact reads. */
OdezzaResult odezza_scoring_template_nvrtc_seconds(const OdezzaScoringTemplate *value,
    double *seconds_ret);
/* Failed creation/read may return a diagnostic handle: only error/destroy valid. */
OdezzaResult odezza_scoring_template_write_error(const OdezzaScoringTemplate *value,
    char *buffer, size_t capacity, size_t *size_ret);
OdezzaResult odezza_scoring_template_destroy(OdezzaScoringTemplate *value);

/* Native analytic trajectory LM. CONSTANT_F32 indexes fitted parameters; fixed
 * coefficients use LITERAL_F32. Each system supplies every RHS in state order.
 * Smooth operators and toggles share the scoring postorder encoding. */
typedef struct OdezzaLmPipeline OdezzaLmPipeline;
typedef struct OdezzaLmShape {
    uint32_t state_count;
    uint32_t parameter_count;
    uint32_t site_patch_capacity;
    uint32_t lanes_per_fit; /* 1, 2, 4 or 8 cooperating lanes per fit. */
} OdezzaLmShape;

typedef struct OdezzaLmPipelineCreateInfo {
    uint32_t sm_version;
    OdezzaLmShape shape;
} OdezzaLmPipelineCreateInfo;

typedef struct OdezzaLmFit {
    const OdezzaAstProgram *rhs; /* shape.state_count programs, borrowed in run */
    CUdeviceptr starts_device;   /* [start_count, parameter_count] FP32 */
    uint64_t start_count;
    uint32_t toggle_bit_count;    /* fits = starts * 2^bits; low bits select toggle */
    CUdeviceptr offsets_device;   /* uint32_t [trajectory_count + 1] */
    CUdeviceptr times_device;     /* FP32 [point_count] */
    CUdeviceptr reference_device; /* FP32 [state_count, point_count] */
    CUdeviceptr weights_device;   /* same shape, zero excludes observation */
    uint32_t trajectory_count, point_count;
    CUdeviceptr lower_device, upper_device; /* FP32 [parameter_count] */
    uint32_t steps_per_interval, max_iterations, max_damping_attempts;
    float initial_damping, max_step, target_mse;
    /* Output rows in [start, permutation] order. Complete initial states must
     * be present at every first point; first points are excluded from scoring. */
    CUdeviceptr parameters_device;              /* FP32 [fits, parameter_count] */
    CUdeviceptr initial_mse_device, mse_device; /* FP32 [fits] */
    CUdeviceptr iterations_device, accepted_device, factorizations_device;
    CUdeviceptr evaluations_device, invalid_device; /* uint32_t [fits] each */
    CUevent input_ready_event;                      /* optional producer dependency */
} OdezzaLmFit;

typedef struct OdezzaLmRunReport {
    uint64_t fit_count;
    size_t completed_system_count;
    double total_seconds, specialization_seconds, module_load_seconds;
    double kernel_seconds; /* sum of overlapping launches, not elapsed time */
    uint32_t peak_inflight_modules, maximum_register_count;
} OdezzaLmRunReport;

/* Lane values double as mask bits: 1|2|4|8. Array indices correspond to those
 * widths in ascending order. Template fields cover creation; execution fields
 * describe the last run, including a failed run. Counts are candidate systems,
 * each of which can contain many starts and toggle permutations. */
typedef struct OdezzaLmShapeReport {
    uint32_t requested_lanes_per_fit, allowed_lanes_mask, available_lanes_mask;
    uint32_t used_lanes_mask, attempted_lanes_mask;
    uint32_t template_registers[4];
    OdezzaResult template_results[4];
    uint64_t system_counts[4], register_rejections[4];
    double template_prepare_seconds;
} OdezzaLmShapeReport;

/* Same ownership model as scoring: caller keeps one CUDA context current;
 * handle owns templates, persistent streams/events, never the context or data.
 * Creation failure may return a diagnostic handle: only error/shape-report/destroy allowed.
 * One host thread at a time per handle. All output ranges in a batch must be
 * disjoint and must not alias inputs. Data stay alive until run returns.
 * Failure drains admitted work; discard the entire failed run's outputs.
 * CUDA_UNFENCED requires retaining handle/workspace/data until context recovery. */
OdezzaResult odezza_lm_pipeline_create(const OdezzaLmPipelineCreateInfo *create_info,
                                       OdezzaLmPipeline **pipeline_ret);
/* Opt-in fallback. Bits may name only wider supported widths than the requested
 * shape. Templates are prepared at creation so caller workspace size is stable.
 * Each system tries widths in ascending order; only REGISTER_PRESSURE retries.
 * All variants share the handle's two execution streams and output layout. */
OdezzaResult odezza_lm_pipeline_create_with_fallback(
    const OdezzaLmPipelineCreateInfo *create_info, uint32_t fallback_lanes_mask,
    OdezzaLmPipeline **pipeline_ret);
OdezzaResult odezza_lm_pipeline_shape_report(const OdezzaLmPipeline *pipeline,
                                             OdezzaLmShapeReport *report_ret);
OdezzaResult odezza_lm_pipeline_workspace_requirements(const OdezzaLmPipeline *pipeline, size_t *size_ret,
                                                       size_t *alignment_ret);
OdezzaResult odezza_lm_pipeline_run(OdezzaLmPipeline *pipeline, const OdezzaLmFit *fits, size_t system_count,
                                    void *workspace, size_t workspace_size, OdezzaLmRunReport *report_ret);
OdezzaResult odezza_lm_pipeline_write_error(const OdezzaLmPipeline *pipeline, char *buffer, size_t capacity,
                                            size_t *size_ret);
OdezzaResult odezza_lm_pipeline_destroy(OdezzaLmPipeline *pipeline);

/*
 * Complete immutable scoring shape and persistent execution resources.
 *
 * The caller must keep the intended CUDA context current on the calling thread
 * for this complete call. Creation generates and compiles the scoring template
 * for sm_version, inspects the resulting CUBIN, specializes fixed_rhs, and
 * creates the persistent streams and events used by every run. CUDA source,
 * NVRTC, CUBIN inspection, and SASS metadata are not part of the public API.
 *
 * The resulting handle is bound to that current context. The caller owns its
 * selection and lifetime, must keep it alive until after pipeline destruction,
 * and must make the same context current for every run and destruction. Odezza
 * never gets, stores, creates, pushes, pops, or sets a context.
 */
typedef struct OdezzaScoringPipelineCreateInfo {
    /* CUDA compute capability encoded as major * 10 + minor (for example 120). */
    uint32_t sm_version;

    /* Active counts may be smaller than reusable compiled-template capacities. */
    size_t state_count;
    size_t state_capacity;
    uint32_t constant_count;
    uint32_t constant_capacity;

    const OdezzaScoringRhs *fixed_rhs;
    size_t fixed_rhs_count;

    uint32_t system_capacity;
    uint32_t shared_patch_capacity;
    uint32_t system_patch_capacity;

    uint32_t worker_count;
    uint32_t cubin_slots_per_worker;
} OdezzaScoringPipelineCreateInfo;

/*
 * Device pointers must belong to the CUDA context that the caller keeps
 * current for the complete synchronous run. Constant banks are
 * [system, bank, active constant], and MSE output is [system, configuration].
 * constant_banks_device may be zero when the active constant count is
 * zero or sampled_parameters_device supplies descriptors. A bulk run may exceed system_capacity; Odezza
 * partitions it into module-sized batches without changing output ordering.
 *
 * Trajectories are ragged and may use irregular times. trajectory_offsets has
 * trajectory_count + 1 entries and partitions concatenated points. The first
 * point of each nonempty trajectory supplies its initial condition. Times are
 * [concatenated point], and reference data is
 * [active state, concatenated point]. By default every point observes every active state.
 * With allow_missing_observations, NaNs at noninitial points are unobserved.
 * All times and every initial state, including singleton trajectories,
 * must be finite; infinities are always invalid. Offsets start at zero, increase strictly, and end at
 * trajectory_point_count. Times increase strictly within each trajectory;
 * the FP32 RK4 step must remain positive. At least one point must be scored.
 * Invalid trajectory data, nonfinite integrated states, or an overflowing
 * FP32 squared-error sum produce FLT_MAX in the affected scores. MSE excludes
 * each trajectory's initial point and averages over the observed remaining state values.
 *
 * Buffer extents must cover these layouts and pointers must be 4-byte aligned.
 * Odezza rejects overflowing byte spans before submitting work; the caller
 * remains responsible for the actual allocations and their CUDA context.
 *
 * The launch's active_toggle_count is passed as a runtime uniform; it does
 * not create another CUDA-template shape. Every template uses the same
 * 32-bit permutation ABI. Low configuration bits select a permutation and
 * the remaining bits select a constant bank. Toggle bit
 * positions are encoded in each specialized AST, and reusing one couples
 * toggle sites. For example, one toggle bit and sixteen constant banks
 * evaluate 2 * 16 configurations; even configurations select choice zero and
 * odd configurations select one.
 */
/* Sampled constants: distribution 0=fixed shift, 1=uniform pool, 2=normal pool.
 * Pool index = offset + bank*stride, independent of module/GPU/permutation.
 * Transform is separately rounded FP32 multiply then add, then explicit clamp.
 * Descriptors are [system,active constant]. Size/alignment: 32/8 bytes.
 * Finite scale/shift, ordered finite clamp bounds and in-range indices required. */
typedef struct OdezzaSampledParameter {
    uint64_t offset;
    uint32_t stride, distribution;
    float scale, shift, lower, upper;
} OdezzaSampledParameter;

typedef struct OdezzaScoringLaunch {
    CUdeviceptr constant_banks_device;
    uint32_t constant_bank_count;
    uint32_t active_toggle_count;
    CUdeviceptr trajectory_offsets_device;
    CUdeviceptr trajectory_times_device;
    CUdeviceptr reference_data_device;
    uint32_t trajectory_count;
    uint32_t trajectory_point_count;
    uint32_t steps_per_observation;
    CUdeviceptr mse_output_device;
    /* Zero-initialize these fields for the original explicit-bank path.
     * When sampled_parameters_device != 0, constant_banks_device is unused.
     * Pools are shared across systems; bulk batching advances only descriptors.
     * This extends the C launch ABI: recompile callers, do not mix old binaries. */
    CUdeviceptr sampled_parameters_device, uniform_pool_device, normal_pool_device;
    uint64_t rng_pool_size;
    /* Opt-in: NaN reference entries after each initial point are unobserved.
     * Initial vectors must remain complete and finite. Infinity is invalid.
     * MSE divides by observed scalar residuals, not total state/point pairs. */
    uint32_t allow_missing_observations;
    /* Optional producer event recorded after all input uploads. Scoring and
     * reducer streams wait on this event, not on unrelated producer work.
     * Keep it alive until synchronous calls return; NULL means inputs ready. */
    CUevent input_ready_event;
    /* Optional CUDA interval profiling; zero keeps the uninstrumented launch.
     * ABI extension: recompile callers. Events are owned by the handle. */
    uint32_t profile_timing;
} OdezzaScoringLaunch;

typedef struct OdezzaScoringRunReport {
    /* On failure these counts describe submitted work, not valid results. */
    size_t system_count;
    size_t module_count;
    uint64_t configuration_count;
    double total_seconds;
    /* Summed per-module host intervals. These overlap specialization threads,
     * CUDA work and each other; do not add them to elapsed time. */
    double queue_wait_seconds, cubin_reset_seconds, specialization_seconds;
    double module_load_seconds, function_lookup_seconds, launch_seconds;
    double completion_wait_seconds, module_unload_seconds;
    /* Optional device-event intervals: sum counts concurrent kernels twice;
     * active is their union, span is first start to last end. Span-active is
     * time with no scoring launch in flight, NOT an SM occupancy measurement.
     * Excludes other kernels, transfers and time before the first launch. */
    double gpu_sum_seconds, gpu_active_seconds, gpu_span_seconds;
    uint64_t profiled_modules;
    uint32_t maximum_register_count, maximum_shared_memory_bytes;
} OdezzaScoringRunReport;

/*
 * On any failure after a handle has been allocated, pipeline_ret receives a
 * failed diagnostic handle. Only odezza_scoring_pipeline_write_error and
 * odezza_scoring_pipeline_destroy may be called on such a handle.
 */
OdezzaResult odezza_scoring_pipeline_create(const OdezzaScoringPipelineCreateInfo *create_info,
                                            OdezzaScoringPipeline **pipeline_ret);
/* Same context/ownership/failure contract, using a prepared immutable template.
 * No NVRTC compilation. Active counts may be smaller than template capacities;
 * all six template shape fields must match create_info exactly. */
OdezzaResult odezza_scoring_pipeline_create_with_template(
    const OdezzaScoringPipelineCreateInfo *create_info, const OdezzaScoringTemplate *value,
    OdezzaScoringPipeline **pipeline_ret);

/* The pipeline's creation context must be current for destruction. */
OdezzaResult odezza_scoring_pipeline_destroy(OdezzaScoringPipeline *pipeline);

/* Exact caller-owned steady-state storage; malloc provides sufficient alignment. */
OdezzaResult odezza_scoring_pipeline_workspace_requirements(const OdezzaScoringPipeline *pipeline,
                                                            size_t *workspace_size_ret,
                                                            size_t *workspace_alignment_ret);

/*
 * Synchronously specializes, loads, executes, and unloads every module needed
 * for systems using streams and events retained by the handle. The caller must
 * keep the pipeline's creation context current on this thread for the whole
 * call. The handle and workspace may serve only one run at a time. Odezza
 * performs no allocation and creates no CUDA resources in this path.
 * A failed run may leave partially written output; discard that run's scores.
 */
OdezzaResult odezza_scoring_pipeline_run(OdezzaScoringPipeline *pipeline, const OdezzaScoringSystem *systems,
                                         size_t system_count, const OdezzaScoringLaunch *launch,
                                         void *workspace, size_t workspace_size,
                                         OdezzaScoringRunReport *report_ret);

/* error_bytes_ret excludes the trailing NUL byte. A NULL buffer measures. */
OdezzaResult odezza_scoring_pipeline_write_error(const OdezzaScoringPipeline *pipeline, char *buffer,
                                                 size_t buffer_size, size_t *error_bytes_ret);

#define ODEZZA_RNG_VERSION 1u
#define ODEZZA_RNG_UNIFORM 1u
#define ODEZZA_RNG_NORMAL 2u
typedef struct OdezzaRngPool OdezzaRngPool;
typedef struct OdezzaRngPoolInfo {
    uint64_t seed, stream, sample_base, size;
    uint32_t distributions;
} OdezzaRngPoolInfo;
typedef struct OdezzaRngPoolView {
    CUdeviceptr uniform_device, normal_device;
    OdezzaRngPoolInfo info;
    double uniform_seconds, normal_seconds;
} OdezzaRngPoolView;
/* Standard Philox4x32-10 integer primitive, CPU-only. */
OdezzaResult odezza_philox4x32(const uint32_t counter[4], const uint32_t key[2], uint32_t out[4]);
/* Exact CPU uniform replay. Stream occupies 63 bits; high counter domain bit
 * separates distributions. Uniform uses 23-bit open-interval midpoint mapping.
 * Normal uses FP32 Box-Muller, whose bitwise replay requires the same GPU math
 * implementation. Gather actual winner constants for portable exact replay. */
OdezzaResult odezza_rng_uniform(uint64_t seed, uint64_t stream, uint64_t sample, float *out);
/* Caller keeps its CUDA context current. Pool owns its allocation and resident
 * generation module; creation fills only requested distribution planes. size is
 * samples PER plane, not bytes or ASTs. Stream <= INT64_MAX. No silent wrapping.
 * Failed creation can return a diagnostic handle; destroy it to release resources. */
OdezzaResult odezza_rng_pool_create(uint32_t sm, const OdezzaRngPoolInfo *info, OdezzaRngPool **out);
OdezzaResult odezza_rng_pool_view(const OdezzaRngPool *pool, OdezzaRngPoolView *out);
OdezzaResult odezza_rng_pool_write_error(const OdezzaRngPool *pool, char *out, size_t capacity);
OdezzaResult odezza_rng_pool_destroy(OdezzaRngPool *pool);
/* Validate a host-side descriptor for bank_count trials against initialized pools. */
OdezzaResult odezza_sampled_parameter_validate(const OdezzaRngPoolInfo *pool, uint32_t bank_count,
                                               const OdezzaSampledParameter *parameter);

typedef struct OdezzaScoreReducer OdezzaScoreReducer;
typedef enum OdezzaScoreGrouping {
    ODEZZA_SCORE_BY_SYSTEM = 0,
    ODEZZA_SCORE_BY_SYSTEM_PERMUTATION = 1,
    ODEZZA_SCORE_GLOBAL = 2
} OdezzaScoreGrouping;

/* Original FP32 score and flat element index relative to mse_output_device.
 * Stable order: (mse, score_index). Unfilled ranks: FLT_MAX, UINT64_MAX.
 * reserved is always zero. These structs have identical host/device layouts. */
typedef struct OdezzaScoreWinner {
    float mse;
    uint32_t reserved;
    uint64_t score_index;
} OdezzaScoreWinner;

typedef struct OdezzaScoreCounts {
    uint64_t valid, invalid, negative;
} OdezzaScoreCounts;

typedef struct OdezzaScoreIndex {
    uint64_t system_index, configuration_index, bank_index;
    uint32_t permutation;
} OdezzaScoreIndex;

typedef struct OdezzaScoreReductionSize {
    uint64_t configuration_count, score_count, group_count;
    uint32_t tiles_per_group;
    size_t workspace_bytes, winner_bytes, count_bytes;
} OdezzaScoreReductionSize;

typedef struct OdezzaScoreReductionReport {
    double kernel_seconds;
    uint64_t group_count, score_count;
} OdezzaScoreReductionReport;

/* Caller keeps the same CUDA context current for create/run/gather/destroy.
 * No context management occurs here. Creation compiles/loads a reusable reducer
 * for this SM and initial k (1..256). CUB block primitives compile through NVRTC
 * using the CUDA/CCCL headers from the build's toolkit include directory.
 * No allocations/module loading occur in set_k/run/gather.
 * One caller at a time per handle. Failed creation may return a diagnostic
 * handle; only write_error/destroy are then permitted. */
OdezzaResult odezza_score_reducer_create(uint32_t sm_version, uint32_t k, OdezzaScoreReducer **out);
/* Change k between completed operations, without recompilation. Query buffer
 * sizes using this k before run; keep k unchanged until its gather completes. */
OdezzaResult odezza_score_reducer_set_k(OdezzaScoreReducer *r, uint32_t k);
OdezzaResult odezza_score_reducer_destroy(OdezzaScoreReducer *r);
OdezzaResult odezza_score_reducer_write_error(const OdezzaScoreReducer *r, char *out, size_t capacity);

/* Layout is derived only from the launch ABI plus the bulk run's system_count.
 * Group order: system; system*permutations+permutation; or one global group.
 * Outputs are [group,k] winners and [group] counts. Negative/nonfinite/FLT_MAX
 * scores are invalid; negative additionally counts finite scores below zero.
 * Requirements and decode do not need CUDA or a current context. */
OdezzaResult odezza_score_reduction_requirements(const OdezzaScoringLaunch *launch, size_t system_count,
                                                 OdezzaScoreGrouping grouping, uint32_t k,
                                                 OdezzaScoreReductionSize *out);
OdezzaResult odezza_score_decode_index(const OdezzaScoringLaunch *launch, size_t system_count,
                                       uint64_t score_index, OdezzaScoreIndex *out);

/* Synchronous; scoring must have completed before this call. All device buffers
 * must be disjoint, workspace/winners/counts 8-byte aligned, and sizes cover
 * requirements. A failed run's outputs must be discarded. Raw scores unmodified. */
OdezzaResult odezza_score_reducer_run(OdezzaScoreReducer *r, const OdezzaScoringLaunch *launch,
                                      size_t system_count, OdezzaScoreGrouping grouping,
                                      CUdeviceptr workspace, size_t workspace_bytes, CUdeviceptr winners,
                                      size_t winner_bytes, CUdeviceptr counts, size_t count_bytes,
                                      OdezzaScoreReductionReport *report);

/* Optional exact gather from [system,bank,constant] into [group,k,constant].
 * Uses winner indices, not ASTs. Unfilled ranks receive NaN. All indices must
 * come from a successful matching reducer run. Output must not alias inputs.
 * Philox reconstruction additionally requires the caller's versioned sampling
 * descriptor; neither RNG streams nor sample-base offsets are implied by layout. */
OdezzaResult odezza_score_reducer_gather(OdezzaScoreReducer *r, const OdezzaScoringLaunch *launch,
                                         size_t system_count, OdezzaScoreGrouping grouping,
                                         uint32_t constant_count, CUdeviceptr winners, size_t winner_bytes,
                                         CUdeviceptr rows, size_t row_bytes);

#ifdef __cplusplus
}
#endif

#endif
