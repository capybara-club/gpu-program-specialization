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
#ifndef SECANT_SINDY_H_INCLUDED
#define SECANT_SINDY_H_INCLUDED

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define SECANT_SINDY_DEC extern "C"
#else
#define SECANT_SINDY_DEC extern
#endif

/** Status returned by every fallible Secant-SINDy operation. */
typedef int32_t SecantSindyResult;

enum {
    SECANT_SINDY_SUCCESS = 0,
    SECANT_SINDY_ERROR_INVALID_VALUE = 1,
    SECANT_SINDY_ERROR_UNSUPPORTED_VERSION = 2,
    SECANT_SINDY_ERROR_UNSUPPORTED_SHAPE = 3,
    SECANT_SINDY_ERROR_OVERFLOW = 4,
    SECANT_SINDY_ERROR_INSUFFICIENT_BUFFER = 5,
    SECANT_SINDY_ERROR_FORMAT = 6,
    SECANT_SINDY_RESULT_COUNT = 7
};

enum {
    SECANT_SINDY_CUDA_SOLVER_RECIPE_VERSION_1 = 1u,
    SECANT_SINDY_CPU_TARGET_STATS_RUN_VERSION_1 = 1u,
    SECANT_SINDY_CPU_SOLVE_RUN_VERSION_1 = 1u,
    SECANT_SINDY_MAX_FEATURE_CAPACITY = 32u,
    SECANT_SINDY_MAX_TARGETS = 32u,
    SECANT_SINDY_TARGET_STAT_SUM_F32 = 0u,
    SECANT_SINDY_TARGET_STAT_SUM_SQUARED_F32 = 1u,
    SECANT_SINDY_TARGET_STAT_COUNT_F32 = 2u
};

/** Read-only contiguous f32 host storage. */
typedef struct SecantSindyConstHostSpanF32 {
    const float* data;
    size_t num_elements;
} SecantSindyConstHostSpanF32;

/** Mutable contiguous f32 host storage. */
typedef struct SecantSindyHostSpanF32 {
    float* data;
    size_t num_elements;
} SecantSindyHostSpanF32;

/** Mutable contiguous u32 host storage. */
typedef struct SecantSindyHostSpanU32 {
    uint32_t* data;
    size_t num_elements;
} SecantSindyHostSpanU32;

/** Mutable contiguous s32 host storage. */
typedef struct SecantSindyHostSpanS32 {
    int32_t* data;
    size_t num_elements;
} SecantSindyHostSpanS32;

/**
 * Compile-time configuration for generated cuSolverDx CUDA source.
 *
 * `feature_capacity` must be in `[1, 32]`. `max_targets` must be in
 * `[1, 32]`; the generator rounds it up to a supported cuSolverDx RHS shape.
 * `max_stlsq_iterations` must be in `[1, feature_capacity]`. A capacity equal
 * to `feature_capacity` permits the worst case in which STLSQ removes one
 * feature per iteration. Compute capability identifies the exact native target
 * used by the cuSolverDx operator, for example `8, 0` or `12, 0`.
 * Supported architecture families match Secant: 8.x, 9.x, 10.x, and 12.x.
 */
typedef struct SecantSindyCudaSolverRecipe {
    uint32_t struct_size;
    uint32_t version;
    uint32_t flags;
    uint32_t reserved;
    size_t feature_capacity;
    size_t max_targets;
    size_t max_stlsq_iterations;
    uint32_t compute_capability_major;
    uint32_t compute_capability_minor;
} SecantSindyCudaSolverRecipe;

/**
 * CPU target-moment request.
 *
 * `targets` is column-major `[target][row]`. The required target extent is
 * `(num_targets - 1) * targets_leading_dimension + num_rows`. `target_stats`
 * is accumulated, not cleared, as `[target][2]`, with a required extent of
 * `(num_targets - 1) * target_stats_leading_dimension + 2`.
 */
typedef struct SecantSindyCpuTargetStatsRun {
    uint32_t struct_size;
    uint32_t version;
    uint32_t flags;
    uint32_t reserved;
    SecantSindyConstHostSpanF32 targets;
    size_t targets_leading_dimension;
    size_t num_rows;
    size_t num_targets;
    SecantSindyHostSpanF32 target_stats;
    size_t target_stats_leading_dimension;
} SecantSindyCpuTargetStatsRun;

/**
 * CPU ridge solve over Secant static Gram-statistics output.
 *
 * `statistics` contains contiguous cohort vectors with the Secant layout:
 *
 * ```text
 * [0, A)                      feature sums
 * [A, A + A*A)              row-major raw Gram matrix
 * [A + A*A, A + A*A + A*T) feature-target sums [feature][target]
 * ```
 *
 * `A = feature_capacity` and `T = num_targets`. Cohorts are formed from the
 * contiguous `num_asts` expression list; only the final cohort may be partial.
 * `target_stats` is `[target][2]` containing target sum and square sum.
 * `alphas` contains `num_sweeps` nonnegative ridge values.
 *
 * Outputs are overwritten in these contiguous layouts:
 *
 * ```text
 * coefficients [sweep][cohort][target][feature capacity]
 * intercepts   [sweep][cohort][target]
 * sse          [sweep][cohort][target]
 * solve_info   [sweep][cohort][target]
 * ```
 *
 * A positive `solve_info` is the one-based Cholesky pivot that failed. Such a
 * numerical failure does not change the function result from success.
 */
typedef struct SecantSindyCpuRidgeRun {
    uint32_t struct_size;
    uint32_t version;
    uint32_t flags;
    uint32_t reserved;
    size_t feature_capacity;
    size_t num_asts;
    size_t num_rows;
    size_t num_targets;
    size_t num_sweeps;
    float scale_epsilon;
    SecantSindyConstHostSpanF32 statistics;
    size_t statistics_leading_dimension;
    SecantSindyConstHostSpanF32 target_stats;
    size_t target_stats_leading_dimension;
    SecantSindyConstHostSpanF32 alphas;
    SecantSindyHostSpanF32 coefficients;
    SecantSindyHostSpanF32 intercepts;
    SecantSindyHostSpanF32 sse;
    SecantSindyHostSpanS32 solve_info;
} SecantSindyCpuRidgeRun;

/**
 * CPU sequential-thresholded least-squares request.
 *
 * Input and primary output layouts match `SecantSindyCpuRidgeRun`.
 * `thresholds` contains one nonnegative threshold per sweep and applies to
 * standardized coefficients. STLSQ starts with every active feature, solves,
 * removes coefficients below the threshold, and repeats until the mask is
 * stable, empty, or `max_iterations` is reached. Additional outputs use
 * `[sweep][cohort][target]` layout.
 */
typedef struct SecantSindyCpuSTLSQRun {
    uint32_t struct_size;
    uint32_t version;
    uint32_t flags;
    uint32_t reserved;
    size_t feature_capacity;
    size_t num_asts;
    size_t num_rows;
    size_t num_targets;
    size_t num_sweeps;
    size_t max_iterations;
    float scale_epsilon;
    SecantSindyConstHostSpanF32 statistics;
    size_t statistics_leading_dimension;
    SecantSindyConstHostSpanF32 target_stats;
    size_t target_stats_leading_dimension;
    SecantSindyConstHostSpanF32 alphas;
    SecantSindyConstHostSpanF32 thresholds;
    SecantSindyHostSpanF32 coefficients;
    SecantSindyHostSpanF32 intercepts;
    SecantSindyHostSpanF32 sse;
    SecantSindyHostSpanS32 solve_info;
    SecantSindyHostSpanU32 active_masks;
    SecantSindyHostSpanS32 active_counts;
    SecantSindyHostSpanS32 iteration_counts;
} SecantSindyCpuSTLSQRun;

/** Returns a process-lifetime symbolic name for a Secant-SINDy result. */
SECANT_SINDY_DEC const char* secant_sindy_result_to_string(SecantSindyResult result);

/** Validates a generated-solver recipe independently of source generation. */
SECANT_SINDY_DEC SecantSindyResult secant_sindy_cuda_solver_recipe_validate(
    const SecantSindyCudaSolverRecipe* recipe
);

/**
 * Returns process-lifetime target-moment CUDA source.
 *
 * The generated source exports this kernel:
 *
 * @code{.cpp}
 * extern "C" __global__ void secant_sindy_cuda_target_stats_f32(
 *     const float* targets,
 *     size_t targets_leading_dimension,
 *     size_t num_rows,
 *     size_t num_targets,
 *     float* target_stats,
 *     size_t target_stats_leading_dimension);
 * @endcode
 *
 * `targets` is `[target][row]`; `target_stats` is `[target][sum, sum_squared]`.
 * The caller clears target statistics before launch. Use a block size that is a
 * multiple of 32 and at most 512, a nonzero x grid sized for the row count, and
 * `gridDim.y >= num_targets`. Multiple CTAs atomically accumulate each target.
 * @param[out] source_size_ret Optional source size, including the trailing NUL.
 * @return NUL-terminated CUDA source valid for the process lifetime.
 */
SECANT_SINDY_DEC const char* secant_sindy_cuda_target_stats_source_get(size_t* source_size_ret);

/**
 * Returns the exact generated cuSolverDx CUDA source size, including the trailing NUL.
 *
 * The source exports `secant_sindy_cuda_ridge_solve_f32` and
 * `secant_sindy_cuda_stlsq_solve_f32`. Both consume immutable Secant statistics,
 * normalize into per-CTA shared storage, and overwrite outputs in
 * `[sweep][cohort][target]` order. Coefficients append a feature-capacity axis.
 *
 * Launch either kernel with `grid = (num_cohorts, 1, 1)`. One CTA normalizes
 * its cohort once, then processes all ridge sweeps and targets before exiting.
 * Both kernels require exactly 32 threads per block.
 * `statistics_leading_dimension` spans one cohort vector;
 * `target_stats_leading_dimension` spans one target and must be at least two.
 * Runtime target and STLSQ-iteration counts must not exceed recipe capacities.
 * No dynamic shared-memory launch allocation is required.
 *
 * The generated translation unit requires C++17 and cuSolverDx. It is designed
 * for NVRTC LTO IR plus nvJitLink and `libcusolverdx.fatbin`, or equivalent NVCC
 * relocatable-device-LTO compilation and device linking.
 */
SECANT_SINDY_DEC SecantSindyResult secant_sindy_cuda_solver_source_size(
    const SecantSindyCudaSolverRecipe* recipe,
    size_t* source_size_ret
);

/** Writes generated cuSolverDx ridge/STLSQ CUDA source to a caller-owned buffer. */
SECANT_SINDY_DEC SecantSindyResult secant_sindy_cuda_solver_source_write(
    const SecantSindyCudaSolverRecipe* recipe,
    char* source,
    size_t source_size
);

/** Accumulates dataset-level target moments on the CPU. */
SECANT_SINDY_DEC SecantSindyResult secant_sindy_cpu_target_stats_run(const SecantSindyCpuTargetStatsRun* run);

/** Solves and scores every requested ridge model using an f64 CPU reference. */
SECANT_SINDY_DEC SecantSindyResult secant_sindy_cpu_ridge_run(const SecantSindyCpuRidgeRun* run);

/** Solves and scores every requested STLSQ model using an f64 CPU reference. */
SECANT_SINDY_DEC SecantSindyResult secant_sindy_cpu_stlsq_run(const SecantSindyCpuSTLSQRun* run);

static inline SecantSindyCudaSolverRecipe secant_sindy_cuda_solver_recipe_init(void) {
#ifdef __cplusplus
    SecantSindyCudaSolverRecipe value = {};
#else
    SecantSindyCudaSolverRecipe value = {0};
#endif
    value.struct_size = (uint32_t)sizeof(value);
    value.version = SECANT_SINDY_CUDA_SOLVER_RECIPE_VERSION_1;
    value.feature_capacity = SECANT_SINDY_MAX_FEATURE_CAPACITY;
    value.max_targets = 1u;
    value.max_stlsq_iterations = SECANT_SINDY_MAX_FEATURE_CAPACITY;
    return value;
}

static inline SecantSindyCpuTargetStatsRun secant_sindy_cpu_target_stats_run_init(void) {
#ifdef __cplusplus
    SecantSindyCpuTargetStatsRun value = {};
#else
    SecantSindyCpuTargetStatsRun value = {0};
#endif
    value.struct_size = (uint32_t)sizeof(value);
    value.version = SECANT_SINDY_CPU_TARGET_STATS_RUN_VERSION_1;
    return value;
}

static inline SecantSindyCpuRidgeRun secant_sindy_cpu_ridge_run_init(void) {
#ifdef __cplusplus
    SecantSindyCpuRidgeRun value = {};
#else
    SecantSindyCpuRidgeRun value = {0};
#endif
    value.struct_size = (uint32_t)sizeof(value);
    value.version = SECANT_SINDY_CPU_SOLVE_RUN_VERSION_1;
    value.feature_capacity = SECANT_SINDY_MAX_FEATURE_CAPACITY;
    value.scale_epsilon = 1.0e-6f;
    return value;
}

static inline SecantSindyCpuSTLSQRun secant_sindy_cpu_stlsq_run_init(void) {
#ifdef __cplusplus
    SecantSindyCpuSTLSQRun value = {};
#else
    SecantSindyCpuSTLSQRun value = {0};
#endif
    value.struct_size = (uint32_t)sizeof(value);
    value.version = SECANT_SINDY_CPU_SOLVE_RUN_VERSION_1;
    value.feature_capacity = SECANT_SINDY_MAX_FEATURE_CAPACITY;
    value.max_iterations = SECANT_SINDY_MAX_FEATURE_CAPACITY;
    value.scale_epsilon = 1.0e-6f;
    return value;
}

#endif /* SECANT_SINDY_H_INCLUDED */
