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
#ifndef SECANT_H_INCLUDED
#define SECANT_H_INCLUDED

#include <stddef.h>
#include <stdint.h>

/**
 * @file secant.h
 * @brief Public AST, CPU reference, direct-CUBIN, and bulk-runner API.
 *
 * Secant programs are return-terminated postorder bytecode streams. CPU entry
 * points interpret those streams directly. CUBIN entry points generate CUDA
 * skeleton source, inspect compiled skeletons, specialize caller-owned CUBIN
 * copies, and execute them through a blocking bulk runner.
 *
 * Recipes and run requests are header-first concrete structures. Generic
 * lifecycle functions read the fixed-width shape tag, validate the complete
 * descriptor, and dispatch to the corresponding implementation. Unless stated
 * otherwise, sizes and strides are measured in elements of the pointed-to
 * type. Device addresses belong to the CUDA context current when a runner is
 * created and used.
 *
 * Descriptor versions are exact. An unknown version returns
 * `SECANT_ERROR_UNSUPPORTED_VERSION`; an unknown shape returns
 * `SECANT_ERROR_UNSUPPORTED_SHAPE`; and unsupported flags or a `struct_size`
 * smaller than the known structure return `SECANT_ERROR_INVALID_VALUE`. A
 * larger `struct_size` is accepted and unknown trailing bytes are ignored.
 * Within a version, appended fields must be optional with zero/default
 * semantics. New required fields or changed field meanings require a new
 * version.
 *
 * @section secant_kernel_shapes Kernel shapes
 *
 * Static materialize consumes column-major `[input][row]` and writes
 * `[ast][row]`. Static SSE consumes `[input][row]` and `[target][row]`, then
 * atomically accumulates `[ast][target]`. Static affine statistics writes
 * prediction moments and prediction-target cross moments per AST. Static Gram
 * statistics evaluates one AST cohort into shared memory and accumulates
 * feature sums, a full Gram matrix, and feature-target cross moments. A separate
 * fixed CUDA source computes target moments once during dataset preprocessing.
 * Toggle SSE evaluates coefficient banks crossed with explicit AST permutations.
 */

#ifdef __cplusplus
#define SECANT_DEC extern "C"
#else
#define SECANT_DEC extern
#endif

/**
 * @name Secant version
 *
 * The version is part of the identity of generated native skeletons. Consumers
 * that persist generated source or binaries must include all three components
 * in their cache key. Secant source-generation or binary-inspection changes
 * that can alter compatibility require a version change.
 * @{
 */
#define SECANT_VERSION_MAJOR 0u
#define SECANT_VERSION_MINOR 3u
#define SECANT_VERSION_PATCH 1u
#define SECANT_VERSION_STRING "0.3.1"
/** @} */

/** Fixed-width status returned by Secant APIs. SECANT_SUCCESS is the only success value. */
typedef int32_t SecantResult;

enum {
    SECANT_SUCCESS = 0,
    SECANT_ERROR_INVALID_VALUE = 1,
    SECANT_ERROR_OVERFLOW = 2,
    SECANT_ERROR_INSUFFICIENT_BUFFER = 3,
    SECANT_ERROR_FORMAT = 4,
    SECANT_ERROR_ALLOCATION_FAILED = 5,
    SECANT_ERROR_COMPILE_FAILED = 6,
    SECANT_ERROR_PARSE_FAILED = 7,
    SECANT_ERROR_SKELETON_NOT_FOUND = 8,
    SECANT_ERROR_BAD_PROGRAM = 9,
    SECANT_ERROR_STACK_OVERFLOW = 10,
    SECANT_ERROR_STACK_UNDERFLOW = 11,
    SECANT_ERROR_TOO_MANY_ARGS = 12,
    SECANT_ERROR_UNSUPPORTED_OP = 13,
    SECANT_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS = 14,
    SECANT_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS = 15,
    SECANT_ERROR_ROUTINE_DEPTH_EXCEEDED = 16,
    SECANT_ERROR_INSUFFICIENT_PATCH_SPACE = 17,
    SECANT_ERROR_REGISTER_OVERFLOW = 18,
    SECANT_ERROR_UNSUPPORTED_ARCHITECTURE = 19,
    SECANT_ERROR_INVALID_STATE = 20,
    SECANT_ERROR_UNSUPPORTED_SHAPE = 21,
    SECANT_ERROR_UNEXPECTED_INSTRUCTION = 22,
    SECANT_ERROR_BACKEND_UNAVAILABLE = 23,
    SECANT_ERROR_BAD_BINARY = 24,
    SECANT_ERROR_BAD_PATCH_SITE = 25,
    SECANT_ERROR_FUNCTION_NOT_FOUND = 26,
    SECANT_ERROR_PATCH_SITE_NOT_FOUND = 27,
    SECANT_ERROR_REGISTER_COUNT_NOT_FOUND = 28,
    SECANT_ERROR_THREAD_FAILED = 29,
    SECANT_ERROR_DRIVER_FAILED = 30,
    SECANT_ERROR_EAGER_LOADING_REQUIRED = 31,
    SECANT_ERROR_UNSUPPORTED_VERSION = 32,
    SECANT_ERROR_COMPLETION_UNKNOWN = 33,
    SECANT_RESULT_COUNT = 34,
    SECANT_RESULT_NUM_ENUMS = SECANT_RESULT_COUNT
};

/** One byte in a return-terminated, variable-length AST program. */
typedef uint8_t SecantAstInstruction;

/** Unsigned index encoded by an input, routine, or routine-argument token. */
typedef uint8_t SecantAstIdx;

/** Fixed-width operation encoded by a variable-length postorder AST instruction. */
typedef uint8_t SecantAstInstructionType;

enum {
    SECANT_AST_INSTRUCTION_TYPE_NONE = 0x80,
    SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32 = 0x81,
    SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32 = 0x82,
    SECANT_AST_INSTRUCTION_TYPE_RETURN_F32 = 0x83,
    SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32 = 0x84,
    SECANT_AST_INSTRUCTION_TYPE_ADD_F32 = 0x85,
    SECANT_AST_INSTRUCTION_TYPE_SUB_F32 = 0x86,
    SECANT_AST_INSTRUCTION_TYPE_MUL_F32 = 0x87,
    SECANT_AST_INSTRUCTION_TYPE_DIV_F32 = 0x88,
    SECANT_AST_INSTRUCTION_TYPE_NEG_F32 = 0x89,
    SECANT_AST_INSTRUCTION_TYPE_SQRT_F32 = 0x8a,
    SECANT_AST_INSTRUCTION_TYPE_RCP_F32 = 0x8b,
    SECANT_AST_INSTRUCTION_TYPE_ABS_F32 = 0x8c,
    SECANT_AST_INSTRUCTION_TYPE_MIN_F32 = 0x8d,
    SECANT_AST_INSTRUCTION_TYPE_MAX_F32 = 0x8e,
    SECANT_AST_INSTRUCTION_TYPE_FMA_F32 = 0x8f,
    SECANT_AST_INSTRUCTION_TYPE_SIN_F32 = 0x90,
    SECANT_AST_INSTRUCTION_TYPE_COS_F32 = 0x91,
    SECANT_AST_INSTRUCTION_TYPE_EX2_F32 = 0x92,
    SECANT_AST_INSTRUCTION_TYPE_LG2_F32 = 0x93,
    SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32 = 0x94,
    SECANT_AST_INSTRUCTION_TYPE_TANH_F32 = 0x95,
    SECANT_AST_INSTRUCTION_TYPE_INPUT_S32 = 0x96,
    SECANT_AST_INSTRUCTION_TYPE_CONSTANT_S32 = 0x97,
    SECANT_AST_INSTRUCTION_TYPE_ROUTINE_S32 = 0x98,
    SECANT_AST_INSTRUCTION_TYPE_RETURN_S32 = 0x99,
    SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_S32 = 0x9a,
    SECANT_AST_INSTRUCTION_TYPE_ADD_S32 = 0x9b,
    SECANT_AST_INSTRUCTION_TYPE_SUB_S32 = 0x9c,
    SECANT_AST_INSTRUCTION_TYPE_MUL_S32 = 0x9d,
    SECANT_AST_INSTRUCTION_TYPE_DIV_S32 = 0x9e,
    SECANT_AST_INSTRUCTION_TYPE_NEG_S32 = 0x9f,
    SECANT_AST_INSTRUCTION_TYPE_ABS_S32 = 0xa0,
    SECANT_AST_INSTRUCTION_TYPE_MIN_S32 = 0xa1,
    SECANT_AST_INSTRUCTION_TYPE_MAX_S32 = 0xa2,
    SECANT_AST_INSTRUCTION_TYPE_FMA_S32 = 0xa3,
    SECANT_AST_INSTRUCTION_TYPE_INPUT_U32 = 0xa4,
    SECANT_AST_INSTRUCTION_TYPE_CONSTANT_U32 = 0xa5,
    SECANT_AST_INSTRUCTION_TYPE_ROUTINE_U32 = 0xa6,
    SECANT_AST_INSTRUCTION_TYPE_RETURN_U32 = 0xa7,
    SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_U32 = 0xa8,
    SECANT_AST_INSTRUCTION_TYPE_ADD_U32 = 0xa9,
    SECANT_AST_INSTRUCTION_TYPE_SUB_U32 = 0xaa,
    SECANT_AST_INSTRUCTION_TYPE_MUL_U32 = 0xab,
    SECANT_AST_INSTRUCTION_TYPE_DIV_U32 = 0xac,
    SECANT_AST_INSTRUCTION_TYPE_MIN_U32 = 0xad,
    SECANT_AST_INSTRUCTION_TYPE_MAX_U32 = 0xae,
    SECANT_AST_INSTRUCTION_TYPE_FMA_U32 = 0xaf,
    SECANT_AST_INSTRUCTION_TYPE_AND_U32 = 0xb0,
    SECANT_AST_INSTRUCTION_TYPE_OR_U32 = 0xb1,
    SECANT_AST_INSTRUCTION_TYPE_XOR_U32 = 0xb2,
    SECANT_AST_INSTRUCTION_TYPE_NOT_U32 = 0xb3,
    SECANT_AST_INSTRUCTION_TYPE_SHL_U32 = 0xb4,
    SECANT_AST_INSTRUCTION_TYPE_SHR_U32 = 0xb5,
    SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32 = 0xb6,
    SECANT_AST_INSTRUCTION_TYPE_EXP_F32 = 0xba,
    SECANT_AST_INSTRUCTION_TYPE_LOG_F32 = 0xbb,
    SECANT_AST_INSTRUCTION_TYPE_BANK_CONSTANT_F32 = 0xbc,
    SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32 = 0xbd,
    SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32 = 0xbe,
    /* Direct leaf: separately rounded scale * bank[slot] + offset. */
    SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32 = 0xbf,
    SECANT_AST_INSTRUCTION_TYPE_ONE_PAST_LAST = 0xc0,
    SECANT_AST_INSTRUCTION_TYPE_NUM_ENUMS = SECANT_AST_INSTRUCTION_TYPE_ONE_PAST_LAST
};

/** Fixed-width tag carried by every public recipe and run descriptor. */
typedef uint32_t SecantKernelShape;

enum {
    SECANT_KERNEL_SHAPE_NONE = 0u,
    SECANT_KERNEL_SHAPE_STATIC_MATERIALIZE_F32 = 1u,
    SECANT_KERNEL_SHAPE_STATIC_SSE_F32 = 2u,
    SECANT_KERNEL_SHAPE_STATIC_AFFINE_STATS_F32 = 5u,
    SECANT_KERNEL_SHAPE_STATIC_GRAM_STATS_F32 = 6u,
    SECANT_KERNEL_SHAPE_TOGGLE_SSE_F32 = 10u
};


/** Fixed inner indices in each static affine AST-statistics vector. */
enum {
    SECANT_AFFINE_AST_STAT_SUM_PREDICTION_F32 = 0u,
    SECANT_AFFINE_AST_STAT_SUM_PREDICTION_SQUARED_F32 = 1u,
    SECANT_AFFINE_AST_STAT_PREDICTION_TARGET_BASE_F32 = 2u
};

/** Inner indices in each static affine target-statistics vector. */
enum {
    SECANT_AFFINE_TARGET_STAT_SUM_F32 = 0u,
    SECANT_AFFINE_TARGET_STAT_SUM_SQUARED_F32 = 1u,
    SECANT_AFFINE_TARGET_STAT_COUNT_F32 = 2u
};

enum {
    SECANT_CUBIN_RECIPE_VERSION_3 = 3u,
    SECANT_CUBIN_RUN_VERSION_3 = 3u,
    SECANT_CPU_RUN_VERSION_3 = 3u,
    SECANT_CUBIN_RUNNER_OPTIONS_VERSION_1 = 1u,
    SECANT_CUBIN_PLAN_INFO_VERSION_1 = 1u,
    SECANT_RUNNER_STATS_VERSION_1 = 1u
};

/** One pointer/count view over return-terminated AST programs. */
typedef struct SecantAstProgramList {
    const SecantAstInstruction* const* items; /**< Program pointers, or NULL when count is zero. */
    size_t count;                            /**< Number of return-terminated programs. */
} SecantAstProgramList;

/**
 * Borrowed device-resident f32 strided storage view.
 *
 * The containing descriptor defines the logical dimensions, layout, and
 * whether the storage is read-only or writable. This view owns no memory.
 */
typedef struct SecantDeviceMatrixF32 {
    uintptr_t address;         /**< CUDA device address. */
    size_t num_elements;       /**< Accessible f32 elements from address. */
    size_t leading_dimension;  /**< Elements between consecutive outer indices. */
} SecantDeviceMatrixF32;

/** Borrowed read-only strided f32 storage view in host memory. */
typedef struct SecantConstHostMatrixF32 {
    const float* data;         /**< First readable f32 element. */
    size_t num_elements;       /**< Accessible f32 elements from data. */
    size_t leading_dimension;  /**< Elements between consecutive outer indices. */
} SecantConstHostMatrixF32;

/** Borrowed writable strided f32 storage view in host memory. */
typedef struct SecantHostMatrixF32 {
    float* data;               /**< First writable f32 element. */
    size_t num_elements;       /**< Accessible f32 elements from data. */
    size_t leading_dimension;  /**< Elements between consecutive outer indices. */
} SecantHostMatrixF32;

/** Borrowed routine and top-level AST programs. */
typedef struct SecantAstProgramSet {
    SecantAstProgramList routines;
    SecantAstProgramList asts;
} SecantAstProgramSet;

/** Common prefix of every concrete CUBIN source/inspection recipe. */
typedef struct SecantCubinRecipeHeader {
    uint32_t struct_size;   /**< Bytes in the complete concrete recipe. */
    uint32_t version;       /**< SECANT_CUBIN_RECIPE_VERSION_* value. */
    SecantKernelShape shape; /**< Concrete recipe type. */
    uint32_t flags;         /**< Must be zero. */
} SecantCubinRecipeHeader;

/**
 * Static-column materialize generation and inspection parameters.
 *
 * Every count and `patch_capacity_instructions` must be nonzero. `num_inputs`
 * must not exceed `SECANT_AST_MAX_INPUTS`. The marker-register requirement
 * `num_inputs + asts_per_kernel` must not exceed 256. All arithmetic is
 * checked for `size_t` overflow.
 */
typedef struct SecantCubinMaterializeRecipe {
    SecantCubinRecipeHeader header;      /**< Versioned common descriptor header. */
    size_t num_kernels;                 /**< Global functions emitted into the module. */
    size_t asts_per_kernel;             /**< AST results packed into each global function. */
    size_t num_inputs;                  /**< Fixed input columns visible to every AST. */
    size_t patch_capacity_instructions; /**< 16-byte SASS instructions reserved per patch island. */
} SecantCubinMaterializeRecipe;

/**
 * Static-column SSE generation and inspection parameters.
 *
 * Every count and `patch_capacity_instructions` must be nonzero.
 * `threads_per_block` must be a multiple of 32 in `[32, 512]` and must not
 * exceed `tile_rows`. The marker-register requirement
 * `num_inputs + num_targets + asts_per_kernel * num_targets` must not exceed
 * 256. All arithmetic is checked for `size_t` overflow.
 */
typedef struct SecantCubinSSERecipe {
    SecantCubinRecipeHeader header;      /**< Versioned common descriptor header. */
    size_t num_kernels;                 /**< Global functions emitted into the module. */
    size_t asts_per_kernel;             /**< AST results packed into each global function. */
    size_t num_inputs;                  /**< Fixed input columns visible to every AST. */
    size_t num_targets;                 /**< Maximum active target-column count supported by the plan. */
    size_t tile_rows;                   /**< Maximum rows assigned to one CTA. */
    size_t threads_per_block;           /**< CUDA threads launched for each CTA. */
    size_t patch_capacity_instructions; /**< 16-byte SASS instructions reserved per patch island. */
} SecantCubinSSERecipe;

/**
 * Static-column affine-statistics generation and inspection parameters.
 *
 * Every count and `patch_capacity_instructions` must be nonzero.
 * `threads_per_block` must be a multiple of 32 in `[32, 512]` and must not
 * exceed `tile_rows`. The marker-register requirement
 * `num_inputs + num_targets + asts_per_kernel * (2 + num_targets)` must not
 * exceed 256. All arithmetic is checked for `size_t` overflow. Target moments
 * are computed by one fixed auxiliary kernel and consume no patch registers.
 * The marker-register bound is a source-level upper bound. Register allocation
 * remains compiler- and architecture-dependent; inspection rejects a generated
 * CUBIN if ptxas inserts spills or unrelated instructions into a marker span.
 */
typedef struct SecantCubinAffineStatsRecipe {
    SecantCubinRecipeHeader header;      /**< Versioned common descriptor header. */
    size_t num_kernels;                 /**< Global functions emitted into the module. */
    size_t asts_per_kernel;             /**< AST results packed into each global function. */
    size_t num_inputs;                  /**< Fixed input columns visible to every AST. */
    size_t num_targets;                 /**< Maximum active target-column count supported by the plan. */
    size_t tile_rows;                   /**< Maximum rows assigned to one CTA. */
    size_t threads_per_block;           /**< CUDA threads launched for each CTA. */
    size_t patch_capacity_instructions; /**< 16-byte SASS instructions reserved per patch island. */
} SecantCubinAffineStatsRecipe;

/**
 * Static-column Gram-statistics generation and inspection parameters.
 *
 * Each kernel represents one AST cohort. `asts_per_kernel` must be in `[1, 32]`.
 * Every other count and `patch_capacity_instructions` must be nonzero.
 * `threads_per_block` must be a multiple of 32 in `[32, 512]` and must not
 * exceed `tile_rows`. The generated kernel stores input columns, AST feature
 * values, and one target tile in shared memory. Their combined portable static
 * shared-memory requirement is
 * `((num_inputs + asts_per_kernel + 1) * tile_rows + 1) * sizeof(float)` and
 * may not exceed 48 KiB. The final scalar is an isolated skeleton keepalive
 * location. The marker-register requirement
 * `num_inputs + asts_per_kernel` must not exceed 256. All arithmetic is checked
 * for `size_t` overflow.
 */
typedef struct SecantCubinGramStatsRecipe {
    SecantCubinRecipeHeader header;      /**< Versioned common descriptor header. */
    size_t num_kernels;                 /**< Independent AST cohorts emitted into the module. */
    size_t asts_per_kernel;             /**< Maximum feature count in each cohort; at most 32. */
    size_t num_inputs;                  /**< Fixed input columns visible to every AST. */
    size_t num_targets;                 /**< Maximum active target-column count supported by the plan. */
    size_t tile_rows;                   /**< Rows cached in shared memory by one CTA. */
    size_t threads_per_block;           /**< CUDA threads launched for each CTA. */
    size_t patch_capacity_instructions; /**< 16-byte SASS instructions reserved per patch island. */
} SecantCubinGramStatsRecipe;

/** Register-bound scoring. All columns and bank slots are explicit AST leaves.
 * Each function packs asts_per_kernel ASTs; num_kernels packs functions per module.
 * Each thread owns a configuration = (bank << toggle_bits) | permutation.
 * All packed ASTs share the same permutation bits and bank-slot registers.
 * Constants are loaded once before the row loop; columns load by fixed offsets.
 * No runtime leaf-binding table exists. Zero constants and zero toggle bits are valid.
 */
typedef struct SecantCubinToggleSSERecipe {
    SecantCubinRecipeHeader header;
    size_t num_kernels;
    size_t asts_per_kernel; /**< Packed AST capacity, 1..32; initializer defaults to 1. */
    size_t num_inputs; /**< Fixed column count shared by all packed ASTs. */
    size_t num_constants; /**< Shared coefficient slots, loaded once before the row loop. */
    size_t num_targets;
    size_t tile_rows;
    size_t threads_per_block;
    size_t patch_capacity_instructions;
} SecantCubinToggleSSERecipe;

/** Common prefix of every concrete CUBIN execution request. */
typedef struct SecantCubinRunHeader {
    uint32_t struct_size;   /**< Bytes in the complete concrete run request. */
    uint32_t version;       /**< SECANT_CUBIN_RUN_VERSION_* value. */
    SecantKernelShape shape; /**< Concrete run type; must match the runner plan. */
    uint32_t flags;         /**< Must be zero. */
} SecantCubinRunHeader;

/**
 * Blocking static materialization request.
 *
 * `input` is column-major `[input][row]`; `output` is `[ast][row]`.
 * The input extent must cover `(num_inputs - 1) * input.leading_dimension +
 * num_rows` elements. Each module starts `output_module_stride` elements after
 * the previous one. A zero module stride intentionally reuses one module-sized
 * output region for every module. With multiple modules, a nonzero stride must
 * cover one complete full-module output span. One module with `module_asts`
 * active programs requires `(module_asts - 1) * output.leading_dimension +
 * num_rows` output elements. The input and output allocations must not overlap.
 * The runner writes only active AST rows; unused slots in a partial final module
 * remain untouched.
 */
typedef struct SecantCubinMaterializeRun {
    SecantCubinRunHeader header;
    SecantAstProgramSet programs;
    SecantDeviceMatrixF32 input;
    size_t num_rows;
    SecantDeviceMatrixF32 output;
    size_t output_module_stride;
} SecantCubinMaterializeRun;

/**
 * Blocking static SSE request.
 *
 * `input` and `targets` are `[column][row]`. The runner clears and writes
 * `output` as `[ast][target]` with target as the inner dimension. Input and
 * target extents use `(count - 1) * leading_dimension + num_rows`. Output must
 * cover `(num_asts - 1) * output.leading_dimension + num_targets`, where
 * `num_targets` may not exceed the recipe capacity. The three allocations must
 * not overlap. Only the compact active output range is cleared; unused plan
 * capacity remains untouched.
 */
typedef struct SecantCubinSSERun {
    SecantCubinRunHeader header;
    SecantAstProgramSet programs;
    SecantDeviceMatrixF32 input;
    SecantDeviceMatrixF32 targets;
    size_t num_rows;
    size_t num_targets;
    SecantDeviceMatrixF32 output;
} SecantCubinSSERun;

/**
 * Blocking static affine-statistics request.
 *
 * `input` and `targets` are `[column][row]`. The runner clears and writes
 * `ast_stats` as `[ast][2 + target]`: prediction sum, prediction square sum,
 * then one prediction-target cross sum per active target. Its leading
 * dimension must be at least `2 + num_targets`, and its extent must cover
 * `(num_asts - 1) * ast_stats.leading_dimension + 2 + num_targets`.
 *
 * Target sum and target square sum are dataset-level preprocessing values and
 * are intentionally not produced by this request. The fixed source returned by
 * `secant_cuda_target_stats_f32_source_get()` computes those values once. No
 * target moment is recomputed by an AST kernel. All active allocations must be
 * disjoint.
 *
 * With `n = num_rows`, these values recover raw SSE, the least-squares affine
 * scale and intercept, and affine-adjusted SSE without materializing
 * predictions.
 */
typedef struct SecantCubinAffineStatsRun {
    SecantCubinRunHeader header;
    SecantAstProgramSet programs;
    SecantDeviceMatrixF32 input;
    SecantDeviceMatrixF32 targets;
    size_t num_rows;
    size_t num_targets;
    SecantDeviceMatrixF32 ast_stats;
} SecantCubinAffineStatsRun;

/**
 * Blocking static Gram-statistics request.
 *
 * `input` and `targets` are `[column][row]`. Each kernel is one cohort with
 * capacity `A = recipe.asts_per_kernel`. The runner clears and writes
 * `statistics` as `[cohort][statistic]`. Every cohort vector has this layout:
 *
 * ```text
 * [0, A)                         feature sums
 * [A, A + A*A)                 full row-major feature Gram matrix
 * [A + A*A, A + A*A + A*T)    feature-target cross sums [feature][target]
 * ```
 *
 * `T` is the active `num_targets`. The leading dimension must cover the full
 * vector based on cohort capacity even when the final cohort has fewer active
 * ASTs. Unused feature rows and columns remain zero. Target-only moments are
 * dataset preprocessing values and are not recomputed by this kernel. All
 * active allocations must be disjoint.
 */
typedef struct SecantCubinGramStatsRun {
    SecantCubinRunHeader header;
    SecantAstProgramSet programs;
    SecantDeviceMatrixF32 input;
    SecantDeviceMatrixF32 targets;
    size_t num_rows;
    size_t num_targets;
    SecantDeviceMatrixF32 statistics;
} SecantCubinGramStatsRun;

/** Coefficients are [bank][slot], shared by every AST in the run.
 * bank_stride >= num_constants. Padding between bank vectors is allowed.
 * num_constants=0 permits a null address and requires bank_stride=0.
 */
typedef struct SecantDeviceConstantBanks {
    uintptr_t address;
    size_t num_elements;
    size_t bank_stride;
} SecantDeviceConstantBanks;

typedef struct SecantConstHostConstantBanks {
    const float* data;
    size_t num_elements;
    size_t bank_stride;
} SecantConstHostConstantBanks;

/** Writes [ast][target][configuration] raw SSE. The inner configuration count
 * is num_banks * 2^toggle_bits. num_banks >= 1 and toggle_bits <= 32.
 * Input and targets are column-major. Output leading_dimension spans configurations.
 * Every AST sees the same bank vector and permutation bits for a configuration.
 * Source AST toggle bits must be below toggle_bits. Inputs, banks, targets and
 * output may not overlap. This operation clears active output, including on CPU.
 */
typedef struct SecantCubinToggleSSERun {
    SecantCubinRunHeader header;
    SecantAstProgramSet programs;
    SecantDeviceMatrixF32 input;
    SecantDeviceMatrixF32 targets;
    SecantDeviceConstantBanks constants;
    size_t num_rows;
    size_t num_targets;
    size_t num_banks;
    uint32_t toggle_bits;
    SecantDeviceMatrixF32 output;
} SecantCubinToggleSSERun;

/** Common prefix of every concrete CPU execution request. */
typedef struct SecantCpuRunHeader {
    uint32_t struct_size;   /**< Bytes in the complete concrete CPU request. */
    uint32_t version;       /**< SECANT_CPU_RUN_VERSION_* value. */
    SecantKernelShape shape; /**< Concrete CPU request type. */
    uint32_t flags;         /**< Must be zero. */
} SecantCpuRunHeader;

/**
 * CPU materialization request with `[input][row]` input and `[ast][row]`
 * output. Input and output must not overlap. Materialization writes active
 * output values and does not preserve their previous contents. The required
 * extents are `(num_inputs - 1) * input.leading_dimension + num_rows` and
 * `(num_asts - 1) * output.leading_dimension + num_rows` elements.
 */
typedef struct SecantCpuMaterializeRun {
    SecantCpuRunHeader header;
    SecantAstProgramSet programs;
    size_t num_inputs;
    SecantConstHostMatrixF32 input;
    size_t num_rows;
    SecantHostMatrixF32 output;
} SecantCpuMaterializeRun;

/**
 * CPU static SSE request. Input, targets, and output must not overlap. Results
 * are added to caller-owned `[ast][target]` output; the CPU path never clears
 * it implicitly. Input and target extents use
 * `(count - 1) * leading_dimension + num_rows`; output requires
 * `(num_asts - 1) * output.leading_dimension + num_targets` elements.
 */
typedef struct SecantCpuSSERun {
    SecantCpuRunHeader header;
    SecantAstProgramSet programs;
    size_t num_inputs;
    size_t num_targets;
    SecantConstHostMatrixF32 input;
    SecantConstHostMatrixF32 targets;
    size_t num_rows;
    SecantHostMatrixF32 output;
} SecantCpuSSERun;

/**
 * CPU static affine-statistics request. Active storage ranges must be disjoint.
 * Results are added to caller-owned `ast_stats`; target moments are managed
 * separately by the caller. The CPU path never clears output implicitly.
 * Extents and statistic ordering match `SecantCubinAffineStatsRun`.
 */
typedef struct SecantCpuAffineStatsRun {
    SecantCpuRunHeader header;
    SecantAstProgramSet programs;
    size_t num_inputs;
    size_t num_targets;
    SecantConstHostMatrixF32 input;
    SecantConstHostMatrixF32 targets;
    size_t num_rows;
    SecantHostMatrixF32 ast_stats;
} SecantCpuAffineStatsRun;

/**
 * CPU static Gram-statistics request. Programs are grouped consecutively into
 * cohorts of `asts_per_cohort`. Results are added to caller-owned
 * `[cohort][statistic]` storage using the layout documented by
 * `SecantCubinGramStatsRun`; the CPU path never clears output implicitly.
 */
typedef struct SecantCpuGramStatsRun {
    SecantCpuRunHeader header;
    SecantAstProgramSet programs;
    size_t asts_per_cohort;
    size_t num_inputs;
    size_t num_targets;
    SecantConstHostMatrixF32 input;
    SecantConstHostMatrixF32 targets;
    size_t num_rows;
    SecantHostMatrixF32 statistics;
} SecantCpuGramStatsRun;

/** Host oracle for the same bank/permutation mapping and SSE layout. */
typedef struct SecantCpuToggleSSERun {
    SecantCpuRunHeader header;
    SecantAstProgramSet programs;
    size_t num_inputs;
    size_t num_constants;
    SecantConstHostMatrixF32 input;
    SecantConstHostMatrixF32 targets;
    SecantConstHostConstantBanks constants;
    size_t num_rows;
    size_t num_targets;
    size_t num_banks;
    uint32_t toggle_bits;
    SecantHostMatrixF32 output;
} SecantCpuToggleSSERun;

/** Extensible worker and stream counts used to create a bulk runner. */
typedef struct SecantCubinRunnerOptions {
    uint32_t struct_size; /**< Bytes in this options structure. */
    uint32_t version;     /**< SECANT_CUBIN_RUNNER_OPTIONS_VERSION_* value. */
    uint32_t flags;       /**< Must be zero. */
    uint32_t reserved;    /**< Must be zero. */
    size_t num_workers;   /**< Nonzero persistent CPU specialization-worker count. */
    size_t num_streams;   /**< Nonzero internal CUDA-stream count. */
} SecantCubinRunnerOptions;

/** Common metadata returned for every inspected CUBIN plan. */
typedef struct SecantCubinPlanInfo {
    uint32_t struct_size;    /**< Bytes in this info structure. */
    uint32_t version;        /**< SECANT_CUBIN_PLAN_INFO_VERSION_* value. */
    SecantKernelShape shape; /**< Shape inspected into the plan. */
    uint32_t flags;          /**< Must be zero. */
    size_t cubin_size;       /**< Bytes in each compatible CUBIN copy. */
} SecantCubinPlanInfo;

/**
 * Versioned timing and work counters returned by one blocking bulk run.
 *
 * A caller passing this optional output must initialize it with
 * `secant_runner_stats_init()`. Validation failures return zero counters.
 * Failures after work starts return the counters collected before failure.
 * Compatible larger structures retain their caller-supplied `struct_size` and
 * unknown trailing bytes.
 */
typedef struct SecantRunnerStats {
    uint32_t struct_size;             /**< Bytes in this statistics structure. */
    uint32_t version;                 /**< SECANT_RUNNER_STATS_VERSION_* value. */
    uint32_t flags;                   /**< Must be zero. */
    uint32_t reserved;                /**< Must be zero. */
    size_t num_modules;               /**< AST modules processed. */
    size_t num_asts;                  /**< Input ASTs specialized. */
    size_t modules_loaded;            /**< CUDA modules successfully loaded. */
    double compile_window_seconds;    /**< Wall-clock compile window including backpressure. */
    double compile_critical_seconds;  /**< Largest cumulative compiler time for one worker. */
    double compile_work_seconds;      /**< Sum of compiler time across all workers. */
    double module_load_seconds;       /**< Wall time spent loading CUDA modules. */
    double completion_wait_seconds;   /**< Wall time waiting for GPU completion events. */
    double module_unload_seconds;     /**< Wall time spent unloading CUDA modules. */
    double runtime_seconds;           /**< Device-event interval containing kernel execution. */
    double total_seconds;             /**< Wall time for the complete blocking call. */
} SecantRunnerStats;

/** Caller-owned immutable metadata parsed from one generated CUBIN template. */
typedef struct SecantCubinPlan SecantCubinPlan;

/** Persistent workers and CUDA resources for one direct-CUBIN kernel shape. */
typedef struct SecantCubinRunnerImpl* SecantCubinRunner;

/** Returns a process-lifetime symbolic name for a Secant result code. */
SECANT_DEC const char* secant_result_to_string(SecantResult result);

/**
 * Returns fixed CUDA source for dataset-level f32 target moments.
 *
 * The returned process-lifetime source defines this kernel:
 *
 * @code{.c}
 * extern "C" __global__ void secant_cuda_target_stats_f32(
 *     const float* targets,
 *     size_t targets_leading_dimension,
 *     size_t num_rows,
 *     size_t num_targets,
 *     float* target_stats,
 *     size_t target_stats_leading_dimension);
 * @endcode
 *
 * `targets` is `[target][row]`. The caller must clear `target_stats`, which is
 * atomically accumulated as `[target][2]`: target sum and target square sum.
 * Launch a multiple of 32 and at most 512 threads per block, set `grid.y` to
 * `num_targets`, and choose any nonzero `grid.x`. A useful default is 128
 * threads and `ceil(num_rows / 4096)` blocks in x. The source is independent
 * of a generated Secant CUBIN and may be compiled, cached, loaded, and launched
 * by the caller once per dataset or trajectory.
 *
 * @param[out] source_size_ret Optional source byte count including trailing NUL.
 * @return Process-lifetime NUL-terminated CUDA source.
 */
SECANT_DEC const char* secant_cuda_target_stats_f32_source_get(size_t* source_size_ret);

static inline SecantCubinMaterializeRecipe secant_cubin_materialize_recipe_init(void) {
#ifdef __cplusplus
    SecantCubinMaterializeRecipe value = {};
#else
    SecantCubinMaterializeRecipe value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CUBIN_RECIPE_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_STATIC_MATERIALIZE_F32;
    value.header.flags = 0u;
    return value;
}

static inline SecantCubinSSERecipe secant_cubin_sse_recipe_init(void) {
#ifdef __cplusplus
    SecantCubinSSERecipe value = {};
#else
    SecantCubinSSERecipe value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CUBIN_RECIPE_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_STATIC_SSE_F32;
    value.header.flags = 0u;
    return value;
}

static inline SecantCubinAffineStatsRecipe secant_cubin_affine_stats_recipe_init(void) {
#ifdef __cplusplus
    SecantCubinAffineStatsRecipe value = {};
#else
    SecantCubinAffineStatsRecipe value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CUBIN_RECIPE_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_STATIC_AFFINE_STATS_F32;
    value.header.flags = 0u;
    return value;
}

static inline SecantCubinGramStatsRecipe secant_cubin_gram_stats_recipe_init(void) {
#ifdef __cplusplus
    SecantCubinGramStatsRecipe value = {};
#else
    SecantCubinGramStatsRecipe value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CUBIN_RECIPE_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_STATIC_GRAM_STATS_F32;
    value.header.flags = 0u;
    return value;
}

static inline SecantCubinMaterializeRun secant_cubin_materialize_run_init(void) {
#ifdef __cplusplus
    SecantCubinMaterializeRun value = {};
#else
    SecantCubinMaterializeRun value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CUBIN_RUN_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_STATIC_MATERIALIZE_F32;
    value.header.flags = 0u;
    return value;
}

static inline SecantCubinSSERun secant_cubin_sse_run_init(void) {
#ifdef __cplusplus
    SecantCubinSSERun value = {};
#else
    SecantCubinSSERun value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CUBIN_RUN_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_STATIC_SSE_F32;
    value.header.flags = 0u;
    return value;
}

static inline SecantCubinAffineStatsRun secant_cubin_affine_stats_run_init(void) {
#ifdef __cplusplus
    SecantCubinAffineStatsRun value = {};
#else
    SecantCubinAffineStatsRun value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CUBIN_RUN_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_STATIC_AFFINE_STATS_F32;
    value.header.flags = 0u;
    return value;
}

static inline SecantCubinGramStatsRun secant_cubin_gram_stats_run_init(void) {
#ifdef __cplusplus
    SecantCubinGramStatsRun value = {};
#else
    SecantCubinGramStatsRun value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CUBIN_RUN_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_STATIC_GRAM_STATS_F32;
    value.header.flags = 0u;
    return value;
}

static inline SecantCpuMaterializeRun secant_cpu_materialize_run_init(void) {
#ifdef __cplusplus
    SecantCpuMaterializeRun value = {};
#else
    SecantCpuMaterializeRun value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CPU_RUN_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_STATIC_MATERIALIZE_F32;
    value.header.flags = 0u;
    return value;
}

static inline SecantCpuSSERun secant_cpu_sse_run_init(void) {
#ifdef __cplusplus
    SecantCpuSSERun value = {};
#else
    SecantCpuSSERun value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CPU_RUN_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_STATIC_SSE_F32;
    value.header.flags = 0u;
    return value;
}

static inline SecantCpuAffineStatsRun secant_cpu_affine_stats_run_init(void) {
#ifdef __cplusplus
    SecantCpuAffineStatsRun value = {};
#else
    SecantCpuAffineStatsRun value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CPU_RUN_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_STATIC_AFFINE_STATS_F32;
    value.header.flags = 0u;
    return value;
}

static inline SecantCpuGramStatsRun secant_cpu_gram_stats_run_init(void) {
#ifdef __cplusplus
    SecantCpuGramStatsRun value = {};
#else
    SecantCpuGramStatsRun value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CPU_RUN_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_STATIC_GRAM_STATS_F32;
    value.header.flags = 0u;
    return value;
}

static inline SecantCubinToggleSSERecipe secant_cubin_toggle_sse_recipe_init(void) {
#ifdef __cplusplus
    SecantCubinToggleSSERecipe value = {};
#else
    SecantCubinToggleSSERecipe value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CUBIN_RECIPE_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_TOGGLE_SSE_F32;
    value.asts_per_kernel = 1u;
    return value;
}

static inline SecantCubinToggleSSERun secant_cubin_toggle_sse_run_init(void) {
#ifdef __cplusplus
    SecantCubinToggleSSERun value = {};
#else
    SecantCubinToggleSSERun value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CUBIN_RUN_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_TOGGLE_SSE_F32;
    return value;
}

static inline SecantCpuToggleSSERun secant_cpu_toggle_sse_run_init(void) {
#ifdef __cplusplus
    SecantCpuToggleSSERun value = {};
#else
    SecantCpuToggleSSERun value = {0};
#endif
    value.header.struct_size = (uint32_t)sizeof(value);
    value.header.version = SECANT_CPU_RUN_VERSION_3;
    value.header.shape = SECANT_KERNEL_SHAPE_TOGGLE_SSE_F32;
    return value;
}

static inline SecantCubinRunnerOptions secant_cubin_runner_options_init(void) {
#ifdef __cplusplus
    SecantCubinRunnerOptions value = {};
#else
    SecantCubinRunnerOptions value = {0};
#endif
    value.struct_size = (uint32_t)sizeof(value);
    value.version = SECANT_CUBIN_RUNNER_OPTIONS_VERSION_1;
    value.flags = 0u;
    value.reserved = 0u;
    value.num_workers = 1u;
    value.num_streams = 1u;
    return value;
}

static inline SecantCubinPlanInfo secant_cubin_plan_info_init(void) {
#ifdef __cplusplus
    SecantCubinPlanInfo value = {};
#else
    SecantCubinPlanInfo value = {0};
#endif
    value.struct_size = (uint32_t)sizeof(value);
    value.version = SECANT_CUBIN_PLAN_INFO_VERSION_1;
    value.flags = 0u;
    return value;
}

static inline SecantRunnerStats secant_runner_stats_init(void) {
#ifdef __cplusplus
    SecantRunnerStats value = {};
#else
    SecantRunnerStats value = {0};
#endif
    value.struct_size = (uint32_t)sizeof(value);
    value.version = SECANT_RUNNER_STATS_VERSION_1;
    value.flags = 0u;
    value.reserved = 0u;
    return value;
}

/**
 * Validates a complete header-first concrete CUBIN recipe.
 *
 * @param[in] recipe Common header at the first byte of a concrete recipe.
 * @return SECANT_SUCCESS or a SecantResult error.
 */
SECANT_DEC SecantResult secant_cubin_recipe_validate(const SecantCubinRecipeHeader* recipe);

/**
 * Measures generated CUDA source, including its trailing NUL byte.
 *
 * @param[in] recipe Valid complete concrete recipe header.
 * @param[out] required_size_ret Required source buffer bytes.
 * @return SECANT_SUCCESS or a SecantResult error.
 */
SECANT_DEC SecantResult secant_cubin_source_size(
    const SecantCubinRecipeHeader* recipe,
    size_t* required_size_ret
);

/**
 * Writes generated NUL-terminated CUDA source to caller-owned storage.
 *
 * The function is a single-pass writer. Any error, including
 * `SECANT_ERROR_INSUFFICIENT_BUFFER`, may leave a partial source prefix in
 * `output`; callers must discard it. Use `secant_cubin_source_size()` first
 * when a complete source buffer is required.
 *
 * @param[in] recipe Valid complete concrete recipe header.
 * @param[out] output Caller-owned source buffer.
 * @param[in] output_size Available bytes in output.
 * @return SECANT_SUCCESS or a SecantResult error.
 */
SECANT_DEC SecantResult secant_cubin_source_write(
    const SecantCubinRecipeHeader* recipe,
    char* output,
    size_t output_size
);

/**
 * Measures caller-owned storage required for an immutable CUBIN plan.
 *
 * `cubin` must be aligned to at least eight bytes and must be the exact binary
 * generated from `recipe`. The returned storage size includes worst-case
 * internal padding, so a caller may pass a `plan_storage` pointer with
 * arbitrary byte alignment to `secant_cubin_plan_init()`.
 *
 * @param[in] recipe Recipe used to generate the compiled skeleton.
 * @param[in] cubin Complete compiled CUBIN image.
 * @param[in] cubin_size Bytes available at cubin.
 * @param[out] required_storage_size_ret Required plan storage bytes.
 * @return SECANT_SUCCESS or a SecantResult error.
 */
SECANT_DEC SecantResult secant_cubin_plan_storage_size(
    const SecantCubinRecipeHeader* recipe,
    const void* cubin,
    size_t cubin_size,
    size_t* required_storage_size_ret
);

/**
 * Initializes an immutable plan in caller-owned storage.
 *
 * The plan points only into plan_storage and stores offsets into compatible
 * CUBIN copies. The caller must keep plan_storage alive and unchanged for the
 * lifetime of the plan and every runner created from it.
 *
 * `cubin` must be aligned to at least eight bytes. `plan_storage` may have
 * arbitrary byte alignment. On failure `*plan_ret` is NULL; plan-storage
 * contents are unspecified and no state is retained.
 *
 * @param[in] recipe Recipe used to generate the compiled skeleton.
 * @param[in] cubin Complete compiled CUBIN image to inspect.
 * @param[in] cubin_size Bytes available at cubin.
 * @param[out] plan_storage Caller-owned storage with the measured size.
 * @param[in] plan_storage_size Bytes available at plan_storage.
 * @param[out] plan_ret Initialized immutable plan on success; NULL on failure.
 * @return SECANT_SUCCESS or a SecantResult error.
 */
SECANT_DEC SecantResult secant_cubin_plan_init(
    const SecantCubinRecipeHeader* recipe,
    const void* cubin,
    size_t cubin_size,
    void* plan_storage,
    size_t plan_storage_size,
    SecantCubinPlan** plan_ret
);

/**
 * Reads common metadata from an immutable plan.
 *
 * @param[in] plan Immutable inspected plan.
 * @param[in,out] info_ret Preinitialized common plan-info descriptor.
 * @return SECANT_SUCCESS or a SecantResult error.
 */
SECANT_DEC SecantResult secant_cubin_plan_info_get(
    const SecantCubinPlan* plan,
    SecantCubinPlanInfo* info_ret
);

/**
 * Reconstructs the concrete recipe stored by an immutable plan.
 *
 * `recipe_ret` must point to a concrete recipe initialized for the plan shape.
 * Both its version and shape are validated before any field is modified. The
 * implementation preserves its header and any unknown trailing bytes.
 *
 * @param[in] plan Immutable inspected plan.
 * @param[in,out] recipe_ret Preinitialized concrete recipe header.
 * @return SECANT_SUCCESS or a SecantResult error.
 */
SECANT_DEC SecantResult secant_cubin_plan_recipe_get(
    const SecantCubinPlan* plan,
    SecantCubinRecipeHeader* recipe_ret
);

/**
 * Writes native AST instructions into a mutable copy of the inspected CUBIN.
 *
 * Specialization is destructive. On error, cubin may contain a partial patch
 * and must not be loaded. The function performs no allocation.
 * `programs->asts.items` is ordered by kernel, then AST slot, and its count
 * must be in `[1, num_kernels * asts_per_kernel]`. Unused trailing slots remain
 * unreachable. Program storage must remain valid until the call returns.
 */
SECANT_DEC SecantResult secant_cubin_specialize_into(
    const SecantCubinPlan* plan,
    const SecantAstProgramSet* programs,
    void* cubin,
    size_t cubin_size
);

/**
 * Creates persistent workers and CUDA resources for an inspected plan.
 *
 * The caller must make the intended CUDA context current before this call and
 * every run/destroy call. Secant does not initialize CUDA or change contexts.
 * CUDA_MODULE_LOADING must select eager loading. plan and its storage must
 * outlive the returned runner.
 *
 * `cubin` must be an unmodified byte-for-byte copy of the exact skeleton CUBIN
 * inspected into `plan`. It is copied before this function returns. On failure
 * partial resources are released and `*runner_ret` is NULL, unless cleanup itself
 * fails. In that case a non-NULL destroy-only handle retains ownership; retry
 * secant_cubin_runner_destroy() in the same current context. Never ignore a
 * non-NULL handle just because creation returned an error.
 *
 * @param[in] plan Immutable inspected plan.
 * @param[in] cubin Unmodified skeleton CUBIN copied into internal worker slots.
 * @param[in] cubin_size Bytes available at cubin; must match the plan.
 * @param[in] options Preinitialized worker/stream options.
 * @param[out] runner_ret Runner on success, or a destroy-only handle if cleanup fails.
 * @return SECANT_SUCCESS or a SecantResult error.
 */
SECANT_DEC SecantResult secant_cubin_runner_create(
    const SecantCubinPlan* plan,
    const void* cubin,
    size_t cubin_size,
    const SecantCubinRunnerOptions* options,
    SecantCubinRunner* runner_ret
);

/**
 * Executes one AST program set over multiple independent data runs.
 *
 * Every entry must have the runner's shape and reference the exact same
 * routine and AST pointer arrays and counts. Data pointers, target pointers,
 * active target counts, banks, permutations, row counts, strides, and output storage may
 * differ. All active output ranges must be mutually disjoint and must not
 * overlap any active read range in any entry. Read-only ranges may be shared
 * between entries. Ordinary shapes specialize and load each AST module once,
 * launch that loaded module for every entry, then wait and unload it. This
 * amortizes specialization and module loading while preserving one output bin
 * per trajectory or dataset.
 */
SECANT_DEC SecantResult secant_cubin_runner_run_batch(
    SecantCubinRunner runner,
    const SecantCubinRunHeader* const* runs,
    size_t num_runs,
    SecantRunnerStats* stats_ret
);

/**
 * Executes one validated shape-specific request and blocks until completion.
 *
 * The run shape must match the runner's plan. Input and output device storage
 * remains caller-owned and must stay valid until this function returns, except
 * for SECANT_ERROR_COMPLETION_UNKNOWN as described below. ASTs
 * are ordered by module, kernel, then AST slot; the final module may be
 * partial. AST and routine storage must remain valid through the blocking call.
 *
 * Validation of headers, counts, extents, overlaps, leaf kinds, and return
 * termination occurs before output clearing or CUDA launch. Full structural
 * AST validation is fused with single-pass specialization. A structural specialization,
 * driver, or worker failure after work begins may leave output partially cleared
 * or written; scores from a failed call must not be consumed. The runner can be
 * reused after validation/specialization failures and driver errors for which
 * completion and module cleanup succeeded. An unload failure makes the handle
 * destroy-only; subsequent runs return SECANT_ERROR_INVALID_STATE.
 *
 * SECANT_ERROR_COMPLETION_UNKNOWN means neither events nor recovery synchronization
 * could confirm completion on every owned stream. The runner is destroy-only,
 * and all device buffers supplied to this call must remain valid until destroy
 * succeeds (or the caller tears down the failed CUDA context). AST/routine host
 * storage is no longer used after return, including on this error. Normal runs
 * use events; stream synchronization is confined to error recovery. A non-null
 * `stats_ret` must be initialized with `secant_runner_stats_init()` and receives
 * zero counters for preflight failures or partial counters for failures after
 * work starts.
 *
 * @param[in,out] runner Opaque runner with no concurrent run in progress.
 * @param[in] run Common header of a complete shape-specific run request.
 * @param[out] stats_ret Optional timing and work counters.
 * @return SECANT_SUCCESS or a SecantResult error.
 */
SECANT_DEC SecantResult secant_cubin_runner_run(
    SecantCubinRunner runner,
    const SecantCubinRunHeader* run,
    SecantRunnerStats* stats_ret
);

/**
 * Stops persistent workers and releases internal CUDA resources.
 *
 * Passing NULL succeeds. Only SECANT_SUCCESS invalidates a non-NULL handle.
 * Teardown failures retain ownership of outstanding resources; retry destroy
 * with the same current context. After teardown starts the handle is destroy-only.
 * SECANT_ERROR_COMPLETION_UNKNOWN retains outstanding device-buffer lifetimes
 * as documented for run. A wrong context or an active run rejects teardown
 * without changing the handle. Calls to destroy must be externally serialized
 * against all use of the handle; rejecting an active run is not lifetime protection
 * against concurrent destruction.
 *
 * @param[in] runner Runner to destroy, or NULL.
 * @return SECANT_SUCCESS or a SecantResult error.
 */
SECANT_DEC SecantResult secant_cubin_runner_destroy(SecantCubinRunner runner);

/**
 * Dispatches one validated shape-specific CPU execution request.
 *
 * Materialize writes output values. Static SSE, affine, and Gram statistics add
 * to existing caller-owned output. Toggle SSE clears and overwrites its active
 * score bins, preserving padding. AST validation may be fused with execution;
 * an invalid program can therefore leave a partially written or accumulated
 * output. Callers must discard output after any error.
 *
 * @param[in] run Common header of a complete shape-specific CPU request.
 * @return SECANT_SUCCESS or a SecantResult error.
 */
SECANT_DEC SecantResult secant_cpu_run(const SecantCpuRunHeader* run);

static inline SecantResult secant_cpu_run_materialize(const SecantCpuMaterializeRun* run) {
    return run == NULL
        ? SECANT_ERROR_INVALID_VALUE
        : secant_cpu_run(&run->header);
}

static inline SecantResult secant_cpu_run_sse(const SecantCpuSSERun* run) {
    return run == NULL
        ? SECANT_ERROR_INVALID_VALUE
        : secant_cpu_run(&run->header);
}

static inline SecantResult secant_cpu_run_affine_stats(const SecantCpuAffineStatsRun* run) {
    return run == NULL
        ? SECANT_ERROR_INVALID_VALUE
        : secant_cpu_run(&run->header);
}

static inline SecantResult secant_cpu_run_gram_stats(const SecantCpuGramStatsRun* run) {
    return run == NULL
        ? SECANT_ERROR_INVALID_VALUE
        : secant_cpu_run(&run->header);
}

static inline SecantResult secant_cubin_runner_run_materialize(
    SecantCubinRunner runner,
    const SecantCubinMaterializeRun* run,
    SecantRunnerStats* stats_ret
) {
    return run == NULL
        ? SECANT_ERROR_INVALID_VALUE
        : secant_cubin_runner_run(runner, &run->header, stats_ret);
}

static inline SecantResult secant_cubin_runner_run_sse(
    SecantCubinRunner runner,
    const SecantCubinSSERun* run,
    SecantRunnerStats* stats_ret
) {
    return run == NULL
        ? SECANT_ERROR_INVALID_VALUE
        : secant_cubin_runner_run(runner, &run->header, stats_ret);
}

static inline SecantResult secant_cubin_runner_run_affine_stats(
    SecantCubinRunner runner,
    const SecantCubinAffineStatsRun* run,
    SecantRunnerStats* stats_ret
) {
    return run == NULL
        ? SECANT_ERROR_INVALID_VALUE
        : secant_cubin_runner_run(runner, &run->header, stats_ret);
}

static inline SecantResult secant_cubin_runner_run_gram_stats(
    SecantCubinRunner runner,
    const SecantCubinGramStatsRun* run,
    SecantRunnerStats* stats_ret
) {
    return run == NULL
        ? SECANT_ERROR_INVALID_VALUE
        : secant_cubin_runner_run(runner, &run->header, stats_ret);
}

static inline SecantResult secant_cpu_run_toggle_sse(const SecantCpuToggleSSERun* run) {
    return secant_cpu_run(run ? &run->header : NULL);
}
static inline SecantResult secant_cubin_runner_run_toggle_sse(SecantCubinRunner runner,
    const SecantCubinToggleSSERun* run, SecantRunnerStats* stats) {
    return secant_cubin_runner_run(runner, run ? &run->header : NULL, stats);
}

#include "secant_instructions.h"

#endif /* SECANT_H_INCLUDED */
