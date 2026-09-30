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
#ifndef SECANT_CUDA_H_INCLUDED
#define SECANT_CUDA_H_INCLUDED

#include "secant.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define SECANT_CUDA_DEC extern "C"
#else
#define SECANT_CUDA_DEC extern
#endif

#define SECANT_CUDA_SSE_MAX_WARPS 16u
#define SECANT_CUDA_DYNAMIC_CONSTANT_SSE_MAX_INPUT_COLUMNS 32u
#define SECANT_CUDA_DYNAMIC_CONSTANT_SSE_MAX_INPUT_CONSTANTS 32u
#define SECANT_CUDA_DYNAMIC_LEAF_SSE_MAX_DYNAMIC_LEAVES 32u
#define SECANT_CUDA_DYNAMIC_LEAF_LM_MAX_PARAMETERS 8u
#define SECANT_CUDA_LM_OPTIMIZER_PARAMETERS 8u
#define SECANT_CUDA_LM_OPTIMIZER_STATISTICS 45u

/**
 * CUDA C++ compiler baseline.
 *
 * This backend is retained for correctness validation and compile/runtime
 * comparisons with native CUBIN specialization. It is not SECANT's primary
 * deployment backend.
 */

/** Status returned by CUDA source generation and compilation calls. */
typedef enum SecantCUDAResult {
    SECANT_CUDA_SUCCESS = 0,
    SECANT_CUDA_ERROR_INVALID_VALUE = 1,
    SECANT_CUDA_ERROR_OVERFLOW = 2,
    SECANT_CUDA_ERROR_INSUFFICIENT_BUFFER = 3,
    SECANT_CUDA_ERROR_FORMAT = 4,
    SECANT_CUDA_ERROR_BAD_PROGRAM = 5,
    SECANT_CUDA_ERROR_STACK_OVERFLOW = 6,
    SECANT_CUDA_ERROR_STACK_UNDERFLOW = 7,
    SECANT_CUDA_ERROR_TOO_MANY_ARGS = 8,
    SECANT_CUDA_ERROR_UNSUPPORTED_OP = 9,
    SECANT_CUDA_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS = 10,
    SECANT_CUDA_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS = 11,
    SECANT_CUDA_ERROR_ROUTINE_DEPTH_EXCEEDED = 12,
    SECANT_CUDA_ERROR_ALLOCATION_FAILED = 13,
    SECANT_CUDA_ERROR_COMPILE_FAILED = 14,
    SECANT_CUDA_ERROR_INVALID_STATE = 15,
    SECANT_CUDA_ERROR_UNSUPPORTED_SHAPE = 16,
    SECANT_CUDA_RESULT_NUM_ENUMS = 17
} SecantCUDAResult;

/** Owned CUDA CUBIN returned by a successful compile call. */
typedef struct SecantCUDACompiledImpl* SecantCUDACompiled;

/** Returns a static symbolic name for a CUDA result code. */
SECANT_CUDA_DEC const char* secant_cuda_result_to_string(SecantCUDAResult result);

/**
 * Renders one AST as an inline f32 CUDA device function.
 *
 * Passing NULL for buffer measures the required bytes, including the trailing
 * NUL. num_variables_ret optionally reports the maximum temporary-variable
 * count used by the rendered expression.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_function_source_generate(
    const char* function_name,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* instructions,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* num_variables_ret
);

/**
 * Renders one static-column/dynamic-constant AST as an inline CUDA function.
 *
 * Static-column instructions address `inputN` parameters. Indexed dynamic-
 * constant instructions address `constantN` parameters. Immediate constants
 * remain embedded in the generated expression. Routines are recursively
 * inlined from their postorder bytecode. Dynamic-column and mixed dynamic-leaf
 * instructions are rejected because this experiment differentiates only with
 * respect to dynamic constants.
 *
 * Passing NULL for buffer measures the exact required bytes, including the
 * trailing NUL. num_variables_ret optionally reports the number of scalar SSA
 * temporaries emitted after routine inlining.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_ast_primal_source_generate(
    const char* function_name,
    size_t num_input_columns,
    size_t num_dynamic_constants,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* ast,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* num_variables_ret
);

/**
 * Renders one AST and its forward derivatives as an inline CUDA function.
 *
 * The generated function returns the primal through `value_ret` and one scalar
 * derivative through `gradientN_ret` for every indexed dynamic constant.
 * Derivatives are fully unrolled: no runtime dual-number arrays or derivative
 * loops are emitted. ABS selects +1 at zero; MIN and MAX select their left
 * operand at equality. Other supported f32 operations use their ordinary
 * forward-mode derivatives.
 *
 * Input, routine, validation, measurement, and temporary-count behavior match
 * secant_cuda_ast_primal_source_generate.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_ast_forward_gradient_source_generate(
    const char* function_name,
    size_t num_input_columns,
    size_t num_dynamic_constants,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* ast,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* num_variables_ret
);

/**
 * Renders a mixed column-or-constant AST and its forward derivatives.
 *
 * Mixed dynamic leaves are supplied through the generated constantN
 * parameters.  The caller is responsible for masking a derivative when that
 * runtime leaf is column-bound.  Repeated occurrences of the same leaf index
 * share, and therefore accumulate into, the same derivative.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_ast_dynamic_leaf_forward_gradient_source_generate(
    const char* function_name,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* ast,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* num_variables_ret
);

/**
 * Renders an LM AST with independently indexed constants and mixed leaves.
 *
 * Indexed dynamic-constant instructions always read constantN and always seed
 * derivative N with one. Indexed mixed constant-or-column instructions read
 * mixedN and seed derivative N with mixed_gradientN, allowing the caller to
 * pass zero for a column binding and one for a constant binding. Repeated
 * indices accumulate into one derivative in both cases.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_ast_lm_forward_gradient_source_generate(
    const char* function_name,
    size_t num_static_input_columns,
    size_t num_dynamic_constants,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* ast,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* num_variables_ret
);

/**
 * Generates static-column materialize CUDA source.
 *
 * ASTs are ordered [kernel][ast]. Passing NULL for buffer measures the exact
 * source size, including the trailing NUL. The generated kernels materialize
 * output as [ast][row].
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_materialize_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
);

/**
 * Generates static-column SSE CUDA source.
 *
 * ASTs are ordered [kernel][ast]. Each block owns one tile of tile_rows and
 * atomically accumulates [ast][target] SSE. Passing NULL for buffer measures
 * the exact source size, including the trailing NUL.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
);

/**
 * Generates tile-static SSE CUDA with per-thread dynamic constants.
 *
 * AST input indices [0, num_input_columns) select columns loaded into the
 * shared-memory row tile. Indices beginning at num_input_columns select the
 * current thread's constant-setting registers. Constant settings are stored
 * as [constant][setting]. Output is [ast][target][setting], with settings
 * contiguous. Every thread reduces complete row tiles for its own settings;
 * no cross-thread SSE reduction is performed. Row-tile blocks atomically add
 * to output_sse, so the caller clears it before a fresh accumulation.
 */
SECANT_CUDA_DEC SecantCUDAResult
secant_cuda_dynamic_constant_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
);

/** Generates the comparison CUDA implementation of mixed dynamic-leaf SSE. */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_dynamic_leaf_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
);

/**
 * Generates tile-static SSE CUDA with one dynamic-leaf setting per warp.
 *
 * The kernel ABI and output layout match
 * secant_cuda_dynamic_leaf_sse_source_generate. Each warp owns one setting,
 * evaluates disjoint rows of the resident tile, reduces SSE with warp
 * shuffles, and lets lane zero atomically add the tile contribution. Dynamic
 * bindings are therefore uniform within a warp. This topology is also the
 * reference scheduling skeleton for future per-state statistics such as LM.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_warp_dynamic_leaf_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
);

/** Number of floats in [SSE, J^T r, packed-upper J^T J] for p parameters. */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_dynamic_leaf_lm_statistics_count(
    size_t num_parameters,
    size_t* num_statistics_ret
);

/** Generates thread-owned mixed-leaf LM normal-equation statistics CUDA. */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_dynamic_leaf_lm_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
);

/** Generates warp-owned mixed-leaf LM normal-equation statistics CUDA. */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_warp_dynamic_leaf_lm_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
);

/**
 * Generates the production thread-owned, one-AST LM optimizer module.
 *
 * Each `secant_lm_statistics_NNN` function owns one AST. Every input column is
 * also a fixed static AST input, while each of the eight indexed leaves can be
 * selected at runtime as either a constant or a dynamically bound column.
 * Grid X partitions static row tiles and grid Y partitions setting groups
 * using the runtime `settings_per_cta` argument. The module also exports one
 * thread-owned `secant_lm_solve_f32` kernel specialized to eight parameters.
 * Statistics are `[AST][45][setting]`; constants are `[AST][setting][8]`.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_lm_optimizer_source_generate(
    size_t num_kernels,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
);

/** Generates only the fixed-eight LM solve kernel shared by source backends. */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_lm_solver_source_generate(
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret
);

/**
 * Generates benchmark CUDA for Philox-sampled mixed dynamic-leaf selection.
 *
 * Every CTA loads the complete runtime row set, which must not exceed
 * `tile_rows`. A thread evaluates `num_setting_passes` Philox settings across
 * every active packed AST and writes its best encoded candidate index:
 *
 *     setting_pass * asts_per_kernel + ast_index
 *
 * Output element `[blockIdx.x * blockDim.x + threadIdx.x]` therefore identifies
 * the thread needed to reconstruct the winning setting. `output_best_sse` may
 * be null. The translation unit also exports
 * `secant_philox_dynamic_leaf_reconstruct`, which regenerates leaf masks and
 * words from those output slots and winner indices.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_philox_dynamic_leaf_select_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
);

/**
 * Generates packed shared-jitter constant-optimizer CUDA source.
 *
 * `current_constants` and `current_constant_scales` are `[ast][constant]`.
 * Their values are embedded in the generated expressions. The fixed reducer
 * is appended to the same translation unit.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_packed_constant_optimizer_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    const float* current_constants,
    const float* current_constant_scales,
    size_t current_constants_leading_dimension,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret,
    uint32_t* max_variables_ret
);

/**
 * Compiles static-column materialize ASTs directly to one owned CUBIN.
 *
 * The compile scratch contains NVRTC option pointers followed by generated
 * source. Only the returned SecantCUDACompiled is allocated internally.
 * log_size_ret receives required log bytes including the NUL when verbose is
 * true or compilation fails. This entire call is the canonical CUDA
 * AST-to-CUBIN timing interval.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_materialize_compile(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    bool verbose,
    void* compile_scratch,
    size_t compile_scratch_size,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantCUDACompiled* compiled_ret
);

/**
 * Compiles static-column SSE ASTs directly to one owned CUBIN.
 *
 * Scratch, logging, ownership, and timing semantics match
 * secant_cuda_materialize_compile.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_sse_compile(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    bool verbose,
    void* compile_scratch,
    size_t compile_scratch_size,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantCUDACompiled* compiled_ret
);

/**
 * Compiles tile-static dynamic-constant SSE ASTs to one owned CUBIN.
 *
 * Input-index, constant-layout, output-layout, scratch, logging, and ownership
 * semantics match secant_cuda_dynamic_constant_sse_source_generate and
 * secant_cuda_sse_compile.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_dynamic_constant_sse_compile(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    bool verbose,
    void* compile_scratch,
    size_t compile_scratch_size,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantCUDACompiled* compiled_ret
);

/** Compiles mixed dynamic-leaf SSE ASTs directly to one owned CUBIN. */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_dynamic_leaf_sse_compile(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    bool verbose,
    void* compile_scratch,
    size_t compile_scratch_size,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantCUDACompiled* compiled_ret
);

/** Compiles the warp-owned mixed dynamic-leaf SSE topology. */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_warp_dynamic_leaf_sse_compile(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    bool verbose,
    void* compile_scratch,
    size_t compile_scratch_size,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantCUDACompiled* compiled_ret
);

/** Compiles thread-owned mixed-leaf LM normal-equation statistics. */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_dynamic_leaf_lm_compile(
    size_t num_kernels, size_t asts_per_kernel,
    size_t num_input_columns, size_t num_static_input_columns,
    size_t num_dynamic_leaves, size_t num_targets,
    size_t tile_rows, size_t threads_per_block,
    const SecantAstInstruction* const* routines, size_t num_routines,
    const SecantAstInstruction* const* asts,
    uint32_t compute_capability_major, uint32_t compute_capability_minor,
    const char* const* nvrtc_options, size_t num_nvrtc_options,
    bool verbose, void* compile_scratch, size_t compile_scratch_size,
    char* log_buffer, size_t log_buffer_size, size_t* log_size_ret,
    SecantCUDACompiled* compiled_ret
);

/** Compiles warp-owned mixed-leaf LM normal-equation statistics. */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_warp_dynamic_leaf_lm_compile(
    size_t num_kernels, size_t asts_per_kernel,
    size_t num_input_columns, size_t num_static_input_columns,
    size_t num_dynamic_leaves, size_t num_targets,
    size_t tile_rows, size_t threads_per_block,
    const SecantAstInstruction* const* routines, size_t num_routines,
    const SecantAstInstruction* const* asts,
    uint32_t compute_capability_major, uint32_t compute_capability_minor,
    const char* const* nvrtc_options, size_t num_nvrtc_options,
    bool verbose, void* compile_scratch, size_t compile_scratch_size,
    char* log_buffer, size_t log_buffer_size, size_t* log_size_ret,
    SecantCUDACompiled* compiled_ret
);

/**
 * Compiles a thread-owned, one-AST-per-function LM optimizer module.
 *
 * The module exports `secant_lm_statistics_NNN` for every AST and one shared
 * `secant_lm_solve_f32` kernel. It is intended to stay loaded throughout the
 * fixed evaluate/solve launch sequence.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_lm_optimizer_compile(
    size_t num_kernels,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    bool verbose,
    void* compile_scratch,
    size_t compile_scratch_size,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantCUDACompiled* compiled_ret
);

/** Compiles benchmark Philox mixed-leaf selection ASTs to one owned CUBIN. */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_philox_dynamic_leaf_select_compile(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_static_input_columns,
    size_t num_dynamic_leaves,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    bool verbose,
    void* compile_scratch,
    size_t compile_scratch_size,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantCUDACompiled* compiled_ret
);

/** Compiles one packed constant-optimizer module using native CUDA C++. */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_packed_constant_optimizer_sse_compile(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    const float* current_constants,
    const float* current_constant_scales,
    size_t current_constants_leading_dimension,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    bool verbose,
    void* compile_scratch,
    size_t compile_scratch_size,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantCUDACompiled* compiled_ret
);

/** Releases one caller reference to an owned compiled CUBIN. */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_compiled_destroy(SecantCUDACompiled compiled);

/**
 * Borrows the CUBIN bytes owned by compiled.
 *
 * The returned pointer remains valid until compiled is destroyed.
 */
SECANT_CUDA_DEC SecantCUDAResult secant_cuda_compiled_binary_get(
    SecantCUDACompiled compiled,
    const void** binary_ret,
    size_t* binary_size_ret
);

#endif /* SECANT_CUDA_H_INCLUDED */
