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
    SecantSSEReductionMode reduction_mode,
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
 * to output_sse, so the caller clears it before a fresh reduction.
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
    SecantSSEReductionMode reduction_mode,
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
    SecantSSEReductionMode reduction_mode,
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
    SecantSSEReductionMode reduction_mode,
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
