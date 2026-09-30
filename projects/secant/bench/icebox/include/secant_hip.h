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
#ifndef SECANT_HIP_H_INCLUDED
#define SECANT_HIP_H_INCLUDED

#include "secant.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define SECANT_HIP_DEC extern "C"
#else
#define SECANT_HIP_DEC extern
#endif

#define SECANT_HIP_SSE_MAX_WAVES 16u
#define SECANT_HIP_DYNAMIC_CONSTANT_SSE_MAX_INPUT_COLUMNS 32u
#define SECANT_HIP_DYNAMIC_CONSTANT_SSE_MAX_INPUT_CONSTANTS 32u

/**
 * HIP C++ compiler baseline.
 *
 * This backend is retained for correctness validation and compile/runtime
 * comparisons with native HSACO specialization. It is not SECANT's primary
 * deployment backend.
 */

/** Status returned by HIP source generation and compilation calls. */
typedef enum SecantHIPResult {
    SECANT_HIP_SUCCESS = 0,
    SECANT_HIP_ERROR_INVALID_VALUE = 1,
    SECANT_HIP_ERROR_OVERFLOW = 2,
    SECANT_HIP_ERROR_INSUFFICIENT_BUFFER = 3,
    SECANT_HIP_ERROR_FORMAT = 4,
    SECANT_HIP_ERROR_BAD_PROGRAM = 5,
    SECANT_HIP_ERROR_STACK_OVERFLOW = 6,
    SECANT_HIP_ERROR_STACK_UNDERFLOW = 7,
    SECANT_HIP_ERROR_TOO_MANY_ARGS = 8,
    SECANT_HIP_ERROR_UNSUPPORTED_OP = 9,
    SECANT_HIP_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS = 10,
    SECANT_HIP_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS = 11,
    SECANT_HIP_ERROR_ROUTINE_DEPTH_EXCEEDED = 12,
    SECANT_HIP_ERROR_ALLOCATION_FAILED = 13,
    SECANT_HIP_ERROR_COMPILE_FAILED = 14,
    SECANT_HIP_ERROR_INVALID_STATE = 15,
    SECANT_HIP_ERROR_UNSUPPORTED_SHAPE = 16,
    SECANT_HIP_RESULT_NUM_ENUMS = 17
} SecantHIPResult;

/** Owned HIP HSACO returned by a successful compile call. */
typedef struct SecantHIPCompiledImpl* SecantHIPCompiled;

/** Returns a static symbolic name for a HIP result code. */
SECANT_HIP_DEC const char* secant_hip_result_to_string(SecantHIPResult result);

/**
 * Renders one AST as an inline f32 HIP device function.
 *
 * Passing NULL for buffer measures the required bytes, including the trailing
 * NUL. num_variables_ret optionally reports the maximum temporary-variable
 * count used by the rendered expression.
 */
SECANT_HIP_DEC SecantHIPResult secant_hip_function_source_generate(
    const char* function_name,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* instructions,
    char* buffer,
    size_t buffer_size,
    size_t* hip_size_ret,
    uint32_t* num_variables_ret
);

/**
 * Generates static-column materialize HIP source.
 *
 * ASTs are ordered [kernel][ast]. Passing NULL for buffer measures the exact
 * source size, including the trailing NUL. The generated kernels materialize
 * output as [ast][row].
 */
SECANT_HIP_DEC SecantHIPResult secant_hip_materialize_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* buffer,
    size_t buffer_size,
    size_t* hip_size_ret,
    uint32_t* max_variables_ret
);

/**
 * Generates static-column SSE HIP source.
 *
 * ASTs are ordered [kernel][ast]. Each block owns one tile of tile_rows and
 * atomically accumulates [ast][target] SSE. Passing NULL for buffer measures
 * the exact source size, including the trailing NUL. For portability across
 * wave32 and wave64 devices, threads_per_block must be 64 through 512 and a
 * multiple of 64.
 */
SECANT_HIP_DEC SecantHIPResult secant_hip_sse_source_generate(
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
    size_t* hip_size_ret,
    uint32_t* max_variables_ret
);

/** Generates tile-static SSE HIP source with per-thread dynamic constants. */
SECANT_HIP_DEC SecantHIPResult
secant_hip_dynamic_constant_sse_source_generate(
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
    size_t* hip_size_ret,
    uint32_t* max_variables_ret
);

/**
 * Compiles static-column materialize ASTs directly to one owned HSACO.
 *
 * The compile scratch contains HIPRTC option pointers followed by generated
 * source. Only the returned SecantHIPCompiled is allocated internally.
 * log_size_ret receives required log bytes including the NUL when verbose is
 * true or compilation fails. This entire call is the canonical HIP
 * AST-to-HSACO timing interval.
 */
SECANT_HIP_DEC SecantHIPResult secant_hip_materialize_compile(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    const char* architecture,
    const char* const* hiprtc_options,
    size_t num_hiprtc_options,
    bool verbose,
    void* compile_scratch,
    size_t compile_scratch_size,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantHIPCompiled* compiled_ret
);

/**
 * Compiles static-column SSE ASTs directly to one owned HSACO.
 *
 * Scratch, logging, ownership, and timing semantics match
 * secant_hip_materialize_compile.
 */
SECANT_HIP_DEC SecantHIPResult secant_hip_sse_compile(
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
    const char* architecture,
    const char* const* hiprtc_options,
    size_t num_hiprtc_options,
    bool verbose,
    void* compile_scratch,
    size_t compile_scratch_size,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantHIPCompiled* compiled_ret
);

/** Compiles dynamic-constant SSE ASTs directly to one owned HSACO. */
SECANT_HIP_DEC SecantHIPResult secant_hip_dynamic_constant_sse_compile(
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
    const char* architecture,
    const char* const* hiprtc_options,
    size_t num_hiprtc_options,
    bool verbose,
    void* compile_scratch,
    size_t compile_scratch_size,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantHIPCompiled* compiled_ret
);

/** Releases one caller reference to an owned compiled HSACO. */
SECANT_HIP_DEC SecantHIPResult secant_hip_compiled_destroy(SecantHIPCompiled compiled);

/**
 * Borrows the HSACO bytes owned by compiled.
 *
 * The returned pointer remains valid until compiled is destroyed.
 */
SECANT_HIP_DEC SecantHIPResult secant_hip_compiled_binary_get(
    SecantHIPCompiled compiled,
    const void** binary_ret,
    size_t* binary_size_ret
);

#endif /* SECANT_HIP_H_INCLUDED */
