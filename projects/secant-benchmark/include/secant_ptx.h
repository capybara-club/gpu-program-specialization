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
#ifndef SECANT_PTX_H_INCLUDED
#define SECANT_PTX_H_INCLUDED

#include "secant.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef SECANT_PTX_MAX_SITES
#define SECANT_PTX_MAX_SITES 16384u
#endif
#ifndef PTX_INJECT_MAX_UNIQUE_INJECTS
#define PTX_INJECT_MAX_UNIQUE_INJECTS SECANT_PTX_MAX_SITES
#endif
#include "ptx_inject.h"

#ifdef __cplusplus
#define SECANT_PTX_DEC extern "C"
#else
#define SECANT_PTX_DEC extern
#endif

#define SECANT_PTX_SSE_MAX_WARPS 16u
#define SECANT_PTX_DYNAMIC_CONSTANT_SSE_MAX_INPUT_COLUMNS 32u
#define SECANT_PTX_DYNAMIC_CONSTANT_SSE_MAX_INPUT_CONSTANTS 32u

/**
 * PTX compiler baseline.
 *
 * This backend is retained for correctness validation and compile/runtime
 * comparisons with native CUBIN specialization. It is not SECANT's primary
 * deployment backend.
 */

/** Status returned by PTX template generation and compilation calls. */
typedef enum SecantPTXResult {
    SECANT_PTX_SUCCESS = 0,
    SECANT_PTX_ERROR_INVALID_VALUE = 1,
    SECANT_PTX_ERROR_OVERFLOW = 2,
    SECANT_PTX_ERROR_INSUFFICIENT_BUFFER = 3,
    SECANT_PTX_ERROR_FORMAT = 4,
    SECANT_PTX_ERROR_BAD_PROGRAM = 5,
    SECANT_PTX_ERROR_STACK_OVERFLOW = 6,
    SECANT_PTX_ERROR_STACK_UNDERFLOW = 7,
    SECANT_PTX_ERROR_TOO_MANY_ARGS = 8,
    SECANT_PTX_ERROR_UNSUPPORTED_OP = 9,
    SECANT_PTX_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS = 10,
    SECANT_PTX_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS = 11,
    SECANT_PTX_ERROR_ROUTINE_DEPTH_EXCEEDED = 12,
    SECANT_PTX_ERROR_ALLOCATION_FAILED = 13,
    SECANT_PTX_ERROR_COMPILE_FAILED = 14,
    SECANT_PTX_ERROR_INVALID_STATE = 15,
    SECANT_PTX_ERROR_UNSUPPORTED_SHAPE = 16,
    SECANT_PTX_RESULT_NUM_ENUMS = 17
} SecantPTXResult;

/** Reusable virtual-PTX template and parsed PTX Inject site metadata. */
typedef struct SecantPTXHandleImpl* SecantPTXHandle;

/** Owned CUBIN produced by nvPTXCompiler. */
typedef struct SecantPTXCompiledImpl* SecantPTXCompiled;

/** Returns a static symbolic name for a PTX result code. */
SECANT_PTX_DEC const char* secant_ptx_result_to_string(SecantPTXResult result);

/**
 * Generates materialize CUDA containing PTX Inject sites but no expressions.
 *
 * Passing NULL for buffer measures the exact source size, including the
 * trailing NUL.
 */
SECANT_PTX_DEC SecantPTXResult secant_ptx_materialize_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret
);

/**
 * Generates SSE CUDA containing PTX Inject sites but no expressions.
 *
 * Passing NULL for buffer measures the exact source size, including the
 * trailing NUL.
 */
SECANT_PTX_DEC SecantPTXResult secant_ptx_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    SecantSSEReductionMode reduction_mode,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret
);

/** Generates tile-static SSE CUDA/PTX Inject source with dynamic constants. */
SECANT_PTX_DEC SecantPTXResult
secant_ptx_dynamic_constant_sse_source_generate(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    SecantSSEReductionMode reduction_mode,
    char* buffer,
    size_t buffer_size,
    size_t* cuda_size_ret
);

/**
 * Creates a reusable static-column materialize PTX template.
 *
 * Creation generates and compiles the reusable CUDA template to virtual PTX,
 * then parses every injection site. This cold-path work is shared by all
 * secant_ptx_compile calls on the returned handle and is intentionally outside
 * the canonical AST-to-CUBIN timing interval.
 */
SECANT_PTX_DEC SecantPTXResult secant_ptx_materialize_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    uint32_t nvrtc_compute_capability_major,
    uint32_t nvrtc_compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    bool verbose,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantPTXHandle* handle_ret
);

/**
 * Creates a reusable static-column SSE PTX template.
 *
 * Ownership, logging, and cold-path timing semantics match
 * secant_ptx_materialize_create.
 */
SECANT_PTX_DEC SecantPTXResult secant_ptx_sse_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    SecantSSEReductionMode reduction_mode,
    uint32_t nvrtc_compute_capability_major,
    uint32_t nvrtc_compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    bool verbose,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantPTXHandle* handle_ret
);

/** Creates a reusable dynamic-constant SSE virtual-PTX template. */
SECANT_PTX_DEC SecantPTXResult secant_ptx_dynamic_constant_sse_create(
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_input_columns,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    SecantSSEReductionMode reduction_mode,
    uint32_t nvrtc_compute_capability_major,
    uint32_t nvrtc_compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    bool verbose,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantPTXHandle* handle_ret
);

/** Releases a PTX template handle after no compile calls are using it. */
SECANT_PTX_DEC SecantPTXResult secant_ptx_handle_destroy(SecantPTXHandle handle);

/**
 * Borrows the prepared virtual PTX owned by handle.
 *
 * The pointer remains valid until secant_ptx_handle_destroy.
 */
SECANT_PTX_DEC SecantPTXResult secant_ptx_template_get(
    SecantPTXHandle handle,
    const char** template_ptx_ret,
    size_t* template_ptx_size_ret
);

/** Reads immutable materialize shape metadata from a prepared handle. */
SECANT_PTX_DEC SecantPTXResult secant_ptx_materialize_info_get(
    SecantPTXHandle handle,
    size_t* num_kernels_ret,
    size_t* asts_per_kernel_ret,
    size_t* num_inputs_ret
);

/** Reads immutable SSE shape metadata from a prepared handle. */
SECANT_PTX_DEC SecantPTXResult secant_ptx_sse_info_get(
    SecantPTXHandle handle,
    size_t* num_kernels_ret,
    size_t* asts_per_kernel_ret,
    size_t* num_inputs_ret,
    size_t* num_targets_ret,
    size_t* tile_rows_ret,
    size_t* threads_per_block_ret
);

/** Reads immutable dynamic-constant SSE shape metadata. */
SECANT_PTX_DEC SecantPTXResult secant_ptx_dynamic_constant_sse_info_get(
    SecantPTXHandle handle,
    size_t* num_kernels_ret,
    size_t* asts_per_kernel_ret,
    size_t* num_input_columns_ret,
    size_t* num_input_constants_ret,
    size_t* num_targets_ret,
    size_t* tile_rows_ret,
    size_t* threads_per_block_ret
);

/**
 * Injects ASTs into a prepared template and compiles one owned CUBIN.
 *
 * ASTs are ordered [kernel][ast]. compile_scratch owns all lowering, rendered
 * stubs, rendered PTX, and nvPTXCompiler option pointers. Only the final cubin
 * is allocated internally. The handle is read-only during this call, so
 * concurrent calls are supported when each call has independent scratch, log,
 * AST, and output storage. This entire call is the canonical PTX AST-to-CUBIN
 * timing interval.
 */
SECANT_PTX_DEC SecantPTXResult secant_ptx_compile(
    SecantPTXHandle handle,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    uint32_t nvptx_compute_capability_major,
    uint32_t nvptx_compute_capability_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    bool verbose,
    void* compile_scratch,
    size_t compile_scratch_size,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantPTXCompiled* compiled_ret
);

/** Releases one caller reference to an nvPTXCompiler CUBIN. */
SECANT_PTX_DEC SecantPTXResult secant_ptx_compiled_destroy(SecantPTXCompiled compiled);

/**
 * Borrows the CUBIN bytes owned by compiled.
 *
 * The returned pointer remains valid until compiled is destroyed.
 */
SECANT_PTX_DEC SecantPTXResult secant_ptx_compiled_binary_get(
    SecantPTXCompiled compiled,
    const void** binary_ret,
    size_t* binary_size_ret
);

#endif /* SECANT_PTX_H_INCLUDED */
