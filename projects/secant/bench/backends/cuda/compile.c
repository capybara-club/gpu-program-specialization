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
#include "secant_cuda.h"

#include <nvrtc.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define _SECANT_CUDA_ERROR_RET(ans) do { SecantCUDAResult secant_cuda_result = (ans); return secant_cuda_result; } while (0)

typedef enum SecantCUDAKernelShape {
    SECANT_CUDA_KERNEL_SHAPE_STATIC_COLUMN_MATERIALIZE = 0,
    SECANT_CUDA_KERNEL_SHAPE_STATIC_COLUMN_SSE = 1,
    SECANT_CUDA_KERNEL_SHAPE_DYNAMIC_CONSTANT_SSE = 2,
    SECANT_CUDA_KERNEL_SHAPE_DYNAMIC_LEAF_SSE = 3,
    SECANT_CUDA_KERNEL_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE = 4,
    SECANT_CUDA_KERNEL_SHAPE_PHILOX_DYNAMIC_LEAF_SELECT = 5,
    SECANT_CUDA_KERNEL_SHAPE_WARP_DYNAMIC_LEAF_SSE = 6,
    SECANT_CUDA_KERNEL_SHAPE_DYNAMIC_LEAF_LM = 7,
    SECANT_CUDA_KERNEL_SHAPE_WARP_DYNAMIC_LEAF_LM = 8,
    SECANT_CUDA_KERNEL_SHAPE_LM_OPTIMIZER = 9
} SecantCUDAKernelShape;

struct SecantCUDACompiledImpl {
    size_t binary_size;
    unsigned char binary[];
};

const char*
secant_cuda_result_to_string(SecantCUDAResult result) {
    static const char* const strings[SECANT_CUDA_RESULT_NUM_ENUMS] = {
        "SECANT_CUDA_SUCCESS",
        "SECANT_CUDA_ERROR_INVALID_VALUE",
        "SECANT_CUDA_ERROR_OVERFLOW",
        "SECANT_CUDA_ERROR_INSUFFICIENT_BUFFER",
        "SECANT_CUDA_ERROR_FORMAT",
        "SECANT_CUDA_ERROR_BAD_PROGRAM",
        "SECANT_CUDA_ERROR_STACK_OVERFLOW",
        "SECANT_CUDA_ERROR_STACK_UNDERFLOW",
        "SECANT_CUDA_ERROR_TOO_MANY_ARGS",
        "SECANT_CUDA_ERROR_UNSUPPORTED_OP",
        "SECANT_CUDA_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS",
        "SECANT_CUDA_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS",
        "SECANT_CUDA_ERROR_ROUTINE_DEPTH_EXCEEDED",
        "SECANT_CUDA_ERROR_ALLOCATION_FAILED",
        "SECANT_CUDA_ERROR_COMPILE_FAILED",
        "SECANT_CUDA_ERROR_INVALID_STATE",
        "SECANT_CUDA_ERROR_UNSUPPORTED_SHAPE"
    };

    return result < SECANT_CUDA_RESULT_NUM_ENUMS
        ? strings[result]
        : "SECANT_CUDA_ERROR_UNKNOWN";
}

static SecantCUDAResult
_secant_cuda_capture_program_log(
    nvrtcProgram program,
    bool compile_succeeded,
    bool verbose,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret
) {
    size_t log_size = 0u;
    nvrtcResult nvrtc_result;

    if (log_size_ret != NULL) {
        *log_size_ret = 0u;
    }
    if (compile_succeeded && !verbose) {
        if (log_buffer != NULL && log_buffer_size != 0u) {
            log_buffer[0] = '\0';
        }
        return SECANT_CUDA_SUCCESS;
    }

    nvrtc_result = nvrtcGetProgramLogSize(program, &log_size);
    if (nvrtc_result != NVRTC_SUCCESS) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_COMPILE_FAILED);
    }
    if (log_size_ret != NULL) {
        *log_size_ret = log_size;
    }
    if (log_buffer == NULL) {
        return SECANT_CUDA_SUCCESS;
    }
    if (log_buffer_size < log_size) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INSUFFICIENT_BUFFER);
    }
    if (log_size == 0u) {
        if (log_buffer_size != 0u) {
            log_buffer[0] = '\0';
        }
        return SECANT_CUDA_SUCCESS;
    }
    if (nvrtcGetProgramLog(program, log_buffer) != NVRTC_SUCCESS) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_COMPILE_FAILED);
    }
    return SECANT_CUDA_SUCCESS;
}

static SecantCUDAResult
_secant_cuda_allocate_compiled(
    size_t binary_size,
    SecantCUDACompiled* compiled_ret
) {
    SecantCUDACompiled compiled;

    if (compiled_ret == NULL || binary_size == 0u ||
        binary_size > SIZE_MAX - sizeof(*compiled)) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    compiled = (SecantCUDACompiled)malloc(sizeof(*compiled) + binary_size);
    if (compiled == NULL) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_ALLOCATION_FAILED);
    }
    compiled->binary_size = binary_size;
    *compiled_ret = compiled;
    return SECANT_CUDA_SUCCESS;
}

static void
_secant_cuda_compiled_release(SecantCUDACompiled compiled) {
    free(compiled);
}

static SecantCUDAResult
_secant_cuda_generate_source(
    SecantCUDAKernelShape kernel_shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_static_inputs,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    const float* current_constants,
    const float* current_constant_scales,
    size_t current_constants_leading_dimension,
    char* source,
    size_t source_capacity,
    size_t* source_size_ret
) {
    switch (kernel_shape) {
        case SECANT_CUDA_KERNEL_SHAPE_STATIC_COLUMN_MATERIALIZE:
            _SECANT_CUDA_ERROR_RET(secant_cuda_materialize_source_generate(
                num_kernels,
                asts_per_kernel,
                num_inputs,
                routines,
                num_routines,
                routine_names,
                asts,
                source,
                source_capacity,
                source_size_ret,
                NULL));
        case SECANT_CUDA_KERNEL_SHAPE_STATIC_COLUMN_SSE:
            _SECANT_CUDA_ERROR_RET(secant_cuda_sse_source_generate(
                num_kernels,
                asts_per_kernel,
                num_inputs,
                num_targets,
                tile_rows,
                threads_per_block,
                routines,
                num_routines,
                routine_names,
                asts,
                source,
                source_capacity,
                source_size_ret,
                NULL));
        case SECANT_CUDA_KERNEL_SHAPE_DYNAMIC_CONSTANT_SSE:
            _SECANT_CUDA_ERROR_RET(
                secant_cuda_dynamic_constant_sse_source_generate(
                    num_kernels,
                    asts_per_kernel,
                    num_inputs,
                    num_input_constants,
                    num_targets,
                    tile_rows,
                    threads_per_block,
                    routines,
                    num_routines,
                    routine_names,
                    asts,
                    source,
                    source_capacity,
                    source_size_ret,
                    NULL));
        case SECANT_CUDA_KERNEL_SHAPE_DYNAMIC_LEAF_SSE:
            _SECANT_CUDA_ERROR_RET(secant_cuda_dynamic_leaf_sse_source_generate(
                num_kernels,
                asts_per_kernel,
                num_inputs,
                num_static_inputs,
                num_input_constants,
                num_targets,
                tile_rows,
                threads_per_block,
                routines,
                num_routines,
                routine_names,
                asts,
                source,
                source_capacity,
                source_size_ret,
                NULL));
        case SECANT_CUDA_KERNEL_SHAPE_WARP_DYNAMIC_LEAF_SSE:
            _SECANT_CUDA_ERROR_RET(secant_cuda_warp_dynamic_leaf_sse_source_generate(
                num_kernels,
                asts_per_kernel,
                num_inputs,
                num_static_inputs,
                num_input_constants,
                num_targets,
                tile_rows,
                threads_per_block,
                routines,
                num_routines,
                routine_names,
                asts,
                source,
                source_capacity,
                source_size_ret,
                NULL));
        case SECANT_CUDA_KERNEL_SHAPE_DYNAMIC_LEAF_LM:
            _SECANT_CUDA_ERROR_RET(secant_cuda_dynamic_leaf_lm_source_generate(
                num_kernels,
                asts_per_kernel,
                num_inputs,
                num_static_inputs,
                num_input_constants,
                num_targets,
                tile_rows,
                threads_per_block,
                routines,
                num_routines,
                asts,
                source,
                source_capacity,
                source_size_ret,
                NULL));
        case SECANT_CUDA_KERNEL_SHAPE_WARP_DYNAMIC_LEAF_LM:
            _SECANT_CUDA_ERROR_RET(secant_cuda_warp_dynamic_leaf_lm_source_generate(
                num_kernels,
                asts_per_kernel,
                num_inputs,
                num_static_inputs,
                num_input_constants,
                num_targets,
                tile_rows,
                threads_per_block,
                routines,
                num_routines,
                asts,
                source,
                source_capacity,
                source_size_ret,
                NULL));
        case SECANT_CUDA_KERNEL_SHAPE_LM_OPTIMIZER:
            _SECANT_CUDA_ERROR_RET(secant_cuda_lm_optimizer_source_generate(
                num_kernels,
                num_inputs,
                num_static_inputs,
                tile_rows,
                threads_per_block,
                routines,
                num_routines,
                asts,
                source,
                source_capacity,
                source_size_ret,
                NULL));
        case SECANT_CUDA_KERNEL_SHAPE_PHILOX_DYNAMIC_LEAF_SELECT:
            _SECANT_CUDA_ERROR_RET(secant_cuda_philox_dynamic_leaf_select_source_generate(
                num_kernels,
                asts_per_kernel,
                num_inputs,
                num_static_inputs,
                num_input_constants,
                tile_rows,
                threads_per_block,
                routines,
                num_routines,
                routine_names,
                asts,
                source,
                source_capacity,
                source_size_ret,
                NULL));
        case SECANT_CUDA_KERNEL_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE:
            _SECANT_CUDA_ERROR_RET(secant_cuda_packed_constant_optimizer_sse_source_generate(
                num_kernels,
                asts_per_kernel,
                num_inputs,
                num_input_constants,
                tile_rows,
                threads_per_block,
                routines,
                num_routines,
                routine_names,
                asts,
                current_constants,
                current_constant_scales,
                current_constants_leading_dimension,
                source,
                source_capacity,
                source_size_ret,
                NULL));
        default:
            _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_UNSUPPORTED_SHAPE);
    }
}

static SecantCUDAResult
_secant_cuda_compile(
    SecantCUDAKernelShape kernel_shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_static_inputs,
    size_t num_input_constants,
    size_t num_targets,
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
) {
    nvrtcProgram program = NULL;
    SecantCUDACompiled compiled = NULL;
    const char** options;
    char architecture[64];
    char* source;
    uintptr_t scratch_begin;
    uintptr_t options_begin;
    size_t options_bytes;
    size_t source_offset;
    size_t source_size = 0u;
    size_t cubin_size = 0u;
    size_t option_idx;
    int architecture_bytes;
    nvrtcResult nvrtc_result;
    SecantCUDAResult result;

    if (num_kernels == 0u || asts_per_kernel == 0u || num_inputs == 0u || asts == NULL ||
        compute_capability_major == 0u || compile_scratch == NULL || compile_scratch_size == 0u ||
        compiled_ret == NULL || (num_routines != 0u && routines == NULL) ||
        (num_nvrtc_options != 0u && nvrtc_options == NULL)) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    *compiled_ret = NULL;
    if (log_size_ret != NULL) {
        *log_size_ret = 0u;
    }
    if (num_nvrtc_options > (size_t)INT_MAX - 2u ||
        num_nvrtc_options + 2u > SIZE_MAX / sizeof(*options)) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
    }
    architecture_bytes = snprintf(
        architecture,
        sizeof(architecture),
        "--gpu-architecture=sm_%u%u",
        compute_capability_major,
        compute_capability_minor);
    if (architecture_bytes < 0 || (size_t)architecture_bytes >= sizeof(architecture)) {
        _SECANT_CUDA_ERROR_RET(
            architecture_bytes < 0 ? SECANT_CUDA_ERROR_FORMAT : SECANT_CUDA_ERROR_OVERFLOW);
    }

    scratch_begin = (uintptr_t)compile_scratch;
    if (scratch_begin > UINTPTR_MAX - (sizeof(void*) - 1u)) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
    }
    options_begin = (scratch_begin + sizeof(void*) - 1u) & ~(uintptr_t)(sizeof(void*) - 1u);
    if (options_begin < scratch_begin) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_OVERFLOW);
    }
    options_bytes = (num_nvrtc_options + 2u) * sizeof(*options);
    source_offset = (size_t)(options_begin - scratch_begin);
    if (source_offset > compile_scratch_size ||
        options_bytes > compile_scratch_size - source_offset) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INSUFFICIENT_BUFFER);
    }
    options = (const char**)options_begin;
    source_offset += options_bytes;
    source = (char*)compile_scratch + source_offset;
    options[0] = "--std=c++11";
    options[1] = architecture;
    for (option_idx = 0u; option_idx < num_nvrtc_options; ++option_idx) {
        if (nvrtc_options[option_idx] == NULL) {
            _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
        }
        options[option_idx + 2u] = nvrtc_options[option_idx];
    }

    result = _secant_cuda_generate_source(
        kernel_shape,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        num_static_inputs,
        num_input_constants,
        num_targets,
        tile_rows,
        threads_per_block,
        routines,
        num_routines,
        routine_names,
        asts,
        current_constants,
        current_constant_scales,
        current_constants_leading_dimension,
        source,
        compile_scratch_size - source_offset,
        &source_size);
    if (result != SECANT_CUDA_SUCCESS) {
        _SECANT_CUDA_ERROR_RET(result);
    }

    nvrtc_result = nvrtcCreateProgram(&program, source, "secant_generated.cu", 0, NULL, NULL);
    if (nvrtc_result != NVRTC_SUCCESS) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_COMPILE_FAILED);
    }
    nvrtc_result = nvrtcCompileProgram(program, (int)(num_nvrtc_options + 2u), options);
    result = _secant_cuda_capture_program_log(
        program,
        nvrtc_result == NVRTC_SUCCESS,
        verbose,
        log_buffer,
        log_buffer_size,
        log_size_ret);
    if (result != SECANT_CUDA_SUCCESS || nvrtc_result != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&program);
        _SECANT_CUDA_ERROR_RET(
            result != SECANT_CUDA_SUCCESS ? result : SECANT_CUDA_ERROR_COMPILE_FAILED);
    }

    nvrtc_result = nvrtcGetCUBINSize(program, &cubin_size);
    if (nvrtc_result != NVRTC_SUCCESS || cubin_size == 0u) {
        nvrtcDestroyProgram(&program);
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_COMPILE_FAILED);
    }
    result = _secant_cuda_allocate_compiled(
        cubin_size,
        &compiled);
    if (result != SECANT_CUDA_SUCCESS) {
        nvrtcDestroyProgram(&program);
        _SECANT_CUDA_ERROR_RET(result);
    }
    nvrtc_result = nvrtcGetCUBIN(program, (char*)compiled->binary);
    nvrtcDestroyProgram(&program);
    if (nvrtc_result != NVRTC_SUCCESS) {
        _secant_cuda_compiled_release(compiled);
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_COMPILE_FAILED);
    }
    *compiled_ret = compiled;
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_cuda_materialize_compile(
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
) {
    _SECANT_CUDA_ERROR_RET(_secant_cuda_compile(
        SECANT_CUDA_KERNEL_SHAPE_STATIC_COLUMN_MATERIALIZE,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        0u,
        0u,
        0u,
        0u,
        0u,
        routines,
        num_routines,
        routine_names,
        asts,
        NULL,
        NULL,
        0u,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        verbose,
        compile_scratch,
        compile_scratch_size,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        compiled_ret));
}

SecantCUDAResult
secant_cuda_sse_compile(
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
) {
    _SECANT_CUDA_ERROR_RET(_secant_cuda_compile(
        SECANT_CUDA_KERNEL_SHAPE_STATIC_COLUMN_SSE,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        0u,
        0u,
        num_targets,
        tile_rows,
        threads_per_block,
        routines,
        num_routines,
        routine_names,
        asts,
        NULL,
        NULL,
        0u,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        verbose,
        compile_scratch,
        compile_scratch_size,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        compiled_ret));
}

SecantCUDAResult
secant_cuda_dynamic_constant_sse_compile(
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
) {
    _SECANT_CUDA_ERROR_RET(_secant_cuda_compile(
        SECANT_CUDA_KERNEL_SHAPE_DYNAMIC_CONSTANT_SSE,
        num_kernels,
        asts_per_kernel,
        num_input_columns,
        num_input_columns,
        num_input_constants,
        num_targets,
        tile_rows,
        threads_per_block,
        routines,
        num_routines,
        routine_names,
        asts,
        NULL,
        NULL,
        0u,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        verbose,
        compile_scratch,
        compile_scratch_size,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        compiled_ret));
}

SecantCUDAResult
secant_cuda_dynamic_leaf_sse_compile(
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
) {
    _SECANT_CUDA_ERROR_RET(_secant_cuda_compile(
        SECANT_CUDA_KERNEL_SHAPE_DYNAMIC_LEAF_SSE,
        num_kernels,
        asts_per_kernel,
        num_input_columns,
        num_static_input_columns,
        num_dynamic_leaves,
        num_targets,
        tile_rows,
        threads_per_block,
        routines,
        num_routines,
        routine_names,
        asts,
        NULL,
        NULL,
        0u,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        verbose,
        compile_scratch,
        compile_scratch_size,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        compiled_ret));
}

SecantCUDAResult
secant_cuda_warp_dynamic_leaf_sse_compile(
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
) {
    _SECANT_CUDA_ERROR_RET(_secant_cuda_compile(
        SECANT_CUDA_KERNEL_SHAPE_WARP_DYNAMIC_LEAF_SSE,
        num_kernels,
        asts_per_kernel,
        num_input_columns,
        num_static_input_columns,
        num_dynamic_leaves,
        num_targets,
        tile_rows,
        threads_per_block,
        routines,
        num_routines,
        routine_names,
        asts,
        NULL,
        NULL,
        0u,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        verbose,
        compile_scratch,
        compile_scratch_size,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        compiled_ret));
}

static SecantCUDAResult
_secant_cuda_dynamic_leaf_lm_compile(
    SecantCUDAKernelShape shape,
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
) {
    _SECANT_CUDA_ERROR_RET(_secant_cuda_compile(
        shape,
        num_kernels,
        asts_per_kernel,
        num_input_columns,
        num_static_input_columns,
        num_dynamic_leaves,
        num_targets,
        tile_rows,
        threads_per_block,
        routines,
        num_routines,
        NULL,
        asts,
        NULL,
        NULL,
        0u,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        verbose,
        compile_scratch,
        compile_scratch_size,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        compiled_ret));
}

SecantCUDAResult
secant_cuda_dynamic_leaf_lm_compile(
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
) {
    return _secant_cuda_dynamic_leaf_lm_compile(
        SECANT_CUDA_KERNEL_SHAPE_DYNAMIC_LEAF_LM,
        num_kernels, asts_per_kernel, num_input_columns, num_static_input_columns,
        num_dynamic_leaves, num_targets, tile_rows, threads_per_block,
        routines, num_routines, asts,
        compute_capability_major, compute_capability_minor,
        nvrtc_options, num_nvrtc_options, verbose,
        compile_scratch, compile_scratch_size, log_buffer, log_buffer_size,
        log_size_ret, compiled_ret);
}

SecantCUDAResult
secant_cuda_warp_dynamic_leaf_lm_compile(
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
) {
    return _secant_cuda_dynamic_leaf_lm_compile(
        SECANT_CUDA_KERNEL_SHAPE_WARP_DYNAMIC_LEAF_LM,
        num_kernels, asts_per_kernel, num_input_columns, num_static_input_columns,
        num_dynamic_leaves, num_targets, tile_rows, threads_per_block,
        routines, num_routines, asts,
        compute_capability_major, compute_capability_minor,
        nvrtc_options, num_nvrtc_options, verbose,
        compile_scratch, compile_scratch_size, log_buffer, log_buffer_size,
        log_size_ret, compiled_ret);
}

SecantCUDAResult
secant_cuda_lm_optimizer_compile(
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
) {
    _SECANT_CUDA_ERROR_RET(_secant_cuda_compile(
        SECANT_CUDA_KERNEL_SHAPE_LM_OPTIMIZER,
        num_kernels,
        1u,
        num_input_columns,
        num_static_input_columns,
        SECANT_CUDA_LM_OPTIMIZER_PARAMETERS,
        1u,
        tile_rows,
        threads_per_block,
        routines,
        num_routines,
        NULL,
        asts,
        NULL,
        NULL,
        0u,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        verbose,
        compile_scratch,
        compile_scratch_size,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        compiled_ret));
}

SecantCUDAResult
secant_cuda_philox_dynamic_leaf_select_compile(
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
) {
    _SECANT_CUDA_ERROR_RET(_secant_cuda_compile(
        SECANT_CUDA_KERNEL_SHAPE_PHILOX_DYNAMIC_LEAF_SELECT,
        num_kernels,
        asts_per_kernel,
        num_input_columns,
        num_static_input_columns,
        num_dynamic_leaves,
        1u,
        tile_rows,
        threads_per_block,
        routines,
        num_routines,
        routine_names,
        asts,
        NULL,
        NULL,
        0u,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        verbose,
        compile_scratch,
        compile_scratch_size,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        compiled_ret));
}

SecantCUDAResult
secant_cuda_packed_constant_optimizer_sse_compile(
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
) {
    _SECANT_CUDA_ERROR_RET(_secant_cuda_compile(
        SECANT_CUDA_KERNEL_SHAPE_PACKED_CONSTANT_OPTIMIZER_SSE,
        num_kernels,
        asts_per_kernel,
        num_input_columns,
        num_input_columns,
        num_input_constants,
        1u,
        tile_rows,
        threads_per_block,
        routines,
        num_routines,
        routine_names,
        asts,
        current_constants,
        current_constant_scales,
        current_constants_leading_dimension,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        verbose,
        compile_scratch,
        compile_scratch_size,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        compiled_ret));
}

SecantCUDAResult
secant_cuda_compiled_destroy(SecantCUDACompiled compiled) {
    if (compiled == NULL) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    _secant_cuda_compiled_release(compiled);
    return SECANT_CUDA_SUCCESS;
}

SecantCUDAResult
secant_cuda_compiled_binary_get(
    SecantCUDACompiled compiled,
    const void** binary_ret,
    size_t* binary_size_ret
) {
    if (compiled == NULL || binary_ret == NULL || binary_size_ret == NULL) {
        _SECANT_CUDA_ERROR_RET(SECANT_CUDA_ERROR_INVALID_VALUE);
    }
    *binary_ret = compiled->binary;
    *binary_size_ret = compiled->binary_size;
    return SECANT_CUDA_SUCCESS;
}

#undef _SECANT_CUDA_ERROR_RET
