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
#include "secant_hip.h"

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__
#endif

#include <hip/hiprtc.h>

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define _SECANT_HIP_ERROR_RET(ans) do { SecantHIPResult secant_hip_result = (ans); return secant_hip_result; } while (0)

typedef enum SecantHIPKernelShape {
    SECANT_HIP_KERNEL_SHAPE_STATIC_COLUMN_MATERIALIZE = 0,
    SECANT_HIP_KERNEL_SHAPE_STATIC_COLUMN_SSE = 1,
    SECANT_HIP_KERNEL_SHAPE_DYNAMIC_CONSTANT_SSE = 2
} SecantHIPKernelShape;

struct SecantHIPCompiledImpl {
    size_t binary_size;
    unsigned char binary[];
};

const char*
secant_hip_result_to_string(SecantHIPResult result) {
    static const char* const strings[SECANT_HIP_RESULT_NUM_ENUMS] = {
        "SECANT_HIP_SUCCESS",
        "SECANT_HIP_ERROR_INVALID_VALUE",
        "SECANT_HIP_ERROR_OVERFLOW",
        "SECANT_HIP_ERROR_INSUFFICIENT_BUFFER",
        "SECANT_HIP_ERROR_FORMAT",
        "SECANT_HIP_ERROR_BAD_PROGRAM",
        "SECANT_HIP_ERROR_STACK_OVERFLOW",
        "SECANT_HIP_ERROR_STACK_UNDERFLOW",
        "SECANT_HIP_ERROR_TOO_MANY_ARGS",
        "SECANT_HIP_ERROR_UNSUPPORTED_OP",
        "SECANT_HIP_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS",
        "SECANT_HIP_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS",
        "SECANT_HIP_ERROR_ROUTINE_DEPTH_EXCEEDED",
        "SECANT_HIP_ERROR_ALLOCATION_FAILED",
        "SECANT_HIP_ERROR_COMPILE_FAILED",
        "SECANT_HIP_ERROR_INVALID_STATE",
        "SECANT_HIP_ERROR_UNSUPPORTED_SHAPE"
    };

    return result < SECANT_HIP_RESULT_NUM_ENUMS
        ? strings[result]
        : "SECANT_HIP_ERROR_UNKNOWN";
}

static SecantHIPResult
_secant_hip_capture_program_log(
    hiprtcProgram program,
    bool compile_succeeded,
    bool verbose,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret
) {
    size_t log_size = 0u;
    hiprtcResult hiprtc_result;

    if (log_size_ret != NULL) {
        *log_size_ret = 0u;
    }
    if (compile_succeeded && !verbose) {
        if (log_buffer != NULL && log_buffer_size != 0u) {
            log_buffer[0] = '\0';
        }
        return SECANT_HIP_SUCCESS;
    }

    hiprtc_result = hiprtcGetProgramLogSize(program, &log_size);
    if (hiprtc_result != HIPRTC_SUCCESS) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_COMPILE_FAILED);
    }
    if (log_size_ret != NULL) {
        *log_size_ret = log_size;
    }
    if (log_buffer == NULL) {
        return SECANT_HIP_SUCCESS;
    }
    if (log_buffer_size < log_size) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_INSUFFICIENT_BUFFER);
    }
    if (log_size == 0u) {
        if (log_buffer_size != 0u) {
            log_buffer[0] = '\0';
        }
        return SECANT_HIP_SUCCESS;
    }
    if (hiprtcGetProgramLog(program, log_buffer) != HIPRTC_SUCCESS) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_COMPILE_FAILED);
    }
    return SECANT_HIP_SUCCESS;
}

static SecantHIPResult
_secant_hip_allocate_compiled(
    size_t binary_size,
    SecantHIPCompiled* compiled_ret
) {
    SecantHIPCompiled compiled;

    if (compiled_ret == NULL || binary_size == 0u ||
        binary_size > SIZE_MAX - sizeof(*compiled)) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }
    compiled = (SecantHIPCompiled)malloc(sizeof(*compiled) + binary_size);
    if (compiled == NULL) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_ALLOCATION_FAILED);
    }
    compiled->binary_size = binary_size;
    *compiled_ret = compiled;
    return SECANT_HIP_SUCCESS;
}

static void
_secant_hip_compiled_release(SecantHIPCompiled compiled) {
    free(compiled);
}

static SecantHIPResult
_secant_hip_generate_source(
    SecantHIPKernelShape kernel_shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    SecantSSEReductionMode reduction_mode,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const char* const* routine_names,
    const SecantAstInstruction* const* asts,
    char* source,
    size_t source_capacity,
    size_t* source_size_ret
) {
    switch (kernel_shape) {
        case SECANT_HIP_KERNEL_SHAPE_STATIC_COLUMN_MATERIALIZE:
            _SECANT_HIP_ERROR_RET(secant_hip_materialize_source_generate(
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
        case SECANT_HIP_KERNEL_SHAPE_STATIC_COLUMN_SSE:
            _SECANT_HIP_ERROR_RET(secant_hip_sse_source_generate(
                num_kernels,
                asts_per_kernel,
                num_inputs,
                num_targets,
                tile_rows,
                threads_per_block,
                reduction_mode,
                routines,
                num_routines,
                routine_names,
                asts,
                source,
                source_capacity,
                source_size_ret,
                NULL));
        case SECANT_HIP_KERNEL_SHAPE_DYNAMIC_CONSTANT_SSE:
            _SECANT_HIP_ERROR_RET(
                secant_hip_dynamic_constant_sse_source_generate(
                    num_kernels,
                    asts_per_kernel,
                    num_inputs - num_input_constants,
                    num_input_constants,
                    num_targets,
                    tile_rows,
                    threads_per_block,
                    reduction_mode,
                    routines,
                    num_routines,
                    routine_names,
                    asts,
                    source,
                    source_capacity,
                    source_size_ret,
                    NULL));
        default:
            _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_UNSUPPORTED_SHAPE);
    }
}

static SecantHIPResult
_secant_hip_compile(
    SecantHIPKernelShape kernel_shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
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
) {
    hiprtcProgram program = NULL;
    SecantHIPCompiled compiled = NULL;
    const char** options;
    char architecture_option[64];
    char* source;
    uintptr_t scratch_begin;
    uintptr_t options_begin;
    size_t options_bytes;
    size_t source_offset;
    size_t source_size = 0u;
    size_t hsaco_size = 0u;
    size_t option_idx;
    int architecture_bytes;
    hiprtcResult hiprtc_result;
    SecantHIPResult result;

    if (num_kernels == 0u || asts_per_kernel == 0u || num_inputs == 0u || asts == NULL ||
        architecture == NULL || architecture[0] == '\0' ||
        compile_scratch == NULL || compile_scratch_size == 0u ||
        compiled_ret == NULL || (num_routines != 0u && routines == NULL) ||
        (num_hiprtc_options != 0u && hiprtc_options == NULL)) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }
    *compiled_ret = NULL;
    if (log_size_ret != NULL) {
        *log_size_ret = 0u;
    }
    if (num_hiprtc_options > (size_t)INT_MAX - 2u ||
        num_hiprtc_options + 2u > SIZE_MAX / sizeof(*options)) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_OVERFLOW);
    }
    architecture_bytes = snprintf(
        architecture_option,
        sizeof(architecture_option),
        "--offload-arch=%s",
        architecture);
    if (architecture_bytes < 0 || (size_t)architecture_bytes >= sizeof(architecture_option)) {
        _SECANT_HIP_ERROR_RET(
            architecture_bytes < 0 ? SECANT_HIP_ERROR_FORMAT : SECANT_HIP_ERROR_OVERFLOW);
    }

    scratch_begin = (uintptr_t)compile_scratch;
    if (scratch_begin > UINTPTR_MAX - (sizeof(void*) - 1u)) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_OVERFLOW);
    }
    options_begin = (scratch_begin + sizeof(void*) - 1u) & ~(uintptr_t)(sizeof(void*) - 1u);
    if (options_begin < scratch_begin) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_OVERFLOW);
    }
    options_bytes = (num_hiprtc_options + 2u) * sizeof(*options);
    source_offset = (size_t)(options_begin - scratch_begin);
    if (source_offset > compile_scratch_size ||
        options_bytes > compile_scratch_size - source_offset) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_INSUFFICIENT_BUFFER);
    }
    options = (const char**)options_begin;
    source_offset += options_bytes;
    source = (char*)compile_scratch + source_offset;
    options[0] = "--std=c++11";
    options[1] = architecture_option;
    for (option_idx = 0u; option_idx < num_hiprtc_options; ++option_idx) {
        if (hiprtc_options[option_idx] == NULL) {
            _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
        }
        options[option_idx + 2u] = hiprtc_options[option_idx];
    }

    result = _secant_hip_generate_source(
        kernel_shape,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        num_input_constants,
        num_targets,
        tile_rows,
        threads_per_block,
        reduction_mode,
        routines,
        num_routines,
        routine_names,
        asts,
        source,
        compile_scratch_size - source_offset,
        &source_size);
    if (result != SECANT_HIP_SUCCESS) {
        _SECANT_HIP_ERROR_RET(result);
    }

    hiprtc_result = hiprtcCreateProgram(&program, source, "secant_generated.hip", 0, NULL, NULL);
    if (hiprtc_result != HIPRTC_SUCCESS) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_COMPILE_FAILED);
    }
    hiprtc_result = hiprtcCompileProgram(program, (int)(num_hiprtc_options + 2u), options);
    result = _secant_hip_capture_program_log(
        program,
        hiprtc_result == HIPRTC_SUCCESS,
        verbose,
        log_buffer,
        log_buffer_size,
        log_size_ret);
    if (result != SECANT_HIP_SUCCESS || hiprtc_result != HIPRTC_SUCCESS) {
        hiprtcDestroyProgram(&program);
        _SECANT_HIP_ERROR_RET(
            result != SECANT_HIP_SUCCESS ? result : SECANT_HIP_ERROR_COMPILE_FAILED);
    }

    hiprtc_result = hiprtcGetCodeSize(program, &hsaco_size);
    if (hiprtc_result != HIPRTC_SUCCESS || hsaco_size == 0u) {
        hiprtcDestroyProgram(&program);
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_COMPILE_FAILED);
    }
    result = _secant_hip_allocate_compiled(
        hsaco_size,
        &compiled);
    if (result != SECANT_HIP_SUCCESS) {
        hiprtcDestroyProgram(&program);
        _SECANT_HIP_ERROR_RET(result);
    }
    hiprtc_result = hiprtcGetCode(program, (char*)compiled->binary);
    hiprtcDestroyProgram(&program);
    if (hiprtc_result != HIPRTC_SUCCESS) {
        _secant_hip_compiled_release(compiled);
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_COMPILE_FAILED);
    }
    *compiled_ret = compiled;
    return SECANT_HIP_SUCCESS;
}

SecantHIPResult
secant_hip_materialize_compile(
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
) {
    _SECANT_HIP_ERROR_RET(_secant_hip_compile(
        SECANT_HIP_KERNEL_SHAPE_STATIC_COLUMN_MATERIALIZE,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        0u,
        0u,
        0u,
        0u,
        SECANT_SSE_REDUCTION_MODE_ATOMIC,
        routines,
        num_routines,
        routine_names,
        asts,
        architecture,
        hiprtc_options,
        num_hiprtc_options,
        verbose,
        compile_scratch,
        compile_scratch_size,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        compiled_ret));
}

SecantHIPResult
secant_hip_sse_compile(
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
) {
    _SECANT_HIP_ERROR_RET(_secant_hip_compile(
        SECANT_HIP_KERNEL_SHAPE_STATIC_COLUMN_SSE,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        0u,
        num_targets,
        tile_rows,
        threads_per_block,
        reduction_mode,
        routines,
        num_routines,
        routine_names,
        asts,
        architecture,
        hiprtc_options,
        num_hiprtc_options,
        verbose,
        compile_scratch,
        compile_scratch_size,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        compiled_ret));
}

SecantHIPResult
secant_hip_dynamic_constant_sse_compile(
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
) {
    size_t num_inputs;

    if (num_input_columns == 0u ||
        num_input_columns >
            SECANT_HIP_DYNAMIC_CONSTANT_SSE_MAX_INPUT_COLUMNS ||
        num_input_constants == 0u ||
        num_input_constants >
            SECANT_HIP_DYNAMIC_CONSTANT_SSE_MAX_INPUT_CONSTANTS ||
        num_targets == 0u || tile_rows == 0u ||
        threads_per_block == 0u || threads_per_block > 1024u ||
        num_input_constants > SIZE_MAX - num_input_columns) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }
    num_inputs = num_input_columns + num_input_constants;
    _SECANT_HIP_ERROR_RET(_secant_hip_compile(
        SECANT_HIP_KERNEL_SHAPE_DYNAMIC_CONSTANT_SSE,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        num_input_constants,
        num_targets,
        tile_rows,
        threads_per_block,
        reduction_mode,
        routines,
        num_routines,
        routine_names,
        asts,
        architecture,
        hiprtc_options,
        num_hiprtc_options,
        verbose,
        compile_scratch,
        compile_scratch_size,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        compiled_ret));
}

SecantHIPResult
secant_hip_compiled_destroy(SecantHIPCompiled compiled) {
    if (compiled == NULL) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }
    _secant_hip_compiled_release(compiled);
    return SECANT_HIP_SUCCESS;
}

SecantHIPResult
secant_hip_compiled_binary_get(
    SecantHIPCompiled compiled,
    const void** binary_ret,
    size_t* binary_size_ret
) {
    if (compiled == NULL || binary_ret == NULL || binary_size_ret == NULL) {
        _SECANT_HIP_ERROR_RET(SECANT_HIP_ERROR_INVALID_VALUE);
    }
    *binary_ret = compiled->binary;
    *binary_size_ret = compiled->binary_size;
    return SECANT_HIP_SUCCESS;
}

#undef _SECANT_HIP_ERROR_RET
