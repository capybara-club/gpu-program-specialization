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
#define PTX_INJECT_IMPLEMENTATION
#include "secant_ptx.h"

#include <nvPTXCompiler.h>
#include <nvrtc.h>

#ifndef AST_PTX_IMPLEMENTATION_ONCE
#define AST_PTX_IMPLEMENTATION
#define _SECANT_PTX_DEFINED_AST_PTX_IMPLEMENTATION
#endif
#define AST_PTX_STACK_DEPTH SECANT_AST_MAX_STACK_DEPTH
#include "ast_ptx.h"
#include "ast_ptx_instructions.h"
#ifdef _SECANT_PTX_DEFINED_AST_PTX_IMPLEMENTATION
#undef _SECANT_PTX_DEFINED_AST_PTX_IMPLEMENTATION
#undef AST_PTX_IMPLEMENTATION
#endif

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _SECANT_PTX_ERROR_RET(ans) do { SecantPTXResult secant_ptx_result = (ans); return secant_ptx_result; } while (0)
#define _SECANT_PTX_CHECK_RET(ans) do { SecantPTXResult secant_ptx_check_result = (ans); if (secant_ptx_check_result != SECANT_PTX_SUCCESS) { _SECANT_PTX_ERROR_RET(secant_ptx_check_result); } } while (0)

typedef enum SecantPTXKernelShape {
    SECANT_PTX_KERNEL_SHAPE_STATIC_COLUMN_MATERIALIZE = 0,
    SECANT_PTX_KERNEL_SHAPE_STATIC_COLUMN_SSE = 1,
    SECANT_PTX_KERNEL_SHAPE_DYNAMIC_CONSTANT_SSE = 2
} SecantPTXKernelShape;

typedef struct SecantPTXSite {
    size_t inject_idx;
    const char* output_register_name;
    const char** input_register_names;
} SecantPTXSite;

struct SecantPTXHandleImpl {
    SecantPTXKernelShape kernel_shape;
    size_t num_kernels;
    size_t asts_per_kernel;
    size_t num_inputs;
    size_t num_input_constants;
    size_t num_targets;
    size_t tile_rows;
    size_t threads_per_block;
    SecantSSEReductionMode reduction_mode;
    PtxInjectHandle inject;
    char* template_ptx;
    size_t template_ptx_size;
    size_t num_sites;
    SecantPTXSite* sites;
    const char** input_register_names;
};

struct SecantPTXCompiledImpl {
    size_t binary_size;
    unsigned char binary[];
};

static SecantPTXResult
_secant_ptx_scratch_allocate(
    void* buffer,
    size_t buffer_size,
    size_t* offset,
    size_t limit,
    size_t allocation_size,
    size_t alignment,
    void** allocation_ret
) {
    uintptr_t address;
    uintptr_t aligned_address;
    size_t padding;

    if (buffer == NULL || offset == NULL || allocation_ret == NULL ||
        alignment == 0u || (alignment & (alignment - 1u)) != 0u ||
        limit > buffer_size || *offset > limit) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    address = (uintptr_t)buffer + *offset;
    if (address < (uintptr_t)buffer ||
        address > UINTPTR_MAX - (alignment - 1u)) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_OVERFLOW);
    }
    aligned_address =
        (address + alignment - 1u) & ~(uintptr_t)(alignment - 1u);
    padding = (size_t)(aligned_address - address);
    if (padding > limit - *offset ||
        allocation_size > limit - *offset - padding) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INSUFFICIENT_BUFFER);
    }
    *offset += padding;
    *allocation_ret = (unsigned char*)buffer + *offset;
    *offset += allocation_size;
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_capture_nvrtc_log(
    nvrtcProgram program,
    bool compile_succeeded,
    bool verbose,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret
) {
    size_t log_size = 0u;

    if (log_size_ret != NULL) {
        *log_size_ret = 0u;
    }
    if (compile_succeeded && !verbose) {
        if (log_buffer != NULL && log_buffer_size != 0u) {
            log_buffer[0] = '\0';
        }
        return SECANT_PTX_SUCCESS;
    }
    if (nvrtcGetProgramLogSize(program, &log_size) != NVRTC_SUCCESS) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_COMPILE_FAILED);
    }
    if (log_size_ret != NULL) {
        *log_size_ret = log_size;
    }
    if (log_buffer == NULL) {
        return SECANT_PTX_SUCCESS;
    }
    if (log_buffer_size < log_size) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INSUFFICIENT_BUFFER);
    }
    if (log_size == 0u) {
        if (log_buffer_size != 0u) {
            log_buffer[0] = '\0';
        }
        return SECANT_PTX_SUCCESS;
    }
    if (nvrtcGetProgramLog(program, log_buffer) != NVRTC_SUCCESS) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_COMPILE_FAILED);
    }
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_capture_nvptx_log(
    nvPTXCompilerHandle compiler,
    bool compile_succeeded,
    bool verbose,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret
) {
    size_t error_size = 0u;
    size_t info_size = 0u;
    size_t log_size;
    nvPTXCompileResult nvptx_result;

    if (log_size_ret != NULL) {
        *log_size_ret = 0u;
    }
    if (compile_succeeded && !verbose) {
        if (log_buffer != NULL && log_buffer_size != 0u) {
            log_buffer[0] = '\0';
        }
        return SECANT_PTX_SUCCESS;
    }
    if (nvPTXCompilerGetErrorLogSize(compiler, &error_size) != NVPTXCOMPILE_SUCCESS ||
        nvPTXCompilerGetInfoLogSize(compiler, &info_size) != NVPTXCOMPILE_SUCCESS) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_COMPILE_FAILED);
    }
    log_size = error_size > 1u ? error_size : info_size;
    if (log_size_ret != NULL) {
        *log_size_ret = log_size;
    }
    if (log_buffer == NULL) {
        return SECANT_PTX_SUCCESS;
    }
    if (log_buffer_size < log_size) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INSUFFICIENT_BUFFER);
    }
    if (log_size <= 1u) {
        if (log_buffer_size != 0u) {
            log_buffer[0] = '\0';
        }
        return SECANT_PTX_SUCCESS;
    }
    nvptx_result = error_size > 1u
        ? nvPTXCompilerGetErrorLog(compiler, log_buffer)
        : nvPTXCompilerGetInfoLog(compiler, log_buffer);
    if (nvptx_result != NVPTXCOMPILE_SUCCESS) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_COMPILE_FAILED);
    }
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_generate_template_source(
    SecantPTXHandle handle,
    char* buffer,
    size_t buffer_size,
    size_t* source_size_ret
) {
    switch (handle->kernel_shape) {
        case SECANT_PTX_KERNEL_SHAPE_STATIC_COLUMN_MATERIALIZE:
            _SECANT_PTX_ERROR_RET(secant_ptx_materialize_source_generate(
                handle->num_kernels,
                handle->asts_per_kernel,
                handle->num_inputs,
                buffer,
                buffer_size,
                source_size_ret));
        case SECANT_PTX_KERNEL_SHAPE_STATIC_COLUMN_SSE:
            _SECANT_PTX_ERROR_RET(secant_ptx_sse_source_generate(
                handle->num_kernels,
                handle->asts_per_kernel,
                handle->num_inputs,
                handle->num_targets,
                handle->tile_rows,
                handle->threads_per_block,
                handle->reduction_mode,
                buffer,
                buffer_size,
                source_size_ret));
        case SECANT_PTX_KERNEL_SHAPE_DYNAMIC_CONSTANT_SSE:
            _SECANT_PTX_ERROR_RET(
                secant_ptx_dynamic_constant_sse_source_generate(
                    handle->num_kernels,
                    handle->asts_per_kernel,
                    handle->num_inputs -
                        handle->num_input_constants,
                    handle->num_input_constants,
                    handle->num_targets,
                    handle->tile_rows,
                    handle->threads_per_block,
                    handle->reduction_mode,
                    buffer,
                    buffer_size,
                    source_size_ret));
        default:
            _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_UNSUPPORTED_SHAPE);
    }
}

static SecantPTXResult
_secant_ptx_compile_template(
    SecantPTXHandle handle,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    bool verbose,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret
) {
    char* source = NULL;
    const char** options = NULL;
    nvrtcProgram program = NULL;
    char architecture[64];
    size_t source_size = 0u;
    size_t rendered_size = 0u;
    size_t ptx_size = 0u;
    size_t option_idx;
    int architecture_bytes;
    nvrtcResult nvrtc_result;
    SecantPTXResult result;

    result = _secant_ptx_generate_template_source(handle, NULL, 0u, &source_size);
    if (result != SECANT_PTX_SUCCESS) {
        _SECANT_PTX_ERROR_RET(result);
    }
    source = (char*)malloc(source_size);
    if (source == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_ALLOCATION_FAILED);
    }
    result = _secant_ptx_generate_template_source(
        handle, source, source_size, &rendered_size);
    if (result != SECANT_PTX_SUCCESS || rendered_size != source_size) {
        free(source);
        _SECANT_PTX_ERROR_RET(
            result != SECANT_PTX_SUCCESS ? result : SECANT_PTX_ERROR_INVALID_STATE);
    }

    if (num_nvrtc_options > (size_t)INT_MAX - 2u ||
        num_nvrtc_options + 2u > SIZE_MAX / sizeof(*options)) {
        free(source);
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_OVERFLOW);
    }
    options = (const char**)malloc((num_nvrtc_options + 2u) * sizeof(*options));
    if (options == NULL) {
        free(source);
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_ALLOCATION_FAILED);
    }
    architecture_bytes = snprintf(
        architecture,
        sizeof(architecture),
        "--gpu-architecture=compute_%u%u",
        compute_capability_major,
        compute_capability_minor);
    if (architecture_bytes < 0 || (size_t)architecture_bytes >= sizeof(architecture)) {
        free(options);
        free(source);
        _SECANT_PTX_ERROR_RET(
            architecture_bytes < 0 ? SECANT_PTX_ERROR_FORMAT : SECANT_PTX_ERROR_OVERFLOW);
    }
    options[0] = "--std=c++11";
    options[1] = architecture;
    for (option_idx = 0u; option_idx < num_nvrtc_options; ++option_idx) {
        if (nvrtc_options[option_idx] == NULL) {
            free(options);
            free(source);
            _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
        }
        options[option_idx + 2u] = nvrtc_options[option_idx];
    }

    nvrtc_result = nvrtcCreateProgram(
        &program, source, "secant_ptx_template_get.cu", 0, NULL, NULL);
    if (nvrtc_result != NVRTC_SUCCESS) {
        result = SECANT_PTX_ERROR_COMPILE_FAILED;
    } else {
        nvrtc_result = nvrtcCompileProgram(
            program, (int)(num_nvrtc_options + 2u), options);
        result = _secant_ptx_capture_nvrtc_log(
            program,
            nvrtc_result == NVRTC_SUCCESS,
            verbose,
            log_buffer,
            log_buffer_size,
            log_size_ret);
        if (result == SECANT_PTX_SUCCESS && nvrtc_result != NVRTC_SUCCESS) {
            result = SECANT_PTX_ERROR_COMPILE_FAILED;
        }
    }
    if (result == SECANT_PTX_SUCCESS &&
        (nvrtcGetPTXSize(program, &ptx_size) != NVRTC_SUCCESS || ptx_size == 0u)) {
        result = SECANT_PTX_ERROR_COMPILE_FAILED;
    }
    if (result == SECANT_PTX_SUCCESS) {
        handle->template_ptx = (char*)malloc(ptx_size);
        if (handle->template_ptx == NULL) {
            result = SECANT_PTX_ERROR_ALLOCATION_FAILED;
        } else {
            handle->template_ptx_size = ptx_size;
            if (nvrtcGetPTX(program, handle->template_ptx) != NVRTC_SUCCESS) {
                result = SECANT_PTX_ERROR_COMPILE_FAILED;
            }
        }
    }

    if (program != NULL) {
        nvrtcDestroyProgram(&program);
    }
    free(options);
    free(source);
    _SECANT_PTX_ERROR_RET(result);
}

static SecantPTXResult
_secant_ptx_prepare_sites(SecantPTXHandle handle) {
    PtxInjectResult inject_result;
    size_t num_injects = 0u;
    size_t site_idx;

    inject_result = ptx_inject_create(&handle->inject, handle->template_ptx);
    if (inject_result != PTX_INJECT_SUCCESS) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_COMPILE_FAILED);
    }
    inject_result = ptx_inject_num_injects(handle->inject, &num_injects);
    if (inject_result != PTX_INJECT_SUCCESS || num_injects != handle->num_sites) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_STATE);
    }
    if (handle->num_sites > SIZE_MAX / sizeof(*handle->sites) ||
        handle->num_inputs > SIZE_MAX / handle->num_sites ||
        handle->num_inputs * handle->num_sites >
            SIZE_MAX / sizeof(*handle->input_register_names)) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_OVERFLOW);
    }

    handle->sites = (SecantPTXSite*)calloc(handle->num_sites, sizeof(*handle->sites));
    handle->input_register_names = (const char**)calloc(
        handle->num_sites * handle->num_inputs,
        sizeof(*handle->input_register_names));
    if (handle->sites == NULL || handle->input_register_names == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_ALLOCATION_FAILED);
    }

    for (site_idx = 0u; site_idx < handle->num_sites; ++site_idx) {
        SecantPTXSite* site = handle->sites + site_idx;
        const size_t kernel_idx = site_idx / handle->asts_per_kernel;
        const size_t ast_idx = site_idx % handle->asts_per_kernel;
        size_t num_args = 0u;
        size_t num_sites = 0u;
        size_t input_idx;
        char site_name[64];
        int name_bytes;
        PtxInjectMutType mut_type;
        const char* register_type;

        name_bytes = snprintf(
            site_name,
            sizeof(site_name),
            "secant_expr_%03zu_%03zu",
            kernel_idx,
            ast_idx);
        if (name_bytes < 0 || (size_t)name_bytes >= sizeof(site_name)) {
            _SECANT_PTX_ERROR_RET(
                name_bytes < 0 ? SECANT_PTX_ERROR_FORMAT : SECANT_PTX_ERROR_OVERFLOW);
        }
        inject_result = ptx_inject_inject_info_by_name(
            handle->inject,
            site_name,
            &site->inject_idx,
            &num_args,
            &num_sites);
        if (inject_result != PTX_INJECT_SUCCESS ||
            num_args != handle->num_inputs + 1u || num_sites == 0u) {
            _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_STATE);
        }

        inject_result = ptx_inject_variable_info_by_name(
            handle->inject,
            site->inject_idx,
            "result",
            NULL,
            &site->output_register_name,
            &mut_type,
            &register_type,
            NULL);
        if (inject_result != PTX_INJECT_SUCCESS ||
            mut_type != PTX_INJECT_MUT_TYPE_OUT ||
            strcmp(register_type, "f32") != 0) {
            _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_STATE);
        }

        site->input_register_names =
            handle->input_register_names + site_idx * handle->num_inputs;
        for (input_idx = 0u; input_idx < handle->num_inputs; ++input_idx) {
            char input_name[64];

            name_bytes = snprintf(input_name, sizeof(input_name), "input%zu", input_idx);
            if (name_bytes < 0 || (size_t)name_bytes >= sizeof(input_name)) {
                _SECANT_PTX_ERROR_RET(
                    name_bytes < 0 ? SECANT_PTX_ERROR_FORMAT : SECANT_PTX_ERROR_OVERFLOW);
            }
            inject_result = ptx_inject_variable_info_by_name(
                handle->inject,
                site->inject_idx,
                input_name,
                NULL,
                site->input_register_names + input_idx,
                &mut_type,
                &register_type,
                NULL);
            if (inject_result != PTX_INJECT_SUCCESS ||
                mut_type != PTX_INJECT_MUT_TYPE_IN ||
                strcmp(register_type, "f32") != 0) {
                _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_STATE);
            }
        }
    }
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_create(
    SecantPTXKernelShape kernel_shape,
    size_t num_kernels,
    size_t asts_per_kernel,
    size_t num_inputs,
    size_t num_input_constants,
    size_t num_targets,
    size_t tile_rows,
    size_t threads_per_block,
    SecantSSEReductionMode reduction_mode,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvrtc_options,
    size_t num_nvrtc_options,
    bool verbose,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    SecantPTXHandle* handle_ret
) {
    SecantPTXHandle handle;
    SecantPTXResult result;

    if (num_kernels == 0u || asts_per_kernel == 0u || num_inputs == 0u ||
        compute_capability_major == 0u || handle_ret == NULL ||
        (reduction_mode != SECANT_SSE_REDUCTION_MODE_ATOMIC &&
         reduction_mode != SECANT_SSE_REDUCTION_MODE_WORKSPACE) ||
        (num_nvrtc_options != 0u && nvrtc_options == NULL) ||
        num_kernels > SIZE_MAX / asts_per_kernel ||
        num_kernels * asts_per_kernel > SECANT_PTX_MAX_SITES) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    *handle_ret = NULL;
    handle = (SecantPTXHandle)calloc(1u, sizeof(*handle));
    if (handle == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_ALLOCATION_FAILED);
    }
    handle->kernel_shape = kernel_shape;
    handle->num_kernels = num_kernels;
    handle->asts_per_kernel = asts_per_kernel;
    handle->num_inputs = num_inputs;
    handle->num_input_constants = num_input_constants;
    handle->num_targets = num_targets;
    handle->tile_rows = tile_rows;
    handle->threads_per_block = threads_per_block;
    handle->reduction_mode = reduction_mode;
    handle->num_sites = num_kernels * asts_per_kernel;

    result = _secant_ptx_compile_template(
        handle,
        compute_capability_major,
        compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        verbose,
        log_buffer,
        log_buffer_size,
        log_size_ret);
    if (result == SECANT_PTX_SUCCESS) {
        result = _secant_ptx_prepare_sites(handle);
    }
    if (result != SECANT_PTX_SUCCESS) {
        (void)secant_ptx_handle_destroy(handle);
        _SECANT_PTX_ERROR_RET(result);
    }
    *handle_ret = handle;
    return SECANT_PTX_SUCCESS;
}

const char*
secant_ptx_result_to_string(SecantPTXResult result) {
    static const char* const strings[SECANT_PTX_RESULT_NUM_ENUMS] = {
        "SECANT_PTX_SUCCESS",
        "SECANT_PTX_ERROR_INVALID_VALUE",
        "SECANT_PTX_ERROR_OVERFLOW",
        "SECANT_PTX_ERROR_INSUFFICIENT_BUFFER",
        "SECANT_PTX_ERROR_FORMAT",
        "SECANT_PTX_ERROR_BAD_PROGRAM",
        "SECANT_PTX_ERROR_STACK_OVERFLOW",
        "SECANT_PTX_ERROR_STACK_UNDERFLOW",
        "SECANT_PTX_ERROR_TOO_MANY_ARGS",
        "SECANT_PTX_ERROR_UNSUPPORTED_OP",
        "SECANT_PTX_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS",
        "SECANT_PTX_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS",
        "SECANT_PTX_ERROR_ROUTINE_DEPTH_EXCEEDED",
        "SECANT_PTX_ERROR_ALLOCATION_FAILED",
        "SECANT_PTX_ERROR_COMPILE_FAILED",
        "SECANT_PTX_ERROR_INVALID_STATE",
        "SECANT_PTX_ERROR_UNSUPPORTED_SHAPE"
    };

    return result < SECANT_PTX_RESULT_NUM_ENUMS ? strings[result] : "SECANT_PTX_ERROR_UNKNOWN";
}

SecantPTXResult
secant_ptx_materialize_create(
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
) {
    _SECANT_PTX_ERROR_RET(_secant_ptx_create(
        SECANT_PTX_KERNEL_SHAPE_STATIC_COLUMN_MATERIALIZE,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        0u,
        0u,
        0u,
        0u,
        SECANT_SSE_REDUCTION_MODE_ATOMIC,
        nvrtc_compute_capability_major,
        nvrtc_compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        verbose,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        handle_ret));
}

SecantPTXResult
secant_ptx_sse_create(
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
) {
    if (num_targets == 0u || tile_rows == 0u || threads_per_block < 32u ||
        threads_per_block > tile_rows ||
        threads_per_block > SECANT_PTX_SSE_MAX_WARPS * 32u ||
        threads_per_block % 32u != 0u) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    _SECANT_PTX_ERROR_RET(_secant_ptx_create(
        SECANT_PTX_KERNEL_SHAPE_STATIC_COLUMN_SSE,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        0u,
        num_targets,
        tile_rows,
        threads_per_block,
        reduction_mode,
        nvrtc_compute_capability_major,
        nvrtc_compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        verbose,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        handle_ret));
}

SecantPTXResult
secant_ptx_dynamic_constant_sse_create(
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
) {
    size_t num_inputs;

    if (num_input_columns == 0u ||
        num_input_columns >
            SECANT_PTX_DYNAMIC_CONSTANT_SSE_MAX_INPUT_COLUMNS ||
        num_input_constants == 0u ||
        num_input_constants >
            SECANT_PTX_DYNAMIC_CONSTANT_SSE_MAX_INPUT_CONSTANTS ||
        num_targets == 0u || tile_rows == 0u ||
        threads_per_block == 0u || threads_per_block > 1024u ||
        num_input_constants > SIZE_MAX - num_input_columns) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    num_inputs = num_input_columns + num_input_constants;
    _SECANT_PTX_ERROR_RET(_secant_ptx_create(
        SECANT_PTX_KERNEL_SHAPE_DYNAMIC_CONSTANT_SSE,
        num_kernels,
        asts_per_kernel,
        num_inputs,
        num_input_constants,
        num_targets,
        tile_rows,
        threads_per_block,
        reduction_mode,
        nvrtc_compute_capability_major,
        nvrtc_compute_capability_minor,
        nvrtc_options,
        num_nvrtc_options,
        verbose,
        log_buffer,
        log_buffer_size,
        log_size_ret,
        handle_ret));
}

SecantPTXResult
secant_ptx_handle_destroy(SecantPTXHandle handle) {
    SecantPTXResult result = SECANT_PTX_SUCCESS;

    if (handle == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    if (handle->inject != NULL &&
        ptx_inject_destroy(handle->inject) != PTX_INJECT_SUCCESS) {
        result = SECANT_PTX_ERROR_INVALID_STATE;
    }
    free(handle->input_register_names);
    free(handle->sites);
    free(handle->template_ptx);
    free(handle);
    _SECANT_PTX_ERROR_RET(result);
}

SecantPTXResult
secant_ptx_template_get(
    SecantPTXHandle handle,
    const char** template_ptx_ret,
    size_t* template_ptx_size_ret
) {
    if (handle == NULL || template_ptx_ret == NULL || template_ptx_size_ret == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    *template_ptx_ret = handle->template_ptx;
    *template_ptx_size_ret = handle->template_ptx_size;
    return SECANT_PTX_SUCCESS;
}

SecantPTXResult
secant_ptx_materialize_info_get(
    SecantPTXHandle handle,
    size_t* num_kernels_ret,
    size_t* asts_per_kernel_ret,
    size_t* num_inputs_ret
) {
    if (handle == NULL ||
        handle->kernel_shape !=
            SECANT_PTX_KERNEL_SHAPE_STATIC_COLUMN_MATERIALIZE ||
        num_kernels_ret == NULL || asts_per_kernel_ret == NULL ||
        num_inputs_ret == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    *num_kernels_ret = handle->num_kernels;
    *asts_per_kernel_ret = handle->asts_per_kernel;
    *num_inputs_ret = handle->num_inputs;
    return SECANT_PTX_SUCCESS;
}

SecantPTXResult
secant_ptx_sse_info_get(
    SecantPTXHandle handle,
    size_t* num_kernels_ret,
    size_t* asts_per_kernel_ret,
    size_t* num_inputs_ret,
    size_t* num_targets_ret,
    size_t* tile_rows_ret,
    size_t* threads_per_block_ret
) {
    if (handle == NULL ||
        handle->kernel_shape !=
            SECANT_PTX_KERNEL_SHAPE_STATIC_COLUMN_SSE ||
        num_kernels_ret == NULL || asts_per_kernel_ret == NULL ||
        num_inputs_ret == NULL || num_targets_ret == NULL ||
        tile_rows_ret == NULL || threads_per_block_ret == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    *num_kernels_ret = handle->num_kernels;
    *asts_per_kernel_ret = handle->asts_per_kernel;
    *num_inputs_ret = handle->num_inputs;
    *num_targets_ret = handle->num_targets;
    *tile_rows_ret = handle->tile_rows;
    *threads_per_block_ret = handle->threads_per_block;
    return SECANT_PTX_SUCCESS;
}

SecantPTXResult
secant_ptx_dynamic_constant_sse_info_get(
    SecantPTXHandle handle,
    size_t* num_kernels_ret,
    size_t* asts_per_kernel_ret,
    size_t* num_input_columns_ret,
    size_t* num_input_constants_ret,
    size_t* num_targets_ret,
    size_t* tile_rows_ret,
    size_t* threads_per_block_ret
) {
    if (handle == NULL ||
        handle->kernel_shape !=
            SECANT_PTX_KERNEL_SHAPE_DYNAMIC_CONSTANT_SSE ||
        num_kernels_ret == NULL || asts_per_kernel_ret == NULL ||
        num_input_columns_ret == NULL ||
        num_input_constants_ret == NULL ||
        num_targets_ret == NULL || tile_rows_ret == NULL ||
        threads_per_block_ret == NULL ||
        handle->num_input_constants > handle->num_inputs) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    *num_kernels_ret = handle->num_kernels;
    *asts_per_kernel_ret = handle->asts_per_kernel;
    *num_input_columns_ret =
        handle->num_inputs - handle->num_input_constants;
    *num_input_constants_ret = handle->num_input_constants;
    *num_targets_ret = handle->num_targets;
    *tile_rows_ret = handle->tile_rows;
    *threads_per_block_ret = handle->threads_per_block;
    return SECANT_PTX_SUCCESS;
}

/* Postorder SECANT AST lowering and PTX Inject rendering. */
static SecantPTXResult
_secant_ptx_map_instruction(SecantAstInstruction instruction, AstPtxInstruction* mapped_ret) {
    AstPtxInstruction mapped;
    AstPtxPtxInstruction ptx_instruction = AST_PTX_PTX_INSTRUCTION_NUM_ENUMS;

    memset(&mapped, 0, sizeof(mapped));
    switch ((SecantAstInstructionType)instruction.instruction_type) {
        case SECANT_AST_INSTRUCTION_TYPE_INPUT_F32:
            mapped = ast_ptx_encode_input(instruction.payload.idx);
            break;
        case SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32:
            mapped = ast_ptx_encode_constant(
                secant_ast_constant_f32_get(&instruction));
            break;
        case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32:
            mapped = ast_ptx_encode_routine(instruction.payload.idx, instruction.aux);
            break;
        case SECANT_AST_INSTRUCTION_TYPE_RETURN_F32:
            mapped = ast_ptx_encode_return;
            break;
        case SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32:
            mapped = ast_ptx_encode_routine_arg(instruction.payload.idx);
            break;
        case SECANT_AST_INSTRUCTION_TYPE_ADD_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_ADD_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_SUB_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_SUB_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_MUL_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_MUL_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_DIV_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_DIV_APPROX_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_NEG_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_NEG_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_SQRT_APPROX_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_RCP_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_RCP_APPROX_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_ABS_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_ABS_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_MIN_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_MIN_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_MAX_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_MAX_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_FMA_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_FMA_RN_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_SIN_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_SIN_APPROX_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_COS_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_COS_APPROX_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_EX2_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_EX2_APPROX_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_LG2_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_LG2_APPROX_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_RSQRT_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_RSQRT_APPROX_FTZ_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_TANH_F32: ptx_instruction = AST_PTX_PTX_INSTRUCTION_TANH_APPROX_F32; break;
        case SECANT_AST_INSTRUCTION_TYPE_NONE:
        case SECANT_AST_INSTRUCTION_TYPE_NUM_ENUMS:
        default:
            _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_UNSUPPORTED_OP);
    }

    if (ptx_instruction != AST_PTX_PTX_INSTRUCTION_NUM_ENUMS) {
        mapped = ast_ptx_encode_ptx_instruction(
            ptx_instruction,
            ast_ptx_ptx_instruction_num_args[ptx_instruction]);
    }
    *mapped_ret = mapped;
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_routine_num_args(
    const SecantAstInstruction* instructions,
    size_t* num_args_ret
) {
    size_t instruction_idx;
    size_t num_args = 0u;

    if (instructions == NULL || num_args_ret == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    for (instruction_idx = 0u;
         instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_idx) {
        const SecantAstInstruction* instruction =
            instructions + instruction_idx;
        const SecantAstInstructionType instruction_type =
            (SecantAstInstructionType)instruction->instruction_type;

        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_ARG_F32) {
            if (instruction->payload.idx >=
                SECANT_AST_MAX_INSTRUCTION_ARGS) {
                _SECANT_PTX_ERROR_RET(
                    SECANT_PTX_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS);
            }
            if ((size_t)instruction->payload.idx + 1u > num_args) {
                num_args = (size_t)instruction->payload.idx + 1u;
            }
        } else if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            *num_args_ret = num_args;
            return SECANT_PTX_SUCCESS;
        }
    }
    _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_BAD_PROGRAM);
}

static SecantPTXResult
_secant_ptx_routine_calls_validate(
    const SecantAstInstruction* instructions,
    const SecantAstInstruction* const* routines,
    size_t num_routines
) {
    size_t instruction_idx;

    if (instructions == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    for (instruction_idx = 0u;
         instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS;
         ++instruction_idx) {
        const SecantAstInstruction* instruction =
            instructions + instruction_idx;
        const SecantAstInstructionType instruction_type =
            (SecantAstInstructionType)instruction->instruction_type;

        if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_ROUTINE_F32) {
            size_t expected_args;

            if (routines == NULL ||
                instruction->payload.idx >= num_routines ||
                routines[instruction->payload.idx] == NULL) {
                _SECANT_PTX_ERROR_RET(
                    SECANT_PTX_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
            }
            _SECANT_PTX_CHECK_RET(_secant_ptx_routine_num_args(
                routines[instruction->payload.idx],
                &expected_args));
            if ((size_t)instruction->aux != expected_args) {
                _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
            }
        } else if (instruction_type == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
            return SECANT_PTX_SUCCESS;
        }
    }
    _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_BAD_PROGRAM);
}

static SecantPTXResult
_secant_ptx_lower_programs(
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const SecantAstInstruction* const* asts,
    size_t num_asts,
    const AstPtxInstruction*** ptx_routines_ret,
    const AstPtxInstruction*** ptx_asts_ret,
    void* compile_buffer,
    size_t compile_buffer_size,
    size_t* compile_buffer_offset,
    size_t compile_buffer_limit
) {
    const size_t num_programs = num_routines + num_asts;
    const SecantAstInstruction* const* programs[2] = { routines, asts };
    const size_t program_counts[2] = { num_routines, num_asts };
    const AstPtxInstruction** pointers;
    size_t group_idx;
    size_t pointer_idx = 0u;

    if (num_programs < num_routines || num_programs > SIZE_MAX / sizeof(*pointers)) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_OVERFLOW);
    }
    _SECANT_PTX_CHECK_RET(_secant_ptx_scratch_allocate(
        compile_buffer,
        compile_buffer_size,
        compile_buffer_offset,
        compile_buffer_limit,
        num_programs * sizeof(*pointers),
        sizeof(void*),
        (void**)&pointers));

    for (group_idx = 0u; group_idx < 2u; ++group_idx) {
        size_t program_idx;

        for (program_idx = 0u; program_idx < program_counts[group_idx]; ++program_idx) {
            size_t instruction_idx;
            AstPtxInstruction* first_instruction = NULL;

            if (programs[group_idx][program_idx] == NULL) {
                _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
            }
            _SECANT_PTX_CHECK_RET(_secant_ptx_routine_calls_validate(
                programs[group_idx][program_idx],
                routines,
                num_routines));
            for (instruction_idx = 0u; instruction_idx < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS; ++instruction_idx) {
                AstPtxInstruction* mapped;

                if (*compile_buffer_offset > compile_buffer_limit ||
                    sizeof(*mapped) > compile_buffer_limit - *compile_buffer_offset) {
                    _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INSUFFICIENT_BUFFER);
                }
                mapped = (AstPtxInstruction*)((unsigned char*)compile_buffer + *compile_buffer_offset);
                *compile_buffer_offset += sizeof(*mapped);
                if (first_instruction == NULL) {
                    first_instruction = mapped;
                }
                _SECANT_PTX_CHECK_RET(_secant_ptx_map_instruction(
                    programs[group_idx][program_idx][instruction_idx],
                    mapped));
                if ((SecantAstInstructionType)programs[group_idx][program_idx][instruction_idx].instruction_type ==
                    SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
                    break;
                }
            }
            if (instruction_idx == SECANT_AST_MAX_PROGRAM_INSTRUCTIONS) {
                _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_BAD_PROGRAM);
            }
            pointers[pointer_idx++] = first_instruction;
        }
    }

    *ptx_routines_ret = pointers;
    *ptx_asts_ret = pointers + num_routines;
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_map_ast_result(AstPtxResult result) {
    switch (result) {
        case AST_PTX_SUCCESS: return SECANT_PTX_SUCCESS;
        case AST_PTX_ERROR_INSUFFICIENT_BUFFER: _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INSUFFICIENT_BUFFER);
        case AST_PTX_ERROR_INVALID_VALUE: _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
        case AST_PTX_ERROR_BAD_INSTRUCTION_MAYBE_FORGOT_RETURN: _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_BAD_PROGRAM);
        case AST_PTX_ERROR_INPUT_IDX_OUT_OF_BOUNDS: _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
        case AST_PTX_ERROR_STACK_OVERFLOW: _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_STACK_OVERFLOW);
        case AST_PTX_ERROR_STACK_UNDERFLOW: _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_STACK_UNDERFLOW);
        case AST_PTX_ERROR_TOO_MANY_ARGS: _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_TOO_MANY_ARGS);
        case AST_PTX_ERROR_UNSUPPORTED_INSTRUCTION: _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_UNSUPPORTED_OP);
        case AST_PTX_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS: _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_ROUTINE_IDX_OUT_OF_BOUNDS);
        case AST_PTX_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS: _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_ROUTINE_ARG_IDX_OUT_OF_BOUNDS);
        case AST_PTX_ERROR_ROUTINE_DEPTH_EXCEEDED: _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_ROUTINE_DEPTH_EXCEEDED);
        case AST_PTX_ERROR_INTERNAL:
        case AST_PTX_ERROR_PTX_INSTRUCTION_IDX_OUT_OF_BOUNDS:
        case AST_PTX_RESULT_NUM_ENUMS:
        default:
            _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_STATE);
    }
}

static SecantPTXResult
_secant_ptx_render_asts(
    SecantPTXHandle handle,
    const AstPtxInstruction* const* ptx_routines,
    size_t num_routines,
    const AstPtxInstruction* const* ptx_asts,
    void* compile_buffer,
    size_t compile_buffer_size,
    size_t* compile_buffer_offset,
    size_t compile_buffer_limit,
    const char** rendered_ptx_ret,
    size_t* rendered_ptx_size_ret
) {
    const SecantPTXHandle backend = handle;
    const char** stubs;
    size_t site_idx;
    size_t rendered_size = 0u;
    PtxInjectResult inject_result;

    if (backend->num_sites > SIZE_MAX / sizeof(*stubs)) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_OVERFLOW);
    }
    _SECANT_PTX_CHECK_RET(_secant_ptx_scratch_allocate(
        compile_buffer,
        compile_buffer_size,
        compile_buffer_offset,
        compile_buffer_limit,
        backend->num_sites * sizeof(*stubs),
        sizeof(void*),
        (void**)&stubs));

    for (site_idx = 0u; site_idx < backend->num_sites; ++site_idx) {
        const SecantPTXSite* site = backend->sites + site_idx;
        size_t written = 0u;
        size_t capacity;
        char* stub;
        AstPtxResult ast_result;

        if (*compile_buffer_offset > compile_buffer_limit) {
            _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_STATE);
        }
        stub = (char*)compile_buffer + *compile_buffer_offset;
        capacity = compile_buffer_limit - *compile_buffer_offset;
        stubs[site->inject_idx] = stub;
        ast_result = ast_ptx_compile(
            site->output_register_name,
            site->input_register_names,
            backend->num_inputs,
            ast_ptx_ptx_instruction_names,
            ast_ptx_ptx_instruction_num_args,
            AST_PTX_PTX_INSTRUCTION_NUM_ENUMS,
            ptx_routines,
            num_routines,
            ptx_asts[site_idx],
            stub,
            capacity,
            &written);
        _SECANT_PTX_CHECK_RET(_secant_ptx_map_ast_result(ast_result));
        if (written == SIZE_MAX || written + 1u > capacity) {
            _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INSUFFICIENT_BUFFER);
        }
        *compile_buffer_offset += written + 1u;
    }

    if (*compile_buffer_offset > compile_buffer_limit) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_STATE);
    }
    *rendered_ptx_ret = (const char*)compile_buffer + *compile_buffer_offset;
    inject_result = ptx_inject_render_ptx(
        backend->inject,
        stubs,
        backend->num_sites,
        (char*)*rendered_ptx_ret,
        compile_buffer_limit - *compile_buffer_offset,
        &rendered_size);
    if (inject_result == PTX_INJECT_ERROR_INSUFFICIENT_BUFFER) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INSUFFICIENT_BUFFER);
    }
    if (inject_result != PTX_INJECT_SUCCESS || rendered_size == SIZE_MAX) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_COMPILE_FAILED);
    }
    if (rendered_size + 1u > compile_buffer_limit - *compile_buffer_offset) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INSUFFICIENT_BUFFER);
    }
    *compile_buffer_offset += rendered_size + 1u;
    *rendered_ptx_size_ret = rendered_size;
    return SECANT_PTX_SUCCESS;
}

static SecantPTXResult
_secant_ptx_allocate_compiled(
    size_t binary_size,
    SecantPTXCompiled* compiled_ret
) {
    SecantPTXCompiled compiled;

    if (binary_size == 0u || compiled_ret == NULL ||
        binary_size > SIZE_MAX - sizeof(*compiled)) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    compiled = (SecantPTXCompiled)malloc(sizeof(*compiled) + binary_size);
    if (compiled == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_ALLOCATION_FAILED);
    }
    compiled->binary_size = binary_size;
    *compiled_ret = compiled;
    return SECANT_PTX_SUCCESS;
}

static void
_secant_ptx_compiled_release(SecantPTXCompiled compiled) {
    free(compiled);
}

static SecantPTXResult
_secant_ptx_compile_cubin(
    const char* rendered_ptx,
    size_t rendered_ptx_size,
    uint32_t compute_capability_major,
    uint32_t compute_capability_minor,
    const char* const* nvptx_options,
    size_t num_nvptx_options,
    bool verbose,
    void* compile_scratch,
    size_t compile_scratch_size,
    size_t* compile_scratch_offset,
    char* log_buffer,
    size_t log_buffer_size,
    size_t* log_size_ret,
    nvPTXCompilerHandle* compiler_ret,
    SecantPTXCompiled* compiled_ret
) {
    const char** options;
    SecantPTXCompiled compiled = NULL;
    char architecture[64];
    int architecture_bytes;
    size_t option_idx;
    size_t cubin_size = 0u;
    nvPTXCompileResult nvptx_result;
    SecantPTXResult result;

    if (num_nvptx_options > (size_t)INT_MAX - 1u ||
        num_nvptx_options + 1u > SIZE_MAX / sizeof(*options)) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_OVERFLOW);
    }
    architecture_bytes = snprintf(
        architecture,
        sizeof(architecture),
        "--gpu-name=sm_%u%u",
        compute_capability_major,
        compute_capability_minor);
    if (architecture_bytes < 0 || (size_t)architecture_bytes >= sizeof(architecture)) {
        _SECANT_PTX_ERROR_RET(
            architecture_bytes < 0 ? SECANT_PTX_ERROR_FORMAT : SECANT_PTX_ERROR_OVERFLOW);
    }
    _SECANT_PTX_CHECK_RET(_secant_ptx_scratch_allocate(
        compile_scratch,
        compile_scratch_size,
        compile_scratch_offset,
        compile_scratch_size,
        (num_nvptx_options + 1u) * sizeof(*options),
        sizeof(void*),
        (void**)&options));
    options[0] = architecture;
    for (option_idx = 0u; option_idx < num_nvptx_options; ++option_idx) {
        if (nvptx_options[option_idx] == NULL) {
            _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
        }
        options[option_idx + 1u] = nvptx_options[option_idx];
    }

    nvptx_result = nvPTXCompilerCreate(compiler_ret, rendered_ptx_size, rendered_ptx);
    if (nvptx_result != NVPTXCOMPILE_SUCCESS) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_COMPILE_FAILED);
    }
    nvptx_result = nvPTXCompilerCompile(
        *compiler_ret, (int)(num_nvptx_options + 1u), options);
    result = _secant_ptx_capture_nvptx_log(
        *compiler_ret,
        nvptx_result == NVPTXCOMPILE_SUCCESS,
        verbose,
        log_buffer,
        log_buffer_size,
        log_size_ret);
    if (result != SECANT_PTX_SUCCESS || nvptx_result != NVPTXCOMPILE_SUCCESS) {
        _SECANT_PTX_ERROR_RET(
            result != SECANT_PTX_SUCCESS ? result : SECANT_PTX_ERROR_COMPILE_FAILED);
    }
    nvptx_result = nvPTXCompilerGetCompiledProgramSize(*compiler_ret, &cubin_size);
    if (nvptx_result != NVPTXCOMPILE_SUCCESS || cubin_size == 0u) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_COMPILE_FAILED);
    }
    _SECANT_PTX_CHECK_RET(
        _secant_ptx_allocate_compiled(cubin_size, &compiled));
    if (nvPTXCompilerGetCompiledProgram(*compiler_ret, compiled->binary) !=
        NVPTXCOMPILE_SUCCESS) {
        _secant_ptx_compiled_release(compiled);
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_COMPILE_FAILED);
    }
    *compiled_ret = compiled;
    return SECANT_PTX_SUCCESS;
}

SecantPTXResult
secant_ptx_compile(
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
) {
    const AstPtxInstruction** ptx_routines;
    const AstPtxInstruction** ptx_asts;
    const char* rendered_ptx;
    size_t rendered_ptx_size = 0u;
    size_t compile_scratch_offset = 0u;
    nvPTXCompilerHandle compiler = NULL;
    SecantPTXCompiled compiled = NULL;
    SecantPTXResult result;

    if (handle == NULL || asts == NULL || nvptx_compute_capability_major == 0u ||
        (num_routines != 0u && routines == NULL) ||
        (num_nvptx_options != 0u && nvptx_options == NULL) ||
        compile_scratch == NULL || compile_scratch_size == 0u ||
        compiled_ret == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    *compiled_ret = NULL;
    if (log_size_ret != NULL) {
        *log_size_ret = 0u;
    }

    result = _secant_ptx_lower_programs(
        routines,
        num_routines,
        asts,
        handle->num_sites,
        &ptx_routines,
        &ptx_asts,
        compile_scratch,
        compile_scratch_size,
        &compile_scratch_offset,
        compile_scratch_size);
    if (result == SECANT_PTX_SUCCESS) {
        result = _secant_ptx_render_asts(
            handle,
            ptx_routines,
            num_routines,
            ptx_asts,
            compile_scratch,
            compile_scratch_size,
            &compile_scratch_offset,
            compile_scratch_size,
            &rendered_ptx,
            &rendered_ptx_size);
    }
    if (result == SECANT_PTX_SUCCESS) {
        result = _secant_ptx_compile_cubin(
            rendered_ptx,
            rendered_ptx_size,
            nvptx_compute_capability_major,
            nvptx_compute_capability_minor,
            nvptx_options,
            num_nvptx_options,
            verbose,
            compile_scratch,
            compile_scratch_size,
            &compile_scratch_offset,
            log_buffer,
            log_buffer_size,
            log_size_ret,
            &compiler,
            &compiled);
    }
    if (compiler != NULL) {
        nvPTXCompilerDestroy(&compiler);
    }
    if (result != SECANT_PTX_SUCCESS) {
        _secant_ptx_compiled_release(compiled);
        _SECANT_PTX_ERROR_RET(result);
    }
    *compiled_ret = compiled;
    return SECANT_PTX_SUCCESS;
}

SecantPTXResult
secant_ptx_compiled_destroy(SecantPTXCompiled compiled) {
    if (compiled == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    _secant_ptx_compiled_release(compiled);
    return SECANT_PTX_SUCCESS;
}

SecantPTXResult
secant_ptx_compiled_binary_get(
    SecantPTXCompiled compiled,
    const void** binary_ret,
    size_t* binary_size_ret
) {
    if (compiled == NULL || binary_ret == NULL || binary_size_ret == NULL) {
        _SECANT_PTX_ERROR_RET(SECANT_PTX_ERROR_INVALID_VALUE);
    }
    *binary_ret = compiled->binary;
    *binary_size_ret = compiled->binary_size;
    return SECANT_PTX_SUCCESS;
}

#undef _SECANT_PTX_CHECK_RET
#undef _SECANT_PTX_ERROR_RET
