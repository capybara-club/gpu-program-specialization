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
#include <kermac.h>
#include <k_internal.h>

#include <nvrtc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct KermacNvrtcCompilerHandleImpl {
    nvrtcProgram program;
};

static
inline
KermacResult
_kermac_nvrtc_result(
    nvrtcResult status,
    const char* file,
    int line
) {
#ifdef KERMAC_DEBUG
    if (status != NVRTC_SUCCESS) {
        const char* error_string = nvrtcGetErrorString(status);
        fprintf(stderr, "KERMAC_NVRTC_CHECK: %s \n  %s %d\n", error_string, file, line);
    }
#endif
    return status == NVRTC_SUCCESS ? KERMAC_SUCCESS : KERMAC_ERROR_NVRTC;
}

#define _KERMAC_NVRTC_CHECK_GOTO(ans, label)                       \
    do {                                                           \
        KermacResult _result =                                     \
            _kermac_nvrtc_result((ans), __FILE__, __LINE__);        \
        if (_result != KERMAC_SUCCESS) {                           \
            result = _result;                                      \
            goto label;                                            \
        }                                                          \
    } while (0)

#define _KERMAC_NVRTC_CHECK_RET(ans)                                \
    do {                                                            \
        KermacResult _result =                                      \
            _kermac_nvrtc_result((ans), __FILE__, __LINE__);         \
        if (_result != KERMAC_SUCCESS) {                            \
            return _result;                                         \
        }                                                           \
    } while (0)

#define _KERMAC_CHECK_GOTO(ans, label)                              \
    do {                                                            \
        KermacResult _result = (ans);                               \
        if (_result != KERMAC_SUCCESS) {                            \
            result = _result;                                       \
            goto label;                                             \
        }                                                           \
    } while (0)

static
inline
bool
_kermac_nvrtc_has_arch_option(
    const char* option
) {
    if (!option) {
        return false;
    }
    if (strncmp(option, "--gpu-architecture", strlen("--gpu-architecture")) == 0) {
        return true;
    }
    if (strncmp(option, "-arch=", strlen("-arch=")) == 0) {
        return true;
    }
    return false;
}

static
inline
KermacResult
_kermac_nvrtc_compiler_create(
    KermacNvrtcCompilerHandle* compiler,
    KermacHandle handle,
    KermacStackAllocator* hsa,
    const char* program_name,
    const char* program_source,
    const char* const* headers,
    const char* const* include_names,
    size_t num_headers,
    const char* const* options,
    size_t num_options,
    int compute_device_major,
    int compute_device_minor,
    bool verbose
) {
    (void)handle;
    KermacResult result = KERMAC_SUCCESS;
    nvrtcProgram program = NULL;

    const char* effective_program_name = program_name;
    if (!effective_program_name || !effective_program_name[0]) {
        effective_program_name = "kermac_nvrtc_program";
    }

    _KERMAC_NVRTC_CHECK_GOTO(
        nvrtcCreateProgram(
            &program,
            program_source,
            effective_program_name,
            (int)num_headers,
            headers,
            include_names
        ),
        cleanup
    );

    bool has_arch_option = false;
    for (size_t i = 0; i < num_options; ++i) {
        if (_kermac_nvrtc_has_arch_option(options[i])) {
            has_arch_option = true;
            break;
        }
    }

    char arch_option_buffer[64];
    const char* arch_option = NULL;
    if (!has_arch_option) {
        snprintf(
            arch_option_buffer,
            sizeof(arch_option_buffer),
            "--gpu-architecture=compute_%d%d",
            compute_device_major,
            compute_device_minor
        );
        arch_option = arch_option_buffer;
    }

    size_t num_compile_options = num_options + (has_arch_option ? 0 : 1);
    const char** compile_options_ptr = NULL;
    KermacTensor compile_options = {0};
    _KERMAC_CHECK_GOTO(
        kermac_tensor_create(
            &compile_options,
            KERMAC_DATA_TYPE_POINTER,
            (KermacExtent){(int64_t)num_compile_options},
            hsa
        ),
        cleanup
    );
    _KERMAC_CHECK_GOTO(
        kermac_memory_pointer(compile_options.memory, (void**)&compile_options_ptr),
        cleanup
    );

    size_t option_index = 0;
    if (!has_arch_option) {
        compile_options_ptr[option_index++] = arch_option;
    }
    for (size_t i = 0; i < num_options; ++i) {
        compile_options_ptr[option_index++] = options[i];
    }

    nvrtcResult compile_status =
        nvrtcCompileProgram(
            program,
            (int)num_compile_options,
            compile_options_ptr
        );

    size_t log_size = 0;
    _KERMAC_NVRTC_CHECK_GOTO(
        nvrtcGetProgramLogSize(program, &log_size),
        cleanup
    );
    if (log_size > 1 && (verbose || compile_status != NVRTC_SUCCESS)) {
        size_t capacity = log_size + 1;
        char* log_ptr = NULL;
        KermacTensor log_tensor = {0};
        _KERMAC_CHECK_GOTO(
            kermac_tensor_create(
                &log_tensor,
                KERMAC_DATA_TYPE_BYTE,
                (KermacExtent){(int64_t)capacity},
                hsa
            ),
            cleanup
        );
        _KERMAC_CHECK_GOTO(
            kermac_memory_pointer(log_tensor.memory, (void**)&log_ptr),
            cleanup
        );
        _KERMAC_NVRTC_CHECK_GOTO(
            nvrtcGetProgramLog(program, log_ptr),
            cleanup
        );
        log_ptr[capacity - 1] = '\0';
        if (compile_status == NVRTC_SUCCESS) {
            fprintf(stdout, "NVRTC info log:\n%s\n", log_ptr);
        } else {
            fprintf(stderr, "NVRTC error log:\n%s\n", log_ptr);
        }
    }

    _KERMAC_NVRTC_CHECK_GOTO(compile_status, cleanup);

    (*compiler)->program = program;
    return KERMAC_SUCCESS;

cleanup:
    if (program != NULL) {
        (void)nvrtcDestroyProgram(&program);
    }
    return result;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_nvrtc_compiler_create(
    KermacNvrtcCompilerHandle* compiler,
    KermacHandle handle,
    KermacStackAllocator* hsa,
    const char* program_name,
    const char* program_source,
    const char* const* headers,
    const char* const* include_names,
    size_t num_headers,
    const char* const* options,
    size_t num_options,
    bool verbose
) {
    if (!compiler) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    *compiler = NULL;

    if (!handle || !hsa || !program_source) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (num_headers > 0 && (!headers || !include_names)) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (num_options > 0 && !options) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (hsa->memory_space != KERMAC_MEMORY_SPACE_HOST) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (hsa->is_dry) {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    KermacNvrtcCompilerHandle h = (KermacNvrtcCompilerHandle)calloc(1, sizeof(*h));
    if (!h) {
        _KERMAC_ERROR( KERMAC_ERROR_OUT_OF_MEMORY );
    }

    *compiler = h;
    int current_stack_counter = hsa->current_stack_counter;
    size_t current_offset = hsa->current_offset;

    int compute_device_major = handle->device_attributes[_KERMAC_DEVICE_ATTRIBUTES_COMPUTE_CAPABILITY_MAJOR];
    int compute_device_minor = handle->device_attributes[_KERMAC_DEVICE_ATTRIBUTES_COMPUTE_CAPABILITY_MINOR];

    KermacResult result =
        _kermac_nvrtc_compiler_create(
            compiler,
            handle,
            hsa,
            program_name,
            program_source,
            headers,
            include_names,
            num_headers,
            options,
            num_options,
            compute_device_major,
            compute_device_minor,
            verbose
        );

    hsa->current_stack_counter = current_stack_counter;
    hsa->current_offset = current_offset;

    if (result != KERMAC_SUCCESS) {
        free(h);
        *compiler = NULL;
    }

    return result;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_nvrtc_compiler_ptx(
    KermacNvrtcCompilerHandle compiler,
    void* buffer,
    size_t buffer_size,
    size_t* buffer_bytes_written_ret
) {
    if (!compiler) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    _KERMAC_NVRTC_CHECK_RET(
        nvrtcGetPTXSize(
            compiler->program,
            buffer_bytes_written_ret
        )
    );

    if (buffer != NULL) {
        if (buffer_size < *buffer_bytes_written_ret) {
            _KERMAC_ERROR( KERMAC_ERROR_INSUFFICIENT_BUFFER );
        }

        _KERMAC_NVRTC_CHECK_RET(
            nvrtcGetPTX(
                compiler->program,
                (char*)buffer
            )
        );
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_nvrtc_compiler_destroy(
    KermacNvrtcCompilerHandle compiler
) {
    if (!compiler) {
        return KERMAC_SUCCESS;
    }

    if (compiler->program != NULL) {
        nvrtcResult status = nvrtcDestroyProgram(&compiler->program);
        free(compiler);
        return _kermac_nvrtc_result(status, __FILE__, __LINE__);
    }

    free(compiler);
    return KERMAC_SUCCESS;
}

#undef _KERMAC_NVRTC_CHECK_GOTO
#undef _KERMAC_NVRTC_CHECK_RET
#undef _KERMAC_CHECK_GOTO
