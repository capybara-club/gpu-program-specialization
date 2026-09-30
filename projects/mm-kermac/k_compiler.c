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
#define PTX_INJECT_MAX_UNIQUE_INJECTS (128*128*4)
#define PTX_INJECT_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>

#include <kermac.h>
#include <k_internal.h>

#include <ptx_inject.h>

#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>

struct KermacCompilerHandleImpl {
    nvPTXCompilerHandle nvptx_compiler;
};

static
inline
KermacResult
_kermac_compiler_create(
    KermacCompilerHandle* compiler,
    KermacHandle handle,
    KermacStackAllocator* hsa,
    PtxInjectHandle ptx_inject,
    const StackPtxCompilerInfo* compiler_info,
    const StackPtxStackInfo* stack_info,
    size_t execution_limit,
    const StackPtxRegister* registers,
    size_t num_registers,
    const StackPtxInstruction* const* stack_ptx_instruction_stubs,
    const StackPtxInstruction* const* routines,
    size_t num_routines,
    const size_t** request_stubs,
    const size_t* request_stub_sizes,
    size_t num_stubs,
    int compute_device_major,
    int compute_device_minor,
    bool verbose
) {
    char** ptx_stubs_ptr = NULL;
    KermacTensor ptx_stubs = {0};
    _KERMAC_CHECK_RET(
        kermac_tensor_create(
            &ptx_stubs,
            KERMAC_DATA_TYPE_POINTER,
            (KermacExtent){num_stubs},
            hsa
        )
    );
    _KERMAC_CHECK_RET( kermac_memory_pointer(ptx_stubs.memory, (void**)&ptx_stubs_ptr) );

    size_t stack_ptx_workspace_size;
    _KERMAC_STACK_PTX_CHECK_RET( 
        stack_ptx_compile_workspace_size(
            compiler_info,
            stack_info, 
            &stack_ptx_workspace_size
        )
    );

    void* stack_ptx_workspace_ptr = NULL;
    KermacTensor stack_ptx_workspace = {0};
    _KERMAC_CHECK_RET(
        kermac_tensor_create(
            &stack_ptx_workspace, 
            KERMAC_DATA_TYPE_BYTE,
            (KermacExtent){stack_ptx_workspace_size}, 
            hsa
        )
    );
    _KERMAC_CHECK_RET( kermac_memory_pointer(stack_ptx_workspace.memory, (void**)&stack_ptx_workspace_ptr) );
 
    for (size_t i = 0; i < num_stubs; i++) {
        size_t capacity = 0;
        size_t required = 0;
        _KERMAC_STACK_PTX_CHECK_RET(
            stack_ptx_compile(
                compiler_info,
                stack_info,
                stack_ptx_instruction_stubs[i],
                registers, num_registers,
                (const StackPtxInstruction**)routines,
                num_routines,
                request_stubs[i],
                request_stub_sizes[i],
                execution_limit,
                stack_ptx_workspace_ptr,
                stack_ptx_workspace_size,
                NULL,
                capacity,
                &required
            )
        );

        capacity = required + 1;

        char* ptx_stub_ptr = NULL;
        KermacTensor ptx_stub = {0};

        _KERMAC_CHECK_RET(
            kermac_tensor_create(
                &ptx_stub, 
                KERMAC_DATA_TYPE_BYTE,
                (KermacExtent){capacity}, 
                hsa
            )
        );
        _KERMAC_CHECK_RET( kermac_memory_pointer(ptx_stub.memory, (void**)&ptx_stub_ptr) );

        _KERMAC_STACK_PTX_CHECK_RET(
            stack_ptx_compile(
                compiler_info,
                stack_info,
                stack_ptx_instruction_stubs[i],
                registers, num_registers,
                (const StackPtxInstruction**)routines,
                num_routines,
                request_stubs[i],
                request_stub_sizes[i],
                execution_limit,
                stack_ptx_workspace_ptr,
                stack_ptx_workspace_size,
                ptx_stub_ptr,
                capacity,
                &required
            )
        );

        ptx_stubs_ptr[i] = ptx_stub_ptr;
    }

    size_t capacity = 0;
    size_t required = 0;

    _KERMAC_PTX_INJECT_CHECK_RET(
        ptx_inject_render_ptx(
            ptx_inject,
            (const char* const*)ptx_stubs_ptr, 
            num_stubs,
            NULL,
            capacity,
            &required
        )
    );

    capacity = required + 1;
    // printf("PTX size: %zu\n", capacity);

    char* rendered_ptx_ptr = NULL;
    KermacTensor rendered_ptx = {0};
    _KERMAC_CHECK_RET(
        kermac_tensor_create(
            &rendered_ptx, 
            KERMAC_DATA_TYPE_BYTE,
            (KermacExtent){capacity}, 
            hsa
        )
    );
    _KERMAC_CHECK_RET( kermac_memory_pointer(rendered_ptx.memory, (void**)&rendered_ptx_ptr) );

    _KERMAC_PTX_INJECT_CHECK_RET(
        ptx_inject_render_ptx(
            ptx_inject,
            (const char* const*)ptx_stubs_ptr, 
            num_stubs,
            rendered_ptx_ptr,
            capacity,
            &required
        )
    );

    // printf("%s\n", rendered_ptx_ptr);
    // exit(1);
    
    char compile_line_buffer[32];
    snprintf(compile_line_buffer, sizeof(compile_line_buffer), "--gpu-name=sm_%d%d", compute_device_major, compute_device_minor);
    const char* ptx_compile_options[] = {
        compile_line_buffer
    };
    const size_t num_ptx_compile_options = sizeof(ptx_compile_options) / sizeof(*ptx_compile_options);

    nvPTXCompilerHandle nvptx_compiler = {0};

    _KERMAC_NVPTX_CHECK_RET(
        nvPTXCompilerCreate(
            &nvptx_compiler,
            required,
            rendered_ptx_ptr
        )
    );
    nvPTXCompileResult compile_result =
        nvPTXCompilerCompile(
            nvptx_compiler,
            num_ptx_compile_options,
            ptx_compile_options
        );

    if (verbose) {
        size_t info_size = 0;
        _KERMAC_NVPTX_CHECK_RET( nvPTXCompilerGetInfoLogSize(nvptx_compiler, &info_size) );
        if (info_size > 1) {
            size_t capacity = info_size + 1;
            char* info_log_ptr = NULL;
            KermacTensor info_log = {0};
            _KERMAC_CHECK_RET(
                kermac_tensor_create(
                    &info_log,
                    KERMAC_DATA_TYPE_BYTE,
                    (KermacExtent){(int64_t)capacity},
                    hsa
                )
            );
            _KERMAC_CHECK_RET( kermac_memory_pointer(info_log.memory, (void**)&info_log_ptr) );
            _KERMAC_NVPTX_CHECK_RET( nvPTXCompilerGetInfoLog(nvptx_compiler, info_log_ptr) );
            info_log_ptr[capacity - 1] = '\0';
            fprintf(stdout, "NVPTX info log:\n%s\n", info_log_ptr);
        }
    }

    if (compile_result != NVPTXCOMPILE_SUCCESS) {
        size_t error_size = 0;
        _KERMAC_NVPTX_CHECK_RET( nvPTXCompilerGetErrorLogSize(nvptx_compiler, &error_size) );
        if (error_size > 1) {
            size_t capacity = error_size + 1;
            char* error_log_ptr = NULL;
            KermacTensor error_log = {0};
            _KERMAC_CHECK_RET(
                kermac_tensor_create(
                    &error_log,
                    KERMAC_DATA_TYPE_BYTE,
                    (KermacExtent){(int64_t)capacity},
                    hsa
                )
            );
            _KERMAC_CHECK_RET( kermac_memory_pointer(error_log.memory, (void**)&error_log_ptr) );
            _KERMAC_NVPTX_CHECK_RET( nvPTXCompilerGetErrorLog(nvptx_compiler, error_log_ptr) );
            error_log_ptr[capacity - 1] = '\0';
            fprintf(stderr, "NVPTX error log:\n%s\n", error_log_ptr);
        }
        _KERMAC_NVPTX_CHECK_RET( nvPTXCompilerDestroy(&nvptx_compiler) );
        _KERMAC_CHECK_RET( KERMAC_ERROR_NVPTX );
    }

    (*compiler)->nvptx_compiler = nvptx_compiler;

    return KERMAC_SUCCESS;
}

static
inline
KermacResult
_kermac_compiler_create_with_routines(
    KermacCompilerHandle* compiler,
    KermacHandle handle,
    KermacStackAllocator* hsa,
    PtxInjectHandle ptx_inject,
    const StackPtxCompilerInfo* compiler_info,
    const StackPtxStackInfo* stack_info,
    size_t execution_limit,
    const StackPtxRegister* registers,
    size_t num_registers,
    const StackPtxInstruction* const* stack_ptx_instruction_stubs,
    const StackPtxInstruction* const* routines,
    size_t num_routines,
    const size_t** request_stubs,
    const size_t* request_stub_sizes,
    size_t num_stubs,
    bool verbose
) {
    if (!compiler) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }
    *compiler = NULL;

    if (!handle || !hsa) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    if (hsa->memory_space != KERMAC_MEMORY_SPACE_HOST) {
        _KERMAC_ERROR( KERMAC_ERROR_WRONG_ALLOCATOR_DEVICE );
    }

    if (hsa->is_dry) {
        _KERMAC_ERROR( KERMAC_ERROR_INCONSISTENT_ALLOCATION );
    }

    KermacCompilerHandle h = (KermacCompilerHandle)calloc(1, sizeof(*h));
    if (!h) {
        _KERMAC_ERROR( KERMAC_ERROR_OUT_OF_MEMORY );
    }

    *compiler = h;
    int current_stack_counter = hsa->current_stack_counter;
    size_t current_offset = hsa->current_offset;

    int compute_device_major = handle->device_attributes[_KERMAC_DEVICE_ATTRIBUTES_COMPUTE_CAPABILITY_MAJOR];
    int compute_device_minor = handle->device_attributes[_KERMAC_DEVICE_ATTRIBUTES_COMPUTE_CAPABILITY_MINOR];

    KermacResult result =
        _kermac_compiler_create(
            compiler,
            handle,
            hsa,
            ptx_inject,
            compiler_info,
            stack_info,
            execution_limit,
            registers,
            num_registers,
            stack_ptx_instruction_stubs,
            routines,
            num_routines,
            request_stubs,
            request_stub_sizes,
            num_stubs,
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
kermac_compiler_create(
    KermacCompilerHandle* compiler,
    KermacHandle handle,
    KermacStackAllocator* hsa,
    PtxInjectHandle ptx_inject,
    const StackPtxCompilerInfo* compiler_info,
    const StackPtxStackInfo* stack_info,
    size_t execution_limit,
    const StackPtxRegister* registers,
    size_t num_registers,
    const StackPtxInstruction* const* stack_ptx_instruction_stubs,
    const size_t** request_stubs,
    const size_t* request_stub_sizes,
    size_t num_stubs,
    bool verbose
) {
    return _kermac_compiler_create_with_routines(
        compiler,
        handle,
        hsa,
        ptx_inject,
        compiler_info,
        stack_info,
        execution_limit,
        registers,
        num_registers,
        stack_ptx_instruction_stubs,
        NULL,
        0,
        request_stubs,
        request_stub_sizes,
        num_stubs,
        verbose
    );
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_compiler_cubin(
    KermacCompilerHandle compiler,
    void* buffer,
	size_t buffer_size,
	size_t* buffer_bytes_written_ret
) {
    if (!compiler) {
        _KERMAC_ERROR( KERMAC_ERROR_INVALID_VALUE );
    }

    _KERMAC_NVPTX_CHECK_RET(
        nvPTXCompilerGetCompiledProgramSize(
            compiler->nvptx_compiler, 
            buffer_bytes_written_ret
        )
    );

    if (buffer != NULL) {
        if (buffer_size < *buffer_bytes_written_ret) {
            _KERMAC_ERROR( KERMAC_ERROR_INSUFFICIENT_BUFFER );
        }

        _KERMAC_NVPTX_CHECK_RET( 
            nvPTXCompilerGetCompiledProgram(
                compiler->nvptx_compiler, 
                buffer
            )
        );
    }

    return KERMAC_SUCCESS;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_compiler_destroy(
    KermacCompilerHandle compiler
) {
    if (!compiler) {
        return KERMAC_SUCCESS;
    }

     _KERMAC_NVPTX_CHECK_RET(
        nvPTXCompilerDestroy(&compiler->nvptx_compiler)
    );

    free(compiler);
    return KERMAC_SUCCESS;
}

static
inline
KermacResult
_kermac_stack_ptx_inject_compile_with_routines(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    PtxInjectHandle ptx_inject,
    const StackPtxCompilerInfo* compiler_info,
    const StackPtxStackInfo* stack_info,
    size_t execution_limit,
    const StackPtxRegister* registers,
    size_t num_registers,
    const StackPtxInstruction* const* stack_ptx_instruction_stubs,
    const StackPtxInstruction* const* routines,
    size_t num_routines,
    const size_t** request_stubs,
    const size_t* request_stub_sizes,
    size_t num_stubs,
    CUmodule* module
) {
    KermacResult result = KERMAC_SUCCESS;

    KermacCompilerHandle compiler = NULL;
    KermacTensor binary_image = KERMAC_ZERO_INIT(KermacTensor);

    size_t required = 0;
    void* binary_image_ptr = NULL;

    result =
        _kermac_compiler_create_with_routines(
            &compiler,
            handle,
            hsa,
            ptx_inject,
            compiler_info,
            stack_info,
            execution_limit,
            registers,
            num_registers,
            stack_ptx_instruction_stubs,
            routines,
            num_routines,
            request_stubs,
            request_stub_sizes,
            num_stubs,
            false
        );
    if (result != KERMAC_SUCCESS) {
        goto cleanup;
    }

    result = kermac_compiler_cubin(compiler, NULL, 0, &required);
    if (result != KERMAC_SUCCESS) {
        goto cleanup;
    }

    result =
        kermac_tensor_create(
            &binary_image, 
            KERMAC_DATA_TYPE_BYTE,
            (KermacExtent){required}, 
            hsa
        );
    if (result != KERMAC_SUCCESS) {
        goto cleanup;
    }

    result = kermac_memory_pointer(binary_image.memory, (void**)&binary_image_ptr);
    if (result != KERMAC_SUCCESS) {
        goto cleanup;
    }

    result = kermac_compiler_cubin(compiler, binary_image_ptr, required, &required);
    if (result != KERMAC_SUCCESS) {
        goto cleanup;
    }

    if (cuModuleLoadDataEx(module, binary_image_ptr, 0, NULL, NULL) != CUDA_SUCCESS) {
        result = KERMAC_ERROR_CUDA;
        goto cleanup;
    }

cleanup:
    if (binary_image.memory.stack_allocator != NULL) {
        KermacResult destroy_result = kermac_tensor_destroy(binary_image);
        if (result == KERMAC_SUCCESS) {
            result = destroy_result;
        }
    }

    if (compiler != NULL) {
        KermacResult destroy_result = kermac_compiler_destroy(compiler);
        if (result == KERMAC_SUCCESS) {
            result = destroy_result;
        }
    }

    return result;
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_stack_ptx_inject_compile(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    PtxInjectHandle ptx_inject,
    const StackPtxCompilerInfo* compiler_info,
    const StackPtxStackInfo* stack_info,
    size_t execution_limit,
    const StackPtxRegister* registers,
    size_t num_registers,
    const StackPtxInstruction* const* stack_ptx_instruction_stubs,
    const size_t** request_stubs,
    const size_t* request_stub_sizes,
    size_t num_stubs,
    CUmodule* module
) {
    return _kermac_stack_ptx_inject_compile_with_routines(
        handle,
        hsa,
        ptx_inject,
        compiler_info,
        stack_info,
        execution_limit,
        registers,
        num_registers,
        stack_ptx_instruction_stubs,
        NULL,
        0,
        request_stubs,
        request_stub_sizes,
        num_stubs,
        module
    );
}

KERMAC_PUBLIC_DEF
KermacResult
kermac_stack_ptx_inject_compile_with_routines(
    KermacHandle handle,
    KermacStackAllocator* hsa,
    PtxInjectHandle ptx_inject,
    const StackPtxCompilerInfo* compiler_info,
    const StackPtxStackInfo* stack_info,
    size_t execution_limit,
    const StackPtxRegister* registers,
    size_t num_registers,
    const StackPtxInstruction* const* stack_ptx_instruction_stubs,
    const StackPtxInstruction* const* routines,
    size_t num_routines,
    const size_t** request_stubs,
    const size_t* request_stub_sizes,
    size_t num_stubs,
    CUmodule* module
) {
    return _kermac_stack_ptx_inject_compile_with_routines(
        handle,
        hsa,
        ptx_inject,
        compiler_info,
        stack_info,
        execution_limit,
        registers,
        num_registers,
        stack_ptx_instruction_stubs,
        routines,
        num_routines,
        request_stubs,
        request_stub_sizes,
        num_stubs,
        module
    );
}
