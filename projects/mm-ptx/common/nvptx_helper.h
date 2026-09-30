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
/*
 * SPDX-FileCopyrightText: 2026 Charles Durham
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <nvPTXCompiler.h>

#include <check_result_helper.h>

static
void
nvptx_print_info_log(
    nvPTXCompilerHandle nvptx_compiler
) {
    size_t info_size;
    nvptxCheck(
        nvPTXCompilerGetInfoLogSize(nvptx_compiler, &info_size)
    );

    if (info_size != 0) {
        char *info_log = (char*)malloc(info_size+1);
        nvptxCheck(nvPTXCompilerGetInfoLog(nvptx_compiler, info_log));
        printf("Error log: %s\n", info_log);
        free(info_log);
    }
}

static
void
nvptx_print_error_log(
    nvPTXCompilerHandle nvptx_compiler
) {
    size_t error_size;
    nvptxCheck(
        nvPTXCompilerGetErrorLogSize(nvptx_compiler, &error_size)
    );

    if (error_size != 0) {
        char *error_log = (char*)malloc(error_size+1);
        nvptxCheck(nvPTXCompilerGetErrorLog(nvptx_compiler, error_log));
        fprintf(stderr, "Error log: %s\n", error_log);
        free(error_log);
    }
}

__attribute__((unused))
static
char *
nvptx_compile(
    int compute_capability_major,
    int compute_capability_minor,
    const char *ptx_code,
    size_t ptx_code_size,
    size_t* sass_image_size_out,
    bool verbose
) {
    char compile_line_buffer[32];
    sprintf(compile_line_buffer, "--gpu-name=sm_%d%d", compute_capability_major, compute_capability_minor);

    const char* ptx_compile_options[] = {
        compile_line_buffer,
        verbose ? "--verbose" : NULL
    };
    const size_t num_ptx_compile_options = verbose ? 2 : 1;

    nvPTXCompilerHandle nvptx_compiler = {0};
    nvptxCheck(
        nvPTXCompilerCreate(
            &nvptx_compiler,
            ptx_code_size,
            ptx_code
        )
    );

    nvPTXCompileResult result = 
        nvPTXCompilerCompile(
            nvptx_compiler,
            num_ptx_compile_options,
            ptx_compile_options
        );

    nvptx_print_info_log(nvptx_compiler);

    if (result != NVPTXCOMPILE_SUCCESS) {
        nvptx_print_error_log(nvptx_compiler);
        assert( false );
        exit(1);
    }

    size_t binary_image_size;
    nvptxCheck(
        nvPTXCompilerGetCompiledProgramSize(
            nvptx_compiler, &binary_image_size
        )
    );

    char *binary_image = (char*)malloc(binary_image_size);
    nvptxCheck(
        nvPTXCompilerGetCompiledProgram(
            nvptx_compiler, binary_image
        )
    );
    nvptxCheck(
        nvPTXCompilerDestroy(&nvptx_compiler)
    );

    if (sass_image_size_out) {
        *sass_image_size_out = binary_image_size;
    }

    return binary_image;
}
