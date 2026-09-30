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
#include <nvrtc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s sm_XX source.cu output.cubin [--direct]\n", argv[0]);
        return 2;
    }
    if (argc == 5 && strcmp(argv[4], "--direct") != 0) {
        fprintf(stderr, "the only supported optional argument is --direct\n");
        return 2;
    }

    FILE *source_file = fopen(argv[2], "rb");
    if (source_file == NULL || fseek(source_file, 0, SEEK_END) != 0) return 3;
    const long source_bytes = ftell(source_file);
    if (source_bytes < 0 || fseek(source_file, 0, SEEK_SET) != 0) return 4;
    char *source = (char *)malloc((size_t)source_bytes + 1u);
    if (source == NULL ||
        fread(source, 1u, (size_t)source_bytes, source_file) != (size_t)source_bytes) {
        return 5;
    }
    source[source_bytes] = '\0';
    fclose(source_file);

    char architecture[64];
    if (snprintf(architecture, sizeof(architecture),
                 "--gpu-architecture=%s", argv[1]) < 0) return 6;
    const char *options[7] = {
        "--std=c++17",
        architecture,
        "--use_fast_math",
        "--generate-line-info",
        "--ptxas-options=--verbose",
        "--ptxas-options=--opt-level=3",
        "-DSECANT_DIRECT_RATES=1"
    };
    const int option_count = argc == 5 ? 7 : 6;

    nvrtcProgram program = NULL;
    nvrtcResult result = nvrtcCreateProgram(
        &program, source, argv[2], 0, NULL, NULL);
    if (result == NVRTC_SUCCESS) {
        result = nvrtcCompileProgram(program, option_count, options);
    }

    size_t log_bytes = 0u;
    if (program != NULL && nvrtcGetProgramLogSize(program, &log_bytes) == NVRTC_SUCCESS &&
        log_bytes > 1u) {
        char *log = (char *)malloc(log_bytes);
        if (log != NULL && nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS) {
            fputs(log, stderr);
        }
        free(log);
    }
    if (result != NVRTC_SUCCESS) {
        fprintf(stderr, "NVRTC: %s\n", nvrtcGetErrorString(result));
        return 7;
    }

    size_t cubin_bytes = 0u;
    if (nvrtcGetCUBINSize(program, &cubin_bytes) != NVRTC_SUCCESS || cubin_bytes == 0u) {
        return 8;
    }
    char *cubin = (char *)malloc(cubin_bytes);
    if (cubin == NULL || nvrtcGetCUBIN(program, cubin) != NVRTC_SUCCESS) return 9;

    FILE *output = fopen(argv[3], "wb");
    if (output == NULL || fwrite(cubin, 1u, cubin_bytes, output) != cubin_bytes) return 10;
    fclose(output);
    free(cubin);
    free(source);
    (void)nvrtcDestroyProgram(&program);
    return 0;
}
