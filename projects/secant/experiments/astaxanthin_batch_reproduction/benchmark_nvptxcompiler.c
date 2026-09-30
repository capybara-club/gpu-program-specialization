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
// Measure the in-process nvPTXCompiler path used by PTX Inject.
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <nvPTXCompiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    double create_ms;
    double compile_ms;
    double artifact_ms;
    double total_ms;
} Sample;

static double now_ms(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        perror("clock_gettime");
        exit(1);
    }
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
}

static int compare_double(const void* left, const void* right) {
    const double a = *(const double*)left;
    const double b = *(const double*)right;
    return (a > b) - (a < b);
}

static double median(double* values, size_t count) {
    qsort(values, count, sizeof(*values), compare_double);
    if ((count & 1U) != 0U) {
        return values[count / 2];
    }
    return (values[count / 2 - 1] + values[count / 2]) * 0.5;
}

static void print_compiler_log(nvPTXCompilerHandle compiler, int error_log) {
    size_t bytes = 0;
    nvPTXCompileResult result = error_log
        ? nvPTXCompilerGetErrorLogSize(compiler, &bytes)
        : nvPTXCompilerGetInfoLogSize(compiler, &bytes);
    if (result != NVPTXCOMPILE_SUCCESS || bytes <= 1) {
        return;
    }
    char* buffer = (char*)malloc(bytes);
    if (buffer == NULL) {
        return;
    }
    result = error_log
        ? nvPTXCompilerGetErrorLog(compiler, buffer)
        : nvPTXCompilerGetInfoLog(compiler, buffer);
    if (result == NVPTXCOMPILE_SUCCESS) {
        fprintf(stderr, "%s\n", buffer);
    }
    free(buffer);
}

static char* read_file(const char* path, size_t* bytes_out) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    const long length = ftell(file);
    if (length < 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    char* data = (char*)malloc((size_t)length + 128);
    if (data == NULL || fread(data, 1, (size_t)length, file) != (size_t)length) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    data[length] = '\0';
    *bytes_out = (size_t)length;
    return data;
}

static int write_file(const char* path, const void* data, size_t bytes) {
    FILE* file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    const int wrote_all = fwrite(data, 1, bytes, file) == bytes;
    const int closed = fclose(file) == 0;
    return wrote_all && closed;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "usage: %s INPUT.ptx sm_NN [repetitions] [warmups] [OUTPUT.cubin]\n", argv[0]);
        return 2;
    }

    const size_t repetitions = argc >= 4 ? strtoul(argv[3], NULL, 10) : 31;
    const size_t warmups = argc >= 5 ? strtoul(argv[4], NULL, 10) : 3;
    if (repetitions == 0) {
        fprintf(stderr, "repetitions must be positive\n");
        return 2;
    }

    size_t base_bytes = 0;
    char* ptx = read_file(argv[1], &base_bytes);
    if (ptx == NULL) {
        return 1;
    }

    char gpu_name[64];
    if (snprintf(gpu_name, sizeof(gpu_name), "--gpu-name=%s", argv[2]) >= (int)sizeof(gpu_name)) {
        free(ptx);
        return 2;
    }
    const char* options[] = {gpu_name};
    const size_t sample_count = warmups + repetitions;
    Sample* samples = (Sample*)calloc(repetitions, sizeof(*samples));
    void* last_cubin = NULL;
    size_t last_cubin_bytes = 0;
    if (samples == NULL) {
        free(ptx);
        return 1;
    }

    for (size_t index = 0; index < sample_count; ++index) {
        // A different trailing comment makes every input byte string unique while
        // leaving the PTX program and generated machine instructions unchanged.
        const int suffix_bytes = snprintf(
            ptx + base_bytes, 128, "\n// secant_compile_nonce_%zu\n", index
        );
        const size_t input_bytes = base_bytes + (size_t)suffix_bytes;
        nvPTXCompilerHandle compiler = NULL;
        const double total_begin = now_ms();
        const double create_begin = total_begin;
        nvPTXCompileResult result = nvPTXCompilerCreate(&compiler, input_bytes, ptx);
        const double create_end = now_ms();
        if (result != NVPTXCOMPILE_SUCCESS) {
            fprintf(stderr, "nvPTXCompilerCreate failed: %d\n", (int)result);
            free(samples);
            free(ptx);
            return 1;
        }

        const double compile_begin = now_ms();
        result = nvPTXCompilerCompile(compiler, 1, options);
        const double compile_end = now_ms();
        if (result != NVPTXCOMPILE_SUCCESS) {
            fprintf(stderr, "nvPTXCompilerCompile failed: %d\n", (int)result);
            print_compiler_log(compiler, 1);
            print_compiler_log(compiler, 0);
            nvPTXCompilerDestroy(&compiler);
            free(samples);
            free(ptx);
            return 1;
        }

        const double artifact_begin = now_ms();
        size_t cubin_bytes = 0;
        result = nvPTXCompilerGetCompiledProgramSize(compiler, &cubin_bytes);
        if (result != NVPTXCOMPILE_SUCCESS) {
            fprintf(stderr, "nvPTXCompilerGetCompiledProgramSize failed: %d\n", (int)result);
            nvPTXCompilerDestroy(&compiler);
            free(samples);
            free(ptx);
            return 1;
        }
        void* cubin = malloc(cubin_bytes);
        if (cubin == NULL) {
            fprintf(stderr, "cannot allocate %zu-byte CUBIN\n", cubin_bytes);
            nvPTXCompilerDestroy(&compiler);
            free(samples);
            free(ptx);
            return 1;
        }
        result = nvPTXCompilerGetCompiledProgram(compiler, cubin);
        const double artifact_end = now_ms();
        if (result != NVPTXCOMPILE_SUCCESS) {
            fprintf(stderr, "nvPTXCompilerGetCompiledProgram failed: %d\n", (int)result);
            free(cubin);
            nvPTXCompilerDestroy(&compiler);
            free(samples);
            free(ptx);
            return 1;
        }
        nvPTXCompilerDestroy(&compiler);
        const double total_end = now_ms();

        free(last_cubin);
        last_cubin = cubin;
        last_cubin_bytes = cubin_bytes;
        if (index >= warmups) {
            Sample* sample = &samples[index - warmups];
            sample->create_ms = create_end - create_begin;
            sample->compile_ms = compile_end - compile_begin;
            sample->artifact_ms = artifact_end - artifact_begin;
            sample->total_ms = total_end - total_begin;
        }
    }

    double* values = (double*)malloc(repetitions * sizeof(*values));
    if (values == NULL) {
        free(last_cubin);
        free(samples);
        free(ptx);
        return 1;
    }
#define PRINT_MEDIAN(field) do { \
        for (size_t i = 0; i < repetitions; ++i) values[i] = samples[i].field; \
        printf("  \"" #field "\": %.6f%s\n", median(values, repetitions), \
               strcmp(#field, "total_ms") == 0 ? "" : ","); \
    } while (0)
    printf("{\n");
    printf("  \"repetitions\": %zu,\n", repetitions);
    printf("  \"warmups\": %zu,\n", warmups);
    printf("  \"ptx_bytes\": %zu,\n", base_bytes);
    printf("  \"cubin_bytes\": %zu,\n", last_cubin_bytes);
    PRINT_MEDIAN(create_ms);
    PRINT_MEDIAN(compile_ms);
    PRINT_MEDIAN(artifact_ms);
    PRINT_MEDIAN(total_ms);
    printf("}\n");
#undef PRINT_MEDIAN

    int exit_code = 0;
    if (argc >= 6 && !write_file(argv[5], last_cubin, last_cubin_bytes)) {
        fprintf(stderr, "cannot write %s\n", argv[5]);
        exit_code = 1;
    }
    free(values);
    free(last_cubin);
    free(samples);
    free(ptx);
    return exit_code;
}
