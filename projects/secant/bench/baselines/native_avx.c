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
#define _POSIX_C_SOURCE 200112L

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define SECANT_PORTABLE_ALU_INCLUDE_NATIVE
#include "../corpus/portable_alu_v1.h"

#define SECANT_NATIVE_INPUTS SECANT_PORTABLE_ALU_NUM_INPUTS
#define SECANT_NATIVE_CASES SECANT_PORTABLE_ALU_NUM_CASES
#define SECANT_NATIVE_MAX_REPEATS 256u

typedef struct {
    double best_seconds;
    double median_seconds;
    size_t repeats;
    size_t iterations_per_sample;
    double checksum;
} SecantNativeTiming;

static double
secant_native_seconds(void) {
    struct timespec value;

    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static int
secant_native_compare_double(const void* lhs, const void* rhs) {
    const double left = *(const double*)lhs;
    const double right = *(const double*)rhs;

    return (left > right) - (left < right);
}

static uint32_t
secant_native_hash32(uint32_t value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

static float
secant_native_input_value(size_t column, size_t row, uint32_t seed) {
    const uint32_t bits = secant_native_hash32(
        (uint32_t)column * 0x9e3779b9u ^
        (uint32_t)row * 0x85ebca6bu ^
        seed * 0xc2b2ae35u ^
        0x51ed270bu);

    return ((float)(bits & 0xffffu) / 32767.5f - 1.0f) * 1.5f;
}

static int
secant_native_allocate(size_t count, float** buffer_ret) {
    if (buffer_ret == NULL || count == 0u || count > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    *buffer_ret = NULL;
    return posix_memalign((void**)buffer_ret, 64u, count * sizeof(float)) == 0;
}

static __m256i
secant_native_tail_mask(size_t lanes) {
    const int32_t values[8] = {
        lanes > 0u ? -1 : 0,
        lanes > 1u ? -1 : 0,
        lanes > 2u ? -1 : 0,
        lanes > 3u ? -1 : 0,
        lanes > 4u ? -1 : 0,
        lanes > 5u ? -1 : 0,
        lanes > 6u ? -1 : 0,
        lanes > 7u ? -1 : 0,
    };

    return _mm256_loadu_si256((const __m256i*)values);
}

static float
secant_native_horizontal_sum(__m256 value) {
    __m128 low = _mm256_castps256_ps128(value);
    __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);

    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

#define SECANT_NATIVE_DEFINE_CASE(case_idx) \
    static void \
    secant_native_materialize_case_##case_idx(const float* input, size_t rows, float* output) { \
        const float* x0_ptr = input; \
        const float* x1_ptr = input + rows; \
        const float* x2_ptr = input + 2u * rows; \
        const float* x3_ptr = input + 3u * rows; \
        const float* x4_ptr = input + 4u * rows; \
        const float* x5_ptr = input + 5u * rows; \
        const float* x6_ptr = input + 6u * rows; \
        const float* x7_ptr = input + 7u * rows; \
        size_t row = 0u; \
        for (; row + 8u <= rows; row += 8u) { \
            const __m256 x0 = _mm256_loadu_ps(x0_ptr + row); \
            const __m256 x1 = _mm256_loadu_ps(x1_ptr + row); \
            const __m256 x2 = _mm256_loadu_ps(x2_ptr + row); \
            const __m256 x3 = _mm256_loadu_ps(x3_ptr + row); \
            const __m256 x4 = _mm256_loadu_ps(x4_ptr + row); \
            const __m256 x5 = _mm256_loadu_ps(x5_ptr + row); \
            const __m256 x6 = _mm256_loadu_ps(x6_ptr + row); \
            const __m256 x7 = _mm256_loadu_ps(x7_ptr + row); \
            const __m256 result = SECANT_PORTABLE_ALU_SIMD_EXPR_##case_idx(x0, x1, x2, x3, x4, x5, x6, x7); \
            _mm256_storeu_ps(output + row, result); \
        } \
        if (row < rows) { \
            const __m256i mask = secant_native_tail_mask(rows - row); \
            const __m256 x0 = _mm256_maskload_ps(x0_ptr + row, mask); \
            const __m256 x1 = _mm256_maskload_ps(x1_ptr + row, mask); \
            const __m256 x2 = _mm256_maskload_ps(x2_ptr + row, mask); \
            const __m256 x3 = _mm256_maskload_ps(x3_ptr + row, mask); \
            const __m256 x4 = _mm256_maskload_ps(x4_ptr + row, mask); \
            const __m256 x5 = _mm256_maskload_ps(x5_ptr + row, mask); \
            const __m256 x6 = _mm256_maskload_ps(x6_ptr + row, mask); \
            const __m256 x7 = _mm256_maskload_ps(x7_ptr + row, mask); \
            const __m256 result = SECANT_PORTABLE_ALU_SIMD_EXPR_##case_idx(x0, x1, x2, x3, x4, x5, x6, x7); \
            _mm256_maskstore_ps(output + row, mask, result); \
        } \
    } \
    static float \
    secant_native_sse_case_##case_idx(const float* input, const float* target, size_t rows) { \
        const float* x0_ptr = input; \
        const float* x1_ptr = input + rows; \
        const float* x2_ptr = input + 2u * rows; \
        const float* x3_ptr = input + 3u * rows; \
        const float* x4_ptr = input + 4u * rows; \
        const float* x5_ptr = input + 5u * rows; \
        const float* x6_ptr = input + 6u * rows; \
        const float* x7_ptr = input + 7u * rows; \
        __m256 accumulator = _mm256_setzero_ps(); \
        size_t row = 0u; \
        for (; row + 8u <= rows; row += 8u) { \
            const __m256 x0 = _mm256_loadu_ps(x0_ptr + row); \
            const __m256 x1 = _mm256_loadu_ps(x1_ptr + row); \
            const __m256 x2 = _mm256_loadu_ps(x2_ptr + row); \
            const __m256 x3 = _mm256_loadu_ps(x3_ptr + row); \
            const __m256 x4 = _mm256_loadu_ps(x4_ptr + row); \
            const __m256 x5 = _mm256_loadu_ps(x5_ptr + row); \
            const __m256 x6 = _mm256_loadu_ps(x6_ptr + row); \
            const __m256 x7 = _mm256_loadu_ps(x7_ptr + row); \
            const __m256 prediction = SECANT_PORTABLE_ALU_SIMD_EXPR_##case_idx(x0, x1, x2, x3, x4, x5, x6, x7); \
            const __m256 error = _mm256_sub_ps(prediction, _mm256_loadu_ps(target + row)); \
            accumulator = _mm256_fmadd_ps(error, error, accumulator); \
        } \
        if (row < rows) { \
            const __m256i mask = secant_native_tail_mask(rows - row); \
            const __m256 x0 = _mm256_maskload_ps(x0_ptr + row, mask); \
            const __m256 x1 = _mm256_maskload_ps(x1_ptr + row, mask); \
            const __m256 x2 = _mm256_maskload_ps(x2_ptr + row, mask); \
            const __m256 x3 = _mm256_maskload_ps(x3_ptr + row, mask); \
            const __m256 x4 = _mm256_maskload_ps(x4_ptr + row, mask); \
            const __m256 x5 = _mm256_maskload_ps(x5_ptr + row, mask); \
            const __m256 x6 = _mm256_maskload_ps(x6_ptr + row, mask); \
            const __m256 x7 = _mm256_maskload_ps(x7_ptr + row, mask); \
            const __m256 prediction = SECANT_PORTABLE_ALU_SIMD_EXPR_##case_idx(x0, x1, x2, x3, x4, x5, x6, x7); \
            __m256 error = _mm256_sub_ps(prediction, _mm256_maskload_ps(target + row, mask)); \
            error = _mm256_and_ps(error, _mm256_castsi256_ps(mask)); \
            accumulator = _mm256_fmadd_ps(error, error, accumulator); \
        } \
        return secant_native_horizontal_sum(accumulator); \
    }

SECANT_PORTABLE_ALU_FOR_EACH_CASE(SECANT_NATIVE_DEFINE_CASE)

#define SECANT_NATIVE_MATERIALIZE_CASE(case_idx) \
    case case_idx: \
        secant_native_materialize_case_##case_idx(input, rows, output); \
        break;

static void
secant_native_materialize_one(size_t ast_idx, const float* input, size_t rows, float* output) {
    switch (ast_idx % SECANT_NATIVE_CASES) {
        SECANT_PORTABLE_ALU_FOR_EACH_CASE(SECANT_NATIVE_MATERIALIZE_CASE)
    }
}

#define SECANT_NATIVE_SSE_CASE(case_idx) \
    case case_idx: \
        return secant_native_sse_case_##case_idx(input, target, rows);

static float
secant_native_sse_one(size_t ast_idx, const float* input, const float* target, size_t rows) {
    switch (ast_idx % SECANT_NATIVE_CASES) {
        SECANT_PORTABLE_ALU_FOR_EACH_CASE(SECANT_NATIVE_SSE_CASE)
    }
    return 0.0f;
}

#define SECANT_NATIVE_SCALAR_CASE(case_idx) \
    case case_idx: \
        return SECANT_PORTABLE_ALU_SCALAR_EXPR_##case_idx( \
            x0, x1, x2, x3, x4, x5, x6, x7);

static float
secant_native_scalar_one(size_t ast_idx, const float* input, size_t rows, size_t row) {
    const float x0 = input[row];
    const float x1 = input[rows + row];
    const float x2 = input[2u * rows + row];
    const float x3 = input[3u * rows + row];
    const float x4 = input[4u * rows + row];
    const float x5 = input[5u * rows + row];
    const float x6 = input[6u * rows + row];
    const float x7 = input[7u * rows + row];

    switch (ast_idx % SECANT_NATIVE_CASES) {
        SECANT_PORTABLE_ALU_FOR_EACH_CASE(SECANT_NATIVE_SCALAR_CASE)
    }
    return 0.0f;
}

static int
secant_native_validate(const float* input, size_t rows, float* output) {
    const size_t check_rows = rows < 257u ? rows : 257u;

    for (size_t ast_idx = 0u; ast_idx < SECANT_NATIVE_CASES; ++ast_idx) {
        secant_native_materialize_one(ast_idx, input, rows, output);
        for (size_t row = 0u; row < check_rows; ++row) {
            const float expected = secant_native_scalar_one(ast_idx, input, rows, row);
            const float difference = fabsf(output[row] - expected);
            const float tolerance = 2.0e-6f * fmaxf(1.0f, fabsf(expected));

            if (difference > tolerance) {
                fprintf(
                    stderr,
                    "validation failed: ast=%zu row=%zu actual=%.9g expected=%.9g\n",
                    ast_idx,
                    row,
                    output[row],
                    expected);
                return 0;
            }
        }
    }
    return 1;
}

static void
secant_native_run_materialize(const float* input, size_t rows, size_t asts, int workers, float* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(workers > 1) num_threads(workers)
#endif
    for (size_t ast_idx = 0u; ast_idx < asts; ++ast_idx) {
        secant_native_materialize_one(ast_idx, input, rows, output + ast_idx * rows);
    }
}

static void
secant_native_run_sse(const float* input, const float* target, size_t rows, size_t asts, int workers, float* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(workers > 1) num_threads(workers)
#endif
    for (size_t ast_idx = 0u; ast_idx < asts; ++ast_idx) {
        output[ast_idx] = secant_native_sse_one(ast_idx, input, target, rows);
    }
}

static double
secant_native_checksum(const float* output, size_t output_count, size_t rows, int materialize) {
    double checksum = 0.0;

    if (materialize) {
        const size_t asts = output_count / rows;
        for (size_t ast_idx = 0u; ast_idx < asts; ++ast_idx) {
            checksum += output[ast_idx * rows] + output[(ast_idx + 1u) * rows - 1u];
        }
    } else {
        for (size_t idx = 0u; idx < output_count; ++idx) {
            checksum += output[idx];
        }
    }
    return checksum;
}

static void
secant_native_compiler_barrier(float* output) {
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("" : : "r"(output) : "memory");
#else
    (void)output;
#endif
}

static SecantNativeTiming
secant_native_time(
    const float* input,
    const float* target,
    size_t rows,
    size_t asts,
    int workers,
    size_t warmups,
    size_t min_repeats,
    double min_seconds,
    int materialize,
    float* output
) {
    double times[SECANT_NATIVE_MAX_REPEATS];
    size_t repeats = 0u;
    double total_seconds = 0.0;
    double checksum = 0.0;
    double estimated_iterations;
    double probe_seconds;
    double seconds_per_sample;
    size_t iterations_per_sample;
    size_t warmup_idx;

    for (warmup_idx = 0u; warmup_idx < warmups; ++warmup_idx) {
        if (materialize) {
            secant_native_run_materialize(input, rows, asts, workers, output);
        } else {
            secant_native_run_sse(input, target, rows, asts, workers, output);
        }
    }

    {
        const double begin = secant_native_seconds();

        if (materialize) {
            secant_native_run_materialize(input, rows, asts, workers, output);
        } else {
            secant_native_run_sse(input, target, rows, asts, workers, output);
        }
        secant_native_compiler_barrier(output);
        probe_seconds = secant_native_seconds() - begin;
    }
    if (probe_seconds <= 0.0) {
        const SecantNativeTiming failed = {0.0, 0.0, 0u, 0u, 0.0};

        return failed;
    }
    seconds_per_sample = min_seconds / (double)min_repeats;
    estimated_iterations = ceil(seconds_per_sample / probe_seconds * 1.1);
    if (estimated_iterations > (double)SIZE_MAX) {
        const SecantNativeTiming failed = {0.0, 0.0, 0u, 0u, 0.0};

        return failed;
    }
    iterations_per_sample = (size_t)estimated_iterations;
    if (iterations_per_sample == 0u) {
        iterations_per_sample = 1u;
    }

    while ((repeats < min_repeats || total_seconds < min_seconds) &&
           repeats < SECANT_NATIVE_MAX_REPEATS) {
        const double begin = secant_native_seconds();
        double seconds;
        size_t iteration;

        for (iteration = 0u; iteration < iterations_per_sample; ++iteration) {
            if (materialize) {
                secant_native_run_materialize(input, rows, asts, workers, output);
            } else {
                secant_native_run_sse(input, target, rows, asts, workers, output);
            }
            secant_native_compiler_barrier(output);
        }
        seconds = secant_native_seconds() - begin;
        checksum += secant_native_checksum(
            output,
            materialize ? asts * rows : asts,
            rows,
            materialize);
        times[repeats++] = seconds / (double)iterations_per_sample;
        total_seconds += seconds;
    }
    qsort(times, repeats, sizeof(times[0]), secant_native_compare_double);
    {
        const SecantNativeTiming timing = {
            times[0],
            times[repeats / 2u],
            repeats,
            iterations_per_sample,
            checksum,
        };
        return timing;
    }
}

static int
secant_native_parse_size(const char* text, size_t* value_ret) {
    char* end = NULL;
    unsigned long long value;

    if (text == NULL || value_ret == NULL || text[0] == '\0') {
        return 0;
    }
    value = strtoull(text, &end, 10);
    if (end == text || *end != '\0' || value == 0u || value > SIZE_MAX) {
        return 0;
    }
    *value_ret = (size_t)value;
    return 1;
}

static int
secant_native_parse_double(const char* text, double* value_ret) {
    char* end = NULL;
    double value;

    if (text == NULL || value_ret == NULL || text[0] == '\0') {
        return 0;
    }
    value = strtod(text, &end);
    if (end == text || *end != '\0' || value <= 0.0) {
        return 0;
    }
    *value_ret = value;
    return 1;
}

static void
secant_native_usage(const char* executable) {
    fprintf(
        stderr,
        "usage: %s --shape materialize|sse --rows N --asts N --workers N "
        "[--warmups N] [--min-repeats N] [--min-seconds S] [--seed N]\n",
        executable);
}

int
main(int argc, char** argv) {
    const char* shape = NULL;
    size_t rows = 0u;
    size_t asts = 0u;
    size_t workers_value = 0u;
    size_t warmups = 2u;
    size_t min_repeats = 5u;
    double min_seconds = 0.25;
    size_t seed_value = 1u;
    float* input = NULL;
    float* target = NULL;
    float* output = NULL;
    float* validation_output = NULL;
    size_t input_count;
    size_t output_count;
    SecantNativeTiming timing;
    int materialize;
    int arg_idx;

    for (arg_idx = 1; arg_idx < argc; ++arg_idx) {
        if (strcmp(argv[arg_idx], "--shape") == 0 && arg_idx + 1 < argc) {
            shape = argv[++arg_idx];
        } else if (strcmp(argv[arg_idx], "--rows") == 0 && arg_idx + 1 < argc) {
            if (!secant_native_parse_size(argv[++arg_idx], &rows)) {
                secant_native_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[arg_idx], "--asts") == 0 && arg_idx + 1 < argc) {
            if (!secant_native_parse_size(argv[++arg_idx], &asts)) {
                secant_native_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[arg_idx], "--workers") == 0 && arg_idx + 1 < argc) {
            if (!secant_native_parse_size(argv[++arg_idx], &workers_value)) {
                secant_native_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[arg_idx], "--warmups") == 0 && arg_idx + 1 < argc) {
            if (!secant_native_parse_size(argv[++arg_idx], &warmups)) {
                secant_native_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[arg_idx], "--min-repeats") == 0 && arg_idx + 1 < argc) {
            if (!secant_native_parse_size(argv[++arg_idx], &min_repeats)) {
                secant_native_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[arg_idx], "--min-seconds") == 0 && arg_idx + 1 < argc) {
            if (!secant_native_parse_double(argv[++arg_idx], &min_seconds)) {
                secant_native_usage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[arg_idx], "--seed") == 0 && arg_idx + 1 < argc) {
            if (!secant_native_parse_size(argv[++arg_idx], &seed_value)) {
                secant_native_usage(argv[0]);
                return 1;
            }
        } else {
            secant_native_usage(argv[0]);
            return 1;
        }
    }
    if (shape == NULL || rows == 0u || asts == 0u || workers_value == 0u ||
        workers_value > INT32_MAX || min_repeats > SECANT_NATIVE_MAX_REPEATS ||
        seed_value > UINT32_MAX || rows > SIZE_MAX / SECANT_NATIVE_INPUTS ||
        asts > SIZE_MAX / rows) {
        secant_native_usage(argv[0]);
        return 1;
    }
    materialize = strcmp(shape, "materialize") == 0;
    if (!materialize && strcmp(shape, "sse") != 0) {
        secant_native_usage(argv[0]);
        return 1;
    }
#if defined(__GNUC__) || defined(__clang__)
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        fprintf(stderr, "AVX2 and FMA are required\n");
        return 1;
    }
#endif
#ifdef _OPENMP
    omp_set_dynamic(0);
#endif
    input_count = SECANT_NATIVE_INPUTS * rows;
    output_count = materialize ? asts * rows : asts;
    if (!secant_native_allocate(input_count, &input) ||
        !secant_native_allocate(rows, &target) ||
        !secant_native_allocate(output_count, &output) ||
        !secant_native_allocate(rows, &validation_output)) {
        fprintf(stderr, "allocation failed\n");
        free(validation_output);
        free(output);
        free(target);
        free(input);
        return 1;
    }
    for (size_t column = 0u; column < SECANT_NATIVE_INPUTS; ++column) {
        for (size_t row = 0u; row < rows; ++row) {
            input[column * rows + row] = secant_native_input_value(
                column,
                row,
                (uint32_t)seed_value);
        }
    }
    secant_native_materialize_one(
        SECANT_PORTABLE_ALU_TARGET_CASE,
        input,
        rows,
        target);
    if (!secant_native_validate(input, rows, validation_output)) {
        free(validation_output);
        free(output);
        free(target);
        free(input);
        return 1;
    }
    memset(output, 0, output_count * sizeof(float));
    timing = secant_native_time(
        input,
        target,
        rows,
        asts,
        (int)workers_value,
        warmups,
        min_repeats,
        min_seconds,
        materialize,
        output);
    if (timing.repeats == 0u) {
        fprintf(stderr, "timing failed\n");
        free(validation_output);
        free(output);
        free(target);
        free(input);
        return 1;
    }

    puts(
        "system,backend,shape,ast_mode,corpus,corpus_hash,seed,rows,asts,workers,execution_mode,"
        "best_seconds,median_seconds,asts_per_second,row_evals_per_second,repeats,checksum,notes");
    printf(
        "native_avx2,cpu,%s,alu,%s,%s,%zu,%zu,%zu,%zu,avx2_fma_openmp,"
        "%.9f,%.9f,%.3f,%.3f,%zu,%.9g,"
        "portable_corpus_generated_code;timed_iterations_per_sample=%zu\n",
        shape,
        SECANT_PORTABLE_ALU_CORPUS_NAME,
        SECANT_PORTABLE_ALU_CORPUS_HASH,
        seed_value,
        rows,
        asts,
        workers_value,
        timing.best_seconds,
        timing.median_seconds,
        (double)asts / timing.median_seconds,
        (double)asts * (double)rows / timing.median_seconds,
        timing.repeats,
        timing.checksum,
        timing.iterations_per_sample);
    free(validation_output);
    free(output);
    free(target);
    free(input);
    return 0;
}
