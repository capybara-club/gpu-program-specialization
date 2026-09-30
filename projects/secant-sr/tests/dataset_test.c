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
#include "dataset.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define TEST_ROWS 37u
#define TEST_BINARY_PATH "secant_sr_dataset_test.secsr"

static int
dataset_test_u32_write(FILE* output, uint32_t value) {
    const uint8_t bytes[] = {
        (uint8_t)(value >> 0u),
        (uint8_t)(value >> 8u),
        (uint8_t)(value >> 16u),
        (uint8_t)(value >> 24u)
    };

    return fwrite(bytes, sizeof(bytes), 1u, output) == 1u;
}

static int
dataset_test_u64_write(FILE* output, uint64_t value) {
    const uint8_t bytes[] = {
        (uint8_t)(value >> 0u),
        (uint8_t)(value >> 8u),
        (uint8_t)(value >> 16u),
        (uint8_t)(value >> 24u),
        (uint8_t)(value >> 32u),
        (uint8_t)(value >> 40u),
        (uint8_t)(value >> 48u),
        (uint8_t)(value >> 56u)
    };

    return fwrite(bytes, sizeof(bytes), 1u, output) == 1u;
}

static int
dataset_test_f32_write(FILE* output, float value) {
    uint32_t bits;

    memcpy(&bits, &value, sizeof(bits));
    return dataset_test_u32_write(output, bits);
}

static int
dataset_binary_test_write(void) {
    static const uint8_t magic[] = {'S', 'E', 'C', 'S', 'R', 'D', 'S', 0};
    static const float values[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        5.0f, 7.0f, 9.0f,
        7.0f, 8.0f,
        9.0f, 10.0f,
        16.0f, 18.0f
    };
    FILE* output = fopen(TEST_BINARY_PATH, "wb");
    size_t value_idx;
    int success = output != NULL;

    if (success) {
        success = fwrite(magic, sizeof(magic), 1u, output) == 1u &&
            dataset_test_u32_write(output, SECANT_SR_DATASET_BINARY_VERSION) &&
            dataset_test_u32_write(output, 2u) &&
            dataset_test_u64_write(output, 3u) &&
            dataset_test_u64_write(output, 2u);
    }
    for (value_idx = 0u; success && value_idx < sizeof(values) / sizeof(values[0]); ++value_idx) {
        success = dataset_test_f32_write(output, values[value_idx]);
    }
    if (output != NULL) {
        success = fclose(output) == 0 && success;
    }
    return success;
}

static int
dataset_binary_test_trailing_byte_append(void) {
    FILE* output = fopen(TEST_BINARY_PATH, "ab");
    int success = output != NULL;

    if (success) {
        success = fputc(0, output) != EOF;
    }
    if (output != NULL) {
        success = fclose(output) == 0 && success;
    }
    return success;
}

static int
dataset_binary_test_run(void) {
    static const float expected_train_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    static const float expected_train_target[] = {5.0f, 7.0f, 9.0f};
    static const float expected_validation_input[] = {7.0f, 8.0f, 9.0f, 10.0f};
    static const float expected_validation_target[] = {16.0f, 18.0f};
    float train_input[6];
    float train_target[3];
    float validation_input[4];
    float validation_target[2];
    size_t num_inputs;
    size_t num_train_rows;
    size_t num_validation_rows;
    double train_ssd;
    double validation_ssd;

    if (!dataset_binary_test_write() ||
        !secant_sr_dataset_binary_info(
            TEST_BINARY_PATH,
            &num_inputs,
            &num_train_rows,
            &num_validation_rows) ||
        num_inputs != 2u || num_train_rows != 3u || num_validation_rows != 2u ||
        !secant_sr_dataset_binary_load(
            TEST_BINARY_PATH,
            2u,
            train_input,
            train_target,
            3u,
            validation_input,
            validation_target,
            2u,
            &train_ssd,
            &validation_ssd) ||
        memcmp(train_input, expected_train_input, sizeof(train_input)) != 0 ||
        memcmp(train_target, expected_train_target, sizeof(train_target)) != 0 ||
        memcmp(validation_input, expected_validation_input, sizeof(validation_input)) != 0 ||
        memcmp(validation_target, expected_validation_target, sizeof(validation_target)) != 0 ||
        fabs(train_ssd - 8.0) > 1.0e-12 || fabs(validation_ssd - 2.0) > 1.0e-12) {
        return 1;
    }
    if (!dataset_binary_test_trailing_byte_append() ||
        secant_sr_dataset_binary_load(
            TEST_BINARY_PATH,
            2u,
            train_input,
            train_target,
            3u,
            validation_input,
            validation_target,
            2u,
            &train_ssd,
            &validation_ssd)) {
        return 1;
    }
    return 0;
}

static int
dataset_test_run(void) {
    static const char* names[] = {
        "nguyen1",
        "nguyen5",
        "rational2",
        "distance2",
        "interaction3",
        "oscillator2",
        "feynman_I_12_1",
        "feynman_I_14_3",
        "feynman_I_14_4",
        "feynman_I_16_6",
        "feynman_I_18_12",
        "feynman_I_47_23",
        "feynman_II_8_31",
        "feynman_II_38_14",
        "feynman_III_15_12",
        "feynman_III_17_37"
    };
    float input_a[SECANT_SR_DATASET_MAX_INPUTS * TEST_ROWS];
    float input_b[SECANT_SR_DATASET_MAX_INPUTS * TEST_ROWS];
    float input_holdout[SECANT_SR_DATASET_MAX_INPUTS * TEST_ROWS];
    float target_a[TEST_ROWS];
    float target_b[TEST_ROWS];
    float target_holdout[TEST_ROWS];
    size_t dataset_idx;

    if (secant_sr_dataset_find("missing") != NULL) {
        return 1;
    }
    for (dataset_idx = 0u; dataset_idx < sizeof(names) / sizeof(names[0]); ++dataset_idx) {
        const SecantSRDataset* dataset = secant_sr_dataset_find(names[dataset_idx]);
        const uint64_t train_seed = UINT64_C(0x747261696e) + dataset_idx;
        const uint64_t holdout_seed = UINT64_C(0x76616c6964617465) + dataset_idx;
        double train_ssd;
        double repeated_ssd;
        double holdout_ssd;
        size_t value_idx;
        int split_differs = 0;

        if (dataset == NULL || dataset->num_inputs == 0u || dataset->num_inputs > SECANT_SR_DATASET_MAX_INPUTS) {
            return 1;
        }
        train_ssd = secant_sr_dataset_fill(dataset, train_seed, input_a, target_a, TEST_ROWS);
        repeated_ssd = secant_sr_dataset_fill(dataset, train_seed, input_b, target_b, TEST_ROWS);
        holdout_ssd = secant_sr_dataset_fill(dataset, holdout_seed, input_holdout, target_holdout, TEST_ROWS);
        if (!(train_ssd > 0.0) || train_ssd != repeated_ssd || !(holdout_ssd > 0.0) ||
            memcmp(input_a, input_b, dataset->num_inputs * TEST_ROWS * sizeof(*input_a)) != 0 ||
            memcmp(target_a, target_b, sizeof(target_a)) != 0) {
            fprintf(stderr, "dataset determinism failed: %s\n", dataset->name);
            return 1;
        }
        for (value_idx = 0u; value_idx < dataset->num_inputs * TEST_ROWS; ++value_idx) {
            if (!isfinite(input_a[value_idx]) || !isfinite(input_holdout[value_idx])) {
                return 1;
            }
            split_differs |= input_a[value_idx] != input_holdout[value_idx];
        }
        for (value_idx = 0u; value_idx < TEST_ROWS; ++value_idx) {
            if (!isfinite(target_a[value_idx]) || !isfinite(target_holdout[value_idx])) {
                return 1;
            }
        }
        if (!split_differs) {
            fprintf(stderr, "train and holdout splits match: %s\n", dataset->name);
            return 1;
        }
    }
    return 0;
}

int
main(void) {
    const int result = dataset_test_run() || dataset_binary_test_run();

    (void)remove(TEST_BINARY_PATH);
    return result;
}
