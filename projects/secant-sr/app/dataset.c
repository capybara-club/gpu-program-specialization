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
#include <stdint.h>
#include <string.h>

static const uint8_t secant_sr_dataset_binary_magic[8] = {'S', 'E', 'C', 'S', 'R', 'D', 'S', 0};

static float
secant_sr_target_nguyen1(const float* x) {
    return x[0] * x[0] * x[0] + x[0] * x[0] + x[0];
}

static float
secant_sr_target_nguyen5(const float* x) {
    return sinf(x[0] * x[0]) * cosf(x[0]) - 1.0f;
}

static float
secant_sr_target_rational2(const float* x) {
    return x[0] / (1.0f + x[1] * x[1]);
}

static float
secant_sr_target_distance2(const float* x) {
    return sqrtf(x[0] * x[0] + x[1] * x[1]);
}

static float
secant_sr_target_interaction3(const float* x) {
    return x[0] * x[1] + x[2] * x[2] - x[2];
}

static float
secant_sr_target_oscillator2(const float* x) {
    return sinf(x[0]) + cosf(x[1]);
}

static float
secant_sr_target_feynman_i_12_1(const float* x) {
    return x[0] * x[1];
}

static float
secant_sr_target_feynman_i_14_3(const float* x) {
    return x[0] * x[1] * x[2];
}

static float
secant_sr_target_feynman_i_14_4(const float* x) {
    return 0.5f * x[0] * x[1] * x[1];
}

static float
secant_sr_target_feynman_i_16_6(const float* x) {
    return (x[2] + x[1]) / (1.0f + x[2] * x[1] / (x[0] * x[0]));
}

static float
secant_sr_target_feynman_i_18_12(const float* x) {
    return x[0] * x[1] * sinf(x[2]);
}

static float
secant_sr_target_feynman_i_47_23(const float* x) {
    return sqrtf(x[0] * x[1] / x[2]);
}

static float
secant_sr_target_feynman_ii_8_31(const float* x) {
    return 0.5f * x[0] * x[1] * x[1];
}

static float
secant_sr_target_feynman_ii_38_14(const float* x) {
    return x[0] / (2.0f * (1.0f + x[1]));
}

static float
secant_sr_target_feynman_iii_15_12(const float* x) {
    return 2.0f * x[0] * (1.0f - cosf(x[1] * x[2]));
}

static float
secant_sr_target_feynman_iii_17_37(const float* x) {
    return x[0] * (1.0f + x[1] * cosf(x[2]));
}

static const SecantSRDataset secant_sr_datasets[] = {
    {
        "nguyen1",
        "x0^3 + x0^2 + x0",
        1u,
        {-1.0f, 0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, 0.0f},
        secant_sr_target_nguyen1
    },
    {
        "nguyen5",
        "sin(x0^2) * cos(x0) - 1",
        1u,
        {-2.0f, 0.0f, 0.0f, 0.0f},
        {2.0f, 0.0f, 0.0f, 0.0f},
        secant_sr_target_nguyen5
    },
    {
        "rational2",
        "x0 / (1 + x1^2)",
        2u,
        {-2.0f, -2.0f, 0.0f, 0.0f},
        {2.0f, 2.0f, 0.0f, 0.0f},
        secant_sr_target_rational2
    },
    {
        "distance2",
        "sqrt(x0^2 + x1^2)",
        2u,
        {-2.0f, -2.0f, 0.0f, 0.0f},
        {2.0f, 2.0f, 0.0f, 0.0f},
        secant_sr_target_distance2
    },
    {
        "interaction3",
        "x0*x1 + x2^2 - x2",
        3u,
        {-2.0f, -2.0f, -2.0f, 0.0f},
        {2.0f, 2.0f, 2.0f, 0.0f},
        secant_sr_target_interaction3
    },
    {
        "oscillator2",
        "sin(x0) + cos(x1)",
        2u,
        {-3.14159265f, -3.14159265f, 0.0f, 0.0f},
        {3.14159265f, 3.14159265f, 0.0f, 0.0f},
        secant_sr_target_oscillator2
    },
    {
        "feynman_I_12_1",
        "x0*x1",
        2u,
        {1.0f, 1.0f, 0.0f, 0.0f},
        {5.0f, 5.0f, 0.0f, 0.0f},
        secant_sr_target_feynman_i_12_1
    },
    {
        "feynman_I_14_3",
        "x0*x1*x2",
        3u,
        {1.0f, 1.0f, 1.0f, 0.0f},
        {5.0f, 5.0f, 5.0f, 0.0f},
        secant_sr_target_feynman_i_14_3
    },
    {
        "feynman_I_14_4",
        "0.5*x0*x1^2",
        2u,
        {1.0f, 1.0f, 0.0f, 0.0f},
        {5.0f, 5.0f, 0.0f, 0.0f},
        secant_sr_target_feynman_i_14_4
    },
    {
        "feynman_I_16_6",
        "(x2+x1)/(1+x2*x1/x0^2)",
        3u,
        {1.0f, 1.0f, 1.0f, 0.0f},
        {5.0f, 5.0f, 5.0f, 0.0f},
        secant_sr_target_feynman_i_16_6
    },
    {
        "feynman_I_18_12",
        "x0*x1*sin(x2)",
        3u,
        {1.0f, 1.0f, 0.0f, 0.0f},
        {5.0f, 5.0f, 5.0f, 0.0f},
        secant_sr_target_feynman_i_18_12
    },
    {
        "feynman_I_47_23",
        "sqrt(x0*x1/x2)",
        3u,
        {1.0f, 1.0f, 1.0f, 0.0f},
        {5.0f, 5.0f, 5.0f, 0.0f},
        secant_sr_target_feynman_i_47_23
    },
    {
        "feynman_II_8_31",
        "0.5*x0*x1^2",
        2u,
        {1.0f, 1.0f, 0.0f, 0.0f},
        {5.0f, 5.0f, 0.0f, 0.0f},
        secant_sr_target_feynman_ii_8_31
    },
    {
        "feynman_II_38_14",
        "x0/(2*(1+x1))",
        2u,
        {1.0f, 1.0f, 0.0f, 0.0f},
        {5.0f, 5.0f, 0.0f, 0.0f},
        secant_sr_target_feynman_ii_38_14
    },
    {
        "feynman_III_15_12",
        "2*x0*(1-cos(x1*x2))",
        3u,
        {1.0f, 1.0f, 1.0f, 0.0f},
        {5.0f, 5.0f, 5.0f, 0.0f},
        secant_sr_target_feynman_iii_15_12
    },
    {
        "feynman_III_17_37",
        "x0*(1+x1*cos(x2))",
        3u,
        {1.0f, 1.0f, 1.0f, 0.0f},
        {5.0f, 5.0f, 5.0f, 0.0f},
        secant_sr_target_feynman_iii_17_37
    }
};

const SecantSRDataset*
secant_sr_dataset_find(const char* name) {
    size_t dataset_idx;

    if (name == NULL) {
        return NULL;
    }
    for (dataset_idx = 0u; dataset_idx < sizeof(secant_sr_datasets) / sizeof(secant_sr_datasets[0]);
         ++dataset_idx) {
        if (strcmp(secant_sr_datasets[dataset_idx].name, name) == 0) {
            return secant_sr_datasets + dataset_idx;
        }
    }
    return NULL;
}

void
secant_sr_datasets_print(FILE* output) {
    size_t dataset_idx;

    if (output == NULL) {
        return;
    }
    for (dataset_idx = 0u; dataset_idx < sizeof(secant_sr_datasets) / sizeof(secant_sr_datasets[0]);
         ++dataset_idx) {
        fprintf(output, "%s\tinputs=%zu\t%s\n",
            secant_sr_datasets[dataset_idx].name,
            secant_sr_datasets[dataset_idx].num_inputs,
            secant_sr_datasets[dataset_idx].expression);
    }
}

static uint64_t
secant_sr_dataset_hash(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}

static float
secant_sr_dataset_unit(uint64_t value) {
    return (float)((double)(secant_sr_dataset_hash(value) >> 11u) * (1.0 / 9007199254740992.0));
}

static double
secant_sr_target_sum_squared_deviation(const float* target, size_t num_rows) {
    double mean = 0.0;
    double sum_squared_deviation = 0.0;
    size_t row;

    for (row = 0u; row < num_rows; ++row) {
        mean += target[row];
    }
    mean /= (double)num_rows;
    for (row = 0u; row < num_rows; ++row) {
        const double difference = (double)target[row] - mean;

        sum_squared_deviation += difference * difference;
    }
    return sum_squared_deviation;
}

static int
secant_sr_dataset_binary_u32_read(FILE* input, uint32_t* value_ret) {
    uint8_t bytes[4];

    if (fread(bytes, sizeof(bytes), 1u, input) != 1u) {
        return 0;
    }
    *value_ret = (uint32_t)bytes[0] |
        ((uint32_t)bytes[1] << 8u) |
        ((uint32_t)bytes[2] << 16u) |
        ((uint32_t)bytes[3] << 24u);
    return 1;
}

static int
secant_sr_dataset_binary_u64_read(FILE* input, uint64_t* value_ret) {
    uint8_t bytes[8];

    if (fread(bytes, sizeof(bytes), 1u, input) != 1u) {
        return 0;
    }
    *value_ret = (uint64_t)bytes[0] |
        ((uint64_t)bytes[1] << 8u) |
        ((uint64_t)bytes[2] << 16u) |
        ((uint64_t)bytes[3] << 24u) |
        ((uint64_t)bytes[4] << 32u) |
        ((uint64_t)bytes[5] << 40u) |
        ((uint64_t)bytes[6] << 48u) |
        ((uint64_t)bytes[7] << 56u);
    return 1;
}

static int
secant_sr_dataset_binary_header_read(
    FILE* input,
    size_t* num_inputs_ret,
    size_t* num_train_rows_ret,
    size_t* num_validation_rows_ret
) {
    uint8_t magic[sizeof(secant_sr_dataset_binary_magic)];
    uint32_t version;
    uint32_t num_inputs;
    uint64_t num_train_rows;
    uint64_t num_validation_rows;

    if (fread(magic, sizeof(magic), 1u, input) != 1u ||
        memcmp(magic, secant_sr_dataset_binary_magic, sizeof(magic)) != 0 ||
        !secant_sr_dataset_binary_u32_read(input, &version) ||
        !secant_sr_dataset_binary_u32_read(input, &num_inputs) ||
        !secant_sr_dataset_binary_u64_read(input, &num_train_rows) ||
        !secant_sr_dataset_binary_u64_read(input, &num_validation_rows) ||
        version != SECANT_SR_DATASET_BINARY_VERSION ||
        num_inputs == 0u || num_inputs > SECANT_SR_DATASET_MAX_INPUTS ||
        num_train_rows < 2u || num_validation_rows < 2u ||
        num_train_rows > SIZE_MAX || num_validation_rows > SIZE_MAX ||
        num_train_rows > SIZE_MAX / sizeof(float) / num_inputs ||
        num_validation_rows > SIZE_MAX / sizeof(float) / num_inputs) {
        return 0;
    }
    *num_inputs_ret = num_inputs;
    *num_train_rows_ret = (size_t)num_train_rows;
    *num_validation_rows_ret = (size_t)num_validation_rows;
    return 1;
}

static int
secant_sr_dataset_binary_f32_read(FILE* input, float* values, size_t num_values) {
    size_t value_idx;

    for (value_idx = 0u; value_idx < num_values; ++value_idx) {
        uint32_t bits;

        if (!secant_sr_dataset_binary_u32_read(input, &bits)) {
            return 0;
        }
        memcpy(values + value_idx, &bits, sizeof(bits));
        if (!isfinite(values[value_idx])) {
            return 0;
        }
    }
    return 1;
}

static int
secant_sr_dataset_binary_end_read(FILE* input) {
    return fgetc(input) == EOF && feof(input) && !ferror(input);
}

int
secant_sr_dataset_binary_info(
    const char* path,
    size_t* num_inputs_ret,
    size_t* num_train_rows_ret,
    size_t* num_validation_rows_ret
) {
    FILE* input;
    int success;

    if (path == NULL || num_inputs_ret == NULL || num_train_rows_ret == NULL || num_validation_rows_ret == NULL) {
        return 0;
    }
    input = fopen(path, "rb");
    if (input == NULL) {
        return 0;
    }
    success = secant_sr_dataset_binary_header_read(
        input,
        num_inputs_ret,
        num_train_rows_ret,
        num_validation_rows_ret);
    fclose(input);
    return success;
}

int
secant_sr_dataset_binary_load(
    const char* path,
    size_t expected_num_inputs,
    float* train_input,
    float* train_target,
    size_t num_train_rows,
    float* validation_input,
    float* validation_target,
    size_t num_validation_rows,
    double* train_sum_squared_deviation_ret,
    double* validation_sum_squared_deviation_ret
) {
    FILE* input;
    size_t file_num_inputs;
    size_t file_num_train_rows;
    size_t file_num_validation_rows;
    int success = 0;

    if (path == NULL || train_input == NULL || train_target == NULL || validation_input == NULL ||
        validation_target == NULL || train_sum_squared_deviation_ret == NULL ||
        validation_sum_squared_deviation_ret == NULL || expected_num_inputs == 0u) {
        return 0;
    }
    input = fopen(path, "rb");
    if (input == NULL) {
        return 0;
    }
    if (secant_sr_dataset_binary_header_read(
            input,
            &file_num_inputs,
            &file_num_train_rows,
            &file_num_validation_rows) &&
        file_num_inputs == expected_num_inputs && file_num_train_rows == num_train_rows &&
        file_num_validation_rows == num_validation_rows &&
        secant_sr_dataset_binary_f32_read(input, train_input, expected_num_inputs * num_train_rows) &&
        secant_sr_dataset_binary_f32_read(input, train_target, num_train_rows) &&
        secant_sr_dataset_binary_f32_read(
            input,
            validation_input,
            expected_num_inputs * num_validation_rows) &&
        secant_sr_dataset_binary_f32_read(input, validation_target, num_validation_rows) &&
        secant_sr_dataset_binary_end_read(input)) {
        *train_sum_squared_deviation_ret = secant_sr_target_sum_squared_deviation(train_target, num_train_rows);
        *validation_sum_squared_deviation_ret =
            secant_sr_target_sum_squared_deviation(validation_target, num_validation_rows);
        success = isfinite(*train_sum_squared_deviation_ret) && *train_sum_squared_deviation_ret >= 0.0 &&
            isfinite(*validation_sum_squared_deviation_ret) && *validation_sum_squared_deviation_ret >= 0.0;
    }
    fclose(input);
    return success;
}

double
secant_sr_dataset_fill(
    const SecantSRDataset* dataset,
    uint64_t split_seed,
    float* input,
    float* target,
    size_t num_rows
) {
    size_t row;

    if (dataset == NULL || dataset->target == NULL || input == NULL || target == NULL || num_rows == 0u) {
        return 0.0;
    }
    for (row = 0u; row < num_rows; ++row) {
        float values[SECANT_SR_DATASET_MAX_INPUTS];
        size_t input_idx;

        for (input_idx = 0u; input_idx < dataset->num_inputs; ++input_idx) {
            const uint64_t coordinate = split_seed ^ ((uint64_t)row + 1u) * UINT64_C(0xd1b54a32d192ed03) ^
                ((uint64_t)input_idx + 1u) * UINT64_C(0x94d049bb133111eb);
            const float unit = secant_sr_dataset_unit(coordinate);

            values[input_idx] = dataset->input_min[input_idx] +
                unit * (dataset->input_max[input_idx] - dataset->input_min[input_idx]);
            input[input_idx * num_rows + row] = values[input_idx];
        }
        target[row] = dataset->target(values);
    }
    return secant_sr_target_sum_squared_deviation(target, num_rows);
}

double
secant_sr_dataset_nguyen1_fill(float* input, float* target, size_t num_rows) {
    size_t row;

    for (row = 0u; row < num_rows; ++row) {
        const float x = -1.0f + 2.0f * (float)row / (float)(num_rows - 1u);

        input[row] = x;
        target[row] = secant_sr_target_nguyen1(&x);
    }
    return secant_sr_target_sum_squared_deviation(target, num_rows);
}
