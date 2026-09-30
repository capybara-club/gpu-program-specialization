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
#ifndef SECANT_SR_DATASET_H_INCLUDED
#define SECANT_SR_DATASET_H_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define SECANT_SR_DATASET_MAX_INPUTS 128u
#define SECANT_SR_DATASET_BINARY_VERSION 1u

typedef float (*SecantSRDatasetTargetFn)(const float* values);

typedef struct SecantSRDataset {
    const char* name;
    const char* expression;
    size_t num_inputs;
    float input_min[SECANT_SR_DATASET_MAX_INPUTS];
    float input_max[SECANT_SR_DATASET_MAX_INPUTS];
    SecantSRDatasetTargetFn target;
} SecantSRDataset;

const SecantSRDataset* secant_sr_dataset_find(const char* name);

void secant_sr_datasets_print(FILE* output);

double secant_sr_dataset_fill(
    const SecantSRDataset* dataset,
    uint64_t split_seed,
    float* input,
    float* target,
    size_t num_rows
);

int secant_sr_dataset_binary_info(
    const char* path,
    size_t* num_inputs_ret,
    size_t* num_train_rows_ret,
    size_t* num_validation_rows_ret
);

int secant_sr_dataset_binary_load(
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
);

double secant_sr_dataset_nguyen1_fill(float* input, float* target, size_t num_rows);

#endif /* SECANT_SR_DATASET_H_INCLUDED */
