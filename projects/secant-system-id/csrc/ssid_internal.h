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
#ifndef SSID_INTERNAL_H
#define SSID_INTERNAL_H

#include "secant_system_id.h"

#include <pthread.h>

typedef struct ssid_kernel_plan {
    char *name;
    uint32_t genome_base;
    uint32_t genome_capacity;
    uint32_t genomes_per_cta;
    uint32_t register_count;
    uint32_t input_count;
    uint32_t output_count;
    uint64_t function_file_offset;
    uint64_t entry_file_offset;
    uint64_t arena_start_file_offset;
    uint32_t arena_instruction_count;
    uint32_t incoming_wait_mask;
    uint64_t continuation_file_offset;
    uint64_t *dispatch_file_offsets;
    uint64_t *dispatch_instruction_words;
    uint32_t dispatch_instruction_count;
    uint64_t *cleanup_file_offsets;
    uint32_t cleanup_count;
    uint64_t *target_table_file_offsets;
    uint32_t target_table_count;
    uint64_t *register_count_file_offsets;
    uint32_t register_count_offset_count;
    uint64_t *register_count_header_file_offsets;
    uint32_t register_count_header_offset_count;
    uint8_t *input_registers;
    uint8_t *output_registers;
    uint8_t *available_registers;
    uint32_t available_register_count;
} ssid_kernel_plan;

struct ssid_template {
    uint32_t architecture;
    uint8_t *cubin;
    size_t cubin_byte_count;
    ssid_kernel_plan *kernels;
    uint32_t kernel_count;
    uint32_t *site_input_offsets;
    uint32_t *site_input_counts;
    uint32_t site_count;
    uint32_t settings_mode;
};

void ssid_set_error(const char *format, ...);
double ssid_monotonic_seconds(void);
int ssid_validate_batch(const ssid_template *template_value, const ssid_genome_batch *batch);

#endif
