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
#define _POSIX_C_SOURCE 200809L

#include "ssid_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static pthread_key_t ssid_error_key;
static pthread_once_t ssid_error_key_once = PTHREAD_ONCE_INIT;

static void ssid_error_buffer_destroy(void *value) {
    free(value);
}

static void ssid_error_key_create(void) {
    pthread_key_create(&ssid_error_key, ssid_error_buffer_destroy);
}

static char *ssid_error_buffer_get(void) {
    char *buffer;
    pthread_once(&ssid_error_key_once, ssid_error_key_create);
    buffer = (char *)pthread_getspecific(ssid_error_key);
    if (buffer == NULL) {
        buffer = (char *)calloc(512, 1);
        if (buffer != NULL) pthread_setspecific(ssid_error_key, buffer);
    }
    return buffer;
}

void ssid_set_error(const char *format, ...) {
    char *buffer = ssid_error_buffer_get();
    va_list arguments;
    if (buffer == NULL) return;
    va_start(arguments, format);
    vsnprintf(buffer, 512, format, arguments);
    va_end(arguments);
}

const char *ssid_last_error(void) {
    const char *buffer = ssid_error_buffer_get();
    return buffer == NULL ? "could not allocate the thread-local error buffer" : buffer;
}

const char *ssid_status_string(int status) {
    switch (status) {
        case SSID_OK: return "ok";
        case SSID_INVALID_ARGUMENT: return "invalid argument";
        case SSID_OUT_OF_MEMORY: return "out of memory";
        case SSID_OUT_OF_RANGE: return "out of range";
        case SSID_UNSUPPORTED: return "unsupported";
        case SSID_QUEUE_FULL: return "queue full";
        case SSID_CUDA_ERROR: return "CUDA error";
        case SSID_INTERNAL_ERROR: return "internal error";
        default: return "unknown status";
    }
}

double ssid_monotonic_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static char *ssid_copy_string(const char *value) {
    size_t length;
    char *result;
    if (value == NULL) return NULL;
    length = strlen(value);
    result = (char *)malloc(length + 1);
    if (result != NULL) memcpy(result, value, length + 1);
    return result;
}

static void *ssid_copy_array(const void *source, size_t count, size_t item_size) {
    void *result;
    if (count == 0) return NULL;
    if (source == NULL || item_size > SIZE_MAX / count) return NULL;
    result = malloc(count * item_size);
    if (result != NULL) memcpy(result, source, count * item_size);
    return result;
}

static int ssid_offset_fits(uint64_t offset, uint64_t byte_count, size_t image_size) {
    return offset <= image_size && byte_count <= image_size - offset;
}

static void ssid_kernel_release(ssid_kernel_plan *kernel) {
    if (kernel == NULL) return;
    free(kernel->name);
    free(kernel->dispatch_file_offsets);
    free(kernel->dispatch_instruction_words);
    free(kernel->cleanup_file_offsets);
    free(kernel->target_table_file_offsets);
    free(kernel->register_count_file_offsets);
    free(kernel->register_count_header_file_offsets);
    free(kernel->input_registers);
    free(kernel->output_registers);
    free(kernel->available_registers);
    memset(kernel, 0, sizeof(*kernel));
}

void ssid_template_destroy(ssid_template *template_value) {
    uint32_t index;
    if (template_value == NULL) return;
    for (index = 0; index < template_value->kernel_count; ++index) ssid_kernel_release(&template_value->kernels[index]);
    free(template_value->kernels);
    free(template_value->site_input_offsets);
    free(template_value->site_input_counts);
    free(template_value->cubin);
    free(template_value);
}

size_t ssid_template_cubin_size(const ssid_template *template_value) {
    return template_value == NULL ? 0 : template_value->cubin_byte_count;
}

static int ssid_copy_kernel(const ssid_template_desc *desc, const ssid_packed_kernel_desc *source, ssid_kernel_plan *target, uint32_t expected_base) {
    uint64_t arena_bytes = (uint64_t)source->arena_instruction_count * 16u;
    uint32_t index;
    if (source->name == NULL || source->name[0] == '\0' || source->genome_base != expected_base || source->genome_capacity == 0 || source->genomes_per_cta == 0 || source->genomes_per_cta > source->genome_capacity) {
        ssid_set_error("invalid packed kernel identity or genome range");
        return SSID_INVALID_ARGUMENT;
    }
    if (source->input_count == 0 || source->output_count != desc->site_count || source->target_table_count != source->genome_capacity || source->dispatch_instruction_count == 0) {
        ssid_set_error("packed kernel dimensions do not match the template");
        return SSID_INVALID_ARGUMENT;
    }
    if (source->dispatch_file_offsets == NULL || source->dispatch_instruction_words == NULL || source->target_table_file_offsets == NULL || source->input_registers == NULL || source->output_registers == NULL || (source->available_register_count != 0 && source->available_registers == NULL) || (source->register_count_offset_count != 0 && source->register_count_file_offsets == NULL) || (source->register_count_header_offset_count != 0 && source->register_count_header_file_offsets == NULL)) {
        ssid_set_error("packed kernel descriptor contains a missing array");
        return SSID_INVALID_ARGUMENT;
    }
    if (!ssid_offset_fits(source->entry_file_offset, 16, desc->cubin_byte_count) || !ssid_offset_fits(source->arena_start_file_offset, arena_bytes, desc->cubin_byte_count) || !ssid_offset_fits(source->continuation_file_offset, 0, desc->cubin_byte_count) || source->entry_file_offset >= source->arena_start_file_offset) {
        ssid_set_error("packed kernel instruction range is outside the CUBIN");
        return SSID_OUT_OF_RANGE;
    }
    for (index = 0; index < source->dispatch_instruction_count; ++index) {
        if (!ssid_offset_fits(source->dispatch_file_offsets[index], 16, desc->cubin_byte_count)) {
            ssid_set_error("dispatch instruction offset is outside the CUBIN");
            return SSID_OUT_OF_RANGE;
        }
    }
    for (index = 0; index < source->cleanup_count; ++index) {
        if (!ssid_offset_fits(source->cleanup_file_offsets[index], 16, desc->cubin_byte_count)) {
            ssid_set_error("cleanup instruction offset is outside the CUBIN");
            return SSID_OUT_OF_RANGE;
        }
    }
    for (index = 0; index < source->target_table_count; ++index) {
        if (!ssid_offset_fits(source->target_table_file_offsets[index], 4, desc->cubin_byte_count)) {
            ssid_set_error("branch-target offset is outside the CUBIN");
            return SSID_OUT_OF_RANGE;
        }
    }
    for (index = 0; index < source->register_count_offset_count; ++index) {
        if (!ssid_offset_fits(source->register_count_file_offsets[index], 4, desc->cubin_byte_count)) {
            ssid_set_error("register-count offset is outside the CUBIN");
            return SSID_OUT_OF_RANGE;
        }
    }
    for (index = 0; index < source->register_count_header_offset_count; ++index) {
        if (!ssid_offset_fits(source->register_count_header_file_offsets[index], 1, desc->cubin_byte_count)) {
            ssid_set_error("register-count section-header offset is outside the CUBIN");
            return SSID_OUT_OF_RANGE;
        }
    }
    target->name = ssid_copy_string(source->name);
    target->dispatch_file_offsets = (uint64_t *)ssid_copy_array(source->dispatch_file_offsets, source->dispatch_instruction_count, sizeof(uint64_t));
    target->dispatch_instruction_words = (uint64_t *)ssid_copy_array(source->dispatch_instruction_words, (size_t)source->dispatch_instruction_count * 2, sizeof(uint64_t));
    target->cleanup_file_offsets = (uint64_t *)ssid_copy_array(source->cleanup_file_offsets, source->cleanup_count, sizeof(uint64_t));
    target->target_table_file_offsets = (uint64_t *)ssid_copy_array(source->target_table_file_offsets, source->target_table_count, sizeof(uint64_t));
    target->register_count_file_offsets = (uint64_t *)ssid_copy_array(source->register_count_file_offsets, source->register_count_offset_count, sizeof(uint64_t));
    target->register_count_header_file_offsets = (uint64_t *)ssid_copy_array(source->register_count_header_file_offsets, source->register_count_header_offset_count, sizeof(uint64_t));
    target->input_registers = (uint8_t *)ssid_copy_array(source->input_registers, source->input_count, sizeof(uint8_t));
    target->output_registers = (uint8_t *)ssid_copy_array(source->output_registers, source->output_count, sizeof(uint8_t));
    target->available_registers = (uint8_t *)ssid_copy_array(source->available_registers, source->available_register_count, sizeof(uint8_t));
    if (target->name == NULL || target->dispatch_file_offsets == NULL || target->dispatch_instruction_words == NULL || target->target_table_file_offsets == NULL || target->input_registers == NULL || target->output_registers == NULL || (source->available_register_count != 0 && target->available_registers == NULL) || (source->cleanup_count != 0 && target->cleanup_file_offsets == NULL) || (source->register_count_offset_count != 0 && target->register_count_file_offsets == NULL) || (source->register_count_header_offset_count != 0 && target->register_count_header_file_offsets == NULL)) {
        ssid_set_error("could not copy packed kernel metadata");
        ssid_kernel_release(target);
        return SSID_OUT_OF_MEMORY;
    }
    target->genome_base = source->genome_base;
    target->genome_capacity = source->genome_capacity;
    target->genomes_per_cta = source->genomes_per_cta;
    target->register_count = source->register_count;
    target->input_count = source->input_count;
    target->output_count = source->output_count;
    target->function_file_offset = source->function_file_offset;
    target->entry_file_offset = source->entry_file_offset;
    target->arena_start_file_offset = source->arena_start_file_offset;
    target->arena_instruction_count = source->arena_instruction_count;
    target->incoming_wait_mask = source->incoming_wait_mask;
    target->continuation_file_offset = source->continuation_file_offset;
    target->dispatch_instruction_count = source->dispatch_instruction_count;
    target->cleanup_count = source->cleanup_count;
    target->target_table_count = source->target_table_count;
    target->register_count_offset_count = source->register_count_offset_count;
    target->register_count_header_offset_count = source->register_count_header_offset_count;
    target->available_register_count = source->available_register_count;
    return SSID_OK;
}

int ssid_template_create(const ssid_template_desc *desc, ssid_template **result) {
    ssid_template *template_value;
    uint32_t index;
    uint32_t expected_base = 0;
    if (result != NULL) *result = NULL;
    if (desc == NULL || result == NULL || desc->abi_version != SSID_TEMPLATE_ABI_VERSION || desc->cubin == NULL || desc->cubin_byte_count == 0 || desc->kernels == NULL || desc->kernel_count == 0 || desc->site_count == 0 || desc->site_input_offsets == NULL || desc->site_input_counts == NULL || desc->settings_mode > SSID_SETTINGS_HASHED_INCUMBENT) {
        ssid_set_error("invalid template descriptor");
        return SSID_INVALID_ARGUMENT;
    }
    if (desc->architecture != 89 && desc->architecture != 90 && desc->architecture != 120) {
        ssid_set_error("C99 SASS writer supports sm_89, sm_90, and sm_120, not sm_%u", desc->architecture);
        return SSID_UNSUPPORTED;
    }
    template_value = (ssid_template *)calloc(1, sizeof(*template_value));
    if (template_value == NULL) return SSID_OUT_OF_MEMORY;
    template_value->architecture = desc->architecture;
    template_value->cubin_byte_count = desc->cubin_byte_count;
    template_value->kernel_count = desc->kernel_count;
    template_value->site_count = desc->site_count;
    template_value->settings_mode = desc->settings_mode;
    template_value->cubin = (uint8_t *)ssid_copy_array(desc->cubin, desc->cubin_byte_count, 1);
    template_value->kernels = (ssid_kernel_plan *)calloc(desc->kernel_count, sizeof(ssid_kernel_plan));
    template_value->site_input_offsets = (uint32_t *)ssid_copy_array(desc->site_input_offsets, desc->site_count, sizeof(uint32_t));
    template_value->site_input_counts = (uint32_t *)ssid_copy_array(desc->site_input_counts, desc->site_count, sizeof(uint32_t));
    if (template_value->cubin == NULL || template_value->kernels == NULL || template_value->site_input_offsets == NULL || template_value->site_input_counts == NULL) {
        ssid_set_error("could not copy template data");
        ssid_template_destroy(template_value);
        return SSID_OUT_OF_MEMORY;
    }
    for (index = 0; index < desc->kernel_count; ++index) {
        int status = ssid_copy_kernel(desc, &desc->kernels[index], &template_value->kernels[index], expected_base);
        if (status != SSID_OK) {
            ssid_template_destroy(template_value);
            return status;
        }
        expected_base += desc->kernels[index].genome_capacity;
    }
    *result = template_value;
    return SSID_OK;
}

int ssid_validate_batch(const ssid_template *template_value, const ssid_genome_batch *batch) {
    uint32_t genome_index;
    uint32_t capacity;
    if (template_value == NULL || batch == NULL || batch->genomes == NULL || batch->asts == NULL || batch->program_bytes == NULL || batch->genome_count == 0 || batch->ast_count == 0 || batch->program_byte_count == 0) {
        ssid_set_error("invalid or empty genome batch");
        return SSID_INVALID_ARGUMENT;
    }
    capacity = template_value->kernels[template_value->kernel_count - 1].genome_base + template_value->kernels[template_value->kernel_count - 1].genome_capacity;
    if (batch->genome_count > capacity) {
        ssid_set_error("%u genomes exceed module capacity %u", batch->genome_count, capacity);
        return SSID_OUT_OF_RANGE;
    }
    for (genome_index = 0; genome_index < batch->genome_count; ++genome_index) {
        const ssid_genome_desc *genome = &batch->genomes[genome_index];
        uint8_t seen[256] = {0};
        uint32_t local_index;
        if (template_value->site_count > 255 || genome->ast_count != template_value->site_count || genome->first_ast > batch->ast_count || genome->ast_count > batch->ast_count - genome->first_ast) {
            ssid_set_error("genome %u does not contain exactly one AST per site", genome_index);
            return SSID_INVALID_ARGUMENT;
        }
        for (local_index = 0; local_index < genome->ast_count; ++local_index) {
            const ssid_ast_desc *ast = &batch->asts[genome->first_ast + local_index];
            if (ast->site_index >= template_value->site_count || seen[ast->site_index]) {
                ssid_set_error("genome %u has a duplicate or invalid site index", genome_index);
                return SSID_INVALID_ARGUMENT;
            }
            seen[ast->site_index] = 1;
            if (ast->byte_count == 0 || ast->byte_offset > batch->program_byte_count || ast->byte_count > batch->program_byte_count - ast->byte_offset) {
                ssid_set_error("genome %u AST %u is outside the program byte stream", genome_index, local_index);
                return SSID_OUT_OF_RANGE;
            }
        }
    }
    return SSID_OK;
}
