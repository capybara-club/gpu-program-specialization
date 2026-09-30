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
#include "s_runner_internal.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static SecantResult s_run_unpack(SecantCubinRunner runner, const SecantCubinRunHeader *h, SRun *r) {
    const SecantCubinPlan *p = runner->plan;
    size_t required, count, width;
    SecantResult result;
    if (!h || h->struct_size < sizeof(*h) || h->flags)
        return SECANT_ERROR_INVALID_VALUE;
    if (h->version != SECANT_CUBIN_RUN_VERSION_3)
        return SECANT_ERROR_UNSUPPORTED_VERSION;
    if ((unsigned)h->shape != (unsigned)p->shape)
        return SECANT_ERROR_UNSUPPORTED_SHAPE;
    memset(r, 0, sizeof(*r));
#define UNPACK(Type, output_member)                                                                          \
    do {                                                                                                     \
        const Type *v = (const Type *)h;                                                                     \
        if (h->struct_size < sizeof(*v))                                                                     \
            return SECANT_ERROR_INVALID_VALUE;                                                               \
        r->programs = v->programs;                                                                           \
        r->input = v->input;                                                                                 \
        r->output = v->output_member;                                                                        \
        r->rows = v->num_rows;                                                                               \
    } while (0)
#define TARGETS(Type)                                                                                        \
    do {                                                                                                     \
        const Type *v = (const Type *)h;                                                                     \
        r->targets = v->targets;                                                                             \
        r->targets_count = v->num_targets;                                                                   \
    } while (0)
    switch (h->shape) {
    case SECANT_KERNEL_SHAPE_STATIC_MATERIALIZE_F32:
        UNPACK(SecantCubinMaterializeRun, output);
        break;
    case SECANT_KERNEL_SHAPE_STATIC_SSE_F32:
        UNPACK(SecantCubinSSERun, output);
        TARGETS(SecantCubinSSERun);
        break;
    case SECANT_KERNEL_SHAPE_STATIC_AFFINE_STATS_F32:
        UNPACK(SecantCubinAffineStatsRun, ast_stats);
        TARGETS(SecantCubinAffineStatsRun);
        break;
    case SECANT_KERNEL_SHAPE_STATIC_GRAM_STATS_F32:
        UNPACK(SecantCubinGramStatsRun, statistics);
        TARGETS(SecantCubinGramStatsRun);
        break;
    case SECANT_KERNEL_SHAPE_TOGGLE_SSE_F32: {
        const SecantCubinToggleSSERun *v = (const SecantCubinToggleSSERun *)h;
        UNPACK(SecantCubinToggleSSERun, output);
        TARGETS(SecantCubinToggleSSERun);
        r->banks = v->constants;
        r->toggle_bits = v->toggle_bits;
        result =
            s_toggle_layout(r->programs.asts.count, v->num_banks, v->toggle_bits, p->num_constant_registers,
                            r->banks.bank_stride, &r->configurations, &r->bank_span);
        if (result != SECANT_SUCCESS)
            return result;
        if (r->targets_count != p->num_targets)
            return SECANT_ERROR_INVALID_VALUE;
        break;
    }
    default:
        return SECANT_ERROR_UNSUPPORTED_SHAPE;
    }
#undef UNPACK
#undef TARGETS
    count = r->programs.asts.count;
    if (!count || !r->programs.asts.items || !r->rows ||
        r->programs.routines.count > SECANT_AST_MAX_ROUTINES ||
        (r->programs.routines.count && !r->programs.routines.items) || r->targets_count > p->num_targets ||
        (p->shape != _SECANT_CUBIN_SHAPE_MATERIALIZE && !r->targets_count) ||
        (p->num_input_registers > p->num_constant_registers && r->input.leading_dimension < r->rows) ||
        (r->targets_count && r->targets.leading_dimension < r->rows))
        return SECANT_ERROR_INVALID_VALUE;
    if (p->shape == _SECANT_CUBIN_SHAPE_MATERIALIZE)
        width = r->rows;
    else if (p->shape == _SECANT_CUBIN_SHAPE_SSE)
        width = r->targets_count;
    else if (p->shape == _SECANT_CUBIN_SHAPE_AFFINE_STATS)
        width = 2 + r->targets_count;
    else if (p->shape == _SECANT_CUBIN_SHAPE_GRAM_STATS) {
        width = p->asts_per_kernel * (1 + p->asts_per_kernel + r->targets_count);
        count = count / p->asts_per_kernel + (count % p->asts_per_kernel != 0);
    } else {
        width = r->configurations;
        if (!s_size_mul(count, r->targets_count, &count))
            return SECANT_ERROR_OVERFLOW;
    }
    r->output_count = count;
    r->output_width = width;
    if (r->output.leading_dimension > SIZE_MAX / sizeof(float))
        return SECANT_ERROR_OVERFLOW;
    if (r->output.leading_dimension < width)
        return SECANT_ERROR_INVALID_VALUE;
    if (!s_span_size(p->num_input_registers - p->num_constant_registers, r->input.leading_dimension, r->rows,
                     &r->input_span) ||
        !s_span_size(r->targets_count, r->targets.leading_dimension, r->rows, &r->target_span) ||
        !s_span_size(count, r->output.leading_dimension, width, &r->output_span))
        return SECANT_ERROR_OVERFLOW;
    if (r->input.num_elements < r->input_span || r->targets.num_elements < r->target_span ||
        r->banks.num_elements < r->bank_span || r->output.num_elements < r->output_span)
        return SECANT_ERROR_INSUFFICIENT_BUFFER;
    required = p->shape == _SECANT_CUBIN_SHAPE_MATERIALIZE ? 256 : p->tile_rows;
    if ((r->rows - 1) / required >= INT_MAX)
        return SECANT_ERROR_OVERFLOW;
    return SECANT_SUCCESS;
}

static SecantResult s_program_preflight(const SecantAstInstruction *program, const SecantCubinPlan *p,
                                        unsigned bits) {
    size_t i, offset = 0, direct = 0;
    if (!program)
        return SECANT_ERROR_BAD_PROGRAM;
    for (i = 0; i < SECANT_AST_MAX_PROGRAM_INSTRUCTIONS; ++i) {
        const uint8_t *ins = program + offset;
        unsigned op = ins[0];
        size_t size = secant_ast_instruction_size_get(ins);
        if (!size)
            return SECANT_ERROR_BAD_PROGRAM;
        if (!secant_internal_ast_instruction_is_f32((SecantAstInstructionType)op))
            return SECANT_ERROR_UNSUPPORTED_OP;
        if (op == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32)
            return SECANT_SUCCESS;
        if (op == SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32) {
            if (ins[1] >= p->num_input_registers - p->num_constant_registers)
                return SECANT_ERROR_BAD_PROGRAM;
            ++direct;
        } else if (op == SECANT_AST_INSTRUCTION_TYPE_BANK_CONSTANT_F32 ||
                   op == SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32) {
            if (p->shape != _SECANT_CUBIN_SHAPE_TOGGLE_SSE || ins[1] >= p->num_constant_registers)
                return SECANT_ERROR_BAD_PROGRAM;
            if (op == SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32 &&
                (!isfinite(secant_ast_affine_bank_scale_get(ins)) ||
                 !isfinite(secant_ast_affine_bank_offset_get(ins))))
                return SECANT_ERROR_BAD_PROGRAM;
            ++direct;
        } else if (op == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32)
            ++direct;
        else {
            if (op == SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32 ||
                op == SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32) {
                if (p->shape != _SECANT_CUBIN_SHAPE_TOGGLE_SSE || ins[1] >= bits ||
                    direct < (op == SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32 ? 2u : 4u) ||
                    (op == SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32 && (ins[2] >= bits || ins[1] == ins[2])))
                    return SECANT_ERROR_BAD_PROGRAM;
            }
            direct = 0;
        }
        offset += size;
    }
    return SECANT_ERROR_BAD_PROGRAM;
}
static SecantResult s_range(uintptr_t address, size_t count, uintptr_t *end) {
    size_t bytes;
    if (!count) {
        *end = address;
        return SECANT_SUCCESS;
    }
    if (!address || address % sizeof(float))
        return SECANT_ERROR_INVALID_VALUE;
    if (!s_size_mul(count, sizeof(float), &bytes) || bytes > UINTPTR_MAX - address)
        return SECANT_ERROR_OVERFLOW;
    *end = address + bytes;
    return SECANT_SUCCESS;
}
static SecantResult s_runs_validate(SecantCubinRunner runner, const SRun *runs, size_t count) {
    size_t i, j, k;
    unsigned bits = 32;
    const SecantAstProgramSet *programs = &runs[0].programs;
    for (i = 0; i < count; ++i) {
        uintptr_t starts[4] = {runs[i].input.address, runs[i].targets.address, runs[i].banks.address,
                               runs[i].output.address};
        size_t sizes[4] = {runs[i].input_span, runs[i].target_span, runs[i].bank_span, runs[i].output_span};
        uintptr_t ends[4];
        if (runs[i].programs.asts.items != programs->asts.items ||
            runs[i].programs.asts.count != programs->asts.count ||
            runs[i].programs.routines.items != programs->routines.items ||
            runs[i].programs.routines.count != programs->routines.count)
            return SECANT_ERROR_INVALID_VALUE;
        if (runs[i].toggle_bits < bits)
            bits = runs[i].toggle_bits;
        for (j = 0; j < 4; ++j) {
            SecantResult result = s_range(starts[j], sizes[j], ends + j);
            if (result != SECANT_SUCCESS)
                return result;
            for (k = 0; k < j; ++k)
                if (sizes[j] && sizes[k] && starts[j] < ends[k] && starts[k] < ends[j])
                    return SECANT_ERROR_INVALID_VALUE;
        }
        for (j = 0; j < count; ++j) {
            uintptr_t output_end;
            SecantResult result;
            if (i == j)
                continue;
            result = s_range(runs[j].output.address, runs[j].output_span, &output_end);
            if (result != SECANT_SUCCESS)
                return result;
            for (k = 0; k < 4; ++k)
                if (sizes[k] && runs[j].output.address < ends[k] && starts[k] < output_end)
                    return SECANT_ERROR_INVALID_VALUE;
        }
    }
    for (i = 0; i < programs->routines.count + programs->asts.count; ++i) {
        const uint8_t *program = i < programs->routines.count
                                     ? programs->routines.items[i]
                                     : programs->asts.items[i - programs->routines.count];
        SecantResult result = s_program_preflight(program, runner->plan, bits);
        if (result != SECANT_SUCCESS)
            return result;
    }
    return SECANT_SUCCESS;
}

SecantResult secant_cubin_runner_create(const SecantCubinPlan *plan, const void *cubin, size_t bytes,
                                        const SecantCubinRunnerOptions *options, SecantCubinRunner *out) {
    SecantCubinRunner r;
    SecantResult result;
    if (!out)
        return SECANT_ERROR_INVALID_VALUE;
    *out = NULL;
    if (!plan || !cubin || bytes != plan->cubin_size || !options || options->struct_size < sizeof(*options))
        return SECANT_ERROR_INVALID_VALUE;
    if (options->version != SECANT_CUBIN_RUNNER_OPTIONS_VERSION_1)
        return SECANT_ERROR_UNSUPPORTED_VERSION;
    if (options->flags || options->reserved || !options->num_workers || !options->num_streams ||
        options->num_workers > 256 || options->num_streams > 1024)
        return SECANT_ERROR_INVALID_VALUE;
    r = (SecantCubinRunner)calloc(1, sizeof(*r));
    if (!r)
        return SECANT_ERROR_ALLOCATION_FAILED;
    r->plan = plan;
    r->worker_count = options->num_workers;
    r->stream_count = options->num_streams;
    r->slot_count = r->worker_count + 2;
    if (!s_size_mul(plan->num_kernels, plan->asts_per_kernel, &r->module_capacity)) {
        free(r);
        return SECANT_ERROR_OVERFLOW;
    }
    r->template_cubin = (unsigned char *)malloc(bytes);
    if (!r->template_cubin) {
        free(r);
        return SECANT_ERROR_ALLOCATION_FAILED;
    }
    memcpy(r->template_cubin, cubin, bytes);
    result = s_pipeline_create(r);
    if (result != SECANT_SUCCESS) {
        SecantResult cleanup = s_pipeline_destroy(r);
        if (cleanup != SECANT_SUCCESS) {
            /* Preserve ownership when cleanup itself fails. Destroy-only handle. */
            *out = r;
            return cleanup;
        }
        return result;
    }
    *out = r;
    return SECANT_SUCCESS;
}
SecantResult secant_cubin_runner_run_batch(SecantCubinRunner r, const SecantCubinRunHeader *const *headers,
                                           size_t count, SecantRunnerStats *stats) {
    SRun *runs;
    size_t i;
    SecantResult result = SECANT_SUCCESS;
    SecantRunnerStats measured = secant_runner_stats_init();
    CUcontext context;
    double begin = s_runner_seconds();
    if (stats) {
        if (stats->struct_size < sizeof(*stats) || stats->flags || stats->reserved)
            return SECANT_ERROR_INVALID_VALUE;
        if (stats->version != SECANT_RUNNER_STATS_VERSION_1)
            return SECANT_ERROR_UNSUPPORTED_VERSION;
        measured.struct_size = stats->struct_size;
        *stats = measured;
    }
    if (!r || !headers || !count || count > SIZE_MAX / sizeof(*runs))
        return SECANT_ERROR_INVALID_VALUE;
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context != r->context)
        return SECANT_ERROR_INVALID_STATE;
    pthread_mutex_lock(&r->mutex);
    if (r->active || r->failed || r->shutdown) {
        pthread_mutex_unlock(&r->mutex);
        return SECANT_ERROR_INVALID_STATE;
    }
    r->active = 1;
    pthread_mutex_unlock(&r->mutex);
    runs = (SRun *)calloc(count, sizeof(*runs));
    if (!runs)
        result = SECANT_ERROR_ALLOCATION_FAILED;
    for (i = 0; result == SECANT_SUCCESS && i < count; ++i)
        result = s_run_unpack(r, headers[i], runs + i);
    if (result == SECANT_SUCCESS)
        result = s_runs_validate(r, runs, count);
    if (result == SECANT_SUCCESS)
        result = s_pipeline_execute(r, runs, count, &measured);
    free(runs);
    pthread_mutex_lock(&r->mutex);
    r->active = 0;
    pthread_mutex_unlock(&r->mutex);
    if (measured.num_modules)
        measured.total_seconds = s_runner_seconds() - begin;
    if (stats)
        *stats = measured;
    return result;
}
SecantResult secant_cubin_runner_run(SecantCubinRunner r, const SecantCubinRunHeader *run,
                                     SecantRunnerStats *stats) {
    return secant_cubin_runner_run_batch(r, &run, 1, stats);
}
SecantResult secant_cubin_runner_destroy(SecantCubinRunner r) {
    CUcontext context;
    if (!r)
        return SECANT_SUCCESS;
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context != r->context)
        return SECANT_ERROR_INVALID_STATE;
    if (r->mutex_ready) {
        pthread_mutex_lock(&r->mutex);
        if (r->active) {
            pthread_mutex_unlock(&r->mutex);
            return SECANT_ERROR_INVALID_STATE;
        }
        pthread_mutex_unlock(&r->mutex);
    }
    return s_pipeline_destroy(r);
}
