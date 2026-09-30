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
#include "internal.h"

SecantSRResult
secant_sr_search_cpu_evaluate(
    SecantSRSearch search,
    const SecantAstInstruction* const* routines,
    size_t num_routines,
    const float* input,
    size_t input_num_elements,
    size_t input_leading_dimension,
    const float* target,
    size_t target_num_elements,
    size_t num_rows,
    double target_sum_squared_deviation,
    float* sse_scratch,
    size_t sse_scratch_count
) {
    const SecantSRPopulation* population;
    SecantCpuSSERun run = secant_cpu_sse_run_init();
    SecantResult secant_result;

    if (search == NULL || target == NULL || target_num_elements < num_rows || sse_scratch == NULL) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INVALID_VALUE);
    }
    population = &search->populations[search->current_population];
    if (sse_scratch_count < population->count) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_INSUFFICIENT_BUFFER);
    }
    memset(sse_scratch, 0, population->count * sizeof(*sse_scratch));
    run.programs.routines.items = routines;
    run.programs.routines.count = num_routines;
    run.programs.asts.items = population->asts;
    run.programs.asts.count = population->count;
    run.num_inputs = search->config.num_inputs;
    run.num_targets = 1u;
    run.input = (SecantConstHostMatrixF32){input, input_num_elements, input_leading_dimension};
    run.targets = (SecantConstHostMatrixF32){target, target_num_elements, num_rows};
    run.num_rows = num_rows;
    run.output = (SecantHostMatrixF32){sse_scratch, sse_scratch_count, 1u};
    secant_result = secant_cpu_run_sse(&run);
    if (secant_result != SECANT_SUCCESS) {
        _SECANT_SR_ERROR_RET(SECANT_SR_ERROR_SECANT);
    }
    return secant_sr_search_scores_set(
        search,
        sse_scratch,
        population->count,
        num_rows,
        target_sum_squared_deviation);
}
