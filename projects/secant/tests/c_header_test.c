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
#include "secant_instructions.h"
#include <stddef.h>
#define ASSERT_LAYOUT(type) typedef char first_##type[(offsetof(type, header) == 0) ? 1 : -1]
ASSERT_LAYOUT(SecantCubinMaterializeRecipe);
ASSERT_LAYOUT(SecantCubinSSERecipe);
ASSERT_LAYOUT(SecantCubinAffineStatsRecipe);
ASSERT_LAYOUT(SecantCubinGramStatsRecipe);
ASSERT_LAYOUT(SecantCubinToggleSSERecipe);
ASSERT_LAYOUT(SecantCubinMaterializeRun);
ASSERT_LAYOUT(SecantCubinSSERun);
ASSERT_LAYOUT(SecantCubinAffineStatsRun);
ASSERT_LAYOUT(SecantCubinGramStatsRun);
ASSERT_LAYOUT(SecantCubinToggleSSERun);
ASSERT_LAYOUT(SecantCpuToggleSSERun);
int main(void) {
    SecantCubinToggleSSERecipe recipe = secant_cubin_toggle_sse_recipe_init();
    SecantCpuToggleSSERun cpu = secant_cpu_toggle_sse_run_init();
    SecantCubinToggleSSERun gpu = secant_cubin_toggle_sse_run_init();
    SecantCubinRunnerOptions options = secant_cubin_runner_options_init();
    SecantRunnerStats stats = secant_runner_stats_init();
    const SecantAstInstruction program[] = {secant_ast_encode_column_f32(1),
                                            secant_ast_encode_bank_constant_f32(0),
                                            secant_ast_encode_toggle2_f32(3), secant_ast_encode_return_f32};
    size_t source_size = 0;
    const char *source = secant_cuda_target_stats_f32_source_get(&source_size);
    return recipe.header.struct_size != sizeof(recipe) || cpu.header.struct_size != sizeof(cpu) ||
           gpu.header.struct_size != sizeof(gpu) ||
           recipe.header.shape != SECANT_KERNEL_SHAPE_TOGGLE_SSE_F32 ||
           cpu.header.shape != recipe.header.shape || gpu.header.shape != recipe.header.shape ||
           stats.struct_size != sizeof(stats) || options.struct_size != sizeof(options) ||
           !options.num_workers || !options.num_streams || sizeof(program) != 7 ||
           secant_ast_instruction_size_get(program + 4) != 2 || program[5] != 3 || !source || !source_size ||
           source[source_size - 1] != '\0' || secant_cpu_run_toggle_sse(NULL) != SECANT_ERROR_INVALID_VALUE ||
           secant_cpu_run_materialize(NULL) != SECANT_ERROR_INVALID_VALUE;
}
