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
SecantResult secant_sr_score_agreement(double gpu_sse, double cpu_sse,
                                       double target_energy, size_t rows,
                                       double tolerance,
                                       SRScoreAgreement *out) {
  double gpu, cpu;
  if (!out || !rows || !isfinite(gpu_sse) || gpu_sse < 0 ||
      !isfinite(cpu_sse) || cpu_sse < 0 || !isfinite(target_energy) ||
      target_energy < 0 || !isfinite(tolerance) || tolerance < 0)
    return SECANT_ERROR_INVALID_VALUE;
  gpu = sqrt(gpu_sse / rows);
  cpu = sqrt(cpu_sse / rows);
  out->rmse_gap = fabs(gpu - cpu);
  out->scale = fmax(1., fmax(sqrt(target_energy / rows), fmax(gpu, cpu)));
  out->relative_rmse_gap = out->rmse_gap / out->scale;
  out->accepted = out->relative_rmse_gap <= tolerance;
  return SECANT_SUCCESS;
}
SecantResult secant_sr_reduce(const float *grid, size_t count, size_t configs,
                              SRScore *out) {
  size_t a, k, elements;
  if (!grid || !out || !count || !configs)
    return SECANT_ERROR_INVALID_VALUE;
  if (!sr_mul(count, configs, &elements))
    return SECANT_ERROR_OVERFLOW;
  for (a = 0; a < count; ++a) {
    SRScore v = {INFINITY, 0, UINT64_MAX, 0};
    for (k = 0; k < configs; ++k) {
      float x = grid[a * configs + k];
      if (!isfinite(x) || x < 0)
        continue;
      ++v.valid_configurations;
      if (x < v.sse) {
        v.sse = x;
        v.configuration = k;
      }
    }
    out[a] = v;
  }
  return SECANT_SUCCESS;
}
SecantResult secant_sr_cpu_score(const SRConfig *c,
                                 const SecantAstProgramSet *asts,
                                 const float *input, const float *target,
                                 size_t rows, const float *banks, float *grid,
                                 size_t elements, SRScore *out) {
  SecantCpuToggleSSERun run = secant_cpu_toggle_sse_run_init();
  size_t configs, n, inputs, bankn;
  SecantResult r = sr_validate_config(c);
  if (r != SECANT_SUCCESS)
    return r;
  if (!asts || !out || !rows)
    return SECANT_ERROR_INVALID_VALUE;
  if (!sr_mul(c->num_banks, (size_t)1 << c->toggle_bits, &configs) ||
      !sr_mul(asts->asts.count, configs, &n) ||
      !sr_mul(c->num_inputs, rows, &inputs) ||
      !sr_mul(c->num_constants, c->num_banks, &bankn))
    return SECANT_ERROR_OVERFLOW;
  if (elements < n)
    return SECANT_ERROR_INSUFFICIENT_BUFFER;
  run.programs = *asts;
  run.num_inputs = c->num_inputs;
  run.num_constants = c->num_constants;
  run.num_targets = 1;
  run.num_rows = rows;
  run.num_banks = c->num_banks;
  run.toggle_bits = c->toggle_bits;
  run.input = (SecantConstHostMatrixF32){input, inputs, rows};
  run.targets = (SecantConstHostMatrixF32){target, rows, rows};
  run.constants =
      (SecantConstHostConstantBanks){banks, bankn, c->num_constants};
  run.output = (SecantHostMatrixF32){grid, elements, configs};
  r = secant_cpu_run_toggle_sse(&run);
  if (r == SECANT_SUCCESS)
    r = secant_sr_reduce(grid, asts->asts.count, configs, out);
  return r;
}
