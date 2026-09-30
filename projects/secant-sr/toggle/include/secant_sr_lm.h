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
#ifndef SECANT_SR_LM_H
#define SECANT_SR_LM_H
#include "secant_sr.h"
#ifdef __cplusplus
extern "C" {
#endif
/* GPU LM is a separate fitting service. It owns no GP policy. The
 * backend is a reusable register-based CUDA evaluator, not native-SASS
 * specialization. It groups work into 1/2/4/8-parameter shapes and rejects
 * compiled kernels that use local memory.
 * Every start retains its own coefficients and damping on the GPU. */
typedef struct SRLMOptions {
  size_t capacity, bindings, starts, iterations;
  unsigned parameters, threads;
  int device;
  uint64_t seed;
  float initial_scale;
  double initial_damping;
} SRLMOptions;
typedef struct SRLMStats {
  double seconds, device_seconds, setup_seconds, nvrtc_seconds;
  uint64_t states, statistics_evaluations, row_evaluations, accepted_steps;
  uint64_t invalid_evaluations;
  size_t bindings, blocked_bindings, inactive_bindings, iterations;
  /* Row evaluations count attempts; invalid domains can exit early.
   * blocks_by_shape is the resource-limited ceiling, not measured occupancy. */
  int registers, local_bytes, shared_bytes;
  int active_blocks_per_sm, multiprocessors;
  int registers_by_shape[4], blocks_by_shape[4];
  uint64_t states_by_shape[4];
} SRLMStats;
typedef struct SRLMCudaImpl *SRLMCuda;
SRLMOptions secant_sr_lm_options_default(void);
/* Copies training data once; CUDA primary-context ownership follows the scorer
 * convention. Destroy before releasing other owners of that context. A failed
 * create may return a non-NULL handle when cleanup must be retried. */
SecantResult secant_sr_lm_cuda_create(const SRConfig *, const SRLMOptions *,
    const float *input, const float *target, size_t rows, SRLMCuda *, SRLMStats *);
SecantResult secant_sr_lm_cuda_destroy(SRLMCuda);
/* Preserves all toggle alternatives, literals and parameter sharing. Binding 0
 * is the incumbent; remaining bindings visit unique assignments of used bits.
 * Start 0 preserves the center for every binding. At most eight active
 * parameters are fitted at once; wider bindings rotate explicit blocks with
 * sequence, never merge or silently drop parameters. Reported counts expose it.
 * max_seconds is a cooperative per-call limit checked between GPU iterations.
 * Results are proposals: rescore through the ordinary scorer before promotion.
 * No holdout data enters this API. Allocations occur only at create. */
SecantResult secant_sr_lm_cuda_fit(SRLMCuda, const SRModel *, size_t count,
    const float *banks, size_t elements, uint64_t sequence, double max_seconds,
    SecantAstProgramSet *, SRLMStats *);
/* Borrowed fitted genome; valid until next fit/destroy. Score is provisional
 * LM error; its winning toggle binding is encoded with bank zero. */
SecantResult secant_sr_lm_cuda_model(SRLMCuda, size_t index, SRModel *);
#ifdef __cplusplus
}
#endif
#endif
