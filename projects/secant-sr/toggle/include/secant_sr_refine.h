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
#ifndef SECANT_SR_REFINE_H
#define SECANT_SR_REFINE_H
#include "secant_sr.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Bounded coefficient fitting, independent of GP and CUDA ownership. A model
 * entry is one start. Duplicate structures with different initial configurations
 * are permitted. All entries share a Philox bank crossed with native toggles. */
typedef enum SRRefineParameters {
  SR_REFINE_ALL_PARAMETERS = 0,
  SR_REFINE_ACTIVE_BLOCK = 1
} SRRefineParameters;
typedef struct SRRefineOptions {
  size_t capacity, trials;
  uint64_t seed;
  float initial_scale, scale_learning_rate, failure_decay, minimum_scale, maximum_scale;
  SRRefineParameters parameters;
} SRRefineOptions;
typedef struct SRRefineStats {
  size_t rounds, accepted;
  uint64_t configurations, finite_configurations, bank_seed;
} SRRefineStats;
typedef struct SRRefinerImpl *SRRefiner;
SRRefineOptions secant_sr_refine_options_default(void);
SecantResult secant_sr_refiner_create(const SRConfig *, const SRRefineOptions *, SRRefiner *);
void secant_sr_refiner_destroy(SRRefiner);
/* Raw references sharing a slot are one parameter. Fitted references sharing
 * slot AND exact value bits are one parameter; crossover may introduce more
 * parameters than bank capacity. ALL rejects this; ACTIVE_BLOCK fits bounded
 * subsets. Neither policy merges distinct parameters to fit the bank. */
SecantResult secant_sr_refine_parameter_count(const SRConfig *, const SRModel *, size_t *);
SecantResult secant_sr_refine_active_parameter_count(const SRConfig *, const SRModel *, size_t *);
/* ALL rejects models wider than the bank. ACTIVE_BLOCK cycles through bounded
 * blocks used by the incumbent binding. Other centers stay fixed; every toggle
 * remains available and tied parameters stay tied. Host parameter identities
 * are independent of runtime bank slots. */
/* Seed copies nodes and coefficients. Source banks are needed only for raw
 * coefficient references. Existing fitted coefficients use their stored value.
 * sequence identifies the batch/generation and changes its deterministic RNG bank.
 * A failed seed invalidates the previous batch. No allocations after create. */
SecantResult secant_sr_refiner_seed(SRRefiner, const SRModel *, size_t count,
                                    const float *source_banks, size_t elements, uint64_t sequence);
/* ask -> score all returned ASTs with returned banks -> tell. Borrowed buffers
 * remain valid through tell. Do not change input rows/targets between rounds.
 * Trial zero is the unchanged center. Nonzero trials use uniform [-1,1) Philox.
 * Bank rows and permutation indices use Secant's ordinary Cartesian layout. */
SecantResult secant_sr_refiner_ask(SRRefiner, SecantAstProgramSet *, const float **banks, size_t *elements);
SecantResult secant_sr_refiner_tell(SRRefiner, const SRScore *, size_t count);
/* Results retain every toggle and adjustable parameter identity. Their fitted
 * values are encoded as affine-bank leaves with zero scale until refined again.
 * configuration is the winning permutation at bank zero, not an index into an
 * expired trial bank. Copy the model before reseeding/destroying the refiner. */
SecantResult secant_sr_refiner_model(SRRefiner, size_t index, SRModel *);
SRRefineStats secant_sr_refiner_stats(SRRefiner);
/* Replayable shared normalized bank; row zero is reserved for the incumbent. */
SecantResult secant_sr_refinement_bank(float *, size_t trials, size_t slots,
                                       uint64_t seed, uint32_t round);
#ifdef __cplusplus
}
#endif
#endif
