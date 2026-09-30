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
#ifndef SECANT_SR_TOGGLE_H
#define SECANT_SR_TOGGLE_H
#include "secant.h"
#ifdef __cplusplus
extern "C" {
#endif
#define SECANT_SR_VERSION "0.3.4"
#define SECANT_SR_MAX_NODES 63u
#define SECANT_SR_MAX_BITS 16u
/* Search nodes are postorder. A selector is one indivisible GP leaf, whose
 * alternatives lower to direct Secant leaves followed by toggle2/toggle4. */
typedef enum SRLeafKind { SR_COLUMN, SR_COEFFICIENT, SR_LITERAL, SR_FITTED_COEFFICIENT } SRLeafKind;
typedef struct SRLeaf {
  SRLeafKind kind;
  uint32_t slot;
  /* For fitted coefficients, slot is a host parameter identity independent of
   * runtime bank width, and value is its center. Other kinds use column/bank slots. */
  float value;
} SRLeaf;
typedef struct SRNode {
  uint8_t op;      /* 0 = leaf group; arithmetic uses Secant opcodes. */
  uint8_t choices; /* leaf: 1, 2 or 4; arithmetic: 0 */
  uint8_t low_bit, high_bit;
  SRLeaf leaf[4];
} SRNode;
typedef struct SRConfig {
  size_t population, num_inputs, num_constants, num_banks;
  uint32_t toggle_bits, max_nodes, max_depth, initial_depth;
  uint32_t tournament, elites;
  uint64_t seed;
  double parsimony, crossover_probability, mutation_probability;
  double toggle_probability, four_way_probability, coefficient_probability;
  /* Optional host-only variation policies; zero preserves the original GP.
   * Fractions apply within mutation/crossover respectively, not all offspring. */
  double toggle_mutation_probability, leaf_mix_probability;
  double power_mutation_probability; /* Shared-subtree square/cube proposals. */
  uint32_t align_crossover_bits;
  /* A nonempty list of Secant arithmetic opcodes, copied at creation. */
  const uint8_t *operators;
  size_t num_operators;
} SRConfig;
/* SSE is over training rows; configuration = bank * 2^bits + permutation.
 * Invalid configurations (nonfinite or negative SSE) never win. All invalid:
 * sse=+infinity, configuration=UINT64_MAX, valid_configurations=0. Ties choose
 * the lowest configuration index, identically on CPU and GPU. */
typedef struct SRScore {
  float sse;
  uint32_t reserved;
  uint64_t configuration, valid_configurations;
} SRScore;
typedef struct SRModel {
  const SRNode *nodes;
  size_t num_nodes;
  SRScore score;
  size_t generation;
  uint64_t fingerprint;
} SRModel;
typedef struct SRProgress {
  size_t generation, structures;
  uint64_t configurations, finite_configurations, rejected_variations;
  uint64_t aligned_crossovers, toggle_mutations, leaf_mixes, reused_toggle_bits;
  uint64_t power_mutations;
  double best_mse, best_r2;
} SRProgress;
typedef struct SRSearchImpl *SRSearch;
SRConfig secant_sr_config_default(void);
/* Allocations are bounded by config and occur once at create. Bank values are
 * copied, immutable, [bank][slot], and shared across every AST and generation.
 * No LM, settings tables or GPU state belongs to this search object. */
SecantResult secant_sr_create(const SRConfig *, const float *banks,
                              size_t bank_elements, SRSearch *);
void secant_sr_destroy(SRSearch);
/* One ask -> successful tell -> advance. Failed scoring leaves the generation
 * awaiting tell; callers must discard partial backend output. */
SecantResult secant_sr_ask(SRSearch, SecantAstProgramSet *);
SecantResult secant_sr_tell(SRSearch, const SRScore *, size_t count,
                            size_t rows, double target_ssd);
SecantResult secant_sr_advance(SRSearch);
SecantResult secant_sr_best(SRSearch, SRModel *);
SRProgress secant_sr_progress(SRSearch);
/* Inspect an evaluated candidate, or accept a verified refinement while retaining
 * its toggles and adjustable coefficient identities. No structural replacement. */
SecantResult secant_sr_candidate(SRSearch, size_t index, SRModel *);
SecantResult secant_sr_accept_refined(SRSearch, size_t index, const SRModel *);

const SRConfig *secant_sr_config(SRSearch);
const float *secant_sr_banks(SRSearch);
/* Explicit seeds are optional; no known solution is injected by the search. */
SecantResult secant_sr_seed(SRSearch, size_t candidate, const SRNode *,
                            size_t count);
SecantResult secant_sr_program_write(const SRConfig *, const SRNode *,
                                     size_t count, uint8_t *output,
                                     size_t capacity, size_t *needed);
SecantResult secant_sr_resolve(const SRConfig *, const SRNode *, size_t count,
                               const float *banks, size_t bank_elements,
                               uint64_t configuration, uint8_t *output,
                               size_t capacity, size_t *needed);
SecantResult secant_sr_format(const SRConfig *, const SRNode *, size_t count,
                              const float *banks, size_t bank_elements,
                              uint64_t configuration, char *output,
                              size_t capacity, size_t *needed);
/* Stable, deterministic host bank generation. This is not Philox. Normal
 * mode uses mean + scale*N(0,1); uniform mode uses low + (high-low)*U[0,1). */
SecantResult secant_sr_bank_generate(float *, size_t banks, size_t slots,
                                     uint64_t seed, int normal,
                                     float low_or_mean, float high_or_scale);
/* CPU scorer uses the same new Secant API, with caller-provided grid scratch.
 * Reduces only after a successful Secant call. No silent CPU fallback from
 * CUDA. */
SecantResult secant_sr_cpu_score(const SRConfig *, const SecantAstProgramSet *,
                                 const float *input, const float *target,
                                 size_t rows, const float *banks, float *grid,
                                 size_t grid_elements, SRScore *);
SecantResult secant_sr_reduce(const float *grid, size_t asts,
                              size_t configurations, SRScore *);
/* Numerical score audit, separate from AST/configuration identity checks.
 * Compare RMSE on the scale max(1, target RMS, either residual RMSE). Near a
 * good fit, relative SSE alone magnifies small f32 prediction differences.
 * No scores are changed; the caller chooses tolerance and retains both. */
typedef struct SRScoreAgreement {
  double rmse_gap, scale, relative_rmse_gap;
  int accepted;
} SRScoreAgreement;
SecantResult secant_sr_score_agreement(double gpu_sse, double cpu_sse,
                                       double target_energy, size_t rows,
                                       double tolerance, SRScoreAgreement *);
#ifdef __cplusplus
}
#endif
#endif
