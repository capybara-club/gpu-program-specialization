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
#ifndef SECANT_SR_TOGGLE_INTERNAL_H
#define SECANT_SR_TOGGLE_INTERNAL_H
#include "secant_instructions.h"
#include "secant_sr.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define SR_PROGRAM_BYTES (SECANT_SR_MAX_NODES * 43u + 1u)
typedef struct SRCandidate {
  SRNode nodes[SECANT_SR_MAX_NODES];
  uint16_t count, depth;
  uint64_t fingerprint;
  SRScore score;
} SRCandidate;
struct SRSearchImpl {
  SRConfig config;
  uint8_t operators[32];
  SRCandidate *population[2], best;
  uint8_t *programs;
  const uint8_t **asts;
  float *banks;
  size_t generation, best_generation, rows;
  double target_ssd;
  int current, scored, has_best, encoded;
  uint64_t rng, rejected;
  SRProgress progress;
};
static inline int sr_mul(size_t a, size_t b, size_t *out) {
  if (a && b > SIZE_MAX / a)
    return 0;
  *out = a * b;
  return 1;
}
static inline uint64_t sr_random(uint64_t *s) {
  uint64_t z = (*s += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}
static inline double sr_unit(uint64_t *s) {
  return (sr_random(s) >> 11) * 0x1.0p-53;
}
unsigned sr_arity(uint8_t op);
SecantResult sr_validate_config(const SRConfig *);
SecantResult sr_validate(const SRConfig *, const SRNode *, size_t,
                         uint16_t *first, uint16_t *depth);
uint64_t sr_fingerprint(const SRNode *, size_t);
unsigned sr_selector_choice(const SRNode *, uint32_t permutation);
void sr_selector_rebind(SRNode *, unsigned low, unsigned high,
                        uint32_t old_permutation, uint32_t new_permutation);
/* Align a transplanted subtree; returns unavoidable external bit reuses. */
unsigned sr_align_subtree(SRNode *, size_t count, size_t start, size_t length,
                          unsigned bits, uint32_t donor_permutation, uint32_t recipient_permutation);
#endif
