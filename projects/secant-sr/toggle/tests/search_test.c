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
#include "secant_sr.h"
#include "../src/internal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "line %d: %s\n", __LINE__, #x);                          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define OK(x) CHECK((x) == SECANT_SUCCESS)
static SRNode col(size_t i) {
  SRNode n;
  memset(&n, 0, sizeof(n));
  n.choices = 1;
  n.leaf[0].kind = SR_COLUMN;
  n.leaf[0].slot = (uint32_t)i;
  return n;
}
static void encode_replay(void) {
  SRConfig c = secant_sr_config_default();
  SRNode n[3];
  uint8_t code[128], resolved[128];
  size_t size;
  float banks[] = {2, 3, 4, 5}, x[] = {1, 2, 3, 4}, y[] = {0, 0}, grid[32],
        static_sse;
  const uint8_t *asts[] = {code}, *static_asts[] = {resolved};
  SecantAstProgramSet programs;
  SRScore scores[1];
  SecantCpuSSERun fixed = secant_cpu_sse_run_init();
  uint64_t i;
  c.num_inputs = 2;
  c.num_constants = 2;
  c.num_banks = 2;
  c.toggle_bits = 2;
  n[0] = col(0);
  n[0].choices = 4;
  n[0].low_bit = 0;
  n[0].high_bit = 1;
  n[0].leaf[1].kind = SR_COEFFICIENT;
  n[0].leaf[1].slot = 0;
  n[0].leaf[2].kind = SR_COLUMN;
  n[0].leaf[2].slot = 1;
  n[0].leaf[3].kind = SR_COEFFICIENT;
  n[0].leaf[3].slot = 1;
  n[1] = col(1);
  n[1].choices = 2;
  n[1].low_bit = 0;
  n[1].leaf[1].kind = SR_COEFFICIENT;
  n[1].leaf[1].slot = 1;
  memset(n + 2, 0, sizeof(*n));
  n[2].op = SECANT_AST_INSTRUCTION_TYPE_MUL_F32;
  OK(secant_sr_program_write(&c, n, 3, NULL, 0, &size));
  CHECK(size == 19);
  CHECK(secant_sr_program_write(&c, n, 3, code, size - 1, &size) ==
        SECANT_ERROR_INSUFFICIENT_BUFFER);
  OK(secant_sr_program_write(&c, n, 3, code, sizeof(code), &size));
  memset(&programs, 0, sizeof(programs));
  programs.asts.items = asts;
  programs.asts.count = 1;
  OK(secant_sr_cpu_score(&c, &programs, x, y, 2, banks, grid, 32, scores));
  for (i = 0; i < 8; ++i) {
    char text[256];
    OK(secant_sr_resolve(&c, n, 3, banks, 4, i, resolved, sizeof(resolved),
                         &size));
    OK(secant_sr_format(&c, n, 3, banks, 4, i, text, sizeof(text), &size));
    CHECK(size == strlen(text) + 1);
    static_sse = 0;
    fixed.programs.asts.items = static_asts;
    fixed.programs.asts.count = 1;
    fixed.num_inputs = 2;
    fixed.num_targets = 1;
    fixed.num_rows = 2;
    fixed.input = (SecantConstHostMatrixF32){x, 4, 2};
    fixed.targets = (SecantConstHostMatrixF32){y, 2, 2};
    fixed.output = (SecantHostMatrixF32){&static_sse, 1, 1};
    OK(secant_cpu_run_sse(&fixed));
    CHECK(static_sse == grid[i]);
  }
  n[0].high_bit = 0;
  CHECK(secant_sr_program_write(&c, n, 3, code, sizeof(code), &size) ==
        SECANT_ERROR_BAD_PROGRAM);
}
static void reductions(void) {
  float values[] = {NAN, 4,        2,  2,   INFINITY, -1,
                    NAN, INFINITY, -1, NAN, NAN,      INFINITY};
  SRScore out[2];
  OK(secant_sr_reduce(values, 2, 6, out));
  CHECK(secant_sr_reduce(values, 1, 0, out) == SECANT_ERROR_INVALID_VALUE);
  CHECK(out[0].configuration == 2 && out[0].sse == 2 &&
        out[0].valid_configurations == 3);
  CHECK(out[1].configuration == UINT64_MAX && out[1].sse == INFINITY &&
        out[1].valid_configurations == 0);
}
static void score_agreement(void) {
  SRScoreAgreement a;
  /* Captured near-fit f32 exp/log cases: pointwise errors were below 1 ppm. */
  OK(secant_sr_score_agreement(.109371393919, .110066466,
                               58.3381519951 * 58.3381519951 * 10000, 10000,
                               2e-5, &a));
  CHECK(a.accepted && a.relative_rmse_gap < 1e-6);
  OK(secant_sr_score_agreement(441.516204834, 441.397827,
                               297.22179497 * 297.22179497 * 10000, 10000, 2e-5,
                               &a));
  CHECK(a.accepted && a.relative_rmse_gap < 1e-6);
  OK(secant_sr_score_agreement(1, 4, 100, 100, 2e-5, &a));
  CHECK(!a.accepted);
  OK(secant_sr_score_agreement(0, 0, 0, 1, 0, &a));
  CHECK(a.accepted);
  CHECK(secant_sr_score_agreement(INFINITY, 1, 1, 1, 2e-5, &a) ==
        SECANT_ERROR_INVALID_VALUE);
  CHECK(secant_sr_score_agreement(1, NAN, 1, 1, 2e-5, &a) ==
        SECANT_ERROR_INVALID_VALUE);
  CHECK(secant_sr_score_agreement(1, 1, 1, 0, 2e-5, &a) ==
        SECANT_ERROR_INVALID_VALUE);
}
static void evolution(void) {
  SRConfig c = secant_sr_config_default();
  SRSearch a, b;
  size_t g, i, configs;
  float banks[8], x[64], y[32], *grid;
  SRScore *scores;
  SecantAstProgramSet pa, pb;
  SRModel best;
  c.population = 64;
  c.elites = 2;
  c.num_inputs = 2;
  c.num_banks = 4;
  c.num_constants = 2;
  c.toggle_bits = 3;
  c.max_nodes = 15;
  c.max_depth = 5;
  OK(secant_sr_bank_generate(banks, 4, 2, 99, 0, -2, 2));
  configs = 4 * 8;
  grid = malloc(c.population * configs * sizeof(float));
  scores = malloc(c.population * sizeof(*scores));
  CHECK(grid && scores);
  for (i = 0; i < 32; ++i) {
    x[i] = (float)i / 31 - .5f;
    x[32 + i] = 1 - (float)i / 31;
    y[i] = x[i] * x[i] + x[32 + i];
  }
  OK(secant_sr_create(&c, banks, 8, &a));
  OK(secant_sr_create(&c, banks, 8, &b));
  CHECK(secant_sr_advance(a) == SECANT_ERROR_INVALID_STATE);
  CHECK(secant_sr_tell(a, scores, 64, 32, 10) == SECANT_ERROR_INVALID_STATE);
  for (g = 0; g < 20; ++g) {
    OK(secant_sr_ask(a, &pa));
    OK(secant_sr_ask(b, &pb));
    for (i = 0; i < c.population; ++i) {
      size_t p = 0;
      for (;;) {
        size_t width = secant_ast_instruction_size_get(pa.asts.items[i] + p);
        CHECK(width && p + width <= SECANT_SR_MAX_NODES * 23 + 1);
        CHECK(!memcmp(pa.asts.items[i] + p, pb.asts.items[i] + p, width));
        if (pa.asts.items[i][p] == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32)
          break;
        p += width;
      }
    }
    OK(secant_sr_cpu_score(&c, &pa, x, y, 32, banks, grid,
                           c.population * configs, scores));
    if (g == 0) {
      SRScore save = scores[63];
      scores[63].configuration = UINT64_MAX;
      CHECK(secant_sr_tell(a, scores, 64, 32, 10) ==
            SECANT_ERROR_INVALID_VALUE);
      CHECK(secant_sr_progress(a).structures == 0);
      scores[63] = save;
    } else {
      CHECK(secant_sr_tell(a, scores, 64, 33, 10) ==
            SECANT_ERROR_INVALID_VALUE);
    }
    OK(secant_sr_tell(a, scores, 64, 32, 10));
    OK(secant_sr_tell(b, scores, 64, 32, 10));
    CHECK(secant_sr_tell(a, scores, 64, 32, 10) == SECANT_ERROR_INVALID_STATE);
    if (g < 19) {
      OK(secant_sr_advance(a));
      OK(secant_sr_advance(b));
    }
  }
  OK(secant_sr_best(a, &best));
  CHECK(isfinite(best.score.sse));
  CHECK(secant_sr_progress(a).configurations == 20 * 64 * 32);
  secant_sr_destroy(a);
  secant_sr_destroy(b);
  free(grid);
  free(scores);
}
/* Force each variation class separately. A crossover must inherit complete
 * selector groups, while mutation must be able to reach both bindings and bits. */
static void selector_variation(void) {
  SRConfig c = secant_sr_config_default();
  SRNode parents[2][3];
  const uint8_t ops[] = {SECANT_AST_INSTRUCTION_TYPE_ADD_F32,
                         SECANT_AST_INSTRUCTION_TYPE_MUL_F32};
  const float banks[] = {2, 3}, x[] = {1, 2, -1, .5f, .25f, -2, 3, 4};
  const float y[] = {0, 1, 2, 3};
  SRScore *scores;
  float *grid;
  int mode;
  size_t i, j;
  c.population = 512; c.elites = 1; c.tournament = 1;
  c.num_inputs = 2; c.num_constants = 2; c.num_banks = 1;
  c.toggle_bits = 4; c.max_nodes = 7; c.max_depth = 4; c.initial_depth = 2;
  c.operators = ops; c.num_operators = sizeof(ops);
  memset(parents, 0, sizeof(parents));
  for (i = 0; i < 2; ++i) {
    for (j = 0; j < 2; ++j) {
      SRNode *n = &parents[i][j];
      n->choices = (i == j) ? 4 : 2;
      n->low_bit = (uint8_t)(i + j); n->high_bit = (uint8_t)(i + j + 1);
      n->leaf[0].kind = SR_COLUMN; n->leaf[0].slot = (uint32_t)j;
      n->leaf[1].kind = SR_COEFFICIENT; n->leaf[1].slot = (uint32_t)i;
      if (n->choices == 4) {
        n->leaf[2].kind = SR_LITERAL; n->leaf[2].value = (float)(7 + i);
        n->leaf[3].kind = SR_COLUMN; n->leaf[3].slot = (uint32_t)(1 - j);
      }
    }
    parents[i][2].op = ops[i];
  }
  scores = malloc(c.population * sizeof(*scores));
  grid = malloc(c.population * 16 * sizeof(*grid));
  CHECK(scores && grid);
  for (mode = 0; mode < 2; ++mode) {
    SRSearch s;
    SecantAstProgramSet programs;
    size_t hybrids = 0, bit_changes = 0, alternative_changes = 0, arity_changes = 0;
    c.crossover_probability = mode == 0 ? 1 : 0;
    c.mutation_probability = mode == 0 ? 0 : 1;
    OK(secant_sr_create(&c, banks, 2, &s));
    for (i = 0; i < c.population; ++i)
      OK(secant_sr_seed(s, i, parents[mode == 0 ? i % 2 : 0], 3));
    OK(secant_sr_ask(s, &programs));
    OK(secant_sr_cpu_score(&c, &programs, x, y, 4, banks, grid, c.population * 16, scores));
    OK(secant_sr_tell(s, scores, c.population, 4, 5));
    OK(secant_sr_advance(s));
    for (i = 1; i < c.population; ++i) {
      const SRCandidate *child = s->population[s->current] + i;
      unsigned origins = 0;
      OK(sr_validate(&c, child->nodes, child->count, NULL, NULL));
      for (j = 0; j < child->count; ++j) {
        const SRNode *n = child->nodes + j;
        size_t p, k;
        if (n->op) continue;
        if (mode == 0) {
          int matched = 0;
          for (p = 0; p < 2; ++p) for (k = 0; k < 2; ++k)
            if (!memcmp(n, &parents[p][k], sizeof(*n))) {
              matched = 1; origins |= 1u << p;
            }
          CHECK(matched); /* Includes the original bit IDs and operand order. */
        } else {
          for (k = 0; k < 2; ++k) {
            const SRNode *old = &parents[0][k];
            if (n->choices == old->choices && !memcmp(n->leaf, old->leaf, sizeof(n->leaf)) &&
                (n->low_bit != old->low_bit || n->high_bit != old->high_bit)) ++bit_changes;
          }
          if (child->count == 3 && j < 2 && n->choices == parents[0][j].choices &&
              memcmp(n->leaf, parents[0][j].leaf, sizeof(n->leaf))) ++alternative_changes;
          if (n->choices == 1) ++arity_changes;
        }
      }
      hybrids += origins == 3;
    }
    if (mode == 0) CHECK(hybrids > 0);
    else CHECK(bit_changes > 0 && alternative_changes > 0 && arity_changes > 0);
    OK(secant_sr_ask(s, &programs));
    OK(secant_sr_cpu_score(&c, &programs, x, y, 4, banks, grid, c.population * 16, scores));
    secant_sr_destroy(s);
  }
  free(scores); free(grid);
}
int main(void) {
  encode_replay();
  reductions();
  score_agreement();
  evolution();
  selector_variation();
  puts("toggle genome, winner replay, reduction and deterministic GP tests "
       "passed");
  return 0;
}
