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
#include "secant_sr_cuda.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "%d: %s\n", __LINE__, #x);                               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define OK(x)                                                                  \
  do {                                                                         \
    SecantResult r = (x);                                                      \
    if (r != SECANT_SUCCESS) {                                                 \
      fprintf(stderr, "%d: %s: %s\n", __LINE__, #x,                            \
              secant_result_to_string(r));                                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
int main(void) {
  SRConfig c = secant_sr_config_default();
  SRSearch search;
  SRCudaScorer gpu;
  SRCudaStats timing;
  SRCudaOptions options = secant_sr_cuda_options_default();
  SecantAstProgramSet programs;
  enum { rows = 31, asts = 129, configs = 5 * 8 };
  float x[3 * rows], y[rows], banks[10], grid[asts * configs];
  SRScore cpu[asts], actual[asts];
  size_t i, j;
  c.population = asts;
  c.num_inputs = 3;
  c.num_constants = 2;
  c.num_banks = 5;
  c.toggle_bits = 3;
  c.max_nodes = 15;
  for (i = 0; i < 3 * rows; ++i)
    x[i] = (float)(i % 17) * .03125f - .2f;
  for (i = 0; i < rows; ++i)
    y[i] = x[i] * x[i + rows] + sinf(x[i + 2 * rows]);
  OK(secant_sr_bank_generate(banks, 5, 2, 98, 1, 0, 1));
  OK(secant_sr_create(&c, banks, 10, &search));
  /* Known ties and all-invalid rows exercise index and validity rules. */
  {
    SRNode n[2];
    memset(n, 0, sizeof(n));
    n[0].choices = 1;
    n[0].leaf[0].kind = SR_LITERAL;
    n[0].leaf[0].value = -1;
    OK(secant_sr_seed(search, 0, n, 1));
    n[1].op = SECANT_AST_INSTRUCTION_TYPE_LOG_F32;
    OK(secant_sr_seed(search, 1, n, 2));
  }
  OK(secant_sr_ask(search, &programs));
  OK(secant_sr_cpu_score(&c, &programs, x, y, rows, banks, grid, asts * configs,
                         cpu));
  options.ast_batch = 19;
  options.score_bytes = 19 * configs * sizeof(float);
  options.asts_per_kernel = 3;
  options.kernels_per_module = 2;
  options.tile_rows = 16;
  options.streams = 3;
  OK(secant_sr_cuda_create(&c, x, y, rows, banks, &options, &gpu, &timing));
  {
    const uint8_t invalid[] = {0xff, SECANT_AST_INSTRUCTION_TYPE_RETURN_F32};
    const uint8_t *bad[] = {invalid};
    SecantAstProgramSet request = programs;
    request.asts.items = bad;
    request.asts.count = 1;
    CHECK(secant_sr_cuda_score(gpu, &request, actual, asts, &timing) !=
          SECANT_SUCCESS);
  }
  for (j = 0; j < 2; ++j) {
    OK(secant_sr_cuda_score(gpu, &programs, actual, asts, &timing));
    CHECK(timing.batches == 7);
    for (i = 0; i < asts; ++i) {
      CHECK(actual[i].valid_configurations == cpu[i].valid_configurations);
      if (!cpu[i].valid_configurations) {
        CHECK(actual[i].configuration == UINT64_MAX &&
              actual[i].sse == INFINITY);
        continue;
      }
      CHECK(actual[i].configuration < configs);
      CHECK(fabsf(actual[i].sse - cpu[i].sse) <=
            2e-4f * (1 + fabsf(cpu[i].sse)));
      CHECK(
          fabsf(actual[i].sse - grid[i * configs + actual[i].configuration]) <=
          2e-4f * (1 + fabsf(actual[i].sse)));
    }
    CHECK(actual[0].configuration == 0);
    CHECK(actual[1].valid_configurations == 0);
  }
  OK(secant_sr_cuda_destroy(gpu));
  secant_sr_destroy(search);
  puts("CUDA SR adapter: CPU agreement, deterministic ties, invalid domains, "
       "partial chunks/modules, "
       "repeated calls passed");
  return 0;
}
