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
#include "engine.h"
#include "lower.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"replay line %d: %s\n",__LINE__,#x);exit(2);}}while(0)
int
main(int argc, char **argv)
{
  FILE *f;
  uint64_t h[10];
  size_t i, n;
  float *x, *y, *banks;
  uint8_t **code;
  SRConfig c = secant_sr_config_default();
  SRCudaOptions o = secant_sr_cuda_options_default();
  SRCudaScorer scorer;
  SRCudaStats setup, stats;
  SRScore *scores;
  SecantAstProgramSet p = {0};
  CHECK(argc == 3);
  f = fopen(argv[1], "rb");
  CHECK(f);
  CHECK(fread(h, sizeof(h), 1, f) == 1);
  CHECK(h[0] == UINT64_C(0x3159414c50455253) && h[1] == 1 && h[2] > 0 && h[2] <= 32 && h[3] <= 16 && h[4] > 0 && h[4] <= 512 && h[5] <= 8 && h[6] > 0 && h[6] <= 1000000 && h[7] > 0 && h[7] <= 65536);
  c.num_inputs = h[2];
  c.num_constants = h[3];
  c.num_banks = h[4];
  c.toggle_bits = (uint32_t) h[5];
  c.population = h[7];
  n = h[6];
  o.asts_per_kernel = (size_t) atoi(argv[2]);
  o.kernels_per_module = 16;
  o.ast_batch = 4096;
  o.score_bytes = 256u * 1024u * 1024u;
  x = malloc(n * c.num_inputs * 4);
  y = malloc(n * 4);
  banks = malloc((c.num_banks * c.num_constants + 1) * 4);
  code = calloc(c.population, sizeof(*code));
  scores = calloc(c.population, sizeof(*scores));
  CHECK(x && y && banks && code && scores);
  CHECK(fread(x, 4, n * c.num_inputs, f) == n * c.num_inputs);
  CHECK(fread(y, 4, n, f) == n);
  CHECK(fread(banks, 4, c.num_banks * c.num_constants, f) == c.num_banks * c.num_constants);
  for (i = 0; i < c.population; ++i) {
    uint32_t bytes;
    BTree t;
    CHECK(fread(&bytes, 4, 1, f) == 1 && bytes && bytes <= B_CODE);
    code[i] = calloc(B_CODE, 1);
    CHECK(code[i]);
    CHECK(fread(code[i], 1, bytes, f) == bytes);
    CHECK(b_parse(code[i], bytes, (unsigned)c.num_inputs, (unsigned)c.num_constants, c.toggle_bits, &t) && t.bytes == bytes);
  }
  CHECK(fgetc(f) == EOF);
  CHECK(!fclose(f));
  p.asts.items = (const uint8_t * const *)code;
  p.asts.count = c.population;
  CHECK(bench_cuda_create(&c, x, y, n, banks, &o, &scorer, &setup) == SECANT_SUCCESS);
  CHECK(bench_cuda_score(scorer, &p, scores, c.population, &stats) == SECANT_SUCCESS);
  printf("{\"event\":\"replay\",\"asts\":%zu,\"configs_per_ast\":%zu,\"rows\":%zu,\"setup_seconds\":%.9g,\"nvrtc_seconds\":%.9g,\"wall_seconds\":%.9g,\"device_seconds\":%.9g,\"specialize_seconds\":%.9g,\"load_graph_seconds\":%.9g,\"transfer_seconds\":%.9g,\"reduction_seconds\":%.9g}\n", c.population, c.num_banks * ((size_t) 1 << c.toggle_bits), n, setup.setup_seconds, setup.nvrtc_seconds, stats.wall_seconds, stats.device_seconds, stats.specialize_work_seconds, stats.module_load_seconds, stats.transfer_seconds, stats.reduction_seconds);
  CHECK(bench_cuda_destroy(scorer) == SECANT_SUCCESS);
  /*
   * Independent CPU checks on evenly spaced AST winners and configurations. They run outside all
   * reported engine/pipeline timings.
   */
  {
    const char *grid_path = getenv("SECANT_BENCH_GRID");
    FILE *grid = grid_path ? fopen(grid_path, "rb") : NULL;
    size_t checked = 0, step = c.population > 128 ? c.population / 128 : 1;
    double energy = 0, max_gap = 0, max_resolved_gap=0;
    size_t sensitive=0,sensitive_ast[512];uint64_t sensitive_q[512];float sensitive_gpu[512];
    uint8_t (*sensitive_code)[B_CODE]=calloc(512,B_CODE);
    const uint8_t *sensitive_ptr[512];
    float *predictions = malloc(n * sizeof(float));
    CHECK(predictions&&sensitive_code);
    CHECK(!grid_path || grid);
    for (i = 0; i < n; ++i)
      energy += (double)y[i] * y[i];
    for (i = 0; i < c.population; i += step) {
      unsigned trial;
      BTree t;
      CHECK(b_parse(code[i], B_CODE, (unsigned)c.num_inputs, (unsigned)c.num_constants, c.toggle_bits, &t));
      for (trial = 0; trial < (grid ? 2u : 1u); ++trial) {
        uint8_t fixed[B_CODE] = {0};
        const uint8_t *asts[] = {fixed};
        double cpu = 0;
        float gpu = scores[i].sse, sequential = 0;
        size_t row;
        uint64_t q = scores[i].configuration;
        SecantCpuMaterializeRun run = secant_cpu_materialize_run_init();
        SRScoreAgreement agreement;
        if (trial) {
          q = (i * UINT64_C(2654435761) + 17) % (c.num_banks * ((size_t) 1 << c.toggle_bits));
          CHECK(!fseek(grid, (long)((i * c.num_banks * ((size_t) 1 << c.toggle_bits) + q) * 4), SEEK_SET));
          CHECK(fread(&gpu, 4, 1, grid) == 1);
        } else if (!scores[i].valid_configurations)
          continue;
        CHECK(b_resolve(&t, (unsigned)q, banks + (q >> c.toggle_bits) * c.num_constants, fixed, sizeof(fixed)));
        run.programs.asts.items = asts;
        run.programs.asts.count = 1;
        run.num_inputs = c.num_inputs;
        run.num_rows = n;
        run.input = (SecantConstHostMatrixF32) {
          x, n * c.num_inputs, n
        };
        run.output = (SecantHostMatrixF32) {
          predictions, n, n
        };
        CHECK(secant_cpu_run_materialize(&run) == SECANT_SUCCESS);
        for(row=0;row<n;++row){
          float residual=predictions[row]-y[row], square=residual*residual;
          cpu+=(double)square;
          sequential+=square;
        }
        if(cpu>FLT_MAX)cpu=INFINITY;
        if (!isfinite(cpu) && !isfinite(gpu)) {
          ++checked;
          continue;
        }
        memset(&agreement,0,sizeof(agreement));
        if (secant_sr_score_agreement(gpu, cpu, energy, n, 2e-5, &agreement)!=SECANT_SUCCESS || !agreement.accepted) {
          fprintf(stderr, "CPU-sensitive ast=%zu configuration=%llu GPU=%g CPU_f64_sum=%g CPU_f32_sum=%g; requires resolved GPU check\n", i, (unsigned long long)q, gpu, cpu, sequential);
          CHECK(sensitive<512);memcpy(sensitive_code[sensitive],fixed,B_CODE);sensitive_ptr[sensitive]=sensitive_code[sensitive];
          sensitive_ast[sensitive]=i;sensitive_q[sensitive]=q;sensitive_gpu[sensitive]=gpu;++sensitive;
        }
        if (agreement.relative_rmse_gap > max_gap)
          max_gap = agreement.relative_rmse_gap;
        ++checked;
      }
    }
    if (grid)
      CHECK(!fclose(grid));
    if(sensitive){
      float *gpu_predictions=malloc(sensitive*n*4);size_t j,row;CHECK(gpu_predictions);
      bench_materialize(c.num_inputs,x,n,sensitive_ptr,sensitive,gpu_predictions);
      for(j=0;j<sensitive;++j){double reference=0;SRScoreAgreement agreement;
        for(row=0;row<n;++row){float d=gpu_predictions[j*n+row]-y[row],v=d*d;reference+=(double)v;}
        if(reference>FLT_MAX)reference=INFINITY;
        if(!isfinite(reference)&&!isfinite(sensitive_gpu[j]))continue;
        CHECK(secant_sr_score_agreement(sensitive_gpu[j],reference,energy,n,2e-5,&agreement)==SECANT_SUCCESS);
        if(!agreement.accepted){fprintf(stderr,"resolved GPU mismatch ast=%zu config=%llu scoring=%g materialization=%g gap=%g\n",sensitive_ast[j],(unsigned long long)sensitive_q[j],sensitive_gpu[j],reference,agreement.relative_rmse_gap);return 2;}
        if(agreement.relative_rmse_gap>max_resolved_gap)max_resolved_gap=agreement.relative_rmse_gap;
      }
      free(gpu_predictions);
    }
    free(predictions);
    printf("{\"event\":\"cpu_audit\",\"reference\":\"f32_predictions_f32_squared_residuals_f64_sum\",\"checked\":%zu,\"max_relative_rmse_gap\":%.9g,\"cpu_agreement_all\":%s,\"resolved_gpu_checks\":%zu,\"resolved_gpu_max_gap\":%.9g,\"cpu_sensitive\":[", checked, max_gap,sensitive?"false":"true",sensitive,max_resolved_gap);
    for(i=0;i<sensitive;++i)printf("%s{\"ast\":%zu,\"configuration\":\"%llu\"}",i?",":"",sensitive_ast[i],(unsigned long long)sensitive_q[i]);
    puts("],\"accepted\":true}");free(sensitive_code);
  }
  for (i = 0; i < c.population; ++i)
    free(code[i]);
  free(code);
  free(scores);
  free(x);
  free(y);
  free(banks);
  return 0;
}
