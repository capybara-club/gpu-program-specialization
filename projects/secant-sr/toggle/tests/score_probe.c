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
#include "dataset.h"
#include "secant.h"
#include "secant_sr.h"
#include "secant_sr_cuda.h"
#include <cuda.h>
#include <math.h>
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define REQUIRE(c)                                                             \
  do {                                                                         \
    if (!(c)) {                                                                \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c);                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define CUDA(c)                                                                \
  do {                                                                         \
    CUresult e = (c);                                                          \
    if (e != CUDA_SUCCESS) {                                                   \
      const char *name = NULL;                                                 \
      cuGetErrorName(e, &name);                                                \
      fprintf(stderr, "line %d: %s: %s\n", __LINE__, #c,                       \
              name ? name : "CUDA error");                                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define SEC(c)                                                                 \
  do {                                                                         \
    SecantResult e = (c);                                                      \
    if (e != SECANT_SUCCESS) {                                                 \
      fprintf(stderr, "line %d: %s: %s\n", __LINE__, #c,                       \
              secant_result_to_string(e));                                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
typedef struct TestRunner {
  void *cubin, *storage;
  size_t bytes;
  SecantCubinPlan *plan;
  SecantCubinRunner runner;
} TestRunner;
static TestRunner build(const SecantCubinRecipeHeader *recipe) {
  TestRunner out;
  char *source, *log;
  size_t size, logsize;
  nvrtcProgram program;
  nvrtcResult status;
  SecantCubinRunnerOptions options = secant_cubin_runner_options_init();
  char architecture[48];
  CUdevice device;
  int major, minor;
  const char *flags[] = {architecture, "--std=c++14", "--use_fast_math",
                         "--fmad=false"};
  CUDA(cuCtxGetDevice(&device));
  CUDA(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CUDA(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  snprintf(architecture, sizeof(architecture), "--gpu-architecture=sm_%d%d",
           major, minor);
  memset(&out, 0, sizeof(out));
  SEC(secant_cubin_source_size(recipe, &size));
  source = (char *)malloc(size);
  REQUIRE(source);
  SEC(secant_cubin_source_write(recipe, source, size));
  REQUIRE(nvrtcCreateProgram(&program, source, "secant_toggle.cu", 0, NULL,
                             NULL) == NVRTC_SUCCESS);
  status = nvrtcCompileProgram(program, 4, flags);
  REQUIRE(nvrtcGetProgramLogSize(program, &logsize) == NVRTC_SUCCESS);
  log = (char *)malloc(logsize);
  REQUIRE(log);
  REQUIRE(nvrtcGetProgramLog(program, log) == NVRTC_SUCCESS);
  if (status != NVRTC_SUCCESS)
    fprintf(stderr, "%s\n", log);
  REQUIRE(status == NVRTC_SUCCESS);
  free(log);
  REQUIRE(nvrtcGetCUBINSize(program, &out.bytes) == NVRTC_SUCCESS);
  out.cubin = malloc(out.bytes);
  REQUIRE(out.cubin);
  REQUIRE(nvrtcGetCUBIN(program, (char *)out.cubin) == NVRTC_SUCCESS);
  nvrtcDestroyProgram(&program);
  free(source);
  if (getenv("SECANT_TEST_DUMP_CUBIN")) {
    FILE *f = fopen(getenv("SECANT_TEST_DUMP_CUBIN"), "wb");
    REQUIRE(f);
    REQUIRE(fwrite(out.cubin, 1, out.bytes, f) == out.bytes);
    REQUIRE(fclose(f) == 0);
  }
  SEC(secant_cubin_plan_storage_size(recipe, out.cubin, out.bytes, &size));
  out.storage = malloc(size);
  REQUIRE(out.storage);
  SEC(secant_cubin_plan_init(recipe, out.cubin, out.bytes, out.storage, size,
                             &out.plan));
  options.num_workers = 3;
  options.num_streams = 2;
  SEC(secant_cubin_runner_create(out.plan, out.cubin, out.bytes, &options,
                                 &out.runner));
  return out;
}
static void destroy(TestRunner *r) {
  SEC(secant_cubin_runner_destroy(r->runner));
  free(r->storage);
  free(r->cubin);
}

int main(int argc, char **argv) {
  size_t cols, rows, test, i, n;
  double ssd, vssd, cpu_sse = 0, gpu_sse = 0, difference = 0, energy = 0,
                    max_abs = 0, max_rel = 0;
  float *x, *y, *vx, *vy, *cpu, *gpu;
  uint8_t ast[2048];
  const uint8_t *asts[] = {ast};
  CUdevice device;
  CUcontext context;
  CUdeviceptr dx, dout;
  SecantCubinMaterializeRecipe recipe = secant_cubin_materialize_recipe_init();
  SecantCpuMaterializeRun cr = secant_cpu_materialize_run_init();
  SecantCubinMaterializeRun gr = secant_cubin_materialize_run_init();
  TestRunner runner;
  SRCudaScorer scorer;
  SRCudaStats stats;
  SRCudaOptions options = secant_sr_cuda_options_default();
  SRConfig config = secant_sr_config_default();
  SRScore score;
  REQUIRE(argc == 3 && strlen(argv[2]) % 2 == 0);
  n = strlen(argv[2]) / 2;
  REQUIRE(n < sizeof(ast));
  for (i = 0; i < n; ++i) {
    unsigned v;
    REQUIRE(sscanf(argv[2] + 2 * i, "%2x", &v) == 1);
    ast[i] = (uint8_t)v;
  }
  REQUIRE(secant_sr_dataset_binary_info(argv[1], &cols, &rows, &test));
  x = malloc(cols * rows * 4);
  y = malloc(rows * 4);
  vx = malloc(cols * test * 4);
  vy = malloc(test * 4);
  cpu = malloc(rows * 4);
  gpu = malloc(rows * 4);
  REQUIRE(x && y && vx && vy && cpu && gpu);
  REQUIRE(secant_sr_dataset_binary_load(argv[1], cols, x, y, rows, vx, vy, test,
                                        &ssd, &vssd));
  CUDA(cuInit(0));
  CUDA(cuDeviceGet(&device, 0));
  CUDA(cuDevicePrimaryCtxRetain(&context, device));
  CUDA(cuCtxSetCurrent(context));
  CUDA(cuMemAlloc(&dx, cols * rows * 4));
  CUDA(cuMemcpyHtoD(dx, x, cols * rows * 4));
  CUDA(cuMemAlloc(&dout, rows * 4));
  recipe.num_inputs = cols;
  recipe.num_kernels = 1;
  recipe.asts_per_kernel = 1;
  recipe.patch_capacity_instructions = 1024;
  runner = build(&recipe.header);
  cr.programs.asts.items = asts;
  cr.programs.asts.count = 1;
  cr.num_inputs = cols;
  cr.num_rows = rows;
  cr.input = (SecantConstHostMatrixF32){x, cols * rows, rows};
  cr.output = (SecantHostMatrixF32){cpu, rows, rows};
  gr.programs = cr.programs;
  gr.num_rows = rows;
  gr.input = (SecantDeviceMatrixF32){dx, cols * rows, rows};
  gr.output = (SecantDeviceMatrixF32){dout, rows, rows};
  SEC(secant_cpu_run_materialize(&cr));
  SEC(secant_cubin_runner_run_materialize(runner.runner, &gr, NULL));
  CUDA(cuMemcpyDtoH(gpu, dout, rows * 4));
  for (i = 0; i < rows; ++i) {
    double d = (double)gpu[i] - cpu[i], a = fabs(d),
           rel = a / fmax(1., fabs(cpu[i]));
    difference += d * d;
    energy += (double)y[i] * y[i];
    cpu_sse += ((double)cpu[i] - y[i]) * ((double)cpu[i] - y[i]);
    gpu_sse += ((double)gpu[i] - y[i]) * ((double)gpu[i] - y[i]);
    if (a > max_abs) {
      max_abs = a;
    }
    if (rel > max_rel) {
      max_rel = rel;
    }
  }
  config.num_inputs = cols;
  config.num_constants = 0;
  config.num_banks = 1;
  config.toggle_bits = 0;
  options.asts_per_kernel = 1;
  options.kernels_per_module = 1;
  options.ast_batch = 1;
  SEC(secant_sr_cuda_create(&config, x, y, rows, NULL, &options, &scorer,
                            &stats));
  SEC(secant_sr_cuda_score(scorer, &cr.programs, &score, 1, &stats));
  printf("{\"cpu_predictions_sse_f64_sum\":%.12g,\"gpu_predictions_sse_f64_"
         "sum\":%.12g,\"gpu_scoring_sse\":%.12g,\"prediction_rmse\":%.12g,"
         "\"prediction_max_abs\":%.12g,\"prediction_max_scaled\":%.12g,"
         "\"target_rms\":%.12g,\"relative_prediction_rmse\":%.12g}\n",
         cpu_sse, gpu_sse, (double)score.sse, sqrt(difference / rows), max_abs,
         max_rel, sqrt(energy / rows), sqrt(difference / energy));
  SEC(secant_sr_cuda_destroy(scorer));
  destroy(&runner);
  CUDA(cuMemFree(dx));
  CUDA(cuMemFree(dout));
  CUDA(cuDevicePrimaryCtxRelease(device));
  free(x);
  free(y);
  free(vx);
  free(vy);
  free(cpu);
  free(gpu);
  return 0;
}
