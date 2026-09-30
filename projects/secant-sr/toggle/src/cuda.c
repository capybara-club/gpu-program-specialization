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
#include "secant_sr_cuda.h"
#include <cuda.h>
#include <nvrtc.h>
#include <stddef.h>
#include <stdio.h>
#include <time.h>
struct SRCudaScorerImpl {
  SRConfig config;
  SRCudaOptions options;
  CUdevice device;
  CUcontext context;
  CUdeviceptr input, target, banks, grid, entries;
  CUmodule reducer;
  CUfunction reduce;
  CUstream stream;
  CUevent done;
  SRScore *host;
  float *base_banks;
  int custom_banks;
  void *cubin, *plan_storage;
  size_t cubin_size, rows, configs, batch, grid_elements;
  SecantCubinPlan *plan;
  SecantCubinRunner runner;
  int retained, pending, failed;
};
typedef char sr_score_abi_size[(sizeof(SRScore) == 24) ? 1 : -1];
typedef char
    sr_score_abi_index[(offsetof(SRScore, configuration) == 8) ? 1 : -1];
static double now(void) {
  struct timespec t;
  if (clock_gettime(CLOCK_MONOTONIC, &t))
    return 0;
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static const char reducer_source[] =
    "struct Entry { float sse; unsigned reserved; unsigned long long "
    "configuration, valid; };\n"
    "extern \"C\" __global__ void best(const float *grid, unsigned long long "
    "configs, Entry *out) {\n"
    "  unsigned t=threadIdx.x; unsigned long long at=(unsigned long "
    "long)blockIdx.x*configs;\n"
    "  float value=__int_as_float(0x7f800000); unsigned long long index=~0ull, "
    "count=0;\n"
    "  for(unsigned long long i=t;i<configs;i+=blockDim.x){float "
    "v=grid[at+i];\n"
    "    if(isfinite(v) && v>=0){++count;if(v<value || (v==value && "
    "i<index)){value=v;index=i;}}}\n"
    "  __shared__ float values[256]; __shared__ unsigned long long "
    "indices[256], counts[256];\n"
    "  values[t]=value;indices[t]=index;counts[t]=count;__syncthreads();\n"
    "  for(unsigned stride=128;stride;stride>>=1){if(t<stride){\n"
    "    float v=values[t+stride];unsigned long long i=indices[t+stride];\n"
    "    if(v<values[t] || (v==values[t] && "
    "i<indices[t])){values[t]=v;indices[t]=i;}\n"
    "    counts[t]+=counts[t+stride];}__syncthreads();}\n"
    "  if(!t){out[blockIdx.x].sse=values[0];out[blockIdx.x].reserved=0;\n"
    "    "
    "out[blockIdx.x].configuration=indices[0];out[blockIdx.x].valid=counts[0];}"
    "\n"
    "}\n";
static SecantResult compile(const char *source, int major, int minor,
                            void **out, size_t *bytes, double *seconds) {
  nvrtcProgram p;
  nvrtcResult status;
  char arch[64];
  const char *options[] = {arch, "--std=c++14", "--use_fast_math",
                           "--fmad=false"};
  double begin = now();
  *out = NULL;
  *bytes = 0;
  snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d", major, minor);
  if (nvrtcCreateProgram(&p, source, "secant_sr.cu", 0, NULL, NULL) !=
      NVRTC_SUCCESS)
    return SECANT_ERROR_COMPILE_FAILED;
  status = nvrtcCompileProgram(p, 4, options);
  if (status != NVRTC_SUCCESS) {
    size_t size = 0;
    if (nvrtcGetProgramLogSize(p, &size) == NVRTC_SUCCESS) {
      char *log = malloc(size);
      if (log) {
        if (nvrtcGetProgramLog(p, log) == NVRTC_SUCCESS)
          fprintf(stderr, "%s\n", log);
        free(log);
      }
    }
  }
  if (status == NVRTC_SUCCESS)
    status = nvrtcGetCUBINSize(p, bytes);
  if (status == NVRTC_SUCCESS) {
    *out = malloc(*bytes);
    if (!*out) {
      nvrtcDestroyProgram(&p);
      return SECANT_ERROR_ALLOCATION_FAILED;
    }
    status = nvrtcGetCUBIN(p, *out);
  }
  nvrtcDestroyProgram(&p);
  *seconds += now() - begin;
  if (status != NVRTC_SUCCESS) {
    free(*out);
    *out = NULL;
    return SECANT_ERROR_COMPILE_FAILED;
  }
  return SECANT_SUCCESS;
}
SRCudaOptions secant_sr_cuda_options_default(void) {
  SRCudaOptions o = {0, 256, 256u * 1024u * 1024u, 8, 16, 128, 128, 3, 8};
  return o;
}
static SecantResult wait_owned(SRCudaScorer s) {
  if (!s->pending)
    return SECANT_SUCCESS;
  if (cuEventRecord(s->done, s->stream) != CUDA_SUCCESS ||
      cuEventSynchronize(s->done) != CUDA_SUCCESS) {
    if (cuStreamSynchronize(s->stream) != CUDA_SUCCESS) {
      s->failed = 1;
      return SECANT_ERROR_COMPLETION_UNKNOWN;
    }
    s->pending = 0;
    return SECANT_ERROR_DRIVER_FAILED;
  }
  s->pending = 0;
  return SECANT_SUCCESS;
}
SecantResult secant_sr_cuda_destroy(SRCudaScorer s) {
  CUcontext current;
  SecantResult r;
  if (!s)
    return SECANT_SUCCESS;
  if (s->context &&
      (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != s->context))
    return SECANT_ERROR_INVALID_STATE;
  s->failed = 1;
  if (s->runner) {
    r = secant_cubin_runner_destroy(s->runner);
    if (r != SECANT_SUCCESS)
      return r;
    s->runner = NULL;
  }
  r = wait_owned(s);
  if (r != SECANT_SUCCESS)
    return r;
#define RELEASE(h, fn)                                                         \
  do {                                                                         \
    if (h) {                                                                   \
      if (fn(h) != CUDA_SUCCESS)                                               \
        return SECANT_ERROR_DRIVER_FAILED;                                     \
      (h) = 0;                                                                 \
    }                                                                          \
  } while (0)
  RELEASE(s->reducer, cuModuleUnload);
  RELEASE(s->done, cuEventDestroy);
  RELEASE(s->stream, cuStreamDestroy);
  RELEASE(s->input, cuMemFree);
  RELEASE(s->target, cuMemFree);
  RELEASE(s->banks, cuMemFree);
  RELEASE(s->grid, cuMemFree);
  RELEASE(s->entries, cuMemFree);
  RELEASE(s->host, cuMemFreeHost);
#undef RELEASE
  if (s->retained) {
    if (cuDevicePrimaryCtxRelease(s->device) != CUDA_SUCCESS)
      return SECANT_ERROR_DRIVER_FAILED;
    s->retained = 0;
  }
  free(s->base_banks);
  free(s->cubin);
  free(s->plan_storage);
  free(s);
  return SECANT_SUCCESS;
}
SecantResult secant_sr_cuda_create(const SRConfig *c, const float *input,
                                   const float *target, size_t rows,
                                   const float *banks, const SRCudaOptions *o,
                                   SRCudaScorer *out, SRCudaStats *stats) {
  SRCudaScorer s = NULL;
  SecantCubinToggleSSERecipe recipe = secant_cubin_toggle_sse_recipe_init();
  SecantCubinRunnerOptions options = secant_cubin_runner_options_init();
  SecantResult r;
  size_t source_size, plan_size, inputs, bankn, score_width, bytes, i;
  int major, minor;
  char *source = NULL;
  void *reducer = NULL;
  size_t reducer_size = 0;
  double begin = now(), compilation = 0;
  if (!out)
    return SECANT_ERROR_INVALID_VALUE;
  *out = NULL;
  if (stats)
    memset(stats, 0, sizeof(*stats));
  r = sr_validate_config(c);
  if (r != SECANT_SUCCESS)
    return r;
  if (!o || !input || !target || !rows || !o->ast_batch || !o->score_bytes ||
      !o->asts_per_kernel || o->asts_per_kernel > 32 ||
      !o->kernels_per_module || !o->workers || !o->streams ||
      o->ast_batch > INT_MAX || o->workers > 256 || o->streams > 1024 ||
      !sr_mul(c->num_inputs, rows, &inputs) ||
      !sr_mul(c->num_constants, c->num_banks, &bankn))
    return SECANT_ERROR_INVALID_VALUE;
  if ((bankn && !banks) || !sr_mul(inputs, sizeof(float), &bytes) ||
      !sr_mul(bankn, sizeof(float), &bytes))
    return SECANT_ERROR_INVALID_VALUE;
  for (i = 0; i < inputs; ++i)
    if (!isfinite(input[i]))
      return SECANT_ERROR_INVALID_VALUE;
  for (i = 0; i < rows; ++i)
    if (!isfinite(target[i]))
      return SECANT_ERROR_INVALID_VALUE;
  for (i = 0; i < bankn; ++i)
    if (!isfinite(banks[i]))
      return SECANT_ERROR_INVALID_VALUE;
  s = calloc(1, sizeof(*s));
  if (!s)
    return SECANT_ERROR_ALLOCATION_FAILED;
  s->config = *c;
  s->options = *o;
  s->rows = rows;
  if (!sr_mul(c->num_banks, (size_t)1 << c->toggle_bits, &s->configs) ||
      !sr_mul(s->configs, sizeof(float), &score_width)) {
    r = SECANT_ERROR_OVERFLOW;
    goto fail;
  }
  s->batch = o->score_bytes / score_width;
  if (s->batch > o->ast_batch)
    s->batch = o->ast_batch;
  if (s->batch > c->population)
    s->batch = c->population;
  if (!s->batch) {
    r = SECANT_ERROR_INSUFFICIENT_BUFFER;
    goto fail;
  }
  if (!sr_mul(s->batch, s->configs, &s->grid_elements)) {
    r = SECANT_ERROR_OVERFLOW;
    goto fail;
  }
#define CUDA(call)                                                             \
  do {                                                                         \
    CUresult e = (call);                                                       \
    if (e != CUDA_SUCCESS) {                                                   \
      const char *name = NULL;                                                 \
      cuGetErrorName(e, &name);                                                \
      fprintf(stderr, "%s: %s\n", #call, name ? name : "CUDA error");          \
      r = SECANT_ERROR_DRIVER_FAILED;                                          \
      goto fail;                                                               \
    }                                                                          \
  } while (0)
#define SEC(call)                                                              \
  do {                                                                         \
    r = (call);                                                                \
    if (r != SECANT_SUCCESS)                                                   \
      goto fail;                                                               \
  } while (0)
  CUDA(cuInit(0));
  CUDA(cuDeviceGet(&s->device, o->device));
  CUDA(cuDevicePrimaryCtxRetain(&s->context, s->device));
  s->retained = 1;
  CUDA(cuCtxSetCurrent(s->context));
  CUDA(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, s->device));
  CUDA(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, s->device));
  recipe.num_inputs = c->num_inputs;
  recipe.num_constants = c->num_constants;
  recipe.num_targets = 1;
  recipe.asts_per_kernel = o->asts_per_kernel;
  recipe.num_kernels = o->kernels_per_module;
  recipe.threads_per_block = o->threads;
  recipe.tile_rows = o->tile_rows;
  recipe.patch_capacity_instructions =
      o->asts_per_kernel * (16 * c->max_nodes + 32);
  SEC(secant_cubin_source_size(&recipe.header, &source_size));
  source = malloc(source_size);
  if (!source) {
    r = SECANT_ERROR_ALLOCATION_FAILED;
    goto fail;
  }
  SEC(secant_cubin_source_write(&recipe.header, source, source_size));
  SEC(compile(source, major, minor, &s->cubin, &s->cubin_size, &compilation));
  free(source);
  source = NULL;
  SEC(secant_cubin_plan_storage_size(&recipe.header, s->cubin, s->cubin_size,
                                     &plan_size));
  s->plan_storage = malloc(plan_size);
  if (!s->plan_storage) {
    r = SECANT_ERROR_ALLOCATION_FAILED;
    goto fail;
  }
  SEC(secant_cubin_plan_init(&recipe.header, s->cubin, s->cubin_size,
                             s->plan_storage, plan_size, &s->plan));
  options.num_workers = o->workers;
  options.num_streams = o->streams;
  SEC(secant_cubin_runner_create(s->plan, s->cubin, s->cubin_size, &options,
                                 &s->runner));
  SEC(compile(reducer_source, major, minor, &reducer, &reducer_size,
              &compilation));
  CUDA(cuModuleLoadData(&s->reducer, reducer));
  free(reducer);
  reducer = NULL;
  CUDA(cuModuleGetFunction(&s->reduce, s->reducer, "best"));
  CUDA(cuStreamCreate(&s->stream, CU_STREAM_NON_BLOCKING));
  CUDA(cuEventCreate(&s->done, CU_EVENT_DISABLE_TIMING));
  CUDA(cuMemAlloc(&s->input, inputs * sizeof(float)));
  CUDA(cuMemcpyHtoD(s->input, input, inputs * sizeof(float)));
  CUDA(cuMemAlloc(&s->target, rows * sizeof(float)));
  CUDA(cuMemcpyHtoD(s->target, target, rows * sizeof(float)));
  if (bankn) {
    CUDA(cuMemAlloc(&s->banks, bankn * sizeof(float)));
    CUDA(cuMemcpyHtoD(s->banks, banks, bankn * sizeof(float)));
    s->base_banks = malloc(bankn * sizeof(float));
    if (!s->base_banks) { r = SECANT_ERROR_ALLOCATION_FAILED; goto fail; }
    memcpy(s->base_banks, banks, bankn * sizeof(float));
  }
  CUDA(cuMemAlloc(&s->grid, s->grid_elements * sizeof(float)));
  CUDA(cuMemAlloc(&s->entries, s->batch * sizeof(SRScore)));
  CUDA(cuMemAllocHost((void **)&s->host, s->batch * sizeof(SRScore)));
  if (stats) {
    stats->setup_seconds = now() - begin;
    stats->nvrtc_seconds = compilation;
  }
  *out = s;
  return SECANT_SUCCESS;
fail:
  free(source);
  free(reducer);
  if (stats) {
    stats->setup_seconds = now() - begin;
    stats->nvrtc_seconds = compilation;
  }
  if (s) {
    SecantResult cleanup = secant_sr_cuda_destroy(s);
    if (cleanup != SECANT_SUCCESS) {
      *out = s;
      return cleanup;
    }
  }
  return r;
#undef SEC
#undef CUDA
}
static SecantResult score_banks(SRCudaScorer s,
                                  const SecantAstProgramSet *programs,
                                  const float *banks, size_t num_banks, size_t bank_elements, int custom,
                                  SRScore *out, size_t capacity,
                                  SRCudaStats *stats) {
  size_t first, n, total_configs;
  double begin = now();
  SecantResult result = SECANT_SUCCESS;
  CUcontext current;
  SRCudaStats totals;
  memset(&totals, 0, sizeof(totals));
  if (stats)
    *stats = totals;
  if (!s || !programs || !out || !programs->asts.count ||
      capacity < programs->asts.count ||
      programs->asts.count > SIZE_MAX / sizeof(*out))
    return SECANT_ERROR_INVALID_VALUE;
  if (s->failed || cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
      current != s->context)
    return SECANT_ERROR_INVALID_STATE;
  if (!num_banks || num_banks > s->config.num_banks ||
      !sr_mul(num_banks, s->config.num_constants, &n) || bank_elements < n ||
      (n && !banks) || !sr_mul(num_banks, (size_t)1 << s->config.toggle_bits, &total_configs))
    return SECANT_ERROR_INVALID_VALUE;
  for (first = 0; first < n; ++first) if (!isfinite(banks[first])) return SECANT_ERROR_INVALID_VALUE;
  if (n && (custom || s->custom_banks)) {
    double tick = now();
    s->pending = 1;
    if (cuMemcpyHtoDAsync(s->banks, banks, n*sizeof(float), s->stream) != CUDA_SUCCESS)
      result = SECANT_ERROR_DRIVER_FAILED;
    {
      SecantResult drained = wait_owned(s);
      if (drained != SECANT_SUCCESS) result = drained;
    }
    totals.transfer_seconds += now()-tick;
    if (result != SECANT_SUCCESS) { s->failed = 1; if (stats) *stats=totals; return result; }
  }
  s->custom_banks = custom;
  for (first = 0; first < programs->asts.count; first += s->batch) {
    size_t count = programs->asts.count - first;
    double tick;
    unsigned long long configs = total_configs;
    SecantRunnerStats measured = secant_runner_stats_init();
    SecantCubinToggleSSERun run = secant_cubin_toggle_sse_run_init();
    void *args[] = {&s->grid, &configs, &s->entries};
    if (count > s->batch)
      count = s->batch;
    run.programs = *programs;
    run.programs.asts.items += first;
    run.programs.asts.count = count;
    run.input = (SecantDeviceMatrixF32){
        s->input, s->rows * s->config.num_inputs, s->rows};
    run.targets = (SecantDeviceMatrixF32){s->target, s->rows, s->rows};
    run.constants = (SecantDeviceConstantBanks){
        s->banks, s->config.num_constants * s->config.num_banks,
        s->config.num_constants};
    run.num_rows = s->rows;
    run.num_targets = 1;
    run.num_banks = num_banks;
    run.toggle_bits = s->config.toggle_bits;
    run.output = (SecantDeviceMatrixF32){s->grid, s->grid_elements, total_configs};
    result = secant_cubin_runner_run_toggle_sse(s->runner, &run, &measured);
    totals.pipeline_seconds += measured.total_seconds;
    totals.device_seconds += measured.runtime_seconds;
    totals.module_load_seconds += measured.module_load_seconds;
    totals.specialize_work_seconds += measured.compile_work_seconds;
    if (result != SECANT_SUCCESS) {
      if (result == SECANT_ERROR_DRIVER_FAILED ||
          result == SECANT_ERROR_COMPLETION_UNKNOWN ||
          result == SECANT_ERROR_INVALID_STATE)
        s->failed = 1;
      break;
    }
    tick = now();
    s->pending = 1;
    if (cuLaunchKernel(s->reduce, (unsigned)count, 1, 1, 256, 1, 1, 0,
                       s->stream, args, NULL) != CUDA_SUCCESS)
      result = SECANT_ERROR_DRIVER_FAILED;
    if (result == SECANT_SUCCESS)
      result = wait_owned(s);
    totals.reduction_seconds += now() - tick;
    if (result != SECANT_SUCCESS) {
      s->failed = 1;
      break;
    }
    tick = now();
    s->pending = 1;
    if (cuMemcpyDtoHAsync(s->host, s->entries, count * sizeof(SRScore),
                          s->stream) != CUDA_SUCCESS)
      result = SECANT_ERROR_DRIVER_FAILED;
    if (result == SECANT_SUCCESS)
      result = wait_owned(s);
    totals.transfer_seconds += now() - tick;
    if (result != SECANT_SUCCESS) {
      s->failed = 1;
      break;
    }
    memcpy(out + first, s->host, count * sizeof(SRScore));
    ++totals.batches;
  }
  if (s->pending) {
    SecantResult drained = wait_owned(s);
    if (drained != SECANT_SUCCESS)
      result = drained;
  }
  totals.wall_seconds = now() - begin;
  if (stats)
    *stats = totals;
  return result;
}

SecantResult secant_sr_cuda_score(SRCudaScorer s, const SecantAstProgramSet *p,
                                  SRScore *out, size_t capacity, SRCudaStats *stats) {
  return score_banks(s, p, s ? s->base_banks : NULL,
      s ? s->config.num_banks : 0, s ? s->config.num_banks*s->config.num_constants : 0,
      0, out, capacity, stats);
}
SecantResult secant_sr_cuda_score_banks(SRCudaScorer s, const SecantAstProgramSet *p,
    const float *banks, size_t num_banks, size_t elements,
    SRScore *out, size_t capacity, SRCudaStats *stats) {
  return score_banks(s, p, banks, num_banks, elements, 1, out, capacity, stats);
}
