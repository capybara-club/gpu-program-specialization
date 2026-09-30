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
#include "legacy_bridge.h"
#include "lower.h"
#include "secant_instructions.h"
#include <cuda.h>
#include <nvrtc.h>
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Fail closed in the isolated experiment process. The production runner's ownership/error API is
 * unchanged; benchmark failures preserve caller logs.
 */
#define CHECK(x) do{if(!(x)){fprintf(stderr,"controlled benchmark %s:%d: %s\n",__FILE__,__LINE__,#x);exit(2);}}while(0)
#define CUDA(x) do{CUresult e=(x);if(e){const char *n=NULL;cuGetErrorName(e,&n);fprintf(stderr,"CUDA %s: %s\n",#x,n?n:"unknown");exit(2);}}while(0)
#define SEC(x) CHECK((x)==SECANT_SUCCESS)
typedef struct BEngine {
  SRConfig c;
  SRCudaOptions o;
  SRCudaScorer native;
  int settings;
  size_t rows, capacity, configs, code_bytes, step, fit_step;
  float *x, *y, *base;
  double target_energy;
  CUdevice device;
  CUcontext context;
  CUstream stream;
  CUevent start, stop;
  CUdeviceptr dx, dy, db, dm, dw, grid, entries;
  uint32_t *masks, *words;
  SRScore *host;
  uint8_t *code;
   uint8_t(*oldcode)[B_CODE];
  const uint8_t **oldasts;
  BTree *trees;
  BChoice *choices;
  void *cubin, *patched, *storage, *legacy_plan, *library;
  size_t cubin_bytes;
  const LegacyAPI *legacy;
  SecantCubinPlan *plan;
  CUmodule reducer;
  CUfunction reduce;
  struct {
    void *cubin, *patched, *plan;
    size_t bytes;
  } templates[4];
  size_t active_slots;
  int major, minor, sm_count, regs, local, blocks;
  double nvrtc_seconds;
} BEngine;
static double
tick(void)
{
  struct timespec t;
  CHECK(!clock_gettime(CLOCK_MONOTONIC, &t));
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static void *
allocate(size_t n, size_t size)
{
  void *p;
  CHECK(!n || size <= SIZE_MAX / n);
  p = calloc(n ? n : 1, size);
  CHECK(p);
  return p;
}
static void
wait_done(BEngine *s)
{
  CUDA(cuEventRecord(s->stop, s->stream));
  CUDA(cuEventSynchronize(s->stop));
}
static void
compile(BEngine *s, const char *source, void **out, size_t * bytes)
{
  nvrtcProgram p;
  nvrtcResult e;
  char arch[64];
  const char *opts[] = {arch, "--std=c++14", "--use_fast_math", "--fmad=false"};
  double t = tick();
  snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%d%d", s->major, s->minor);
  CHECK(nvrtcCreateProgram(&p, source, "controlled.cu", 0, NULL, NULL) == NVRTC_SUCCESS);
  e = nvrtcCompileProgram(p, 4, opts);
  if (e != NVRTC_SUCCESS) {
    size_t n;
    char *log;
    CHECK(nvrtcGetProgramLogSize(p, &n) == NVRTC_SUCCESS);
    log = allocate(n, 1);
    CHECK(nvrtcGetProgramLog(p, log) == NVRTC_SUCCESS);
    fprintf(stderr, "%s\n", log);
    free(log);
  }
  CHECK(e == NVRTC_SUCCESS);
  CHECK(nvrtcGetCUBINSize(p, bytes) == NVRTC_SUCCESS);
  *out = allocate(*bytes, 1);
  CHECK(nvrtcGetCUBIN(p, *out) == NVRTC_SUCCESS);
  CHECK(nvrtcDestroyProgram(&p) == NVRTC_SUCCESS);
  s->nvrtc_seconds += tick() - t;
}
static const char reduce_source[] =
"struct E{float s;unsigned pad;unsigned long long i,n;};\n"
"extern \"C\" __global__ void best(const float *g,unsigned long long c,E *out){\n"
"unsigned t=threadIdx.x;unsigned long long b=(unsigned long long)blockIdx.x*c,k=~0ull,n=0;float v=__int_as_float(0x7f800000);\n"
"for(unsigned long long j=t;j<c;j+=256){float x=g[b+j];if(isfinite(x)&&x>=0){++n;if(x<v||(x==v&&j<k)){v=x;k=j;}}}\n"
"__shared__ E a[256];a[t]={v,0,k,n};__syncthreads();\n"
"for(unsigned d=128;d;d>>=1){if(t<d){E x=a[t+d];if(x.s<a[t].s||(x.s==a[t].s&&x.i<a[t].i)){a[t].s=x.s;a[t].i=x.i;}a[t].n+=x.n;}__syncthreads();}\n"
"if(!t)out[blockIdx.x]=a[0];}\n";
void bench_materialize(size_t columns,const float *input,size_t rows,const uint8_t *const *asts,size_t count,float *output) {
  BEngine s={0};SecantCubinMaterializeRecipe recipe=secant_cubin_materialize_recipe_init();
  SecantCubinMaterializeRun run=secant_cubin_materialize_run_init();
  SecantCubinRunnerOptions options=secant_cubin_runner_options_init();
  SecantCubinRunner runner;char *source;size_t n;
  CUDA(cuInit(0));CUDA(cuDeviceGet(&s.device,0));CUDA(cuDevicePrimaryCtxRetain(&s.context,s.device));CUDA(cuCtxSetCurrent(s.context));
  CUDA(cuDeviceGetAttribute(&s.major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,s.device));CUDA(cuDeviceGetAttribute(&s.minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,s.device));
  recipe.num_inputs=columns;recipe.num_kernels=8;recipe.asts_per_kernel=1;recipe.patch_capacity_instructions=256;
  SEC(secant_cubin_source_size(&recipe.header,&n));source=allocate(n,1);SEC(secant_cubin_source_write(&recipe.header,source,n));compile(&s,source,&s.cubin,&s.cubin_bytes);free(source);
  SEC(secant_cubin_plan_storage_size(&recipe.header,s.cubin,s.cubin_bytes,&n));s.storage=allocate(n,1);SEC(secant_cubin_plan_init(&recipe.header,s.cubin,s.cubin_bytes,s.storage,n,&s.plan));
  options.num_workers=1;options.num_streams=2;SEC(secant_cubin_runner_create(s.plan,s.cubin,s.cubin_bytes,&options,&runner));
  CUDA(cuMemAlloc(&s.dx,columns*rows*4));CUDA(cuMemAlloc(&s.grid,count*rows*4));CUDA(cuMemcpyHtoD(s.dx,input,columns*rows*4));
  run.programs.asts.items=asts;run.programs.asts.count=count;run.input=(SecantDeviceMatrixF32){s.dx,columns*rows,rows};run.output=(SecantDeviceMatrixF32){s.grid,count*rows,rows};run.num_rows=rows;
  SEC(secant_cubin_runner_run_materialize(runner,&run,NULL));CUDA(cuMemcpyDtoH(output,s.grid,count*rows*4));
  SEC(secant_cubin_runner_destroy(runner));CUDA(cuMemFree(s.dx));CUDA(cuMemFree(s.grid));CUDA(cuDevicePrimaryCtxRelease(s.device));free(s.storage);free(s.cubin);
}
SecantResult
bench_cuda_create(const SRConfig *c, const float *x, const float *y, size_t rows, const float *banks, const SRCudaOptions *o, SRCudaScorer *out, SRCudaStats *stats)
{
  BEngine *s = allocate(1, sizeof(*s));
  const char *mode = getenv("SECANT_BENCH_MODE");
  size_t i;
  double begin = tick();
  char *source = NULL;
  CHECK(c && o && out && stats && rows && c->num_inputs && c->num_inputs <= 32 && c->num_constants <= 16 && c->num_banks && c->num_banks <= 512 && c->toggle_bits <= 8);
  CHECK(o->asts_per_kernel && o->asts_per_kernel <= 8 && o->kernels_per_module && o->kernels_per_module <= 64 && o->threads == 128 && o->tile_rows == 128);
  memset(stats, 0, sizeof(*stats));
  s->c = *c;
  s->o = *o;
  s->rows = rows;
  s->capacity = o->asts_per_kernel * o->kernels_per_module;
  s->configs = c->num_banks * ((size_t) 1 << c->toggle_bits);
  CHECK(s->capacity <= o->score_bytes / 4 / s->configs);
  s->x = allocate(rows * c->num_inputs, 4);
  s->y = allocate(rows, 4);
  s->base = allocate(c->num_banks * c->num_constants, 4);
  memcpy(s->x, x, rows * c->num_inputs * 4);
  memcpy(s->y, y, rows * 4);
  if (c->num_constants)
    memcpy(s->base, banks, c->num_banks * c->num_constants * 4);
  for (i = 0; i < rows; ++i)
    s->target_energy += (double)y[i] * y[i];
  if (mode && !strcmp(mode, "native")) {
    SEC(secant_sr_cuda_create(c, x, y, rows, banks, o, &s->native, stats));
    *out = (SRCudaScorer)s;
    return SECANT_SUCCESS;
  }
  CHECK(mode && (!strcmp(mode, "settings") || !strcmp(mode, "toggles")));
  s->settings = !strcmp(mode, "settings");
  CUDA(cuInit(0));
  CUDA(cuDeviceGet(&s->device, o->device));
  CUDA(cuDevicePrimaryCtxRetain(&s->context, s->device));
  CUDA(cuCtxSetCurrent(s->context));
  CUDA(cuDeviceGetAttribute(&s->major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, s->device));
  CUDA(cuDeviceGetAttribute(&s->minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, s->device));
  CUDA(cuDeviceGetAttribute(&s->sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, s->device));
  CUDA(cuStreamCreate(&s->stream, CU_STREAM_NON_BLOCKING));
  CUDA(cuEventCreate(&s->start, 0));
  CUDA(cuEventCreate(&s->stop, 0));
  CUDA(cuMemAlloc(&s->dx, rows * c->num_inputs * 4));
  CUDA(cuMemAlloc(&s->dy, rows * 4));
  CUDA(cuMemcpyHtoD(s->dx, x, rows * c->num_inputs * 4));
  CUDA(cuMemcpyHtoD(s->dy, y, rows * 4));
  CUDA(cuMemAlloc(&s->db, c->num_banks * (c->num_constants ? c->num_constants : 1) * 4));
  CUDA(cuMemAlloc(&s->grid, s->capacity * s->configs * 4));
  CUDA(cuMemAlloc(&s->entries, s->capacity * sizeof(SRScore)));
  CUDA(cuMemHostAlloc((void **)&s->host, s->capacity * sizeof(SRScore), 0));
  s->trees = allocate(s->capacity, sizeof(BTree));
  s->oldcode = allocate(s->capacity, B_CODE);
  s->oldasts = allocate(s->capacity, sizeof(void *));
  s->choices = allocate(o->kernels_per_module * 32, sizeof(BChoice));
  if (s->settings) {
    LegacyEntry entry;
    LegacyShape shape = {c->num_inputs, o->kernels_per_module, o->asts_per_kernel, o->tile_rows, o->threads, 256 * o->asts_per_kernel, 0};
    const char *lib = getenv("SECANT_BENCH_LEGACY");
    CHECK(lib);
    s->library = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!s->library)
      fprintf(stderr, "%s\n", dlerror());
    CHECK(s->library);
    *(void **)(&entry) = dlsym(s->library, "secant_bench_legacy_v2");
    CHECK(entry);
    s->legacy = entry();
    CHECK(s->legacy->version == 2);
    for (i = 0; i < 4; ++i) {
      shape.slots = (size_t) 4 << i;
      CHECK(!s->legacy->source(&shape, &source));
      compile(s, source, &s->templates[i].cubin, &s->templates[i].bytes);
      free(source);
      CHECK(!s->legacy->plan(&shape, s->templates[i].cubin, s->templates[i].bytes, &s->templates[i].plan));
      s->templates[i].patched = allocate(s->templates[i].bytes, 1);
    }
    CUDA(cuMemHostAlloc((void **)&s->masks, o->kernels_per_module * s->configs * 4, 0));
    CUDA(cuMemHostAlloc((void **)&s->words, o->kernels_per_module * s->configs * 32 * 4, 0));
    CUDA(cuMemAlloc(&s->dm, o->kernels_per_module * s->configs * 4));
    CUDA(cuMemAlloc(&s->dw, o->kernels_per_module * s->configs * 32 * 4));
  } else {
    SecantCubinToggleSSERecipe r = secant_cubin_toggle_sse_recipe_init();
    size_t n;
    r.num_inputs = c->num_inputs;
    r.num_constants = c->num_constants;
    r.num_targets = 1;
    r.num_kernels = o->kernels_per_module;
    r.asts_per_kernel = o->asts_per_kernel;
    r.tile_rows = o->tile_rows;
    r.threads_per_block = o->threads;
    r.patch_capacity_instructions = 256 * o->asts_per_kernel;
    SEC(secant_cubin_source_size(&r.header, &n));
    source = allocate(n, 1);
    SEC(secant_cubin_source_write(&r.header, source, n));
    compile(s, source, &s->cubin, &s->cubin_bytes);
    free(source);
    SEC(secant_cubin_plan_storage_size(&r.header, s->cubin, s->cubin_bytes, &n));
    s->storage = allocate(n, 1);
    SEC(secant_cubin_plan_init(&r.header, s->cubin, s->cubin_bytes, s->storage, n, &s->plan));
  }
  if (!s->settings)
    s->patched = allocate(s->cubin_bytes, 1);
  {
    void *code;
    size_t n;
    compile(s, reduce_source, &code, &n);
    CUDA(cuModuleLoadData(&s->reducer, code));
    free(code);
    CUDA(cuModuleGetFunction(&s->reduce, s->reducer, "best"));
  }
  s->blocks = INT32_MAX;
  stats->nvrtc_seconds = s->nvrtc_seconds;
  stats->setup_seconds = tick() - begin;
  *out = (SRCudaScorer)s;
  printf("{\"event\":\"comparison_backend\",\"engine\":\"%s\",\"scheduler\":\"common_graph_serial_modules\",\"packed\":%zu,\"kernels\":%zu,\"legacy_slots\":%u}\n", mode, o->asts_per_kernel, o->kernels_per_module, s->settings ? 32 : 0);
  fflush(stdout);
  return SECANT_SUCCESS;
}
static void
capture(BEngine *s, const SecantAstProgramSet *p, const float *banks, size_t nb, int fit)
{
  const char *dir = getenv("SECANT_BENCH_CAPTURE");
  size_t step = fit ? s->fit_step++ : s->step++, i;
  char path[4096];
  FILE *f;
  if (!dir || (!fit && step != 0 && step != 8 && step != 32) || (fit && step != 0))
    return;
  CHECK(snprintf(path, sizeof(path), "%s/%s-%03zu.bin", dir, fit ? "fit" : "generation", step) < (int)sizeof(path));
  f = fopen(path, "wbx");
  CHECK(f);
  {
    uint64_t h[10] = {UINT64_C(0x3159414c50455253), 1, s->c.num_inputs, s->c.num_constants, nb, s->c.toggle_bits, s->rows, p->asts.count, step, (unsigned)fit};
    CHECK(fwrite(h, sizeof(h), 1, f) == 1);
  }
  CHECK(fwrite(s->x, 4, s->rows * s->c.num_inputs, f) == s->rows * s->c.num_inputs);
  CHECK(fwrite(s->y, 4, s->rows, f) == s->rows);
  CHECK(fwrite(banks, 4, nb * s->c.num_constants, f) == nb * s->c.num_constants);
  for (i = 0; i < p->asts.count; ++i) {
    BTree t;
    uint32_t n;
    CHECK(b_parse(p->asts.items[i], SECANT_SR_MAX_NODES * 43 + 1, (unsigned)s->c.num_inputs, (unsigned)s->c.num_constants, s->c.toggle_bits, &t));
    n = (uint32_t) t.bytes;
    CHECK(fwrite(&n, 4, 1, f) == 1);
    CHECK(fwrite(p->asts.items[i], 1, n, f) == n);
  }
  CHECK(!fclose(f));
}
static CUgraphExec graph(BEngine *s, CUmodule module, size_t count, size_t nb){
  CUgraph g;
  CUgraphExec executable;
  size_t k, configs = nb * ((size_t) 1 << s->c.toggle_bits), input_ld = s->rows, cols = s->c.num_inputs, word_ld = s->active_slots,
   targets = 1, bank_stride = s->c.num_constants;
  unsigned bits = s->c.toggle_bits;
  CUDA(cuGraphCreate(&g, 0));
  for (k = 0; k < (count + s->o.asts_per_kernel - 1) / s->o.asts_per_kernel; ++k) {
    CUDA_KERNEL_NODE_PARAMS params;
    CUgraphNode node;
    CUfunction fn;
    char name[100];
    int regs, local, blocks;
    size_t active = s->settings ? (count - k * s->o.asts_per_kernel < s->o.asts_per_kernel ? count - k * s->o.asts_per_kernel : s->o.asts_per_kernel) : count;
    CUdeviceptr output = s->grid + (s->settings ? k * s->o.asts_per_kernel * configs * 4 : 0), mask = s->dm + k * s->configs * 4,
     words = s->dw + k * s->configs * s->active_slots * 4;
    void *oldargs[] = {&s->dx, &cols, &input_ld, &mask, &words, &word_ld, &configs, &s->dy, &input_ld, &s->rows, &active, &targets, &output, &configs};
    void *newargs[] = {&s->dx, &input_ld, &s->dy, &input_ld, &s->db, &bank_stride, &s->rows, &configs, &bits, &output, &configs, &active};
    snprintf(name, sizeof(name), "%s_%03zu", s->settings ? "secant_cubin_dynamic_leaf_sse" : "secant_cubin_toggle_sse", k);
    CUDA(cuModuleGetFunction(&fn, module, name));
    memset(&params, 0, sizeof(params));
    params.func = fn;
    params.gridDimX = (unsigned)((s->rows + s->o.tile_rows - 1) / s->o.tile_rows);
    params.gridDimY = s->settings ? 1 : (unsigned)((configs + s->o.threads - 1) / s->o.threads);
    params.gridDimZ = 1;
    params.blockDimX = (unsigned)s->o.threads;
    params.blockDimY = params.blockDimZ = 1;
    params.sharedMemBytes = s->settings ? (unsigned)(((cols | 1) + 1) * s->o.tile_rows * 4) : 0;
    params.kernelParams = s->settings ? oldargs : newargs;
    CUDA(cuGraphAddKernelNode(&node, g, NULL, 0, &params));
    CUDA(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
    CUDA(cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn));
    CUDA(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, (int)s->o.threads, params.sharedMemBytes));
    if (regs > s->regs)
      s->regs = regs;
    if (local > s->local)
      s->local = local;
    if (blocks < s->blocks)
      s->blocks = blocks;
  }
  CUDA(cuGraphInstantiateWithFlags(&executable, g, 0));
  CUDA(cuGraphDestroy(g));
  return executable;
}
static double
execute_graph(BEngine *s, CUgraphExec g, size_t elements)
{
  float ms;
  CUDA(cuMemsetD32Async(s->grid, 0, elements, s->stream));
  CUDA(cuEventRecord(s->start, s->stream));
  CUDA(cuGraphLaunch(g, s->stream));
  wait_done(s);
  CUDA(cuEventElapsedTime(&ms, s->start, s->stop));
  return ms * .001;
}
static int
compare_double(const void *a, const void *b)
{
double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
static SecantResult
score(BEngine *s, const SecantAstProgramSet *p, const float *banks, size_t nb, SRScore *out, size_t capacity, SRCudaStats *stats, int fit)
{
  double begin = tick(), resident = 0, prepare = 0;
  size_t first, i, table_bytes = 0, configs = nb * ((size_t) 1 << s->c.toggle_bits);
  FILE *dump = NULL;
  const char *dump_path = getenv("SECANT_BENCH_GRID"), *samples = getenv("SECANT_BENCH_SAMPLES");
  unsigned repeats = samples ? (unsigned)atoi(samples) : 0;
  CHECK(p && out && stats && capacity >= p->asts.count && nb && nb <= s->c.num_banks && repeats <= 31);
  memset(stats, 0, sizeof(*stats));
  capture(s, p, banks, nb, fit);
  if (s->native)
    return fit ? secant_sr_cuda_score_banks(s->native, p, banks, nb, nb * s->c.num_constants, out, capacity, stats) : secant_sr_cuda_score(s->native, p, out, capacity, stats);
  if (dump_path) {
    dump = fopen(dump_path, "wbx");
    CHECK(dump);
  }
  if (!s->settings) {
    double t = tick();
    CUDA(cuMemcpyHtoD(s->db, banks, nb * s->c.num_constants * 4));
    stats->transfer_seconds += tick() - t;
  }
  for (first = 0; first < p->asts.count; first += s->capacity) {
    size_t count = p->asts.count - first, k, modules;
    double t = tick();
    CUmodule module;
    CUgraphExec g;
    SecantAstProgramSet part = *p;
    if (count > s->capacity)
      count = s->capacity;
    part.asts.items += first;
    part.asts.count = count;
    modules = (count + s->o.asts_per_kernel - 1) / s->o.asts_per_kernel;
    if (s->settings) {
      size_t needed = 0, used[64], shape;
      for (k = 0; k < modules; ++k) {
        size_t slots = 0, j;
        for (j = k * s->o.asts_per_kernel; j < count && j < (k + 1) * s->o.asts_per_kernel; ++j) {
          CHECK(b_parse(part.asts.items[j], SECANT_SR_MAX_NODES * 43 + 1, (unsigned)s->c.num_inputs, (unsigned)s->c.num_constants, s->c.toggle_bits, s->trees + j));
          if (!b_lower(s->trees + j, s->oldcode[j], B_CODE, s->choices + k * 32, &slots, 32)) {
            fprintf(stderr, "unsupported: kernel %zu exceeds 32 dynamic slots; no ASTs discarded\n", k);
            exit(3);
          } s->oldasts[j] = s->oldcode[j];
        }
        used[k] = slots;
        if (slots > needed)
          needed = slots;
      }
      for (shape = 0; shape < 3 && ((size_t) 4 << shape) < needed; ++shape) {
      }
      s->active_slots = (size_t) 4 << shape;
      s->cubin = s->templates[shape].cubin;
      s->cubin_bytes = s->templates[shape].bytes;
      s->patched = s->templates[shape].patched;
      s->legacy_plan = s->templates[shape].plan;
      for (k = 0; k < modules; ++k)
        CHECK(b_tables(s->choices + k * 32, used[k], s->active_slots, banks, nb, s->c.num_constants, s->c.toggle_bits, s->masks + k * s->configs, s->words + k * s->configs * s->active_slots));
      table_bytes += modules * s->configs * (1 + s->active_slots) * 4;
    }
    prepare += tick() - t;
    t = tick();
    if (s->settings) {
      CUDA(cuMemcpyHtoDAsync(s->dm, s->masks, modules * s->configs * 4, s->stream));
      CUDA(cuMemcpyHtoDAsync(s->dw, s->words, modules * s->configs * s->active_slots * 4, s->stream));
      wait_done(s);
    }
    stats->transfer_seconds += tick() - t;
    t = tick();
    memcpy(s->patched, s->cubin, s->cubin_bytes);
    if (s->settings)
      CHECK(!s->legacy->specialize(s->legacy_plan, s->oldasts, count, s->patched, s->cubin_bytes));
    else
      SEC(secant_cubin_specialize_into(s->plan, &part, s->patched, s->cubin_bytes));
    stats->specialize_work_seconds += tick() - t;
    t = tick();
    CUDA(cuModuleLoadData(&module, s->patched));
    g = graph(s, module, count, nb);
    stats->module_load_seconds += tick() - t;
    stats->device_seconds += execute_graph(s, g, count * configs);
    if (repeats) {
      double durations[31];
      for (i = 0; i < repeats; ++i)
        durations[i] = execute_graph(s, g, count * configs);
      qsort(durations, repeats, sizeof(double), compare_double);
      resident += durations[repeats / 2];
    }
    t = tick(); {
      unsigned long long n = configs;
      void *args[] = {&s->grid, &n, &s->entries};
      CUDA(cuLaunchKernel(s->reduce, (unsigned)count, 1, 1, 256, 1, 1, 0, s->stream, args, NULL));
      wait_done(s);
    } stats->reduction_seconds += tick() - t;
    t = tick();
    CUDA(cuMemcpyDtoHAsync(s->host, s->entries, count * sizeof(SRScore), s->stream));
    wait_done(s);
    memcpy(out + first, s->host, count * sizeof(SRScore));
    stats->transfer_seconds += tick() - t;
    if (dump) {
      float *grid = allocate(count * configs, 4);
      CUDA(cuMemcpyDtoH(grid, s->grid, count * configs * 4));
      CHECK(fwrite(grid, 4, count * configs, dump) == count * configs);
      free(grid);
    }
    CUDA(cuGraphExecDestroy(g));
    CUDA(cuModuleUnload(module));
    ++stats->batches;
  }
  if (dump)
    CHECK(!fclose(dump));
  stats->wall_seconds = tick() - begin;
  stats->pipeline_seconds = stats->wall_seconds;
  {
    const char *path = getenv("SECANT_BENCH_STATS");
    if (path) {
      FILE *f = fopen(path, "a");
      CHECK(f);
      fprintf(f, "{\"fit\":%d,\"asts\":%zu,\"configs\":%zu,\"rows\":%zu,\"prepare_seconds\":%.9g,\"wall_seconds\":%.9g,\"resident_seconds\":%.9g,\"samples\":%u,\"registers_max\":%d,\"local_bytes_max\":%d,\"active_blocks_per_sm_min\":%d,\"sm_count\":%d,\"slot_table_bytes\":%zu}\n", fit, p->asts.count, configs, s->rows, prepare, stats->wall_seconds, resident, repeats, s->regs, s->local, s->blocks, s->sm_count, table_bytes);
      CHECK(!fclose(f));
    }
  }
  return SECANT_SUCCESS;
}
SecantResult
bench_cuda_score(SRCudaScorer h, const SecantAstProgramSet *p, SRScore *out, size_t n, SRCudaStats *stats)
{
  BEngine *s = (BEngine *)h;
  return score(s, p, s->base, s->c.num_banks, out, n, stats, 0);
}
SecantResult
bench_cuda_score_banks(SRCudaScorer h, const SecantAstProgramSet *p, const float *banks, size_t nb, size_t elements, SRScore *out, size_t n, SRCudaStats *stats)
{
  BEngine *s = (BEngine *)h;
  CHECK(elements == nb * s->c.num_constants);
  return score(s, p, banks, nb, out, n, stats, 1);
}
SecantResult
bench_cuda_destroy(SRCudaScorer h)
{
  BEngine *s = (BEngine *)h;
  if (!s)
    return SECANT_SUCCESS;
  if (s->native) {
    SEC(secant_sr_cuda_destroy(s->native));
  } else {
    CUDA(cuModuleUnload(s->reducer));
    CUDA(cuMemFree(s->dx));
    CUDA(cuMemFree(s->dy));
    CUDA(cuMemFree(s->db));
    CUDA(cuMemFree(s->grid));
    CUDA(cuMemFree(s->entries));
    CUDA(cuMemFreeHost(s->host));
    if (s->settings) {
      size_t i;
      CUDA(cuMemFree(s->dm));
      CUDA(cuMemFree(s->dw));
      CUDA(cuMemFreeHost(s->masks));
      CUDA(cuMemFreeHost(s->words));
      for (i = 0; i < 4; ++i) {
        s->legacy->destroy(s->templates[i].plan);
        free(s->templates[i].cubin);
        free(s->templates[i].patched);
      } CHECK(!dlclose(s->library));
      s->cubin = NULL;
      s->patched = NULL;
    }
    CUDA(cuEventDestroy(s->start));
    CUDA(cuEventDestroy(s->stop));
    CUDA(cuStreamDestroy(s->stream));
    CUDA(cuDevicePrimaryCtxRelease(s->device));
  }
  free(s->x);
  free(s->y);
  free(s->base);
  free(s->trees);
  free(s->oldcode);
  free(s->oldasts);
  free(s->choices);
  free(s->storage);
  free(s->cubin);
  free(s->patched);
  free(s);
  return SECANT_SUCCESS;
}
