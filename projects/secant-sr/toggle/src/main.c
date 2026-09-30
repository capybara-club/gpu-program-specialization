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
#include "secant_sr.h"
#include "secant_sr_refine.h"
#include "secant_sr_archive.h"
#include "secant_sr_lm.h"
#include "secant_instructions.h"
#ifdef SR_HAS_CUDA
#include "secant_sr_cuda.h"
#endif
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef struct FitChoice { size_t index; SRModel model; } FitChoice;
static int fit_order(const void *left, const void *right) {
  const FitChoice *a=left, *b=right;
  if (a->model.score.sse != b->model.score.sse) return a->model.score.sse < b->model.score.sse ? -1 : 1;
  if (a->model.fingerprint != b->model.fingerprint) return a->model.fingerprint < b->model.fingerprint ? -1 : 1;
  return a->index < b->index ? -1 : a->index != b->index;
}
static double seconds(void) {
  struct timespec t;
  if (clock_gettime(CLOCK_MONOTONIC, &t))
    return 0;
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static int integer(const char *s, uint64_t *v) {
  char *end;
  unsigned long long x;
  if (!s || !*s || *s == '-')
    return 0;
  errno = 0;
  x = strtoull(s, &end, 10);
  if (errno || *end)
    return 0;
  *v = x;
  return 1;
}
static int real(const char *s, double *v) {
  char *end;
  errno = 0;
  *v = strtod(s, &end);
  return !errno && end != s && !*end && isfinite(*v);
}
static void number(double v) {
  if (isfinite(v))
    printf("%.12g", v);
  else
    printf("null");
}
static void hex(const uint8_t *p, size_t n) {
  size_t i;
  for (i = 0; i < n; ++i)
    printf("%02x", p[i]);
}
static double score(const uint8_t *ast, size_t inputs, const float *x,
                    const float *y, size_t rows) {
  const uint8_t *asts[] = {ast};
  float sse = 0;
  SecantCpuSSERun run = secant_cpu_sse_run_init();
  run.programs.asts.items = asts;
  run.programs.asts.count = 1;
  run.num_inputs = inputs;
  run.num_targets = 1;
  run.num_rows = rows;
  run.input = (SecantConstHostMatrixF32){x, inputs * rows, rows};
  run.targets = (SecantConstHostMatrixF32){y, rows, rows};
  run.output = (SecantHostMatrixF32){&sse, 1, 1};
  if (secant_cpu_run_sse(&run) != SECANT_SUCCESS)
    return INFINITY;
  return sse;
}
static void help(void) {
  puts("Secant-SR 0.3: explicit-toggle genetic programming\n"
       "  --backend cuda|cpu --problem nguyen1 | --data dataset.bin\n"
       "  --population 1024 --generations 100 --seed 42 --seconds 60\n"
       "  --banks 64 --constants 4 --toggle-bits 6 --bank-seed N\n"
       "  --refine-rounds 0 --refine-budget 128 --refine-scale 1\n"
       "  --lm-iterations 0 --lm-budget 128 --lm-bindings 32 --lm-starts 4\n"
       "  --lm-parameters 8 --lm-interval 1 --lm-scale 1 --lm-threads 64\n"
       "  --refine-parameters all|active-block --power-mutation-probability 0\n"
       "  --finalists 0  (passive archive; emits winning bindings for external polish)\n"
       "  --align-crossover-bits 0 --toggle-mutation-probability 0 --leaf-mix-probability 0\n"
       "  --distribution uniform|normal --constant-min -2 --constant-max 2\n"
       "  --mean 0 --stddev 1 --rows 512 --validation-rows 512\n"
       "  --max-nodes 31 --max-depth 7 --initial-depth 3 --elites 8\n"
       "  --operators add,sub,mul,sin,cos --stop-nmse 1e-10\n"
       "  --toggle-probability 0.8 --four-way-probability 0.3 "
       "--coefficient-probability 0.25\n"
       "  --ast-batch 256 --score-mib 256 --pack 8 --kernels 16 --tile-rows "
       "128\n"
       "  --threads 128 --workers 3 --streams 8 --device 0\n"
       "stdout: JSONL progress and final replayable model; stderr: "
       "diagnostics.\n"
       "CPU and CUDA are explicit backends; GPU LM is opt-in, separate from random refinement.");
}
static int ops(const char *text, uint8_t *out, size_t *count) {
  static const struct {
    const char *name;
    uint8_t op;
  } table[] = {{"add", SECANT_AST_INSTRUCTION_TYPE_ADD_F32},
               {"sub", SECANT_AST_INSTRUCTION_TYPE_SUB_F32},
               {"mul", SECANT_AST_INSTRUCTION_TYPE_MUL_F32},
               {"div", SECANT_AST_INSTRUCTION_TYPE_DIV_F32},
               {"sin", SECANT_AST_INSTRUCTION_TYPE_SIN_F32},
               {"cos", SECANT_AST_INSTRUCTION_TYPE_COS_F32},
               {"tanh", SECANT_AST_INSTRUCTION_TYPE_TANH_F32},
               {"neg", SECANT_AST_INSTRUCTION_TYPE_NEG_F32},
               {"abs", SECANT_AST_INSTRUCTION_TYPE_ABS_F32},
               {"sqrt", SECANT_AST_INSTRUCTION_TYPE_SQRT_F32},
               {"exp", SECANT_AST_INSTRUCTION_TYPE_EXP_F32},
               {"log", SECANT_AST_INSTRUCTION_TYPE_LOG_F32},
               {"min", SECANT_AST_INSTRUCTION_TYPE_MIN_F32},
               {"max", SECANT_AST_INSTRUCTION_TYPE_MAX_F32}};
  const char *at = text;
  size_t n = 0;
  while (*at) {
    const char *end = strchr(at, ',');
    size_t size = end ? (size_t)(end - at) : strlen(at), i;
    for (i = 0; i < sizeof(table) / sizeof(*table); ++i)
      if (strlen(table[i].name) == size && !memcmp(at, table[i].name, size))
        break;
    if (i == sizeof(table) / sizeof(*table) || n == 32)
      return 0;
    out[n++] = table[i].op;
    if (!end)
      break;
    at = end + 1;
    if (!*at)
      return 0;
  }
  *count = n;
  return n != 0;
}
int main(int argc, char **argv) {
  SRConfig c = secant_sr_config_default();
  SRSearch search = NULL;
  SRArchive archive = NULL;
  size_t finalist_capacity = 0;
  double archive_time = 0;
  SRModel best;
  SRScoreAgreement agreement = {0};
  double train_energy = 0;
  const double audit_tolerance = 2e-5;
  SRProgress progress;
  const SecantSRDataset *dataset = NULL;
  const char *problem = "nguyen1", *path = NULL, *backend = "cuda",
             *reason = "generation_budget";
  size_t rows = 512, valrows = 512, generations = 100, ast_batch = 256,
         score_mib = 256, g, i, configs, grid_count, bank_elements;
  uint64_t bank_seed = 0;
  int has_bank_seed = 0, normal = 0, status = 1;
  uint8_t operators[32], resolved[SECANT_AST_MAX_PROGRAM_BYTES], genotype[SECANT_AST_MAX_PROGRAM_BYTES];
  size_t resolved_bytes = 0, genotype_bytes = 0, text_size;
  char expression[8192];
  float *x = NULL, *y = NULL, *vx = NULL, *vy = NULL, *banks = NULL,
        *grid = NULL;
  SRScore *scores = NULL;
  SRRefiner refiner = NULL;
  SRRefineOptions refine_options = secant_sr_refine_options_default();
  SRLMOptions lm_options = secant_sr_lm_options_default();
  SRLMStats lm_initial = {0};
  size_t lm_iterations = 0, lm_interval = 1, lm_models = 0, lm_promoted = 0;
  size_t lm_blocked = 0, lm_inactive = 0;
  uint64_t lm_states = 0, lm_evaluations = 0, lm_rows = 0, lm_steps = 0, lm_rescore_configs = 0;
  uint64_t lm_invalid = 0, lm_shape_states[4] = {0};
  double lm_seconds = 0, lm_device = 0, lm_rescore = 0;
  size_t refine_rounds = 0, refine_budget = 128, refined_models = 0, refine_skipped_capacity = 0, refine_skipped_inactive = 0;
  uint64_t refinement_configs = 0;
  double refinement_time = 0;
  FitChoice *fit_choices = NULL;
  SRModel *fit_models = NULL;
  SRScore *fit_scores = NULL;
  double train_ssd = 0, val_ssd = 0, stop = 1e-10, limit = 60, low = -2,
         high = 2, mean = 0, stddev = 1;
  double started = seconds(), setup = 0, build_time = 0, score_time = 0,
         selection_time = 0, validation_time = 0;
  double pipeline_time = 0, device_time = 0, load_time = 0, reduce_time = 0,
         transfer_time = 0, nvrtc_time = 0;
  double train_sse = INFINITY, val_sse = INFINITY;
  float checkpoint_sse = INFINITY;
  double result_ready = 0, teardown_time = 0;
  SecantResult result = SECANT_SUCCESS;
#ifdef SR_HAS_CUDA
  SRCudaOptions gpu_options = secant_sr_cuda_options_default();
  SRCudaScorer gpu = NULL;
  SRLMCuda lm = NULL;
  SRCudaStats initial;
#endif
  for (i = 1; i < (size_t)argc; ++i) {
    const char *arg = argv[i], *value;
    uint64_t iv;
    double dv;
    if (!strcmp(arg, "--help")) {
      help();
      return 0;
    }
    if (!strcmp(arg, "--list-problems")) {
      secant_sr_datasets_print(stdout);
      return 0;
    }
    if (i + 1 >= (size_t)argc) {
      fprintf(stderr, "missing value for %s\n", arg);
      goto cleanup;
    }
    value = argv[++i];
    if (!strcmp(arg, "--backend")) {
      backend = value;
      continue;
    }
    if (!strcmp(arg, "--problem")) {
      problem = value;
      continue;
    }
    if (!strcmp(arg, "--data")) {
      path = value;
      continue;
    }
    if (!strcmp(arg, "--operators")) {
      if (!ops(value, operators, &c.num_operators))
        goto bad;
      c.operators = operators;
      continue;
    }
    if (!strcmp(arg, "--distribution")) {
      if (!strcmp(value, "normal"))
        normal = 1;
      else if (!strcmp(value, "uniform"))
        normal = 0;
      else
        goto bad;
      continue;
    }
#define UINTOPT(name, destination, max)                                        \
  if (!strcmp(arg, name)) {                                                    \
    if (!integer(value, &iv) || iv > (uint64_t)(max))                          \
      goto bad;                                                                \
    (destination) = iv;                                                        \
    continue;                                                                  \
  }
#define REALOPT(name, destination)                                             \
  if (!strcmp(arg, name)) {                                                    \
    if (!real(value, &dv))                                                     \
      goto bad;                                                                \
    (destination) = dv;                                                        \
    continue;                                                                  \
  }
    UINTOPT("--refine-rounds", refine_rounds, 1000);
    UINTOPT("--lm-iterations", lm_iterations, 1000);
    UINTOPT("--lm-budget", lm_options.capacity, 1000000);
    UINTOPT("--lm-bindings", lm_options.bindings, 65536);
    UINTOPT("--lm-starts", lm_options.starts, 128);
    UINTOPT("--lm-parameters", lm_options.parameters, 8);
    UINTOPT("--lm-threads", lm_options.threads, 128);
    UINTOPT("--lm-interval", lm_interval, 1000000);
    REALOPT("--lm-scale", lm_options.initial_scale);
    UINTOPT("--finalists", finalist_capacity, 256);
    if (!strcmp(arg,"--refine-parameters")) {
      if(!strcmp(value,"all")) refine_options.parameters=SR_REFINE_ALL_PARAMETERS;
      else if(!strcmp(value,"active-block")) refine_options.parameters=SR_REFINE_ACTIVE_BLOCK;
      else goto bad;
      continue;
    }
    UINTOPT("--align-crossover-bits", c.align_crossover_bits, 1);
    REALOPT("--toggle-mutation-probability", c.toggle_mutation_probability);
    REALOPT("--leaf-mix-probability", c.leaf_mix_probability);
    REALOPT("--power-mutation-probability", c.power_mutation_probability);
    UINTOPT("--refine-budget", refine_budget, 1000000);
    REALOPT("--refine-scale", refine_options.initial_scale);
    UINTOPT("--population", c.population, 1000000);
    UINTOPT("--generations", generations, 1000000);
    UINTOPT("--rows", rows, 10000000);
    UINTOPT("--validation-rows", valrows, 10000000);
    UINTOPT("--seed", c.seed, UINT64_MAX);
    if (!strcmp(arg, "--bank-seed")) {
      if (!integer(value, &bank_seed))
        goto bad;
      has_bank_seed = 1;
      continue;
    }
    UINTOPT("--constants", c.num_constants, 128);
    UINTOPT("--banks", c.num_banks, 1000000);
    UINTOPT("--toggle-bits", c.toggle_bits, SECANT_SR_MAX_BITS);
    UINTOPT("--max-nodes", c.max_nodes, SECANT_SR_MAX_NODES);
    UINTOPT("--max-depth", c.max_depth, 32);
    UINTOPT("--initial-depth", c.initial_depth, 32);
    UINTOPT("--elites", c.elites, 1000000);
    UINTOPT("--ast-batch", ast_batch, 1000000);
    UINTOPT("--score-mib", score_mib, 65536);
    REALOPT("--stop-nmse", stop);
    REALOPT("--seconds", limit);
    REALOPT("--constant-min", low);
    REALOPT("--constant-max", high);
    REALOPT("--mean", mean);
    REALOPT("--stddev", stddev);
    REALOPT("--parsimony", c.parsimony);
    REALOPT("--toggle-probability", c.toggle_probability);
    REALOPT("--four-way-probability", c.four_way_probability);
    REALOPT("--coefficient-probability", c.coefficient_probability);
#ifdef SR_HAS_CUDA
    UINTOPT("--pack", gpu_options.asts_per_kernel, 32);
    UINTOPT("--kernels", gpu_options.kernels_per_module, 1024);
    UINTOPT("--tile-rows", gpu_options.tile_rows, 1000000);
    UINTOPT("--threads", gpu_options.threads, 1024);
    UINTOPT("--workers", gpu_options.workers, 256);
    UINTOPT("--streams", gpu_options.streams, 1024);
    UINTOPT("--device", gpu_options.device, INT_MAX);
#endif
#undef UINTOPT
#undef REALOPT
  bad:
    fprintf(stderr, "invalid option or value: %s %s\n", arg, value);
    goto cleanup;
  }
  if (strcmp(backend, "cpu") && strcmp(backend, "cuda")) {
    fprintf(stderr, "backend must be cpu or cuda\n");
    goto cleanup;
  }
  if (lm_iterations && (strcmp(backend,"cuda") || refine_rounds || !lm_interval || !lm_options.capacity)) {
    fprintf(stderr,"LM requires CUDA, a positive budget/interval, and refine-rounds=0\n");
    goto cleanup;
  }
#ifndef SR_HAS_CUDA
  if (!strcmp(backend, "cuda")) {
    fprintf(stderr, "CUDA backend was not built; select --backend cpu "
                    "explicitly or build on a CUDA host\n");
    goto cleanup;
  }
#endif
  if (rows < 2 || valrows < 2 || !generations || !ast_batch || !score_mib ||
      !c.population || !c.num_banks || stop < 0 || limit <= 0) {
    fprintf(stderr, "counts and time budget must be positive; both datasets "
                    "need at least two rows\n");
    goto cleanup;
  }
  if (path) {
    if (!secant_sr_dataset_binary_info(path, &c.num_inputs, &rows, &valrows)) {
      fprintf(stderr, "invalid dataset header\n");
      goto cleanup;
    }
  } else {
    dataset = secant_sr_dataset_find(problem);
    if (!dataset) {
      fprintf(stderr, "unknown problem %s\n", problem);
      goto cleanup;
    }
    c.num_inputs = dataset->num_inputs;
  }
  if (rows > SIZE_MAX / sizeof(float) / c.num_inputs ||
      valrows > SIZE_MAX / sizeof(float) / c.num_inputs)
    goto cleanup;
#define ALLOC(p, n)                                                            \
  do {                                                                         \
    p = malloc((n) * sizeof(*(p)));                                            \
    if (!(p)) {                                                                \
      result = SECANT_ERROR_ALLOCATION_FAILED;                                 \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)
#define SEC(call)                                                              \
  do {                                                                         \
    result = (call);                                                           \
    if (result != SECANT_SUCCESS) {                                            \
      fprintf(stderr, "%s: %s\n", #call, secant_result_to_string(result));     \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)
  ALLOC(x, rows * c.num_inputs);
  ALLOC(y, rows);
  ALLOC(vx, valrows * c.num_inputs);
  ALLOC(vy, valrows);
  if (path) {
    if (!secant_sr_dataset_binary_load(path, c.num_inputs, x, y, rows, vx, vy,
                                       valrows, &train_ssd, &val_ssd)) {
      fprintf(stderr, "invalid dataset body\n");
      goto cleanup;
    }
  } else {
    train_ssd = secant_sr_dataset_fill(dataset, UINT64_C(1839271), x, y, rows);
    val_ssd =
        secant_sr_dataset_fill(dataset, UINT64_C(9182371), vx, vy, valrows);
  }
  configs = c.num_banks * ((size_t)1 << c.toggle_bits);
  if (ast_batch > c.population)
    ast_batch = c.population;
  if (configs > score_mib * 1024u * 1024u / sizeof(float)) {
    fprintf(stderr, "score pool cannot hold one AST's configurations\n");
    goto cleanup;
  }
  if (ast_batch > score_mib * 1024u * 1024u / sizeof(float) / configs)
    ast_batch = score_mib * 1024u * 1024u / sizeof(float) / configs;
  grid_count = ast_batch * configs;
  if (!has_bank_seed)
    bank_seed = c.seed ^ UINT64_C(0x89abcdef12345678);
  bank_elements = c.num_constants * c.num_banks;
  if (bank_elements)
    ALLOC(banks, bank_elements);
  SEC(secant_sr_bank_generate(banks, c.num_banks, c.num_constants, bank_seed,
                              normal, (float)(normal ? mean : low),
                              (float)(normal ? stddev : high)));
  SEC(secant_sr_create(&c, banks, bank_elements, &search));
  ALLOC(scores, c.population);
  if (refine_rounds) {
    if (!refine_budget || !c.num_constants || c.num_banks < 2) {
      fprintf(stderr, "refinement requires a positive budget, coefficient slots and at least two banks\n");
      goto cleanup;
    }
    if (refine_budget > c.population) refine_budget = c.population;
    refine_options.capacity = refine_budget;
    refine_options.trials = c.num_banks;
    refine_options.seed = c.seed ^ UINT64_C(0x7068696c6f785352);
    SEC(secant_sr_refiner_create(&c, &refine_options, &refiner));
    ALLOC(fit_choices, c.population); ALLOC(fit_models, refine_budget); ALLOC(fit_scores, refine_budget);
  }
  if (!strcmp(backend, "cpu")) {
    ALLOC(grid, grid_count);
  }
#ifdef SR_HAS_CUDA
  else {
    gpu_options.ast_batch = ast_batch;
    gpu_options.score_bytes = score_mib * 1024u * 1024u;
    SEC(secant_sr_cuda_create(&c, x, y, rows, banks, &gpu_options, &gpu,
                              &initial));
    nvrtc_time = initial.nvrtc_seconds;
    if(lm_iterations) {
      if(lm_options.capacity>c.population) lm_options.capacity=c.population;
      lm_options.iterations=lm_iterations;
      lm_options.device=gpu_options.device;
      lm_options.seed=c.seed ^ UINT64_C(0x4c4d5f746f67676c);
      SEC(secant_sr_lm_cuda_create(&c,&lm_options,x,y,rows,&lm,&lm_initial));
      ALLOC(fit_choices,c.population);ALLOC(fit_models,lm_options.capacity);ALLOC(fit_scores,lm_options.capacity);
    }
  }
#endif
  if(finalist_capacity) SEC(secant_sr_archive_create(&c,finalist_capacity,&archive));
  setup = seconds() - started;
  printf("{\"event\":\"start\",\"version\":\"%s\",\"backend\":\"%s\","
         "\"population\":%zu,\"inputs\":%zu,"
         "\"rows\":%zu,\"validation_rows\":%zu,\"banks\":%zu,\"constants\":%zu,"
         "\"toggle_bits\":%u,"
         "\"configs_per_ast\":%zu,\"ast_batch\":%zu,\"seed\":\"%llu\",\"bank_"
         "seed\":\"%llu\",\"setup_"
         "seconds\":%.9g,\"nvrtc_seconds\":%.9g}\n",
         SECANT_SR_VERSION, backend, c.population, c.num_inputs, rows, valrows,
         c.num_banks, c.num_constants, c.toggle_bits, configs, ast_batch,
         (unsigned long long)c.seed, (unsigned long long)bank_seed, setup,
         nvrtc_time);
  fflush(stdout);
#ifdef SR_HAS_CUDA
  if(lm) printf("{\"event\":\"lm_setup\",\"backend\":\"cuda_register_interpreter_v1\",\"seconds\":%.9g,\"nvrtc_seconds\":%.9g,\"registers\":%d,\"local_bytes\":%d,\"shared_bytes\":%d}\n",
      lm_initial.setup_seconds,lm_initial.nvrtc_seconds,lm_initial.registers,lm_initial.local_bytes,lm_initial.shared_bytes);
#endif
  for (g = 0; g < generations; ++g) {
    SecantAstProgramSet programs;
    double t = seconds(), generation_begin = t;
    SEC(secant_sr_ask(search, &programs));
    build_time += seconds() - t;
    t = seconds();
    if (!strcmp(backend, "cpu")) {
      size_t first;
      for (first = 0; first < c.population; first += ast_batch) {
        SecantAstProgramSet chunk = programs;
        chunk.asts.items += first;
        chunk.asts.count = c.population - first;
        if (chunk.asts.count > ast_batch)
          chunk.asts.count = ast_batch;
        SEC(secant_sr_cpu_score(&c, &chunk, x, y, rows, banks, grid, grid_count,
                                scores + first));
      }
    }
#ifdef SR_HAS_CUDA
    else {
      SRCudaStats timing;
      SEC(secant_sr_cuda_score(gpu, &programs, scores, c.population, &timing));
      pipeline_time += timing.pipeline_seconds;
      device_time += timing.device_seconds;
      load_time += timing.module_load_seconds;
      reduce_time += timing.reduction_seconds;
      transfer_time += timing.transfer_seconds;
    }
#endif
    score_time += seconds() - t;
    t = seconds();
    SEC(secant_sr_tell(search, scores, c.population, rows, train_ssd));
    selection_time += seconds() - t;
    progress = secant_sr_progress(search);
    if (refiner && progress.best_mse * rows / (train_ssd > 0 ? train_ssd : rows) > stop &&
        seconds()-started < limit) {
      size_t selected = 0, j, round;
      double fit_begin = seconds();
      for (j=0; j<c.population; ++j) {
        fit_choices[j].index = j;
        SEC(secant_sr_candidate(search, j, &fit_choices[j].model));
      }
      qsort(fit_choices, c.population, sizeof(*fit_choices), fit_order);
      for (j=0; j<c.population && selected<refine_budget; ++j) {
        size_t parameters;
        if (!fit_choices[j].model.score.valid_configurations) continue;
        SEC(secant_sr_refine_parameter_count(&c, &fit_choices[j].model, &parameters));
        if (!parameters) continue;
        if (refine_options.parameters==SR_REFINE_ACTIVE_BLOCK) {
          size_t active;
          SEC(secant_sr_refine_active_parameter_count(&c,&fit_choices[j].model,&active));
          if(!active) { ++refine_skipped_inactive;continue; }
        } else if (parameters > c.num_constants) { ++refine_skipped_capacity; continue; }
        /* Avoid spending the bounded fitting budget on exact duplicate genomes. */
        if (selected && fit_choices[j].model.fingerprint == fit_models[selected-1].fingerprint) continue;
        fit_choices[selected] = fit_choices[j];
        fit_models[selected++] = fit_choices[j].model;
      }
      if (selected) {
        SEC(secant_sr_refiner_seed(refiner, fit_models, selected, banks, bank_elements, g));
        for (round=0; round<refine_rounds; ++round) {
          SecantAstProgramSet fitting;
          const float *jitter;
          size_t jitter_elements;
          SEC(secant_sr_refiner_ask(refiner, &fitting, &jitter, &jitter_elements));
          if (!strcmp(backend, "cpu")) {
            size_t first;
            for (first=0; first<selected; first+=ast_batch) {
              SecantAstProgramSet chunk=fitting;
              chunk.asts.items += first; chunk.asts.count=selected-first;
              if (chunk.asts.count>ast_batch) chunk.asts.count=ast_batch;
              SEC(secant_sr_cpu_score(&c, &chunk, x, y, rows, jitter, grid, grid_count, fit_scores+first));
            }
          }
#ifdef SR_HAS_CUDA
          else {
            SRCudaStats timing;
            SEC(secant_sr_cuda_score_banks(gpu, &fitting, jitter, refine_options.trials,
                jitter_elements, fit_scores, refine_budget, &timing));
            pipeline_time += timing.pipeline_seconds; device_time += timing.device_seconds;
            load_time += timing.module_load_seconds; reduce_time += timing.reduction_seconds;
            transfer_time += timing.transfer_seconds;
          }
#endif
          SEC(secant_sr_refiner_tell(refiner, fit_scores, selected));
          if (seconds()-started >= limit) break;
        }
        for (j=0; j<selected; ++j) {
          SRModel fitted;
          SEC(secant_sr_refiner_model(refiner, j, &fitted));
          SEC(secant_sr_accept_refined(search, fit_choices[j].index, &fitted));
        }
        refinement_configs += secant_sr_refiner_stats(refiner).configurations;
        refined_models += selected;
      }
      refinement_time += seconds()-fit_begin;
    }
 #ifdef SR_HAS_CUDA
    if(lm && g%lm_interval==0 && progress.best_mse*rows/(train_ssd>0?train_ssd:rows)>stop && seconds()-started<limit) {
      size_t selected=0,j,k;
      double fit_begin=seconds();
      for(j=0;j<c.population;++j) {
        fit_choices[j].index=j;
        SEC(secant_sr_candidate(search,j,&fit_choices[j].model));
      }
      qsort(fit_choices,c.population,sizeof(*fit_choices),fit_order);
      for(j=0;j<c.population && selected<lm_options.capacity;++j) {
        size_t parameters;
        if(!fit_choices[j].model.score.valid_configurations) continue;
        SEC(secant_sr_refine_parameter_count(&c,&fit_choices[j].model,&parameters));
        if(!parameters) continue;
        for(k=0;k<selected;++k) if(fit_models[k].fingerprint==fit_choices[j].model.fingerprint) break;
        if(k<selected) continue;
        fit_choices[selected]=fit_choices[j];fit_models[selected++]=fit_choices[j].model;
      }
      if(selected && seconds()-started<limit) {
        SecantAstProgramSet fitting;
        SRLMStats timing;
        SRCudaStats rescore;
        double t;
        SEC(secant_sr_lm_cuda_fit(lm,fit_models,selected,banks,bank_elements,g,fmax(1e-6,limit-(seconds()-started)),&fitting,&timing));
        lm_states+=timing.states;lm_evaluations+=timing.statistics_evaluations;
        lm_rows+=timing.row_evaluations;lm_steps+=timing.accepted_steps;
        lm_invalid+=timing.invalid_evaluations;
        for(k=0;k<4;++k)lm_shape_states[k]+=timing.states_by_shape[k];
        lm_device+=timing.device_seconds;lm_blocked+=timing.blocked_bindings;lm_inactive+=timing.inactive_bindings;
        /* Fitted centers are embedded in the AST; a single bank is sufficient.
         * All native toggle bindings are rescored before accepting any proposal. */
        t=seconds();
        SEC(secant_sr_cuda_score_banks(gpu,&fitting,banks,1,c.num_constants,fit_scores,selected,&rescore));
        lm_rescore+=seconds()-t;lm_rescore_configs+=(uint64_t)selected<<c.toggle_bits;
        for(j=0;j<selected;++j) {
          SRModel fitted;
          if(!fit_scores[j].valid_configurations || fit_scores[j].sse>=fit_models[j].score.sse) continue;
          SEC(secant_sr_lm_cuda_model(lm,j,&fitted));
          fitted.score=fit_scores[j];fitted.score.valid_configurations=1;
          SEC(secant_sr_accept_refined(search,fit_choices[j].index,&fitted));++lm_promoted;
        }
        lm_models+=selected;
      }
      lm_seconds+=seconds()-fit_begin;
    }
 #endif
    if(archive){
      double begin=seconds();size_t j;
      for(j=0;j<c.population;++j){SRModel model;SEC(secant_sr_candidate(search,j,&model));SEC(secant_sr_archive_offer(archive,&model,banks,bank_elements));}
      archive_time+=seconds()-begin;
    }
    progress = secant_sr_progress(search);
    if(secant_sr_best(search,&best)==SECANT_SUCCESS && best.score.sse<checkpoint_sse) {
      size_t checkpoint_bytes;
      SEC(secant_sr_resolve(&c,best.nodes,best.num_nodes,banks,bank_elements,best.score.configuration,resolved,sizeof(resolved),&checkpoint_bytes));
      printf("{\"event\":\"incumbent\",\"generation\":%zu,\"elapsed_seconds\":%.9g,\"train_nmse\":%.12g,\"resolved_ast_hex\":\"",
          g,seconds()-started,best.score.sse/(train_ssd>0?train_ssd:rows));
      hex(resolved,checkpoint_bytes);printf("\"}\n");checkpoint_sse=best.score.sse;
    }
    printf("{\"event\":\"generation\",\"generation\":%zu,\"ast_occurrences\":%"
           "zu,\"configurations\":\"%"
           "llu\",\"finite_configurations\":\"%llu\",\"best_mse\":",
           g, progress.structures, (unsigned long long)progress.configurations,
           (unsigned long long)progress.finite_configurations);
    number(progress.best_mse);
    printf(",\"refinement_configurations\":\"%llu\",\"refinement_seconds\":%.9g",
           (unsigned long long)refinement_configs, refinement_time);
    printf(",\"lm_seconds\":%.9g,\"lm_device_seconds\":%.9g,\"lm_models\":%zu,\"lm_promoted\":%zu,\"lm_row_evaluations\":\"%llu\"",
        lm_seconds,lm_device,lm_models,lm_promoted,(unsigned long long)lm_rows);
    printf(",\"generation_seconds\":%.9g,\"elapsed_seconds\":%.9g}\n",
           seconds() - generation_begin, seconds() - started);
    fflush(stdout);
    if (progress.best_mse * rows / (train_ssd > 0 ? train_ssd : rows) <= stop) {
      reason = "training_threshold";
      break;
    }
    if (seconds() - started >= limit) {
      reason = "time_budget";
      break;
    }
    if (g + 1 < generations) {
      t = seconds();
      SEC(secant_sr_advance(search));
      build_time += seconds() - t;
    }
  }
  if(archive){
    size_t j,k;double begin=seconds();SRArchiveStats stats=secant_sr_archive_stats(archive);
    printf("{\"event\":\"finalists\",\"capacity\":%zu,\"diversity\":\"ordered_structure_inputs_literals_parameter_sharing\",\"visited\":\"%llu\",\"duplicates\":\"%llu\",\"score_pruned\":\"%llu\",\"models\":[",finalist_capacity,(unsigned long long)stats.visited,(unsigned long long)stats.duplicates,(unsigned long long)stats.score_pruned);
    for(j=0;j<secant_sr_archive_count(archive);++j){
      SRModel model;size_t n,bytes;uint8_t encoded[SECANT_AST_MAX_PROGRAM_BYTES];
      SEC(secant_sr_archive_model(archive,j,&model));
      printf("%s{\"generation\":%zu,\"gpu_sse\":%.9g,\"permutation\":0,\"coefficients\":[",j?",":"",model.generation,(double)model.score.sse);
      for(k=0;k<c.num_constants;++k)printf("%s0",k?",":"");
      printf("],\"fitted_leaves\":[");
      for(k=0,n=0;k<model.num_nodes;++k)if(!model.nodes[k].op&&model.nodes[k].leaf[0].kind==SR_FITTED_COEFFICIENT){
        const SRLeaf *l=model.nodes[k].leaf;
        printf("%s{\"node\":%zu,\"alternative\":0,\"slot\":%u,\"value\":%.9g}",n++?",":"",k,l->slot,(double)l->value);
      }
      SEC(secant_sr_program_write(&c,model.nodes,model.num_nodes,encoded,sizeof(encoded),&bytes));
      printf("],\"genotype_hex\":\"");hex(encoded,bytes);
      SEC(secant_sr_resolve(&c,model.nodes,model.num_nodes,banks,bank_elements,0,encoded,sizeof(encoded),&bytes));
      printf("\",\"resolved_ast_hex\":\"");hex(encoded,bytes);printf("\"}");
    }
    archive_time+=seconds()-begin;
    printf("],\"seconds\":%.9g}\n",archive_time);
  }
  result = secant_sr_best(search, &best);
  if (result != SECANT_SUCCESS) {
    printf("{\"event\":\"result\",\"status\":\"no_finite_model\",\"solved\":"
           "false}\n");
    status = 0;
    goto cleanup;
  }
  {
    double t = seconds();
    SEC(secant_sr_resolve(&c, best.nodes, best.num_nodes, banks, bank_elements,
                          best.score.configuration, resolved, sizeof(resolved),
                          &resolved_bytes));
    SEC(secant_sr_program_write(&c, best.nodes, best.num_nodes, genotype,
                                sizeof(genotype), &genotype_bytes));
    SEC(secant_sr_format(&c, best.nodes, best.num_nodes, banks, bank_elements,
                         best.score.configuration, expression,
                         sizeof(expression), &text_size));
    train_sse = score(resolved, c.num_inputs, x, y, rows);
    val_sse = score(resolved, c.num_inputs, vx, vy, valrows);
    for (i = 0; i < rows; ++i)
      train_energy += (double)y[i] * y[i];
    if (secant_sr_score_agreement(best.score.sse, train_sse, train_energy, rows,
                                  audit_tolerance,
                                  &agreement) != SECANT_SUCCESS ||
        !agreement.accepted) {
      fprintf(stderr, "winner replay mismatch: GPU/CPU %.9g %.9g\n",
              best.score.sse, train_sse);
      fprintf(
          stderr,
          "audit_expression=%s\naudit_configuration=%llu\naudit_coefficients=",
          expression, (unsigned long long)best.score.configuration);
      for (i = 0; i < c.num_constants; ++i)
        fprintf(stderr, "%s%.9g", i ? "," : "",
                banks[(best.score.configuration >> c.toggle_bits) *
                          c.num_constants +
                      i]);
      fprintf(stderr, "\naudit_genotype=");
      for (i = 0; i < genotype_bytes; ++i)
        fprintf(stderr, "%02x", genotype[i]);
      fprintf(stderr, "\naudit_resolved=");
      for (i = 0; i < resolved_bytes; ++i)
        fprintf(stderr, "%02x", resolved[i]);
      fputc('\n', stderr);
      goto cleanup;
    }
    validation_time = seconds() - t;
  }
  result_ready = seconds() - started;
#ifdef SR_HAS_CUDA
  if(lm) {
    double t=seconds();SEC(secant_sr_lm_cuda_destroy(lm));lm=NULL;teardown_time+=seconds()-t;
  }
  if (gpu) {
    double t = seconds();
    SEC(secant_sr_cuda_destroy(gpu));
    gpu = NULL;
    teardown_time += seconds() - t;
  }
#endif
  printf("{\"event\":\"result\",\"criterion\":\"train_and_validation_nmse\","
         "\"symbolic_equivalence\":\"not_"
         "checked\",\"version\":\"%s\",\"backend\":\"%s\",\"status\":\"%s\","
         "\"solved\":%s,"
         "\"expression\":\"%s\",\"train_mse\":",
         SECANT_SR_VERSION, backend, reason,
         train_sse / (train_ssd > 0 ? train_ssd : rows) <= stop &&
                 val_sse / (val_ssd > 0 ? val_ssd : valrows) <= stop
             ? "true"
             : "false",
         expression);
  number(train_sse / rows);
  printf(",\"validation_mse\":");
  number(val_sse / valrows);
  printf(",\"train_nmse\":");
  number(train_sse / (train_ssd > 0 ? train_ssd : rows));
  printf(",\"validation_nmse\":");
  number(val_sse / (val_ssd > 0 ? val_ssd : valrows));
  printf(",\"lm\":{\"backend\":\"cuda_register_interpreter_v1\",\"iterations\":%zu,\"interval\":%zu,\"budget\":%zu,\"bindings\":%zu,\"starts\":%zu,\"parameter_block\":%u,\"models\":%zu,\"promoted\":%zu,\"blocked_bindings\":%zu,\"inactive_bindings\":%zu,\"states\":\"%llu\",\"statistics_evaluations\":\"%llu\",\"row_evaluations\":\"%llu\",\"accepted_steps\":\"%llu\",\"rescore_configurations\":\"%llu\",\"seconds\":%.9g,\"device_seconds\":%.9g,\"rescore_seconds\":%.9g,\"setup_seconds\":%.9g,\"nvrtc_seconds\":%.9g,\"registers\":%d,\"local_bytes\":%d,\"shared_bytes\":%d,\"start_rng\":\"philox4x32_10_host\"}",
      lm_iterations,lm_interval,lm_options.capacity,lm_options.bindings,lm_options.starts,lm_options.parameters,lm_models,lm_promoted,lm_blocked,lm_inactive,
      (unsigned long long)lm_states,(unsigned long long)lm_evaluations,(unsigned long long)lm_rows,(unsigned long long)lm_steps,(unsigned long long)lm_rescore_configs,
      lm_seconds,lm_device,lm_rescore,lm_initial.setup_seconds,lm_initial.nvrtc_seconds,lm_initial.registers,lm_initial.local_bytes,lm_initial.shared_bytes);
  printf(",\"lm_resources\":{\"multiprocessors\":%d,\"invalid_evaluations\":\"%llu\",\"shapes\":[",lm_initial.multiprocessors,(unsigned long long)lm_invalid);
  for(i=0;i<4;++i)printf("%s{\"parameter_capacity\":%u,\"lanes_per_row\":%u,\"registers\":%d,\"local_bytes\":0,\"active_blocks_per_sm\":%d,\"states\":\"%llu\"}",
      i?",":"",1u<<(unsigned)i,i==3?2:1,lm_initial.registers_by_shape[i],lm_initial.blocks_by_shape[i],(unsigned long long)lm_shape_states[i]);
  printf("]}");
  printf(",\"refinement\":{\"rounds_per_generation\":%zu,\"budget\":%zu,"
         "\"trials\":%zu,\"models\":%zu,\"configurations\":\"%llu\","
         "\"skipped_parameter_capacity\":%zu,\"seconds\":%.9g,"
         "\"rng\":\"philox4x32_10_host\",\"policy\":\"adaptive_winner\",\"parameters\":\"%s\",\"skipped_inactive\":%zu}",
         refine_rounds, refine_budget, c.num_banks, refined_models,
         (unsigned long long)refinement_configs, refine_skipped_capacity, refinement_time,
         refine_options.parameters==SR_REFINE_ACTIVE_BLOCK?"active-block":"all",refine_skipped_inactive);
  printf(",\"variation\":{\"align_crossover_bits\":%u,\"toggle_mutation_probability\":%.9g,"
         "\"leaf_mix_probability\":%.9g,\"aligned_crossovers\":\"%llu\",\"toggle_mutations\":\"%llu\","
         "\"leaf_mixes\":\"%llu\",\"reused_toggle_bits\":\"%llu\",\"power_mutation_probability\":%.9g,\"power_mutations\":\"%llu\"}",
         c.align_crossover_bits, c.toggle_mutation_probability, c.leaf_mix_probability,
         (unsigned long long)progress.aligned_crossovers, (unsigned long long)progress.toggle_mutations,
         (unsigned long long)progress.leaf_mixes, (unsigned long long)progress.reused_toggle_bits,
         c.power_mutation_probability,(unsigned long long)progress.power_mutations);
  printf(",\"score_audit\":{\"policy\":\"scaled_rmse_v1\",\"gpu_sse\":%.12g,"
         "\"cpu_sse\":%.12g,\"relative_rmse_gap\":%.12g,\"tolerance\":%.12g,"
         "\"scale\":%.12g,\"accepted\":true}",
         (double)best.score.sse, train_sse, agreement.relative_rmse_gap,
         audit_tolerance, agreement.scale);
  printf(",\"configuration\":\"%llu\",\"bank_index\":\"%llu\",\"permutation\":%"
         "u,\"coefficients\":[",
         (unsigned long long)best.score.configuration,
         (unsigned long long)(best.score.configuration >> c.toggle_bits),
         (unsigned)(best.score.configuration &
                    (((uint64_t)1 << c.toggle_bits) - 1)));
  for (i = 0; i < c.num_constants; ++i)
    printf("%s%.9g", i ? "," : "",
           banks[(best.score.configuration >> c.toggle_bits) * c.num_constants +
                 i]);
  printf("],\"coefficient_semantics\":\"raw_bank_vector\",\"fitted_leaves\":[");
  {
    size_t node, alternative, count = 0;
    for (node = 0; node < best.num_nodes; ++node)
      for (alternative = 0; alternative < best.nodes[node].choices; ++alternative) {
        const SRLeaf *leaf = best.nodes[node].leaf + alternative;
        if (leaf->kind == SR_FITTED_COEFFICIENT) {
          printf("%s{\"node\":%zu,\"alternative\":%zu,\"slot\":%u,\"value\":%.9g}",
              count++ ? "," : "", node, alternative, leaf->slot, (double)leaf->value);
        }
      }
  }
  printf("],\"genotype_hex\":\"");
  hex(genotype, genotype_bytes);
  printf("\",\"resolved_ast_hex\":\"");
  hex(resolved, resolved_bytes);
  printf("\",\"ast_occurrences\":%zu,\"configurations\":\"%llu\",\"rejected_"
         "variations\":\"%llu\","
         "\"timing\":{\"total_seconds\":%.9g,\"setup_seconds\":%.9g,\"nvrtc_"
         "seconds\":%.9g,\"generation_"
         "seconds\":%.9g,\"scoring_seconds\":%.9g,\"selection_seconds\":%.9g,"
         "\"pipeline_seconds\":%.9g,"
         "\"device_seconds\":%.9g,\"module_load_seconds\":%.9g,\"reduction_"
         "seconds\":%.9g,\"transfer_"
         "seconds\":%.9g,\"validation_seconds\":%.9g,\"result_ready_seconds\":%"
         ".9g,\"teardown_seconds\":%."
         "9g}}\n",
         progress.structures, (unsigned long long)progress.configurations,
         (unsigned long long)progress.rejected_variations, seconds() - started,
         setup, nvrtc_time, build_time, score_time, selection_time,
         pipeline_time, device_time, load_time, reduce_time, transfer_time,
         validation_time, result_ready, teardown_time);
  status = 0;
cleanup:
#ifdef SR_HAS_CUDA
  if(lm) {
    SecantResult cleanup=secant_sr_lm_cuda_destroy(lm);
    if(cleanup!=SECANT_SUCCESS) {fprintf(stderr,"LM cleanup retained resources: %s\n",secant_result_to_string(cleanup));status=1;}
  }
  if (gpu) {
    SecantResult cleanup = secant_sr_cuda_destroy(gpu);
    if (cleanup != SECANT_SUCCESS) {
      fprintf(stderr,
              "CUDA cleanup retained resources: %s; worker process must exit\n",
              secant_result_to_string(cleanup));
      status = 1;
    }
  }
#endif
  secant_sr_refiner_destroy(refiner);
  secant_sr_archive_destroy(archive);
  free(fit_choices); free(fit_models); free(fit_scores);
  secant_sr_destroy(search);
  free(x);
  free(y);
  free(vx);
  free(vy);
  free(banks);
  free(grid);
  free(scores);
  return status;
#undef SEC
#undef ALLOC
}
