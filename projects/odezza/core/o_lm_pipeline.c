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
#define _POSIX_C_SOURCE 200809L
#include "o_odezza_internal.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define O_LM_STREAMS 2u
#define TRY O_RETURN_IF_ERROR

typedef struct OLmSlot {
  CUstream stream;
  CUevent start, end;
  CUmodule module;
  int active;
} OLmSlot;
typedef struct OLmTemplate {
  OdezzaLmShape shape;
  OdezzaNvrtcCompilation *compilation;
  char *source;
  void *cubin;
  size_t cubin_size, workspace_size;
  OLmInspection *inspection;
  OdezzaResult result;
} OLmTemplate;
struct OdezzaLmPipeline {
  OdezzaLmPipelineCreateInfo info;
  OLmTemplate templates[4];
  OdezzaLmShapeReport shapes;
  size_t workspace_size;
  uint32_t maximum_lanes;
  OLmSlot slots[O_LM_STREAMS];
  pthread_mutex_t mutex;
  int mutex_ready, ready, poisoned;
  char error[512];
};
static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
static OdezzaResult cuda_error(OdezzaLmPipeline *p, CUresult r,
                               const char *where) {
  const char *name = NULL, *text = NULL;
  if (r == CUDA_SUCCESS)
    return ODEZZA_SUCCESS;
  cuGetErrorName(r, &name);
  cuGetErrorString(r, &text);
  snprintf(p->error, sizeof(p->error), "%s: %s: %s", where,
           name ? name : "CUDA error", text ? text : "");
  return ODEZZA_ERROR_CUDA;
}
static OdezzaResult retire(OdezzaLmPipeline *p, OLmSlot *slot,
                           OdezzaLmRunReport *report) {
  float ms;
  OdezzaResult r;
  if (!slot->active) {
    if (slot->module && !p->poisoned) {
      r = cuda_error(p, cuModuleUnload(slot->module),
                     "LM unlaunched module cleanup");
      if (!r)
        slot->module = NULL;
      return r;
    }
    return p->poisoned ? ODEZZA_ERROR_CUDA_UNFENCED : ODEZZA_SUCCESS;
  }
  r = cuda_error(p, cuEventSynchronize(slot->end), "LM completion event");
  if (r) {
    p->poisoned = 1;
    return ODEZZA_ERROR_CUDA_UNFENCED;
  }
  if (report) {
    r = cuda_error(p, cuEventElapsedTime(&ms, slot->start, slot->end),
                   "LM event timing");
    if (r)
      return r;
    report->kernel_seconds += (double)ms / 1000.;
    ++report->completed_system_count;
  }
  slot->active = 0;
  r = cuda_error(p, cuModuleUnload(slot->module), "LM module unload");
  if (!r)
    slot->module = NULL;
  return r;
}
static OdezzaResult prepare_template(OdezzaLmPipeline *p, OLmTemplate *t) {
  size_t size;
  char arch[40];
  const char *options[4];
  OdezzaResult compilation_result;
  TRY(o_lm_generate_cuda(&t->shape, NULL, 0, &size));
  t->source = malloc(size);
  if (!t->source)
    return ODEZZA_ERROR_ALLOCATION;
  TRY(o_lm_generate_cuda(&t->shape, t->source, size, &size));
  snprintf(arch, sizeof(arch), "--gpu-architecture=sm_%u", p->info.sm_version);
  options[0] = arch;
  options[1] = "--std=c++17";
  options[2] = "--use_fast_math";
  options[3] = "--ptxas-options=-v";
  TRY(odezza_nvrtc_compilation_create(t->source, "odezza_lm.cu", options, 4,
                                      &t->compilation));
  TRY(odezza_nvrtc_compilation_result(t->compilation, &compilation_result));
  if (compilation_result) {
    size_t n;
    char *log = NULL;
    if (!odezza_nvrtc_compilation_log_size(t->compilation, &n) &&
        n < SIZE_MAX && (log = malloc(n + 1))) {
      if (!odezza_nvrtc_compilation_write_log(t->compilation, log, n + 1))
        snprintf(p->error, sizeof(p->error), "NVRTC: %.490s", log);
      free(log);
    }
    return compilation_result;
  }
  TRY(odezza_nvrtc_compilation_cubin_size(t->compilation, &t->cubin_size));
  t->cubin = malloc(t->cubin_size);
  t->inspection = malloc(sizeof(*t->inspection));
  if (!t->cubin || !t->inspection)
    return ODEZZA_ERROR_ALLOCATION;
  TRY(odezza_nvrtc_compilation_write_cubin(t->compilation, t->cubin,
                                           t->cubin_size));
  compilation_result = o_lm_inspect(t->cubin, t->cubin_size, &t->shape,
                                    t->inspection);
  if (compilation_result) {
    snprintf(p->error, sizeof(p->error),
             "LM template inspection failed: states=%u parameters=%u "
             "registers=%u result=%d",
             t->shape.state_count, t->shape.parameter_count,
             t->inspection->registers, compilation_result);
    return compilation_result;
  }
  TRY(o_size_multiply(t->shape.site_patch_capacity,
                      sizeof(OSassInstruction), &size));
  TRY(o_size_add(size, sizeof(OLmExpressions), &size));
  TRY(o_size_add(size, t->cubin_size, &t->workspace_size));
  return ODEZZA_SUCCESS;
}
static OdezzaResult create(OdezzaLmPipeline *p) {
  CUmoduleLoadingMode mode;
  uint32_t i;
  OdezzaResult last = ODEZZA_ERROR_REGISTER_PRESSURE;
  TRY(cuda_error(p, cuModuleGetLoadingMode(&mode), "cuModuleGetLoadingMode"));
  if (mode != CU_MODULE_EAGER_LOADING) {
    snprintf(p->error, sizeof(p->error), "Set CUDA_MODULE_LOADING=EAGER before CUDA initialization");
    return ODEZZA_ERROR_UNSUPPORTED;
  }
  for (i = 0; i < 4u; ++i) {
    OLmTemplate *t = &p->templates[i];
    uint32_t width = 1u << i;
    double started;
    t->result = ODEZZA_ERROR_UNSUPPORTED;
    p->shapes.template_results[i] = t->result;
    if (!(width & p->shapes.allowed_lanes_mask)) continue;
    p->maximum_lanes = width;
    t->shape = p->info.shape;
    t->shape.lanes_per_fit = width;
    started = now();
    t->result = prepare_template(p, t);
    p->shapes.template_prepare_seconds += now() - started;
    p->shapes.template_results[i] = t->result;
    if (t->inspection) p->shapes.template_registers[i] = t->inspection->registers;
    if (t->result) {
      if (t->result != ODEZZA_ERROR_REGISTER_PRESSURE) return t->result;
      last = t->result;
      continue;
    }
    p->shapes.available_lanes_mask |= width;
    if (t->workspace_size > p->workspace_size) p->workspace_size = t->workspace_size;
  }
  if (!p->shapes.available_lanes_mask) return last;
  p->error[0] = 0;
  for (i = 0; i < O_LM_STREAMS; ++i) {
    TRY(cuda_error(p,
                   cuStreamCreate(&p->slots[i].stream, CU_STREAM_NON_BLOCKING),
                   "LM stream create"));
    TRY(cuda_error(p, cuEventCreate(&p->slots[i].start, 0),
                   "LM start event create"));
    TRY(cuda_error(p, cuEventCreate(&p->slots[i].end, 0),
                   "LM completion event create"));
  }
  p->ready = 1;
  return ODEZZA_SUCCESS;
}
OdezzaResult odezza_lm_pipeline_create_with_fallback(const OdezzaLmPipelineCreateInfo *info,
                                       uint32_t fallback, OdezzaLmPipeline **out) {
  OdezzaLmPipeline *p;
  OdezzaResult r;
  if (!out)
    return ODEZZA_ERROR_INVALID_ARGUMENT;
  *out = NULL;
  if (!info)
    return ODEZZA_ERROR_INVALID_ARGUMENT;
  TRY(o_lm_validate_shape(&info->shape));
  if ((fallback & ~15u) || (fallback & (2u * info->shape.lanes_per_fit - 1u)))
    return ODEZZA_ERROR_INVALID_ARGUMENT;
  if (!o_sass_architecture_supported(info->sm_version))
    return ODEZZA_ERROR_UNSUPPORTED;
  p = calloc(1, sizeof(*p));
  if (!p)
    return ODEZZA_ERROR_ALLOCATION;
  *out = p;
  p->info = *info;
  p->shapes.requested_lanes_per_fit = info->shape.lanes_per_fit;
  p->shapes.allowed_lanes_mask = info->shape.lanes_per_fit | fallback;
  if (pthread_mutex_init(&p->mutex, NULL))
    return ODEZZA_ERROR_THREAD;
  p->mutex_ready = 1;
  r = create(p);
  if (r && !p->error[0])
    snprintf(p->error, sizeof(p->error), "LM creation failed, result %d", r);
  return r;
}
OdezzaResult odezza_lm_pipeline_create(const OdezzaLmPipelineCreateInfo *info,
                                       OdezzaLmPipeline **out) {
  return odezza_lm_pipeline_create_with_fallback(info, 0u, out);
}
OdezzaResult odezza_lm_pipeline_shape_report(const OdezzaLmPipeline *pipeline,
                                             OdezzaLmShapeReport *out) {
  OdezzaLmPipeline *p = (OdezzaLmPipeline *)pipeline;
  if (!p || !out || !p->mutex_ready) return ODEZZA_ERROR_INVALID_ARGUMENT;
  if (pthread_mutex_trylock(&p->mutex)) return ODEZZA_ERROR_BUSY;
  *out = p->shapes;
  pthread_mutex_unlock(&p->mutex);
  return ODEZZA_SUCCESS;
}
OdezzaResult
odezza_lm_pipeline_workspace_requirements(const OdezzaLmPipeline *p,
                                          size_t *size, size_t *alignment) {
  if (!p || !p->ready || !size || !alignment)
    return ODEZZA_ERROR_INVALID_ARGUMENT;
  *size = p->workspace_size;
  *alignment = 8;
  return ODEZZA_SUCCESS;
}
/* Native pointer spans cannot prove allocation validity, but catch arithmetic
 * overflow, misalignment and aliases before loading or launching any module. */
static OdezzaResult span(CUdeviceptr p, uint64_t bytes) {
  if (!p || p % 4u || !bytes || bytes > UINT64_MAX - p)
    return ODEZZA_ERROR_INVALID_ARGUMENT;
  return ODEZZA_SUCCESS;
}
static OdezzaResult validate(const OdezzaLmPipeline *p, const OdezzaLmFit *f,
                             uint64_t *fits) {
  uint64_t count, m;
  size_t i;
  CUdeviceptr outputs[8] = {f->parameters_device,  f->initial_mse_device,
                            f->mse_device,         f->iterations_device,
                            f->accepted_device,    f->factorizations_device,
                            f->evaluations_device, f->invalid_device};
  if (!f->rhs || !f->start_count || f->toggle_bit_count > 32u ||
      !f->trajectory_count || f->point_count <= f->trajectory_count ||
      f->trajectory_count == UINT32_MAX || !f->steps_per_interval ||
      !f->max_iterations || !f->max_damping_attempts ||
      !isfinite(f->initial_damping) || f->initial_damping <= 0 ||
      !isfinite(f->max_step) || f->max_step <= 0 || !isfinite(f->target_mse) ||
      f->target_mse < 0)
    return ODEZZA_ERROR_INVALID_ARGUMENT;
  m = UINT64_C(1) << f->toggle_bit_count;
  if (f->start_count > UINT64_MAX / m)
    return ODEZZA_ERROR_OVERFLOW;
  /* Phase zero evaluates the current point, followed by damping proposals.
   * Bound both the inclusive phase loop and its public uint32 counters. */
  if ((uint64_t)f->max_iterations * ((uint64_t)f->max_damping_attempts + 1u) >
      UINT32_MAX)
    return ODEZZA_ERROR_OVERFLOW;
  count = f->start_count * m;
  if (count > UINT64_C(2147483647) * (32u / p->maximum_lanes) ||
      count > SIZE_MAX / (4u * p->info.shape.parameter_count))
    return ODEZZA_ERROR_OVERFLOW;
  TRY(span(f->starts_device,
           4 * f->start_count * p->info.shape.parameter_count));
  TRY(span(f->offsets_device, 4 * ((uint64_t)f->trajectory_count + 1u)));
  TRY(span(f->times_device, 4 * (uint64_t)f->point_count));
  TRY(span(f->reference_device,
           4 * (uint64_t)f->point_count * p->info.shape.state_count));
  TRY(span(f->weights_device,
           4 * (uint64_t)f->point_count * p->info.shape.state_count));
  TRY(span(f->lower_device, 4 * p->info.shape.parameter_count));
  TRY(span(f->upper_device, 4 * p->info.shape.parameter_count));
  for (i = 0; i < 8; ++i)
    TRY(span(outputs[i],
             4 * count * (i == 0 ? p->info.shape.parameter_count : 1u)));
  *fits = count;
  return ODEZZA_SUCCESS;
}
typedef struct OLmSpan {
  CUdeviceptr address;
  uint64_t bytes;
} OLmSpan;
static void spans(const OdezzaLmPipeline *p, const OdezzaLmFit *f,
                  uint64_t count, OLmSpan out[15]) {
  uint64_t params = p->info.shape.parameter_count, points = f->point_count,
           states = p->info.shape.state_count;
  CUdeviceptr addresses[15] = {f->starts_device,         f->offsets_device,
                               f->times_device,          f->reference_device,
                               f->weights_device,        f->lower_device,
                               f->upper_device,          f->parameters_device,
                               f->initial_mse_device,    f->mse_device,
                               f->iterations_device,     f->accepted_device,
                               f->factorizations_device, f->evaluations_device,
                               f->invalid_device};
  uint64_t lengths[15] = {4 * f->start_count * params,
                          4 * ((uint64_t)f->trajectory_count + 1),
                          4 * points,
                          4 * points * states,
                          4 * points * states,
                          4 * params,
                          4 * params,
                          4 * count * params,
                          4 * count,
                          4 * count,
                          4 * count,
                          4 * count,
                          4 * count,
                          4 * count,
                          4 * count};
  unsigned i;
  for (i = 0; i < 15; ++i) {
    out[i].address = addresses[i];
    out[i].bytes = lengths[i];
  }
}
static OdezzaResult validate_aliases(const OdezzaLmPipeline *p,
                                     const OdezzaLmFit *fits, size_t count) {
  size_t i, j;
  unsigned a, b;
  uint64_t ni, nj;
  OLmSpan left[15], right[15];
  for (i = 0; i < count; ++i) {
    TRY(validate(p, &fits[i], &ni));
    spans(p, &fits[i], ni, left);
    for (j = 0; j < count; ++j) {
      TRY(validate(p, &fits[j], &nj));
      spans(p, &fits[j], nj, right);
      for (a = 7; a < 15; ++a)
        for (b = 0; b < 15; ++b) {
          if (i == j && a == b)
            continue;
          if (left[a].address < right[b].address + right[b].bytes &&
              right[b].address < left[a].address + left[a].bytes)
            return ODEZZA_ERROR_INVALID_ARGUMENT;
        }
    }
  }
  return ODEZZA_SUCCESS;
}
static OdezzaResult available_slot(OdezzaLmPipeline *p,
                                   OdezzaLmRunReport *report, OLmSlot **out) {
  unsigned i;
  for (;;) {
    for (i = 0; i < O_LM_STREAMS; ++i) {
      OLmSlot *slot = &p->slots[i];
      CUresult state;
      if (!slot->active) {
        *out = slot;
        return ODEZZA_SUCCESS;
      }
      state = cuEventQuery(slot->end);
      if (state == CUDA_ERROR_NOT_READY)
        continue;
      TRY(retire(p, slot, report));
      *out = slot;
      return ODEZZA_SUCCESS;
    }
    {
      struct timespec pause = {0, 50000};
      nanosleep(&pause, NULL);
    }
  }
}
static OdezzaResult launch(OdezzaLmPipeline *p, const OLmTemplate *template, OLmSlot *slot,
                           const OdezzaLmFit *f, uint64_t count, void *image,
                           OdezzaLmRunReport *report) {
  CUfunction function;
  double t;
  size_t shared;
  void *args[] = {(void *)&f->starts_device,
                  (void *)&f->start_count,
                  (void *)&f->offsets_device,
                  (void *)&f->times_device,
                  (void *)&f->reference_device,
                  (void *)&f->weights_device,
                  (void *)&f->trajectory_count,
                  (void *)&f->point_count,
                  (void *)&f->toggle_bit_count,
                  (void *)&f->steps_per_interval,
                  (void *)&f->max_iterations,
                  (void *)&f->max_damping_attempts,
                  (void *)&f->initial_damping,
                  (void *)&f->lower_device,
                  (void *)&f->upper_device,
                  (void *)&f->max_step,
                  (void *)&f->target_mse,
                  (void *)&f->parameters_device,
                  (void *)&f->initial_mse_device,
                  (void *)&f->mse_device,
                  (void *)&f->iterations_device,
                  (void *)&f->accepted_device,
                  (void *)&f->factorizations_device,
                  (void *)&f->evaluations_device,
                  (void *)&f->invalid_device};
  CUresult cr;
  OdezzaResult r;
  shared = 4 * ((size_t)f->point_count * (2 * p->info.shape.state_count + 1u) +
                (size_t)f->trajectory_count + 1u);
  if (shared > 48u * 1024u)
    return ODEZZA_ERROR_UNSUPPORTED;
  t = now();
  TRY(cuda_error(p, cuModuleLoadData(&slot->module, image), "LM module load"));
  report->module_load_seconds += now() - t;
  TRY(cuda_error(
      p, cuModuleGetFunction(&function, slot->module, "odezza_trajectory_lm"),
      "LM function lookup"));
  if (f->input_ready_event)
    TRY(cuda_error(p, cuStreamWaitEvent(slot->stream, f->input_ready_event, 0),
                   "LM input dependency"));
  TRY(cuda_error(p, cuEventRecord(slot->start, slot->stream),
                 "LM start event"));
  cr = cuLaunchKernel(function,
                      (unsigned int)((count + 32u / template->shape.lanes_per_fit - 1u) /
                                     (32u / template->shape.lanes_per_fit)), 1, 1, 32, 1,
                      1, (unsigned int)shared, slot->stream, args, NULL);
  if (cr == CUDA_SUCCESS)
    cr = cuEventRecord(slot->end, slot->stream);
  r = cuda_error(p, cr, "LM launch/completion record");
  if (r) {
    /* Only a failed event fence requires a stream fence. Do not reclaim an
     * unfenced module or caller workspace if CUDA cannot establish safety. */
    if (cuStreamSynchronize(slot->stream) != CUDA_SUCCESS) {
      p->poisoned = 1;
      return ODEZZA_ERROR_CUDA_UNFENCED;
    }
    return r;
  }
  slot->active = 1;
  return ODEZZA_SUCCESS;
}
static OdezzaResult run(OdezzaLmPipeline *p, const OdezzaLmFit *f, size_t n,
                        void *workspace, OdezzaLmRunReport *r) {
  size_t i;
  uint64_t count, total = 0;
  OLmExpressions *expressions = workspace;
  OSassInstruction *code = (OSassInstruction *)(expressions + 1);
  void *image = code + p->info.shape.site_patch_capacity;
  for (i = 0; i < n; ++i) {
    TRY(validate(p, &f[i], &count));
    if (count > UINT64_MAX - total)
      return ODEZZA_ERROR_OVERFLOW;
    total += count;
  }
  TRY(validate_aliases(p, f, n));
  for (i = 0; i < n; ++i) {
    OLmSlot *slot;
    uint32_t regs = 0, j, chosen = 0, inflight = 0;
    OdezzaResult specialized = ODEZZA_ERROR_REGISTER_PRESSURE;
    OLmTemplate *selected = NULL;
    double t;
    TRY(available_slot(p, r, &slot));
    TRY(validate(p, &f[i], &count));
    t = now();
    for (j = 0; j < 4u; ++j) {
      OLmTemplate *candidate = &p->templates[j];
      if (candidate->result) continue;
      p->shapes.attempted_lanes_mask |= 1u << j;
      specialized = o_lm_specialize(candidate->cubin, candidate->cubin_size,
          candidate->inspection, f[i].rhs, f[i].toggle_bit_count,
          expressions, code, image, &regs);
      if (specialized == ODEZZA_ERROR_REGISTER_PRESSURE) {
        ++p->shapes.register_rejections[j];
        continue;
      }
      if (!specialized) { selected = candidate; chosen = j; }
      break;
    }
    r->specialization_seconds += now() - t;
    if (specialized) return specialized;
    if (regs > r->maximum_register_count)
      r->maximum_register_count = regs;
    TRY(launch(p, selected, slot, &f[i], count, image, r));
    p->shapes.used_lanes_mask |= 1u << chosen;
    ++p->shapes.system_counts[chosen];
    r->fit_count += count;
    for (j = 0; j < O_LM_STREAMS; ++j)
      if (p->slots[j].active)
        ++inflight;
    if (inflight > r->peak_inflight_modules)
      r->peak_inflight_modules = inflight;
  }
  return ODEZZA_SUCCESS;
}
OdezzaResult odezza_lm_pipeline_run(OdezzaLmPipeline *p,
                                    const OdezzaLmFit *fits, size_t count,
                                    void *workspace, size_t bytes,
                                    OdezzaLmRunReport *report) {
  OdezzaResult result, drain;
  OdezzaLmRunReport local;
  double t;
  uint32_t i;
  if (!p || !p->ready || !fits || !count || !workspace ||
      (uintptr_t)workspace % 8u)
    return ODEZZA_ERROR_INVALID_ARGUMENT;
  if (p->poisoned)
    return ODEZZA_ERROR_CUDA_UNFENCED;
  if (bytes < p->workspace_size)
    return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
  if (pthread_mutex_trylock(&p->mutex))
    return ODEZZA_ERROR_BUSY;
  memset(&local, 0, sizeof(local));
  p->error[0] = 0;
  p->shapes.used_lanes_mask = p->shapes.attempted_lanes_mask = 0;
  memset(p->shapes.system_counts, 0, sizeof(p->shapes.system_counts));
  memset(p->shapes.register_rejections, 0, sizeof(p->shapes.register_rejections));
  t = now();
  result = run(p, fits, count, workspace, &local);
  for (i = 0; i < O_LM_STREAMS; ++i) {
    drain = retire(p, &p->slots[i], &local);
    if (!result || drain == ODEZZA_ERROR_CUDA_UNFENCED)
      result = drain;
  }
  local.total_seconds = now() - t;
  if (result && !p->error[0])
    snprintf(p->error, sizeof(p->error),
             "LM run failed, result %d; discard outputs", result);
  if (report)
    *report = local;
  pthread_mutex_unlock(&p->mutex);
  return result;
}
OdezzaResult odezza_lm_pipeline_write_error(const OdezzaLmPipeline *p,
                                            char *buffer, size_t capacity,
                                            size_t *size) {
  size_t n;
  if (!p || !size)
    return ODEZZA_ERROR_INVALID_ARGUMENT;
  n = strlen(p->error);
  *size = n;
  if (!buffer)
    return ODEZZA_SUCCESS;
  if (capacity <= n)
    return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
  memcpy(buffer, p->error, n + 1);
  return ODEZZA_SUCCESS;
}
static OdezzaResult release_resources(OdezzaLmPipeline *p) {
  uint32_t i;
  for (i = 0; i < O_LM_STREAMS; ++i) {
    OLmSlot *slot = &p->slots[i];
    TRY(retire(p, slot, NULL));
    if (slot->start) {
      TRY(cuda_error(p, cuEventDestroy(slot->start), "LM start event destroy"));
      slot->start = NULL;
    }
    if (slot->end) {
      TRY(cuda_error(p, cuEventDestroy(slot->end),
                     "LM completion event destroy"));
      slot->end = NULL;
    }
    if (slot->stream) {
      TRY(cuda_error(p, cuStreamDestroy(slot->stream), "LM stream destroy"));
      slot->stream = NULL;
    }
  }
  for (i = 0; i < 4u; ++i) {
    OLmTemplate *t = &p->templates[i];
    if (t->compilation) {
      OdezzaResult result = odezza_nvrtc_compilation_destroy(t->compilation);
      t->compilation = NULL;
      if (result) return result;
    }
  }
  return ODEZZA_SUCCESS;
}
OdezzaResult odezza_lm_pipeline_destroy(OdezzaLmPipeline *p) {
  OdezzaResult result;
  if (!p)
    return ODEZZA_SUCCESS;
  if (p->mutex_ready && pthread_mutex_trylock(&p->mutex))
    return ODEZZA_ERROR_BUSY;
  if (p->poisoned) {
    if (p->mutex_ready)
      pthread_mutex_unlock(&p->mutex);
    return ODEZZA_ERROR_CUDA_UNFENCED;
  }
  p->ready = 0;
  result = release_resources(p);
  if (result != ODEZZA_SUCCESS) {
    if (p->mutex_ready)
      pthread_mutex_unlock(&p->mutex);
    return result;
  }
  {
    uint32_t i;
    for (i = 0; i < 4u; ++i) {
      free(p->templates[i].source);
      free(p->templates[i].cubin);
      free(p->templates[i].inspection);
    }
  }
  if (p->mutex_ready) {
    pthread_mutex_unlock(&p->mutex);
    pthread_mutex_destroy(&p->mutex);
  }
  free(p);
  return ODEZZA_SUCCESS;
}
