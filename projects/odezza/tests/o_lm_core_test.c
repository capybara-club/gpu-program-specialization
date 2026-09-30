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
/* A public-header-only consumer. The test owns CUDA context/data, never
 * streams. */
#include "odezza.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x)                                                               \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "failed line %d: %s\n", __LINE__, #x);                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define CUDA(x) CHECK((x) == CUDA_SUCCESS)
static CUdeviceptr allocation(size_t n, const void *data) {
  CUdeviceptr p;
  CUDA(cuMemAlloc(&p, n));
  if (data)
    CUDA(cuMemcpyHtoD(p, data, n));
  return p;
}
/* Independent CPU residuals and finite-difference LM proposal. Uses the same
 * RK4 discretization, no generated partials or GPU optimizer code. */
static void cpu_residuals(const double p[3], const float *times,
                          const float *reference, double residual[60]) {
  double state[3] = {1, 1, 1};
  int point, step, stage, s;
  for (point = 1; point < 21; ++point) {
    double h = ((double)times[point] - times[point-1]) / 4.;
    for (step = 0; step < 4; ++step) {
      double k[4][3], x[3];
      for (stage = 0; stage < 4; ++stage) {
        for (s = 0; s < 3; ++s)
          x[s] = state[s] + (stage ? h*(stage == 3 ? 1. : .5)*k[stage-1][s] : 0.);
        k[stage][0] = p[0]*x[0] + 2.*x[1];
        k[stage][1] = p[1]*x[1] + 3.*x[0];
        k[stage][2] = p[2]*x[2];
      }
      for (s = 0; s < 3; ++s)
        state[s] += h*(k[0][s]+2.*k[1][s]+2.*k[2][s]+k[3][s])/6.;
    }
    for (s = 0; s < 3; ++s)
      residual[(point-1)*3+s] = state[s] - reference[s*21+point];
  }
}
static void cpu_proposal(const float *starts, const float *times,
                          const float *reference, double proposal[3]) {
  double p[3], r[60], j[3][60], lo[60], hi[60], a[3][4] = {{0}};
  int x, y, k;
  for (x = 0; x < 3; ++x) p[x] = starts[x];
  cpu_residuals(p, times, reference, r);
  for (x = 0; x < 3; ++x) {
    p[x] -= 1e-5; cpu_residuals(p, times, reference, lo);
    p[x] += 2e-5; cpu_residuals(p, times, reference, hi);
    p[x] = starts[x];
    for (k = 0; k < 60; ++k) j[x][k] = (hi[k]-lo[k])/2e-5;
  }
  for (x = 0; x < 3; ++x) {
    for (k = 0; k < 60; ++k) a[x][3] -= j[x][k]*r[k];
    for (y = 0; y < 3; ++y)
      for (k = 0; k < 60; ++k) a[x][y] += j[x][k]*j[y][k];
    a[x][x] *= 1.001;
  }
  for (x = 0; x < 3; ++x) {
    double pivot = a[x][x];
    for (y = x; y < 4; ++y) a[x][y] /= pivot;
    for (k = 0; k < 3; ++k) if (k != x) {
      double scale = a[k][x];
      for (y = x; y < 4; ++y) a[k][y] -= scale*a[x][y];
    }
  }
  for (x = 0; x < 3; ++x)
    proposal[x] = fmin(0., fmax(-2., starts[x]+fmin(1., fmax(-1., a[x][3]))));
}
#ifdef ODEZZA_LM_FAULT_TEST
extern int o_test_fail_load, o_test_fail_launch, o_test_fail_event;
extern int o_test_fail_specialize;
void o_test_check_resources(int expected_streams);
#endif
int main(int argc, char **argv) {
  CUdevice device;
  CUcontext context;
  OdezzaLmPipeline *lm = NULL;
  OdezzaLmPipelineCreateInfo info = {120, {3, 3, 256, 1}};
  uint8_t code[3][6] = {{0x81, 0, 0x82, 0, 0x92, 0x80},
                        {0x81, 1, 0x82, 1, 0x92, 0x80},
                        {0x81, 2, 0x82, 2, 0x92, 0x80}};
  OdezzaAstProgram rhs[3];
  OdezzaLmFit fit[3];
  OdezzaLmRunReport report;
  OdezzaLmShapeReport shapes;
  float starts[67 * 3], reference[3 * 21], weights[3 * 21], times[21],
      lower[3] = {-2, -2, -2}, upper[3] = {0, 0, 0};
  float target[3] = {-.3f, -.6f, -.9f}, out[67], coefficients[67 * 3];
  uint32_t offsets[2] = {0, 21};
  size_t bytes, alignment;
  void *workspace;
  int i, j, k, major, minor;
  OdezzaResult result;
  CUDA(cuInit(0));
  CUDA(cuDeviceGet(&device, 0));
  CUDA(cuDevicePrimaryCtxRetain(&context, device));
  CUDA(cuCtxSetCurrent(context));
  CUDA(cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CUDA(cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  info.sm_version = (uint32_t)(major * 10 + minor);
  if (argc > 1)
    info.shape.lanes_per_fit = (uint32_t)atoi(argv[1]);
  /* The current eight-state/six-parameter template reaches 255 registers on
   * SM 89/120. This is a resource rejection, not corrupt cubin input. Destroy
   * the failed creation, then verify ordinary fits still work in this context. */
  if (info.sm_version == 89u || info.sm_version == 120u) {
    OdezzaLmPipelineCreateInfo large = {info.sm_version, {8, 6, 256, 1}};
    OdezzaLmPipeline *rejected = NULL;
    char error[512];
    size_t error_size;
    CHECK(odezza_lm_pipeline_create(&large, &rejected) ==
          ODEZZA_ERROR_REGISTER_PRESSURE);
    CHECK(odezza_lm_pipeline_write_error(rejected, error, sizeof(error),
                                        &error_size) == ODEZZA_SUCCESS);
    CHECK(strstr(error, "registers=255") != NULL);
    CHECK(odezza_lm_pipeline_destroy(rejected) == ODEZZA_SUCCESS);
  }
  for (i = 0; i < 3; ++i) {
    rhs[i].bytes = code[i];
    rhs[i].byte_count = 6;
  }
  for (i = 0; i < 67; ++i)
    for (j = 0; j < 3; ++j)
      starts[i * 3 + j] = -.1f - .02f * (float)((i + j) % 40);
  for (i = 0; i < 21; ++i) {
    times[i] = .05f * i;
    for (j = 0; j < 3; ++j) {
      reference[j * 21 + i] = (float)exp(target[j] * times[i]);
      weights[j * 21 + i] = 1;
    }
  }
  /* Coupled analytic trajectories exercise simultaneous RK4 sensitivities.
   * The diagonal-only fixture cannot detect an in-place stage update. */
  if (argc > 2 && !strcmp(argv[2], "coupled")) {
    static uint8_t coupled[2][14] = {
      {0x81,0,0x82,0,0x92,0x83,0,0,0,0x40,0x81,1,0x92,0x90},
      {0x81,1,0x82,1,0x92,0x83,0,0,0x40,0x40,0x81,0,0x92,0x90}};
    static uint8_t programs[2][15];
    for (j = 0; j < 2; ++j) {
      memcpy(programs[j], coupled[j], 14); programs[j][14] = 0x80;
      rhs[j].bytes = programs[j]; rhs[j].byte_count = 15;
    }
    for (i = 0; i < 21; ++i) {
      double m = (target[0] + target[1]) / 2., d = (target[0] - target[1]) / 2.;
      double q = sqrt(d*d + 6.), t = times[i], e = exp(m*t);
      reference[i] = (float)(e*(cosh(q*t) + (d+2.)*sinh(q*t)/q));
      reference[21+i] = (float)(e*(cosh(q*t) + (3.-d)*sinh(q*t)/q));
    }
  }
  memset(fit, 0, sizeof(fit));
  fit[0].rhs = rhs;
  fit[0].start_count = 67;
  fit[0].trajectory_count = 1;
  fit[0].point_count = 21;
  fit[0].steps_per_interval = 4;
  fit[0].max_iterations = 24;
  fit[0].max_damping_attempts = 8;
  fit[0].initial_damping = .001f;
  fit[0].max_step = 1;
  fit[0].target_mse = 1e-13f;
  fit[0].starts_device = allocation(sizeof(starts), starts);
  fit[0].offsets_device = allocation(sizeof(offsets), offsets);
  fit[0].times_device = allocation(sizeof(times), times);
  fit[0].reference_device = allocation(sizeof(reference), reference);
  fit[0].weights_device = allocation(sizeof(weights), weights);
  fit[0].lower_device = allocation(sizeof(lower), lower);
  fit[0].upper_device = allocation(sizeof(upper), upper);
  for (k = 0; k < 3; ++k) {
    fit[k] = fit[0];
    fit[k].parameters_device = allocation(sizeof(coefficients), NULL);
    fit[k].initial_mse_device = allocation(sizeof(out), NULL);
    fit[k].mse_device = allocation(sizeof(out), NULL);
    fit[k].iterations_device = allocation(sizeof(out), NULL);
    fit[k].accepted_device = allocation(sizeof(out), NULL);
    fit[k].factorizations_device = allocation(sizeof(out), NULL);
    fit[k].evaluations_device = allocation(sizeof(out), NULL);
    fit[k].invalid_device = allocation(sizeof(out), NULL);
  }
  result = argc > 3 ? odezza_lm_pipeline_create_with_fallback(&info, 2u*info.shape.lanes_per_fit, &lm)
                    : odezza_lm_pipeline_create(&info, &lm);
  if (result) {
    char error[512];
    size_t n = 0;
    if (lm) {
      odezza_lm_pipeline_write_error(lm, error, sizeof(error), &n);
      fprintf(stderr, "%s\n", error);
    }
    CHECK(result == 0);
  }
  CHECK(odezza_lm_pipeline_workspace_requirements(lm, &bytes, &alignment) == 0);
  CHECK(alignment <= 8);
  workspace = malloc(bytes);
  CHECK(workspace);
  CHECK(odezza_lm_pipeline_run(lm, fit, 3, workspace, bytes - 1, &report) ==
        ODEZZA_ERROR_INSUFFICIENT_BUFFER);
  {
    OdezzaLmFit alias = fit[0];
    alias.mse_device = alias.initial_mse_device;
    CHECK(odezza_lm_pipeline_run(lm, &alias, 1, workspace, bytes, &report) ==
          ODEZZA_ERROR_INVALID_ARGUMENT);
  }
  {
    OdezzaLmFit huge = fit[0];
    huge.max_damping_attempts = UINT32_MAX;
    CHECK(odezza_lm_pipeline_run(lm, &huge, 1, workspace, bytes, &report) ==
          ODEZZA_ERROR_OVERFLOW);
    CHECK(report.fit_count == 0);
    huge.max_damping_attempts = 8;
    huge.max_iterations = UINT32_MAX;
    CHECK(odezza_lm_pipeline_run(lm, &huge, 1, workspace, bytes, &report) ==
          ODEZZA_ERROR_OVERFLOW);
    CHECK(report.fit_count == 0);
  }
  result = odezza_lm_pipeline_run(lm, fit, 3, workspace, bytes, &report);
  if (result) {
    char error[512];
    size_t n;
    odezza_lm_pipeline_write_error(lm, error, sizeof(error), &n);
    fprintf(stderr, "%s\n", error);
    CHECK(!result);
  }
  CHECK(report.completed_system_count == 3 && report.fit_count == 201 &&
        report.peak_inflight_modules == 2);
  CHECK(odezza_lm_pipeline_shape_report(lm, &shapes) == 0);
  CHECK(shapes.used_lanes_mask == info.shape.lanes_per_fit);
  for (k = 0; k < 3; ++k) {
    CUDA(cuMemcpyDtoH(out, fit[k].mse_device, sizeof(out)));
    CUDA(cuMemcpyDtoH(coefficients, fit[k].parameters_device,
                      sizeof(coefficients)));
    for (i = 0; i < 67; ++i) {
      if (!isfinite(out[i]) || out[i] >= 1e-10f)
        fprintf(stderr, "width=%u system=%d start=%d mse=%g coefficients=%g,%g,%g\n",
                info.shape.lanes_per_fit, k, i, out[i], coefficients[3*i],
                coefficients[3*i+1], coefficients[3*i+2]);
      CHECK(isfinite(out[i]) && out[i] < 1e-10f);
      for (j = 0; j < 3; ++j)
        CHECK(fabsf(coefficients[3 * i + j] - target[j]) < 1e-4f);
    }
  }
  printf("native_lm_public_api: fits=%llu kernel_seconds=%.9f "
         "wall_seconds=%.9f registers=%u "
         "peak_modules=%u\n",
         (unsigned long long)report.fit_count, report.kernel_seconds,
         report.total_seconds, report.maximum_register_count,
         report.peak_inflight_modules);
  CHECK(odezza_lm_pipeline_run(lm, fit, 1, workspace, bytes, &report) == 0);
  if (argc > 2 && !strcmp(argv[2], "coupled")) {
    double expected[3];
    cpu_proposal(starts, times, reference, expected);
    fit[0].max_iterations = 1;
    CHECK(odezza_lm_pipeline_run(lm, fit, 1, workspace, bytes, &report) == 0);
    CUDA(cuMemcpyDtoH(coefficients, fit[0].parameters_device, sizeof(coefficients)));
    for (j = 0; j < 3; ++j) {
      printf("coupled first proposal p%d: GPU %.9g CPU %.9g\n", j, coefficients[j], expected[j]);
      CHECK(fabs(coefficients[j] - expected[j]) < 5e-5);
    }
    fit[0].max_iterations = 24;
  }
#ifdef ODEZZA_LM_FAULT_TEST
  o_test_fail_specialize = 1;
  result = odezza_lm_pipeline_run(lm, fit, 3, workspace, bytes, &report);
  o_test_fail_specialize = 0;
  if (argc > 3) {
    CHECK(result == ODEZZA_SUCCESS);
    CHECK(odezza_lm_pipeline_shape_report(lm, &shapes) == 0);
    CHECK(shapes.used_lanes_mask == (info.shape.lanes_per_fit | 2u*info.shape.lanes_per_fit));
    CHECK(report.completed_system_count == 3 && report.fit_count == 201);
    CHECK(shapes.register_rejections[info.shape.lanes_per_fit == 1 ? 0 : info.shape.lanes_per_fit == 2 ? 1 : 2] == 1);
    CUDA(cuMemcpyDtoH(out, fit[0].mse_device, sizeof(out)));
    for (i = 0; i < 67; ++i) CHECK(isfinite(out[i]) && out[i] < 1e-10f);
  } else CHECK(result == ODEZZA_ERROR_REGISTER_PRESSURE);
  o_test_check_resources(2);
  for (k = 0; k < 8; ++k) {
    o_test_fail_load = k < 2 ? k + 1 : 0;
    o_test_fail_launch = k >= 2 && k < 4 ? k - 1 : 0;
    o_test_fail_event = k >= 4 ? k - 3 : 0;
    CHECK(odezza_lm_pipeline_run(lm, fit, 3, workspace, bytes, &report) ==
          ODEZZA_ERROR_CUDA);
    o_test_fail_load = o_test_fail_launch = o_test_fail_event = 0;
    o_test_check_resources(2);
    CHECK(odezza_lm_pipeline_run(lm, fit, 1, workspace, bytes, &report) ==
          ODEZZA_SUCCESS);
    o_test_check_resources(2);
  }
  puts("8 native driver failure/cleanup/reuse checks passed");
#endif
  CHECK(odezza_lm_pipeline_destroy(lm) == 0);
#ifdef ODEZZA_LM_FAULT_TEST
  o_test_check_resources(0);
#endif
  free(workspace);
  for (k = 0; k < 3; ++k) {
    CUDA(cuMemFree(fit[k].parameters_device));
    CUDA(cuMemFree(fit[k].initial_mse_device));
    CUDA(cuMemFree(fit[k].mse_device));
    CUDA(cuMemFree(fit[k].iterations_device));
    CUDA(cuMemFree(fit[k].accepted_device));
    CUDA(cuMemFree(fit[k].factorizations_device));
    CUDA(cuMemFree(fit[k].evaluations_device));
    CUDA(cuMemFree(fit[k].invalid_device));
  }
  CUDA(cuMemFree(fit[0].starts_device));
  CUDA(cuMemFree(fit[0].offsets_device));
  CUDA(cuMemFree(fit[0].times_device));
  CUDA(cuMemFree(fit[0].reference_device));
  CUDA(cuMemFree(fit[0].weights_device));
  CUDA(cuMemFree(fit[0].lower_device));
  CUDA(cuMemFree(fit[0].upper_device));
  CUDA(cuDevicePrimaryCtxRelease(device));
  return 0;
}
