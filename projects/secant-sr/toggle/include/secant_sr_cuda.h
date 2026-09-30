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
#ifndef SECANT_SR_CUDA_H
#define SECANT_SR_CUDA_H
#include "secant_sr.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct SRCudaOptions {
  int device;
  size_t ast_batch, score_bytes, asts_per_kernel, kernels_per_module;
  size_t tile_rows, threads, workers, streams;
} SRCudaOptions;
typedef struct SRCudaStats {
  double wall_seconds, pipeline_seconds, device_seconds,
      specialize_work_seconds;
  double module_load_seconds, reduction_seconds, transfer_seconds;
  double setup_seconds, nvrtc_seconds;
  size_t batches;
} SRCudaStats;
typedef struct SRCudaScorerImpl *SRCudaScorer;
SRCudaOptions secant_sr_cuda_options_default(void);
/* Retains this device's primary context and makes it current. Data and banks
 * are copied once; evaluation requires that context to remain current. The
 * score pool is bounded by score_bytes and ast_batch. No per-generation device
 * malloc. On cleanup failure, a non-NULL output retains ownership for destroy
 * retry. */
SecantResult secant_sr_cuda_create(const SRConfig *, const float *input,
                                   const float *target, size_t rows,
                                   const float *banks, const SRCudaOptions *,
                                   SRCudaScorer *, SRCudaStats *);
SecantResult secant_sr_cuda_score(SRCudaScorer, const SecantAstProgramSet *,
                                  SRScore *, size_t capacity, SRCudaStats *);
/* Score an explicit shared bank (up to the configured bank capacity). Copies
 * only the active bank bytes; uses existing data/score pools and the same
 * template/runner. The ordinary score entry restores the original bank. */
SecantResult secant_sr_cuda_score_banks(SRCudaScorer, const SecantAstProgramSet *,
    const float *banks, size_t num_banks, size_t bank_elements,
    SRScore *, size_t capacity, SRCudaStats *);
/* A failure retains the handle and all not-yet-released resources. */
SecantResult secant_sr_cuda_destroy(SRCudaScorer);
#ifdef __cplusplus
}
#endif
#endif
