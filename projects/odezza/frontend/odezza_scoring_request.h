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
#ifndef ODEZZA_SCORING_REQUEST_H
#define ODEZZA_SCORING_REQUEST_H
#include "odezza_request.h"
#include "odezza.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Runtime preparation owns native handles/cache I/O. Unlike the allocation-free
 * parsers, setup may allocate (including CUDA/NVRTC). run uses caller workspace.
 * Caller owns/current-selects the CUDA context and all device buffers. */
typedef struct OdrScoringSession OdrScoringSession;
/* Build native descriptors in caller memory; instruction bytes are borrowed
 * from batch. Both arenas must remain alive until the scoring call completes. */
OdrResult odr_scoring_pack_systems(const OdrBatch *batch,void *arena,size_t capacity,
    size_t *required_bytes,const OdezzaScoringSystem **out,OdrError *error);
typedef struct OdrScoringOptions {
    OdezzaScoringPipelineCreateInfo shape;
    const char *cache_directory; /* existing directory; NULL disables persistence */
    uint32_t maximum_patch_capacity;
} OdrScoringOptions;
typedef struct OdrScoringStats {
    uint64_t cache_hits, cache_misses, cache_invalidations, capacity_growths;
    uint64_t run_attempts, retry_requested_configurations;
    uint64_t cache_evictions, cache_oversized;
    double nvrtc_seconds, template_prepare_seconds, prespecialize_seconds;
    OdezzaScoringTemplateInfo actual_shape;
} OdrScoringStats;
OdezzaResult odr_scoring_create(const OdrStatic *fixed,const OdrScoringOptions *options,
    OdrScoringSession **out);
OdezzaResult odr_scoring_workspace(const OdrScoringSession *session,size_t *bytes,size_t *alignment);
/* On capacity growth the original ASTs are reused. If the grown pipeline needs
 * more workspace, returns INSUFFICIENT_BUFFER and the new required_bytes; caller
 * retries the same batch. Discard scores from failed attempts. */
OdezzaResult odr_scoring_run(OdrScoringSession *session,const OdezzaScoringSystem *systems,
    size_t count,const OdezzaScoringLaunch *launch,void *workspace,size_t capacity,
    size_t *required_bytes,OdezzaScoringRunReport *report);
OdezzaResult odr_scoring_stats(const OdrScoringSession *session,OdrScoringStats *out);
const char *odr_scoring_error(const OdrScoringSession *session);
OdezzaResult odr_scoring_destroy(OdrScoringSession *session);
#ifdef __cplusplus
}
#endif
#endif
