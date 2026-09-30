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
#ifndef ODEZZA_ADMISSION_H
#define ODEZZA_ADMISSION_H
#include "odezza_request.h"
/* Operator policy; customer data stays in RAM. All counts reserve declared
 * family budgets, not inferred cardinalities or a promise of wall time. */
typedef struct TrialAdmissionPolicy {
    OdrAllocationInfo maximum;
    uint64_t rollout_work,host_bytes;
} TrialAdmissionPolicy;
typedef struct TrialAdmission {
    OdrAllocationInfo allocated;
    uint64_t rollout_work,host_bytes,host_lower_bound;
} TrialAdmission;
void trial_admission_defaults(TrialAdmissionPolicy *policy);
/* Overrides ODZ_ADMISSION_MAX_{FAMILIES,SKELETONS,VARIANTS,CONFIGURATIONS,
 * DERIVATIONS,EXPANSION_STEPS,ROLLOUT_WORK,HOST_BYTES}; invalid values fail. */
int trial_admission_environment(TrialAdmissionPolicy *policy,OdrError *error);
/* NULL means accepted. Stable ASCII rejection code otherwise; no allocation. */
const char *trial_admission_check(OdrJson request,const OdrRk4Work *work,
    const TrialAdmissionPolicy *policy,TrialAdmission *out,OdrError *error);
/* Produces a JSON object. Returns zero if buffer is insufficient. */
size_t trial_admission_json(const TrialAdmission *value,char *out,size_t capacity);
size_t trial_admission_policy_json(const TrialAdmissionPolicy *value,char *out,size_t capacity);
#endif
