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
#include "admission.h"
#include <assert.h>
#include <string.h>
#include <stdio.h>
static OdrJson json(const char *s) {OdrJson j={s,strlen(s)};return j;}
int main(void) {
    TrialAdmissionPolicy p;TrialAdmission a;OdrError e={0};OdrRk4Work work={1,4,1,3};
    const char *r;char buffer[768];OdrAllocationInfo info;
    const char *request="{\"grammar\":{\"families\":[{\"limits\":{\"max_configurations\":1024000000}},{\"limits\":{\"max_configurations\":1024000000}}]}}";
    trial_admission_defaults(&p);
    assert(!trial_admission_check(json(request),&work,&p,&a,&e));
    assert(a.allocated.configurations==UINT64_C(2048000000)&&a.rollout_work==UINT64_C(8192000000));
    assert(trial_admission_json(&a,buffer,sizeof(buffer)));
    assert(!trial_admission_json(&a,buffer,8));
    p.maximum.configurations=2000000000;
    r=trial_admission_check(json(request),&work,&p,&a,&e);
    assert(r&&!strcmp(r,"search_configurations_limit"));
    trial_admission_defaults(&p);p.rollout_work=8000000000;
    assert(!strcmp(trial_admission_check(json(request),&work,&p,&a,&e),"rollout_work_limit"));
    p.maximum.configurations=UINT64_MAX;p.rollout_work=UINT64_MAX;
    assert(!strcmp(trial_admission_check(json("{\"grammar\":{\"limits\":{\"max_configurations\":18446744073709551615}}}"),&work,&p,&a,&e),"rollout_work_overflow"));
    trial_admission_defaults(&p);
    assert(!strcmp(trial_admission_check(json("{\"execution\":{\"max_host_bytes\":1024}}"),&work,&p,&a,&e),"host_minimum_exceeds_reservation"));
    assert(!strcmp(trial_admission_check(json("{\"execution\":{\"max_host_bytes\":2147483649}}"),&work,&p,&a,&e),"host_reservation_limit"));
    assert(odr_request_allocations(json("{\"families\":[{\"limits\":{\"max_configurations\":8}},{\"limits\":{\"max_configurations\":8}}],\"limits\":{\"max_configurations\":15}}"),&info,&e)!=ODR_OK);
    assert(odr_request_allocations(json("{\"families\":[{\"limits\":{\"max_configurations\":18446744073709551615}},{\"limits\":{\"max_configurations\":1}}]}"),&info,&e)==ODR_OVERFLOW);
    assert(odr_request_allocations(json("{\"families\":[{\"limits\":{\"max_configurations\":8}},{}]}"),&info,&e)!=ODR_OK);
    assert(odr_request_allocations(json("{\"families\":[{},{}],\"limits\":{\"max_configurations\":11}}"),&info,&e)==ODR_OK&&info.configurations==11);
    assert(odr_request_allocations(json("{\"families\":[{\"limits\":{\"max_configurations\":3}},{\"limits\":{\"max_configurations\":5}}],\"limits\":{\"max_configurations\":20}}"),&info,&e)==ODR_OK&&info.configurations==8);
    assert(odr_request_allocations(json("{\"allocation\":{\"redistribute_unused\":true}}"),&info,&e)==ODR_UNSUPPORTED);
    puts("allocation sums/conflicts/overflow, billion-config admission, rollout and host ceilings passed");return 0;
}
