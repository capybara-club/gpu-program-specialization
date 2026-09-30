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
#include "odr_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
void trial_admission_defaults(TrialAdmissionPolicy *p) {
    const TrialAdmissionPolicy defaults={{64,100000000,10000000,
        UINT64_C(1000000000000),1000000000,UINT64_C(10000000000)},
        UINT64_C(1000000000000000),UINT64_C(2147483648)};
    *p=defaults;
}
int trial_admission_environment(TrialAdmissionPolicy *p,OdrError *e) {
    static const char *names[]={"FAMILIES","SKELETONS","VARIANTS","CONFIGURATIONS",
        "DERIVATIONS","EXPANSION_STEPS","ROLLOUT_WORK","HOST_BYTES"};
    uint64_t *fields[]={&p->maximum.families,&p->maximum.skeletons,&p->maximum.variants,
        &p->maximum.configurations,&p->maximum.derivations,&p->maximum.expansion_steps,
        &p->rollout_work,&p->host_bytes};
    size_t i;
    for(i=0;i<8;i++) {
        char name[80],*end;const char *s;unsigned long long v;
        snprintf(name,sizeof(name),"ODZ_ADMISSION_MAX_%s",names[i]);s=getenv(name);
        if(!s)continue;
        errno=0;v=strtoull(s,&end,10);
        if(!*s||strspn(s,"0123456789")!=strlen(s)||*end||errno||!v||v>UINT64_MAX) {
            odr_error(e,ODR_SCHEMA,0,"invalid operator admission ceiling");return 1;
        }
        *fields[i]=(uint64_t)v;
    }
    return 0;
}
static int positive(OdrJson obj,const char *key,uint64_t fallback,uint64_t *out) {
    OdrJson v;*out=fallback;
    return jget(obj,key,&v)&&(!ju64(v,out)||!*out);
}
const char *trial_admission_check(OdrJson request,const OdrRk4Work *work,
    const TrialAdmissionPolicy *p,TrialAdmission *out,OdrError *e) {
    OdrJson execution={0};uint64_t unit,dedup;
    memset(out,0,sizeof(*out));
    if(odr_request_allocations(request,&out->allocated,e))return "invalid_search_allocation";
#define LIMIT(field) if(out->allocated.field>p->maximum.field) { \
    odr_error(e,ODR_CAPACITY,0,"allocated " #field " exceeds operator ceiling");return "search_" #field "_limit"; }
    LIMIT(families);LIMIT(skeletons);LIMIT(variants);LIMIT(configurations);
    LIMIT(derivations);LIMIT(expansion_steps);
#undef LIMIT
    unit=work->steps_per_configuration;
    if(unit<work->point_count)unit=work->point_count;
    if(unit&&out->allocated.configurations>UINT64_MAX/unit) {
        odr_error(e,ODR_OVERFLOW,0,"configuration allocation times trajectory work overflows");return "rollout_work_overflow";
    }
    out->rollout_work=unit*out->allocated.configurations;
    if(out->rollout_work>p->rollout_work) {
        odr_error(e,ODR_CAPACITY,0,"allocated configurations times max(steps,points) exceeds operator ceiling");return "rollout_work_limit";
    }
    (void)jget(request,"execution",&execution);
    if(positive(execution,"max_host_bytes",UINT64_C(2147483648),&out->host_bytes)||
       positive(execution,"dedup_bytes_per_family",UINT64_C(67108864),&dedup))return "invalid_host_allocation";
    if(out->host_bytes>p->host_bytes)return "host_reservation_limit";
    /* Deliberately a lower bound. The worker measures grammar, pages, retention
     * and pipeline storage exactly before scoring; no second AST expansion. */
    if(out->allocated.families>(UINT64_MAX-request.size)/dedup)return "host_reservation_overflow";
    out->host_lower_bound=out->allocated.families*dedup+request.size;
    if(out->host_lower_bound>out->host_bytes)return "host_minimum_exceeds_reservation";
    return NULL;
}
static size_t format(const OdrAllocationInfo *a,uint64_t work,uint64_t host,
    uint64_t lower,int with_lower,char *out,size_t capacity) {
    int n=snprintf(out,capacity,"{\"families\":%llu,\"skeletons\":%llu,\"variants\":%llu,"
        "\"configurations\":%llu,\"derivations\":%llu,\"expansion_steps\":%llu,"
        "\"rollout_work\":%llu,\"host_bytes\":%llu",
        (unsigned long long)a->families,(unsigned long long)a->skeletons,
        (unsigned long long)a->variants,(unsigned long long)a->configurations,
        (unsigned long long)a->derivations,(unsigned long long)a->expansion_steps,
        (unsigned long long)work,(unsigned long long)host);
    int tail;
    if(n<0||(size_t)n>=capacity)return 0;
    if(with_lower)tail=snprintf(out+n,capacity-(size_t)n,",\"host_lower_bound\":%llu}",(unsigned long long)lower);
    else tail=snprintf(out+n,capacity-(size_t)n,"}");
    return tail>=0&&(size_t)tail<capacity-(size_t)n?(size_t)n+(size_t)tail:0;
}
size_t trial_admission_json(const TrialAdmission *v,char *out,size_t capacity) {
    return format(&v->allocated,v->rollout_work,v->host_bytes,v->host_lower_bound,1,out,capacity);
}
size_t trial_admission_policy_json(const TrialAdmissionPolicy *v,char *out,size_t capacity) {
    return format(&v->maximum,v->rollout_work,v->host_bytes,0,0,out,capacity);
}
