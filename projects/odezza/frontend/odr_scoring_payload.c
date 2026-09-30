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
#include "odr_internal.h"
#include "odezza_scoring_request.h"
OdrResult odr_scoring_pack_systems(const OdrBatch *batch,void *arena,size_t capacity, size_t *required,const OdezzaScoringSystem **out,OdrError *error) {
    Arena a= {
        (unsigned char *)arena,capacity,0,ODR_OK
    };
    OdezzaScoringSystem *systems;
    size_t i,j;
    if(out)*out=NULL;
    if(required)*required=0;
    if(!batch||!required||!out||!batch->count||!batch->candidates)return odr_error(error,ODR_SCHEMA,0,"nonempty batch required");
    if(arena&&(uintptr_t)arena%odr_arena_alignment())return odr_error(error,ODR_SCHEMA,0,"unaligned system arena");
    systems=odr_take(&a,batch->count,sizeof(*systems));
    for(i=0;i<batch->count;i++) {
        const OdrCandidate *c=&batch->candidates[i];
        OdezzaScoringRhs *rhs=odr_take(&a,c->rhs_count,sizeof(*rhs));
        for(j=0;j<c->rhs_count;j++) {
            if(c->rhs[j].state_index>UINT8_MAX)return odr_error(error,ODR_SCHEMA,0,"native state index overflow");
            if(rhs) {
                rhs[j].state_index=(uint8_t)c->rhs[j].state_index;
                rhs[j].program.bytes=c->rhs[j].program.bytes;
                rhs[j].program.byte_count=c->rhs[j].program.byte_count;
            }
        }
        if(systems) {
            systems[i].rhs=rhs;
            systems[i].rhs_count=c->rhs_count;
        }
    }
    *required=a.used;
    if(a.result)return a.result;
    if(arena&&capacity<a.used)return odr_error(error,ODR_BUFFER,0,"native descriptor arena too small");
    *out=systems;
    return ODR_OK;
}
