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
#include "odezza_request.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static OdrJson json(const char *s) {
    OdrJson j= {
        s,strlen(s)
    };
    return j;
}
int main(void) {
    const char *s="{\"problem\":{\"states\":[\"x\",\"y\"],\"known_rhs\":{\"x\":\"-0.5*x+sin(y)\"},\"trajectories\":[{\"initial\":[1,2],\"times\":[0,0.1,0.3],\"values\":[[1,null],[0.9,null],[0.8,1.5]]},{\"initial\":[2,3],\"times\":[0,0.25],\"values\":[[2,3],[1.8,2.9]],\"mask\":[[true,true],[true,false]]}]}}";
    const OdrTrajectories *t;
    const OdrStatic *fixed;
    size_t size,again;
    unsigned char *p;
    OdrError e;
    OdrResult r;
    OdrRk4Layout layout;
    assert(odr_trajectories_parse(json(s),NULL,0,&size,&t,&e)==ODR_OK&&t==NULL);
    p=malloc(size+32);
    assert(p);
    memset(p,0xcd,size+32);
    assert(odr_trajectories_parse(json(s),p,size-1,&again,&t,&e)==ODR_BUFFER&&again==size&&t==NULL);
    assert(p[size-1]==0xcd&&p[size]==0xcd);
    r=odr_trajectories_parse(json(s),p,size,&again,&t,&e);
    if(r)fprintf(stderr,"%s\n",e.message);
    assert(r==ODR_OK);
    assert(t->state_count==2&&t->trajectory_count==2&&t->point_count==5&&t->observed_count==4);
    assert(t->offsets[0]==0&&t->offsets[1]==3&&t->offsets[2]==5);
    assert(t->values[0]==1&&t->values[3]==2&&t->values[5]==2&&isnan(t->values[6])&&isnan(t->values[9]));
    assert(!t->info[0].uniform_spacing&&t->info[1].uniform_spacing&&t->allow_missing_observations);
    assert(odr_trajectories_rk4(t,0.125f,0,&layout,&e)==ODR_OK&&layout.steps_per_observation==2&&layout.steps_per_configuration==6);
    assert(odr_trajectories_rk4(t,0.125f,5,&layout,&e)==ODR_CAPACITY);
    free(p);
    assert(odr_static_parse(json(s),NULL,0,&size,&fixed,&e)==ODR_OK);
    p=malloc(size+16);
    memset(p,0xcd,size+16);
    assert(odr_static_parse(json(s),p,size-1,&again,&fixed,&e)==ODR_BUFFER&&fixed==NULL&&again==size&&p[size-1]==0xcd);
    assert(odr_static_parse(json(s),p,size,&again,&fixed,&e)==ODR_OK&&fixed->rhs_count==1&&fixed->rhs[0].state_index==0);
    assert(fixed->rhs[0].program.bytes[fixed->rhs[0].program.byte_count-1]==0x80);
    free(p);
    assert(odr_trajectories_parse(json("{\"states\":[\"x\"],\"states\":[\"y\"]}"),NULL,0,&size,&t,&e)==ODR_JSON);
    assert(odr_trajectories_parse(json("{\"states\":[\"x\"],\"trajectories\":[{\"initial\":[1],\"times\":[100000000,100000001],\"values\":[[1],[2]]}]}"),NULL,0,&size,&t,&e)==ODR_SCHEMA);
    assert(odr_trajectories_parse(json("{\"states\":[\"x\"],\"trajectories\":[{\"initial\":[1],\"times\":[0,1],\"values\":[[2],[2]]}]}"),NULL,0,&size,&t,&e)==ODR_SCHEMA);
    assert(odr_static_parse(json("{\"states\":[\"x\"],\"known_rhs\":{\"x\":\"unknown(x)\"}}"),NULL,0,&size,&fixed,&e)==ODR_AST);
    puts("trajectory/static arena tests passed");
    return 0;
}
