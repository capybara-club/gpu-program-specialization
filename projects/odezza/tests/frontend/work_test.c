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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void check(const char *s,OdrResult expected,unsigned subdivisions,uint64_t steps) {
    OdrJson j={s,strlen(s)};OdrRk4Work work={0};OdrRk4Layout prepared={0};OdrError e={0};
    OdrResult r=odr_request_rk4_work(j,&work,&e);
    assert(r==expected);
    if(r==ODR_OK||r==ODR_CAPACITY) {
        size_t bytes;void *arena;const OdrTrajectories *t;
        assert(work.steps_per_observation==subdivisions&&work.steps_per_configuration==steps);
        assert(odr_trajectories_parse(j,NULL,0,&bytes,NULL,&e)==ODR_OK);
        arena=malloc(bytes);assert(arena);
        assert(odr_trajectories_parse(j,arena,bytes,&bytes,&t,&e)==ODR_OK);
        /* Fixtures here use exactly representable dt=.125. */
        assert(odr_trajectories_rk4(t,.125f,0,&prepared,&e)==ODR_OK);
        assert(t->trajectory_count==work.trajectory_count&&t->point_count==work.point_count);
        assert(prepared.steps_per_observation==work.steps_per_observation);
        assert(prepared.steps_per_configuration==work.steps_per_configuration);
        free(arena);
    }
}
int main(void) {
    const char *base="{\"problem\":{\"states\":[\"x\"],\"trajectories\":[{\"initial\":[0],\"times\":[0,.125,.5],\"values\":[[0],[1],[null]]},{\"initial\":[1],\"times\":[0,.25],\"values\":[[1],[2]]}]},\"grammar\":{\"integration\":{\"dt\":.125}}}";
    char json[4096];const char *p=base;size_t n=0;
    /* JSON numbers need leading zero. */
    while(*p) {if(*p=='.')json[n++]='0';json[n++]=*p++;}json[n]=0;
    check(json,ODR_OK,3,9);
    {
        char limited[4096];char *where;
        strcpy(limited,json);where=strstr(limited,"\"dt\":0.125");assert(where);
        strcpy(where,"\"dt\":0.125,\"max_steps\":8}}}");check(limited,ODR_CAPACITY,3,9);
        strcpy(where,"\"dt\":0.125,\"max_steps\":9}}}");check(limited,ODR_OK,3,9);
    }
    check("{\"problem\":{\"states\":[\"x\"],\"trajectories\":[{\"initial\":[0],\"times\":[100000000,100000001],\"values\":[[0],[1]]}]},\"grammar\":{\"integration\":{\"dt\":0.125}}}",ODR_SCHEMA,0,0);
    check("{\"problem\":{\"states\":[\"x\"],\"trajectories\":[{\"initial\":[0],\"times\":[0,1],\"values\":[[0],[1]]}]},\"grammar\":{\"integration\":{\"dt\":1e-30}}}",ODR_OVERFLOW,0,0);
    check("{\"grammar\":{\"integration\":{\"dt\":0}}}",ODR_SCHEMA,0,0);
    check("{\"grammar\":{\"integration\":{\"dt\":0.1,\"method\":\"lsoda\"}}}",ODR_UNSUPPORTED,0,0);
    puts("allocation-free RK4 preflight matches prepared FP32 layout; limits/irregular/masked/overflow cases passed");
    return 0;
}
