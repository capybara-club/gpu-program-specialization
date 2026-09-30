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
#include "odz_sizing.h"
#include "odezza_runtime.h"
#include <math.h>
#include <string.h>
int odz_sizing_init(OdzSizing *s,uint32_t sms,uint32_t threads_per_sm,uint64_t work_units) {
    uint64_t target;
    memset(s,0,sizeof(*s));
    if(!sms||!threads_per_sm||!work_units||work_units>ODZ_MAX_STEPS_PER_CONFIGURATION)return 1;
    s->sms=sms;s->threads_per_sm=threads_per_sm;
    s->wave_configurations=(uint64_t)sms*threads_per_sm;
    /* Bound even future/unexpected hardware descriptions. This is a service
     * policy ceiling, not a requested population or an allocation size. */
    if(s->wave_configurations>ODZ_MAX_PARALLEL_CONFIGURATIONS/2)return 1;
    target=1;
    while(target<2*s->wave_configurations)target*=2;
    s->target_configurations=target;
    s->work_configurations=ODZ_BASE_TILE_WORK_UNITS/work_units;
    if(s->work_configurations<target)s->work_configurations=target;
    return 0;
}
uint64_t odz_sizing_chunk(const OdzSizing *s,uint64_t ceiling,int automatic,double rate,double seconds) {
    uint64_t limit=ceiling<s->work_configurations?ceiling:s->work_configurations;
    double proposed=rate*seconds;
    if(automatic&&isfinite(proposed)&&proposed>0&&proposed<(double)limit) {
        uint64_t measured=(uint64_t)proposed;
        if(measured<s->target_configurations)measured=s->target_configurations;
        if(measured<limit)limit=measured;
    }
    return limit;
}
uint64_t odz_sizing_banks(uint64_t remaining,uint64_t ceiling) {
    uint64_t slices;
    if(!remaining||!ceiling)return 0;
    slices=remaining/ceiling+(remaining%ceiling!=0);
    return remaining/slices+(remaining%slices!=0);
}
void odz_sizing_record(OdzSizing *s,uint64_t systems,uint64_t banks,uint64_t permutations,
    int memory_limited,int explicit_limited,int available_limited,int balanced,
    double call_seconds,double target_seconds) {
    uint64_t configurations=systems*banks*permutations;
    uint64_t blocks=systems*((banks*permutations+ODZ_SCORING_BLOCK_THREADS-1)/ODZ_SCORING_BLOCK_THREADS);
    unsigned bin=0;
    double upper=0.000001;
    if(!s->minimum_configurations||configurations<s->minimum_configurations)s->minimum_configurations=configurations;
    if(systems>s->maximum_systems)s->maximum_systems=systems;
    if(!s->minimum_blocks||blocks<s->minimum_blocks)s->minimum_blocks=blocks;
    if(blocks>s->maximum_blocks)s->maximum_blocks=blocks;
    s->total_blocks+=blocks;
    /* Total tile geometry can reveal scarcity; it cannot prove that individual
     * module launches overlap or that their register-limited occupancy is high. */
    if(configurations<s->wave_configurations) {
        ++s->underfilled_tiles;s->underfilled_configurations+=configurations;
        s->underfilled_call_seconds+=call_seconds;
    }
    s->memory_limited_tiles+=!!memory_limited;
    s->explicit_limited_tiles+=!!explicit_limited;
    s->available_limited_tiles+=!!available_limited;
    s->balanced_tiles+=!!balanced;
    if(call_seconds>target_seconds)++s->over_target_seconds;
    if(call_seconds>s->maximum_call_seconds)s->maximum_call_seconds=call_seconds;
    while(bin<31&&upper<call_seconds) {++bin;upper*=2;}
    ++s->call_histogram[bin];
}
double odz_sizing_percentile(const OdzSizing *s,unsigned percent) {
    uint64_t total=0,seen=0,target;
    unsigned i;
    double upper=0.000001;
    for(i=0;i<32;i++)total+=s->call_histogram[i];
    if(!total||!percent||percent>100)return 0;
    target=(total/100)*percent+((total%100)*percent+99)/100;
    for(i=0;i<32;i++,upper*=2) {
        seen+=s->call_histogram[i];
        if(seen>=target)return i==31?fmax(upper,s->maximum_call_seconds):upper;
    }
    return s->maximum_call_seconds;
}
