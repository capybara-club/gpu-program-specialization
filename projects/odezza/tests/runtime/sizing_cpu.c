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
#include <assert.h>
#include <math.h>
#include <stdio.h>
int main(void) {
    OdzSizing s;
    uint64_t work,bank,ceiling;
    assert(!odz_sizing_init(&s,84,1536,5120));
    assert(s.target_configurations==262144&&s.work_configurations==262144);
    assert(odz_sizing_chunk(&s,16777216,1,1,0.05)==262144);
    assert(odz_sizing_chunk(&s,4096,1,1,0.05)==4096);
    assert(odz_sizing_chunk(&s,4096,0,0,0.05)==4096);
    assert(!odz_sizing_init(&s,84,1536,4));
    assert(odz_sizing_chunk(&s,16777216,1,0,.05)>=1024*2048);
    assert(!odz_sizing_init(&s,1,128,65536));
    assert(s.work_configurations==1024);
    assert(odz_sizing_init(&s,UINT32_MAX,UINT32_MAX,1));
    assert(odz_sizing_init(&s,1,128,0));
    assert(odz_sizing_init(&s,1,128,65537));
    for(work=1;work<=65536;work*=2) {
        assert(!odz_sizing_init(&s,84,1536,work));
        assert(s.work_configurations>=2*s.wave_configurations);
        assert(s.work_configurations*work<=ODZ_MAX_PARALLEL_CONFIGURATIONS*ODZ_MAX_STEPS_PER_CONFIGURATION);
        assert(odz_sizing_chunk(&s,UINT64_MAX,1,INFINITY,1)==s.work_configurations);
    }
    /* Exhaust every row count/ceiling pair: no lost/duplicate rows, and no tiny
     * last slice. This is the old 40*13107+8 tail failure in miniature. */
    for(bank=1;bank<=1000;bank++)for(ceiling=1;ceiling<=128;ceiling++) {
        uint64_t at=0,minimum=UINT64_MAX,maximum=0,count=0;
        while(at<bank) {
            uint64_t n=odz_sizing_banks(bank-at,ceiling);
            assert(n&&n<=ceiling&&n<=bank-at);
            if(n<minimum)minimum=n;
            if(n>maximum)maximum=n;
            at+=n;count++;
        }
        assert(at==bank&&maximum-minimum<=1);
        assert(count==bank/ceiling+(bank%ceiling!=0));
    }
    assert(odz_sizing_banks(524288,13107)==12788);
    assert(odz_sizing_banks(UINT64_MAX,UINT64_MAX)==UINT64_MAX);
    assert(odz_sizing_banks(UINT64_MAX,2)==2);
    assert(!odz_sizing_banks(10,0));
    assert(!odz_sizing_init(&s,84,1536,5120));
    odz_sizing_record(&s,1,16,1,0,1,1,0,.0002,.05);
    odz_sizing_record(&s,1,262144,1,0,0,0,0,.09,.05);
    assert(s.underfilled_tiles==1&&s.underfilled_configurations==16);
    assert(s.minimum_blocks==1&&s.maximum_blocks==2048);
    assert(s.over_target_seconds==1&&odz_sizing_percentile(&s,95)>=.09);
    odz_sizing_record(&s,2048,1,1,0,0,1,0,.002,.05);
    assert(s.underfilled_tiles==2); /* many blocks, but only one active lane each */
    puts("sizing: geometry, limits, overflow, exact balanced coverage, duration reporting passed");
    return 0;
}
