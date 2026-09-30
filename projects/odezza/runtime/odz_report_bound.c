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
#include "odz_internal.h"
/* Deliberately conservative, based only on the parsed grammar. No AST generation
 * and no loss of requested winners/tags. Escaped strings use their actual bound. */
static uint64_t text_bytes(const char *s) {
    uint64_t n=2;const unsigned char *p=(const unsigned char *)s;
    for(;p&&*p;p++)n+=*p<32?6:(*p=='"'||*p=='\\'?2:1);
    return n;
}
static uint64_t tag_bytes(const char *const *tags,size_t count) {
    uint64_t n=0;size_t i;for(i=0;i<count;i++)n+=1+text_bytes(tags[i]);return n;
}
int odz_report_preflight(OdzJob *j) {
    uint64_t total=32768+ODZ_DEVICES*2048,maximum=0,rank_slots=j->global.k,tags_k=0;
    size_t i,r,a,x;uint64_t fixed_bytes=0;
    for(i=0;i<j->fixed->rhs_count;i++)fixed_bytes+=3*j->fixed->rhs[i].program.byte_count;
    for(i=0;i<j->fixed->state_count;i++)total+=text_bytes(j->fixed->states[i])+1;
    total+=(uint64_t)j->trajectories->trajectory_count*192;
    for(i=0;i<j->tag_count;i++){total+=text_bytes(j->tags[i].name)+16;tags_k+=j->tags[i].ranking.k;}
    for(i=0;i<j->family_count;i++) {
        const Family *f=&j->grammar->families[i];uint64_t slot_count=0,slot_name=2,locals=0,tags=tag_bytes(f->tags,f->tag_count),candidate;
        for(x=0;x<f->slot_count;x++)if(f->slots[x].kind!=4) {
            uint64_t len=text_bytes(f->slots[x].axis.name);++slot_count;if(len>slot_name)slot_name=len;
        }
        for(r=0;r<f->rule_count;r++)for(a=0;a<f->rules[r].count;a++) {
            const Alt *alt=&f->rules[r].alternatives[a];tags+=tag_bytes(alt->tags,alt->tag_count);
            for(x=0;x<alt->local_count;x++)if(alt->locals[x].kind!=4) {
                uint64_t len=text_bytes(alt->locals[x].axis.name);++locals;if(len>slot_name)slot_name=len;
            }
        }
        if(tags)j->provenance_possible=1;
        slot_count+=locals*j->grammar->max_nodes;
        if(slot_count>253-2*j->fixed->state_count)slot_count=253-2*j->fixed->state_count;
        candidate=1024+text_bytes(f->name)+tags+slot_count*(512+slot_name)+
            2*(fixed_bytes+5*(uint64_t)j->grammar->max_nodes+j->fixed->state_count)+4*j->fixed->state_count;
        total+=text_bytes(f->name)*2+768+candidate*j->families[i].ranking.k;
        rank_slots+=j->families[i].ranking.k;if(candidate>maximum)maximum=candidate;
    }
    rank_slots+=tags_k;
    total+=maximum*(j->global.k+tags_k)+rank_slots*80;
    j->report_bound=total;
    if(total>j->options.report_bytes) {
        j->report_rejected=1;
        return odz_fail(j,"report_size_limit: conservative bound %llu bytes exceeds %llu; reduce retained k, tag output, or grammar max_nodes",
            (unsigned long long)total,(unsigned long long)j->options.report_bytes);
    }
    if(rank_slots) {
        size_t capacity=1;
        while(capacity<rank_slots*2)capacity*=2;
        j->report_seen=odz_alloc(j,capacity*sizeof(*j->report_seen));
        if(!j->report_seen)return 1;
        j->report_seen_capacity=capacity;
    }
    return 0;
}
