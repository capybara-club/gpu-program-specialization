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
#define _POSIX_C_SOURCE 200809L
#include "odezza_request.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec+t.tv_nsec*1e-9;
}

int main(int argc,char **argv) {
    FILE *file;
    long len;
    char *text;
    OdrJson json;
    const OdrGrammar *g;
    OdrProducer *p;
    OdrProducerOptions o= {
        0
    };
    OdrError e= {
        0
    };
    OdrBatch b;
    void *ga,*pa,*ba;
    size_t gn,pn,bn,need;
    uint64_t next=0,bytes=0,checksum=0;
    double start=now(),prepared,generated;
    OdrResult r;
    if(argc!=2)return 2;
    file=fopen(argv[1],"rb");
    if(!file)return 2;
    fseek(file,0,SEEK_END);
    len=ftell(file);
    rewind(file);
    text=malloc((size_t)len+1);
    if(!text||fread(text,1,(size_t)len,file)!=(size_t)len)return 2;
    fclose(file);
    text[len]=0;
    json.data=text;
    json.size=(size_t)len;
    if(odr_grammar_parse(json,NULL,NULL,0,&gn,&g,&e))return 1;
    ga=malloc(gn);
    if(odr_grammar_parse(json,NULL,ga,gn,&need,&g,&e))return 1;
    o.max_asts=1000000;
    o.dedup_capacity=128u*1024u*1024u;
    if(odr_producer_create(g,&o,NULL,0,&pn,&p,&e))return 1;
    pa=malloc(pn);
    if(odr_producer_create(g,&o,pa,pn,&need,&p,&e))return 1;
    if(odr_batch_requirements(p,256,&bn))return 1;
    ba=malloc(bn);
    prepared=now()-start;
    start=now();
    do {
        size_t i,j;
        r=odr_producer_next(p,next,256,ba,bn,&need,&b,&e);
        if(r) {
            fprintf(stderr,"%d %s at %llu\n",r,e.message,(unsigned long long)next);
            return 1;
        }
        for(i=0;i<b.count;i++)for(j=0;j<b.candidates[i].rhs_count;j++) {
            const OdrProgram *a=&b.candidates[i].rhs[j].program;
            bytes+=a->byte_count;
            checksum=checksum*31+a->bytes[a->byte_count/2];
        }
        next=b.next_index;
    }
    while(!b.exhausted);
    generated=now()-start;
    printf("{\"asts\":%llu,\"attempts\":%llu,\"duplicates\":%llu,\"pruned\":%llu,\"prepare_seconds\":%.9f,\"generate_seconds\":%.9f,\"asts_per_second\":%.3f,\"instruction_bytes\":%llu,\"plan_arena_bytes\":%zu,\"producer_arena_bytes\":%zu,\"batch_arena_bytes\":%zu,\"checksum\":%llu}\n",(unsigned long long)next,(unsigned long long)b.attempts,(unsigned long long)b.duplicates,(unsigned long long)b.pruned,prepared,generated,next/generated,(unsigned long long)bytes,gn,pn,bn,(unsigned long long)checksum);
    free(ba);
    free(pa);
    free(ga);
    free(text);
    return next==1000000?0:1;
}
