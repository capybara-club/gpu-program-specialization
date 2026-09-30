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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv) {
    FILE *f;
    long len;
    char *text;
    OdrJson j;
    OdrError error= {
        0
    };
    const OdrGrammar *g;
    OdrProducer *p;
    OdrProducerOptions options= {
        0
    };
    size_t bytes,need,i;
    void *plan,*producer,*batch;
    OdrBatch b;
    OdrResult r;
    uint64_t next=0,total=0;
    size_t requested=argc>3?(size_t)strtoull(argv[3],NULL,10):16;
    if(argc<2)return 2;
    f=fopen(argv[1],"rb");
    if(!f)return 2;
    fseek(f,0,SEEK_END);
    len=ftell(f);
    rewind(f);
    text=malloc((size_t)len+1);
    if(!text||fread(text,1,(size_t)len,f)!=(size_t)len)return 2;
    fclose(f);
    text[len]=0;
    j.data=text;
    j.size=(size_t)len;
    r=odr_grammar_parse(j,NULL,NULL,0,&bytes,&g,&error);
    if(r) {
        fprintf(stderr,"measure %d: %s\n",r,error.message);
        return 1;
    }
    plan=malloc(bytes);
    r=odr_grammar_parse(j,NULL,plan,bytes,&need,&g,&error);
    if(r) {
        fprintf(stderr,"parse %d: %s\n",r,error.message);
        return 1;
    }
    options.max_asts=requested;
    options.family_index=argc>2?(uint32_t)strtoul(argv[2],NULL,10):0;
    r=odr_producer_create(g,&options,NULL,0,&bytes,&p,&error);
    if(r)return 1;
    producer=malloc(bytes);
    r=odr_producer_create(g,&options,producer,bytes,&need,&p,&error);
    if(r)return 1;
    odr_batch_requirements(p,16,&bytes);
    batch=malloc(bytes);
    while(total<requested) {
        size_t n=requested-total<16?(size_t)(requested-total):16;
        r=odr_producer_next(p,next,n,batch,bytes,&need,&b,&error);
        if(r) {
            fprintf(stderr,"produce %d: %s\n",r,error.message);
            return 1;
        }
        for(i=0;i<b.count;i++) {
            const OdrCandidate *c=&b.candidates[i];
            size_t x,y;
            printf("%llu %u %u %llu",(unsigned long long)c->index,c->slot_count,c->toggle_bits,(unsigned long long)c->numeric_count);
            for(x=0;x<c->rhs_count;x++) {
                putchar(' ');
                for(y=0;y<c->rhs[x].program.byte_count;y++)printf("%02x",c->rhs[x].program.bytes[y]);
            }
            putchar('\n');
        }
        total+=b.count;
        next=b.next_index;
        if(b.exhausted)break;
    }
    fprintf(stderr,"total=%llu attempts=%llu duplicates=%llu pruned=%llu\n",(unsigned long long)total,(unsigned long long)b.attempts,(unsigned long long)b.duplicates,(unsigned long long)b.pruned);
    free(batch);
    free(producer);
    free(plan);
    free(text);
    return 0;
}
