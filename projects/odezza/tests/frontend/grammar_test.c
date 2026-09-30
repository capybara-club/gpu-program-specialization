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
static OdrJson json(const char *s) {
    OdrJson j= {
        s,strlen(s)
    };
    return j;
}
static unsigned alias_count;
static void alias(void *user,uint64_t prior,uint64_t derivation,uint64_t variant,
                  const char *const *tags,size_t count) {
    (void)user;(void)variant;
    assert(prior==(alias_count?2u:0u)&&derivation==(alias_count?5u:2u));
    assert(count==1&&!strcmp(tags[0],"duplicate"));
    ++alias_count;
}
int main(void) {
    const char *s="{\"states\":[\"x\",\"y\"],\"rules\":{\"R(q)\":[\"q\",\"sin(q)\",{\"expr\":\"q\",\"tags\":[\"duplicate\"]}]},\"shapes\":{\"S\":[\"x\",\"y\"]},\"rhs\":{\"x\":\"R(shape.S)+shape.S\",\"y\":\"-const.a*y\"},\"constants\":{\"a\":{\"values\":[0.5,1,2]}},\"limits\":{\"max_skeletons\":20}}";
    const OdrGrammar *g;
    OdrProducer *producer;
    OdrProducerOptions options= {
        0
    };
    OdrError e;
    OdrBatch b;
    size_t n,need;
    void *plan,*cursor,*batch;
    OdrResult r;
    uint64_t next=0;
    unsigned counts=0;
    r=odr_grammar_parse(json(s),NULL,NULL,0,&n,&g,&e);
    if(r)fprintf(stderr,"measure: %s\n",e.message);
    assert(r==ODR_OK&&g==NULL);
    plan=malloc(n);
    r=odr_grammar_parse(json(s),NULL,plan,n,&need,&g,&e);
    if(r)fprintf(stderr,"parse: %s\n",e.message);
    assert(r==ODR_OK&&need==n);
    assert(odr_grammar_family_count(g)==1);
    options.max_asts=20;
    options.on_duplicate=alias;
    assert(odr_producer_create(g,&options,NULL,0,&n,&producer,&e)==ODR_OK);
    cursor=malloc(n);
    assert(odr_producer_create(g,&options,cursor,n,&need,&producer,&e)==ODR_OK);
    assert(odr_batch_requirements(producer,1,&n)==ODR_OK);
    batch=malloc(n);
    do {
        assert(odr_producer_next(producer,next,1,NULL,0,&need,&b,&e)==ODR_OK);
        assert(odr_producer_next(producer,next,1,batch,n-1,&need,&b,&e)==ODR_BUFFER);
        r=odr_producer_next(producer,next,1,batch,n,&need,&b,&e);
        if(r)fprintf(stderr,"generate: %s\n",e.message);
        assert(r==ODR_OK);
        if(b.count) {
            const OdrCandidate *c=&b.candidates[0];
            assert(c->index==next&&c->numeric_count==3&&c->slot_count==1&&c->rhs_count==2);
            assert(c->rhs[0].program.bytes[0]==0x81);
            assert(c->rhs[0].program.bytes[1]==c->rhs[0].program.bytes[c->rhs[0].program.byte_count-3]);
            ++counts;
        }
        next=b.next_index;
    }
    while(!b.exhausted);
    assert(counts==4&&b.duplicates==2&&alias_count==2);
    assert(odr_producer_next(producer,0,1,batch,n,&need,&b,&e)==ODR_INDEX);
    free(batch);
    free(cursor);
    free(plan);
    puts("prepared grammar cursor/shared-choice/dedup tests passed");
    return 0;
}
