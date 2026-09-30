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
typedef struct Fixture {
    void *plan,*cursor,*batch;
    size_t bytes;
    const OdrGrammar *g;
    OdrProducer *p;
}

Fixture;

static OdrJson json(const char *s) {
    OdrJson j= {
        s,strlen(s)
    };
    return j;
}

static OdrResult plan(const char *s,Fixture *f) {
    size_t n;
    OdrError e;
    OdrResult r;
    memset(f,0,sizeof(*f));
    r=odr_grammar_parse(json(s),NULL,NULL,0,&n,&f->g,&e);
    if(r)return r;
    f->plan=malloc(n);
    assert(f->plan);
    r=odr_grammar_parse(json(s),NULL,f->plan,n,&n,&f->g,&e);
    return r;
}

static void cursor(Fixture *f,OdrProducerOptions *o,size_t count) {
    size_t n;
    OdrError e;
    assert(odr_producer_create(f->g,o,NULL,0,&n,&f->p,&e)==ODR_OK);
    f->cursor=malloc(n);
    assert(f->cursor);
    assert(odr_producer_create(f->g,o,f->cursor,n,&n,&f->p,&e)==ODR_OK);
    assert(odr_batch_requirements(f->p,count,&f->bytes)==ODR_OK);
    f->batch=malloc(f->bytes);
    assert(f->batch);
}

static OdrResult next(Fixture *f,uint64_t start,size_t count,OdrBatch *b) {
    size_t n;
    OdrError e;
    return odr_producer_next(f->p,start,count,f->batch,f->bytes,&n,b,&e);
}

static void cleanup(Fixture *f) {
    free(f->batch);
    free(f->cursor);
    free(f->plan);
}

static void bad(const char *s) {
    Fixture f;
    assert(plan(s,&f)!=ODR_OK);
    cleanup(&f);
}

static uint64_t digest(uint64_t h,const void *p,size_t n) {
    const unsigned char *b=p;
    size_t i;
    for(i=0;i<n;i++)h=(h^b[i])*UINT64_C(1099511628211);
    return h;
}

static uint64_t sampled(size_t chunk) {
    const char *s="{\"states\":[\"x\",\"y\"],\"rules\":{\"R\":[\"x\",\"y\",\"sin(R)\",\"R+R\",\"R*R\"]},\"rhs\":{\"x\":\"R\",\"y\":\"-y\"},\"expansion\":{\"strategy\":\"sample\",\"seed\":125,\"max_nodes\":15,\"max_depth\":5,\"max_expansion_depth\":5},\"limits\":{\"max_skeletons\":50,\"max_variants\":50}}";
    Fixture f;
    OdrProducerOptions o= {
        0
    };
    OdrBatch b;
    uint64_t index=0,h=1;
    size_t i,j;
    o.max_attempts_per_batch=3;
    assert(plan(s,&f)==ODR_OK);
    cursor(&f,&o,chunk);
    do {
        assert(next(&f,index,chunk,&b)==ODR_OK);
        for(i=0;i<b.count;i++) {
            const OdrCandidate *c=&b.candidates[i];
            h=digest(h,&c->derivation_index,sizeof(c->derivation_index));
            for(j=0;j<c->rhs_count;j++)h=digest(h,c->rhs[j].program.bytes,c->rhs[j].program.byte_count);
        }
        index=b.next_index;
    }
    while(!b.exhausted);
    assert(index==50);
    cleanup(&f);
    return h;
}


static uint64_t skeleton_cap(unsigned cap,size_t chunk) {
    char text[2048],limit[128];Fixture f;OdrBatch b;OdrProducerOptions o={0};
    uint64_t index=0,h=1;size_t i;OdrAllocationInfo a;
    snprintf(limit,sizeof(limit),",\"limits\":{\"max_configurations_per_skeleton\":%u}",cap);
    snprintf(text,sizeof(text),"{\"states\":[\"x\",\"y\",\"z\",\"u\",\"v\",\"w\"],"
        "\"constants\":{\"c\":{\"values\":[1,2,3,4]}},"
        "\"leaves\":{\"a\":{\"states\":[\"x\",\"y\",\"z\",\"u\",\"v\",\"w\"],\"coverage\":\"explicit\",\"groups\":[[\"x\",\"y\"],[\"z\",\"u\"],[\"v\",\"w\"]]}},"
        "\"rules\":{\"R\":[\"const.c*leaf.a\",\"sin(const.c*leaf.a)\"]},"
        "\"rhs\":{\"x\":\"R\",\"y\":\"y\",\"z\":\"z\",\"u\":\"u\",\"v\":\"v\",\"w\":\"w\"}%s}",cap?limit:"");
    assert(odr_request_allocations(json(text),&a,NULL)==ODR_OK);
    assert(plan(text,&f)==ODR_OK);cursor(&f,&o,chunk);
    do {
        assert(next(&f,index,chunk,&b)==ODR_OK);
        for(i=0;i<b.count;i++) {
            const OdrCandidate *c=&b.candidates[i];
            assert(c->bank_start==0&&c->numeric_count==4&&c->toggle_bits==1);
            if(cap==10)assert(c->bank_count==(c->variant_index==0?4:1)&&c->variant_index<2);
            h=digest(h,&c->derivation_index,sizeof(c->derivation_index));
            h=digest(h,&c->variant_index,sizeof(c->variant_index));
            h=digest(h,&c->bank_count,sizeof(c->bank_count));
        }
        index=b.next_index;
    }while(!b.exhausted);
    if(cap==10)assert(index==4&&b.configurations_reserved==20&&b.configuration_limited_derivations==2);
    if(cap==1)assert(index==0&&b.configurations_reserved==0&&b.configuration_limited_derivations==2);
    if(cap==8)assert(index==2&&b.configurations_reserved==16&&b.configuration_limited_derivations==2);
    if(!cap)assert(index==6&&b.configurations_reserved==48&&!b.configuration_limited_derivations);
    cleanup(&f);return h;
}

int main(void) {
    Fixture f;
    OdrBatch b;
    OdrProducerOptions o= {
        0
    };
    OdrFamilyInfo fi;
    size_t n;
    assert(sampled(1)==sampled(7));
    assert(skeleton_cap(10,1)==skeleton_cap(10,7));
    assert(skeleton_cap(1,1)==skeleton_cap(1,7));
    assert(skeleton_cap(8,1)==skeleton_cap(8,7));
    assert(skeleton_cap(0,1)==skeleton_cap(0,7));
    assert(plan("{\"states\":[\"x\",\"y\"],\"rules\":{\"R\":[\"x\",\"sin(x)\"]},\"rhs\":{\"x\":\"R\",\"y\":\"y\"}}",&f)==ODR_OK);
    o.max_asts=10;
    o.dedup_capacity=6;
    cursor(&f,&o,2);
    assert(next(&f,0,2,&b)==ODR_CAPACITY&&b.count==1&&b.next_index==1);
    assert(next(&f,1,2,&b)==ODR_CAPACITY&&b.count==0&&b.next_index==1);
    assert(odr_batch_requirements(f.p,SIZE_MAX,&n)==ODR_OVERFLOW);
    cleanup(&f);
    memset(&o,0,sizeof(o));
    o.max_attempts_per_batch=1;
    assert(plan("{\"states\":[\"x\",\"y\"],\"rules\":{\"R\":[\"x\",\"x\",\"y\"]},\"rhs\":{\"x\":\"R\",\"y\":\"y\"}}",&f)==ODR_OK);
    cursor(&f,&o,3);
    assert(next(&f,0,3,&b)==ODR_OK&&b.count==1&&b.yielded);
    assert(next(&f,1,3,&b)==ODR_OK&&b.count==0&&b.yielded&&!b.exhausted);
    assert(next(&f,1,3,&b)==ODR_OK&&b.count==1&&b.exhausted);
    cleanup(&f);
    assert(plan("{\"states\":[\"x\",\"y\"],\"rhs\":{\"x\":\"x+y\",\"y\":\"y\"},\"limits\":{\"max_expansion_steps\":1}}",&f)==ODR_OK);
    cursor(&f,&o,2);
    assert(next(&f,0,2,&b)==ODR_OK&&b.count==0&&b.stop_reason==ODR_STOP_EXPANSION_STEPS&&b.expansion_steps==1);
    cleanup(&f);
    assert(plan("{\"states\":[\"x\",\"y\"],\"constants\":{\"c\":{\"values\":[1,2,3,4]}},\"leaves\":{\"a\":{\"states\":[\"x\",\"y\"]}},\"rhs\":{\"x\":\"const.c*leaf.a\",\"y\":\"y\"},\"families\":[{\"id\":\"a\"},{\"id\":\"b\"}],\"limits\":{\"max_configurations\":11}}",&f)==ODR_OK);
    assert(odr_grammar_family_info(f.g,0,&fi)==ODR_OK&&fi.max_configurations==6);
    assert(odr_grammar_family_info(f.g,1,&fi)==ODR_OK&&fi.max_configurations==5);
    cursor(&f,&o,1);
    assert(next(&f,0,1,&b)==ODR_OK&&b.candidates[0].numeric_count==4&&b.candidates[0].bank_count==3&&b.configurations_reserved==6);
    cleanup(&f);
    assert(plan("{\"states\":[\"x\"],\"rng_banks\":{\"u\":{\"base\":\"uniform01\",\"count\":3}},\"rules\":{\"R\":{\"expr\":\"rng.u*x+param.p\",\"locals\":{\"constants\":{\"s\":{\"value\":2}},\"parameters\":{\"p\":{\"initial\":1}},\"rng\":{\"u\":{\"bank\":\"u\",\"transform\":{\"kind\":\"affine\",\"scale\":\"const.s\"}}}}}},\"rhs\":{\"x\":\"R+R\"}}",&f)==ODR_OK);
    cursor(&f,&o,1);
    assert(next(&f,0,1,&b)==ODR_OK&&b.count==1&&b.candidates[0].slot_count==6&&b.candidates[0].numeric_count==9);
    {
        const OdrAxis *a=b.candidates[0].slots;
        assert(a[2].kind==2&&a[3].kind==2&&a[2].scale_slot==0&&a[3].scale_slot==1);
        assert(a[2].instance!=a[3].instance&&a[2].numeric_stride==3&&a[3].numeric_stride==1);
    }
    cleanup(&f);
    bad("{\"states\":[\"x\"],\"rhs\":{\"x\":\"x\",\"z\":\"x\"}}");
    bad("{\"states\":[\"x\"],\"rhs\":{\"x\":\"x\"},\"rules\":{\"R(a,a)\":\"a\"}}");
    bad("{\"states\":[\"x\"],\"rhs\":{\"x\":\"x\"},\"rules\":{\"R(a)\":\"a\",\"R(b)\":\"b\"}}");
    bad("{\"states\":[\"x\",\"y\",\"z\"],\"rhs\":{\"x\":\"leaf.a\",\"y\":\"y\",\"z\":\"z\"},\"leaves\":{\"a\":{\"states\":[\"x\",\"y\"],\"coverage\":\"explicit\",\"groups\":[[\"x\",\"z\"]]}}}");
    bad("{\"states\":[\"x\"],\"rhs\":{\"x\":\"x\"},\"families\":[{\"id\":\"a\",\"limits\":{\"max_skeletons\":10}},{\"id\":\"b\",\"limits\":{\"max_skeletons\":10}}],\"limits\":{\"max_skeletons\":19}}");
    bad("{\"states\":[\"x\"],\"rhs\":{\"x\":\"x\"},\"allocation\":{\"redistribute_unused\":true}}");
    bad("{\"states\":[\"x\"],\"rules\":{\"R\":{\"choices\":[\"x\"],\"weights\":[1]}},\"rhs\":{\"x\":\"R\"}}");
    bad("{\"states\":[\"x\"],\"shapes\":{\"S\":{\"choices\":[\"x\"],\"tags\":[\"ignored\"]}},\"rhs\":{\"x\":\"shape.S\"}}");
    assert(plan("{\"states\":[\"x\"],\"rules\":{\"R\":{\"choices\":[\"x\",\"sin(x)\"]}},\"shapes\":{\"S\":{\"choices\":[\"R\"]}},\"rhs\":{\"x\":\"shape.S\"}}",&f)==ODR_OK);
    cleanup(&f);
    puts("chunk invariance, partial failures, bounded yields, local RNG and family budgets passed");
    return 0;
}
