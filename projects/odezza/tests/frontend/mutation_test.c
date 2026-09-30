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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Bounded deterministic mutation smoke test; run this under ASan/UBSan too.
 * Invalid documents may fail in measure or build. Either must respect the arena. */
static uint32_t rng=0x632a10bdu;
static uint32_t random32(void) {
    rng^=rng<<13; rng^=rng>>17; rng^=rng<<5;
    return rng;
}
static OdrResult parse(unsigned phase,OdrJson j,void *arena,size_t capacity,size_t *needed) {
    const OdrTrajectories *t=NULL;
    const OdrStatic *s=NULL;
    const OdrGrammar *g=NULL;
    OdrError e;
    OdrResult r;
    if(phase==0) r=odr_trajectories_parse(j,arena,capacity,needed,&t,&e);
    else if(phase==1) r=odr_static_parse(j,arena,capacity,needed,&s,&e);
    else r=odr_grammar_parse(j,NULL,arena,capacity,needed,&g,&e);
    if(r!=ODR_OK) assert(!t&&!s&&!g);
    return r;
}
int main(void) {
    static const char *seeds[]={
        "{\"problem\":{\"states\":[\"x\",\"y\"],\"known_rhs\":{\"y\":\"-.2*y+sin(x)\"},\"trajectories\":[{\"initial\":[1,2],\"times\":[0,0.1],\"values\":[[1,2],[.9,null]]}]},\"grammar\":{\"states\":[\"x\",\"y\"],\"rhs\":{\"x\":\"-const.c*x\",\"y\":\"-y\"},\"constants\":{\"c\":{\"values\":[1,2,3]}}}}",
        "{\"states\":[\"x\"],\"rules\":{\"R(z)\":[\"z\",\"sin(hole(R,z))\"]},\"rhs\":{\"x\":\"hole(R,x)\"},\"expansion\":{\"max_nodes\":15,\"max_depth\":5}}",
        "{\"states\":[\"x\"],\"rng_banks\":{\"u\":{\"base\":\"uniform01\",\"count\":9}},\"rng\":{\"u\":{\"bank\":\"u\",\"transform\":{\"kind\":\"affine\",\"scale\":2}}},\"rhs\":{\"x\":\"rng.u*x\"}}",
        "{\"states\":[\"x\"],\"known_rhs\":{\"x\":\"pow(1+x,3)\"},\"trajectories\":[{\"initial\":[1],\"times\":[0,0.1],\"values\":[[1],[0.9]]}]}",
        "{\"states\":[\"x\",\"y\"],\"leaves\":{\"s\":{\"states\":[\"x\",\"y\"],\"arity\":2}},\"rhs\":{\"x\":\"leaf.s\",\"y\":\"-y\"},\"retain\":{\"global\":2}}"
    };
    size_t iteration,measured=0,built=0;
    for(iteration=0;iteration<6000;iteration++) {
        char input[4096];
        const char *source=seeds[iteration%5];
        size_t length=strlen(source),edit,nedits=iteration<5||iteration%8==0?0:1+random32()%5;
        unsigned phase;
        memcpy(input,source,length);
        for(edit=0;edit<nedits;edit++) {
            size_t at=random32()%(length+1);
            unsigned op=random32()%4;
            if(op==0&&at<length) {memmove(input+at,input+at+1,length-at-1);--length;}
            else if(op==1&&length+1<sizeof(input)) {memmove(input+at+1,input+at,length-at);input[at]=(char)random32();++length;}
            else if(op==2&&at<length) input[at]=(char)random32();
            else length=at;
        }
        for(phase=0;phase<3;phase++) {
            OdrJson j={input,length};
            size_t required=0,again=0,k;
            unsigned char *arena;
            OdrResult r=parse(phase,j,NULL,0,&required);
            if(r!=ODR_OK||!required||required>4u*1024u*1024u)continue;
            ++measured;
            arena=malloc(required+64);assert(arena);
            memset(arena,0xa7,required+64);
            r=parse(phase,j,arena,required-1,&again);
            assert(r!=ODR_OK);
            for(k=required-1;k<required+64;k++)assert(arena[k]==0xa7);
            r=parse(phase,j,arena,required,&again);
            if(r==ODR_OK)++built;
            for(k=required;k<required+64;k++)assert(arena[k]==0xa7);
            free(arena);
        }
    }
    assert(measured>10&&built>10);
    printf("18000 mutation measurement calls; %zu measured and %zu built arenas passed\n",measured,built);
    return 0;
}
