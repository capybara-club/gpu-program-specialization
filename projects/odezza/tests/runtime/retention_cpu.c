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
#include <assert.h>
#include "retention_support.h"
static void toggle_tail(void) {
    OdzJob *j=calloc(1,sizeof(*j));OdrStatic fixed={0};OdrCandidate c={0};
    OdrRhs known[3]={0},rhs={0};OdrAxis slot={0};Kept *k=NULL;
    unsigned char zero[]={0x83,0,0,0,0,0x80};
    unsigned char quad[]={0x82,0,0x81,0,0x81,1,0x81,2,0x81,3,0x85,0,1,0x90,0x80};
    unsigned char binary[]={0x82,0,0x81,0,0x81,1,0x84,0,0x90,0x80};
    float value=3.25f;unsigned i,kind,permutation;
    assert(j);j->fixed=&fixed;j->local_k=2;fixed.state_count=4;fixed.rhs=known;fixed.rhs_count=3;
    for(i=0;i<3;i++){known[i].state_index=(uint8_t)i;known[i].program.bytes=zero;known[i].program.byte_count=sizeof(zero);}
    rhs.state_index=3;c.rhs=&rhs;c.rhs_count=1;c.slot_count=1;c.slots=&slot;slot.name="rng.a";
    for(kind=0;kind<2;kind++) {
        rhs.program.bytes=kind?quad:binary;rhs.program.byte_count=kind?sizeof(quad):sizeof(binary);
        for(permutation=0;permutation<(kind?4u:2u);permutation++) {
            c.index=kind;
            assert(!odz_offer(j,&c,permutation,0,0,&value,&k));
            assert(k->numeric_size==47&&k->structure_size==44);
            assert(k->numeric[44]==permutation&&k->structure[41]==permutation);
            assert(k->numeric[46]==0x80&&k->structure[43]==0x80);
            odz_keep_unref(j,k);
        }
    }
    assert(j->memory.total==0);free(j);
}
int main(void){
    toggle_tail();
    OdzJob *j=calloc(1,sizeof(*j));OdrStatic fixed={0};OdrGrammar grammar={0};Family family={0};FamilyRun run={0};
    OdrCandidate c={0};OdrRhs rhs={0};OdrAxis slots[16]={0};float values[16];unsigned char bytes[64];
    Kept *rows[256];const char *base[]={"base"},*late[]={"late","base"};void *pair[]={j,0};size_t i,n=0;
    assert(j);pthread_mutex_init(&j->mutex,NULL);j->fixed=&fixed;j->grammar=&grammar;j->families=&run;j->family_count=1;
    fixed.state_count=1;grammar.family_count=1;grammar.families=&family;family.max_variants=1;
    j->tag_count=1;j->tags[0].name=odz_alloc(j,5);memcpy((char*)j->tags[0].name,"late",5);
    j->tags[0].ranking.k=1;j->tags[0].ranking.rows=odz_alloc(j,sizeof(Kept*));
    for(i=0;i<16;i++){bytes[n++]=0x82;bytes[n++]=(unsigned char)i;if(i)bytes[n++]=0x90;slots[i].name="const.c";slots[i].count=256;slots[i].numeric_stride=1;}
    bytes[n++]=0x80;rhs.program.bytes=bytes;rhs.program.byte_count=n;c.rhs=&rhs;c.rhs_count=1;c.slot_count=16;c.slots=slots;c.tags=base;c.tag_count=1;
    /* Alias before scoring must preserve base-before-alias tag ordering. */
    odz_alias(pair,0,1,0,late,2);
    for(i=0;i<256;i++){
        size_t k;for(k=0;k<16;k++)values[k]=(float)(i+k);
        assert(!odz_offer(j,&c,0,i,(float)i,values,&rows[i]));
        assert(rows[i]->numeric_size==5+16*5+15+1);
        assert(rows[i]->shared==rows[0]->shared);
        assert(rows[i]->shared->provenance->count==2);
        assert(!strcmp(rows[i]->shared->provenance->names[0],"base"));
    }
    assert(j->memory.peak_total<120000);
    printf("256 rows / 16 constants retained allocation peak: %llu bytes\n",(unsigned long long)j->memory.peak_total);
    /* A late alias promotes an archived result after it lost a global contest. */
    run.archive=odz_alloc(j,sizeof(Kept*));run.archive[0]=rows[0];++rows[0]->refs;
    odz_alias(pair,0,2,0,late,2);assert(j->tags[0].ranking.count==1&&j->tags[0].ranking.rows[0]==rows[0]);
    for(i=0;i<256;i++)odz_keep_unref(j,rows[i]);
    odz_retention_close(j);assert(j->memory.total==0);
    for(i=0;i<ODZ_RETAIN_BUCKETS;i++)assert(!j->shared[i]&&!j->provenance[i]);
    j->tag_count=0;c.tags=NULL;c.tag_count=0;j->local_k=1;
    assert(!odz_offer(j,&c,0,7,.5f,values,&rows[0]));
    assert(rows[0]->shared->inline_storage&&rows[0]->numeric_size==101);
    odz_keep_unref(j,rows[0]);assert(j->memory.total==0);
    pthread_mutex_destroy(&j->mutex);free(j);puts("PASS shared structures, exact numeric payloads, early/late aliases, cleanup");return 0;
}
