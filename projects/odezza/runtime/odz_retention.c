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
static int policy(OdrJson value,Ranking *rank,OdzJob *j) {
    OdrJson v;
    uint64_t k=0;
    rank->unit=1;
    if(jtype(value)=='{') {
        if(!jkeys(value,"k,unit",NULL)||!jget(value,"k",&v)||!ju64(v,&k))return odz_fail(j,"retention policy requires k");
        if(jget(value,"unit",&v)) {
            if(jeq(v,"numeric_candidate"))rank->unit=0;
            else if(jeq(v,"resolved_structure"))rank->unit=1;
            else if(jeq(v,"variant"))rank->unit=2;
            else if(jeq(v,"evaluation_row"))rank->unit=3;
            else return odz_fail(j,"unknown retention unit");
        }
    }
    else if(!ju64(value,&k))return odz_fail(j,"invalid retention policy");
    if(k>ODZ_K)return odz_fail(j,"retention k exceeds 256");
    rank->k=(uint32_t)k;
    rank->rows=odz_alloc(j,(size_t)k*sizeof(*rank->rows));
    if(k&&!rank->rows)return 1;
    if(!rank->unit&&k>j->local_k)j->local_k=(uint32_t)k;
    if(rank->unit==3&&k>j->row_k)j->row_k=(uint32_t)k;
    return 0;
}

int odz_retention_parse(OdzJob *j) {
    OdrJson retain=j->grammar->info.retention,v;
    size_t i;
    j->local_k=1;
    if(!retain.data) {
        j->global.k=16;
        j->global.unit=1;
        j->global.rows=odz_alloc(j,16*sizeof(Kept*));
        return j->global.rows?0:1;
    }
    if(!jkeys(retain,"global,per_family,by_tag",NULL))return odz_fail(j,"unsupported retention fields");
    if(jget(retain,"global",&v)&&policy(v,&j->global,j))return 1;
    if(jget(retain,"per_family",&v))for(i=0;i<j->family_count;i++)if(policy(v,&j->families[i].ranking,j))return 1;
    if(jget(retain,"by_tag",&v)) {
        Ji it;
        OdrJson k,p;
        if(jtype(v)!='{')return odz_fail(j,"by_tag must be an object");
        jiter(v,&it);
        while(jnext(&it,&k,&p)) {
            size_t length;
            char *name;
            if(j->tag_count==ODZ_TAGS)return odz_fail(j,"at most 64 retained tags supported");
            if(!jstring(k,NULL,0,&length)||!length)return odz_fail(j,"empty tag");
            name=odz_alloc(j,length+1);
            if(!name)return 1;
            jstring(k,name,length+1,NULL);
            j->tags[j->tag_count].name=name;
            ++j->tag_count;
            if(policy(p,&j->tags[j->tag_count-1].ranking,j))return 1;
        }
    }
    for(i=0;i<j->tag_count;i++) {
        Ranking *r=&j->tags[i].ranking;
        uint32_t need=r->unit==0||r->unit==3?r->k:(r->k?1u:0u);
        uint32_t *cap=r->unit==3?&j->tag_row_k:&j->tag_local_k;
        if(need>*cap)*cap=need;
    }
    /* Raw-only jobs need no numerical-distinct collection or continuation. */
    {
        int distinct=j->global.k&&j->global.unit!=3;
        for(i=0;i<j->family_count;i++)if(j->families[i].ranking.k&&j->families[i].ranking.unit!=3)distinct=1;
        for(i=0;i<j->tag_count;i++)if(j->tags[i].ranking.k&&j->tags[i].ranking.unit!=3)distinct=1;
        if(!distinct)j->local_k=0;
    }
    return 0;
}

int odz_rank_order(const Kept *a,const Kept *b) {
    if(a->mse!=b->mse)return a->mse<b->mse?-1:1;
    if(a->family!=b->family)return a->family<b->family?-1:1;
    if(a->ast!=b->ast)return a->ast<b->ast?-1:1;
    if(a->bank!=b->bank)return a->bank<b->bank?-1:1;
    return a->permutation<b->permutation?-1:a->permutation!=b->permutation;
}

int odz_same_numeric(const Kept *a,const Kept *b) {
    return a->numeric_size==b->numeric_size&&!memcmp(a->numeric,b->numeric,a->numeric_size);
}

static int same(const Kept *a,const Kept *b,unsigned unit) {
    if(unit==3)return a->family==b->family&&a->ast==b->ast&&a->bank==b->bank&&a->permutation==b->permutation;
    if(!unit)return odz_same_numeric(a,b);
    if(unit==1)return a->structure_size==b->structure_size&&!memcmp(a->structure,b->structure,a->structure_size);
    return a->family==b->family&&a->ast==b->ast;
}

static void shared_unref(OdzJob *j,KeptShared *s);

void odz_keep_unref(OdzJob *j,Kept *k) {
    if(k&&!--k->refs) {
        shared_unref(j,k->shared);
        odz_host_free(j,k);
    }
}

void odz_rank_offer(OdzJob *j,Ranking *r,Kept *k) {
    size_t i,pos=r->count;
    if(!r->k)return;
    for(i=0;i<r->count;i++)if(same(r->rows[i],k,r->unit)) {
        if(odz_rank_order(r->rows[i],k)<=0)return;
        pos=i;
        break;
    }
    if(pos==r->count&&r->count==r->k) {
        if(odz_rank_order(r->rows[r->count-1],k)<=0)return;
        pos=r->count-1;
    }
    if(pos<r->count) {
        odz_keep_unref(j,r->rows[pos]);
        for(i=pos+1;i<r->count;i++)r->rows[i-1]=r->rows[i];
        --r->count;
    }
    pos=0;
    while(pos<r->count&&odz_rank_order(r->rows[pos],k)<=0)++pos;
    for(i=r->count;i>pos;i--)r->rows[i]=r->rows[i-1];
    r->rows[pos]=k;
    ++k->refs;
    ++r->count;
}

static void append32(unsigned char *p,uint32_t n) {
    unsigned i;
    for(i=0;i<4;i++)p[i]=(unsigned char)(n>>(8*i));
}

static size_t resolve(const OdrProgram *program,const float *values,uint32_t permutation,unsigned char *out,int *remap,unsigned *next,size_t *peak) {
    size_t at=0,n=0;
    if(peak)*peak=0;
    while(at<program->byte_count) {
        unsigned op=program->bytes[at++];
        if(op==0x82) {
            unsigned slot=program->bytes[at++];
            if(values) {
                uint32_t bits;memcpy(&bits,&values[slot],4);
                if(out){out[n]=0x83;append32(out+n+1,bits);}n+=5;
            } else {
                if(remap[slot]<0)remap[slot]=(int)(*next)++;
                if(out){out[n]=0x82;out[n+1]=(unsigned char)remap[slot];}n+=2;
            }
        } else if(op==0x84||op==0x85) {
            unsigned width=op==0x84?1:2,choice=0,i,total=1u<<width;
            for(i=0;i<width;i++)choice|=((permutation>>program->bytes[at++])&1u)<<i;
            if(n<2*total)return 0;
            if(out){out[n-2*total]=0x81;out[n-2*total+1]=out[n-2*total+2*choice+1];}
            n-=2*(total-1);
        } else {
            unsigned extra=op==0x81?1:op==0x83?4:0;
            if(out){out[n]=(unsigned char)op;if(extra)memcpy(out+n+1,program->bytes+at,extra);}
            ++n;n+=extra;at+=extra;
        }
        if(peak&&n>*peak)*peak=n;
    }
    return n;
}

static uint64_t tag_mask(OdzJob *j,const char *const *tags,size_t count) {
    uint64_t mask=0;
    size_t i,k;
    for(i=0;i<j->tag_count;i++)for(k=0;k<count;k++)if(!strcmp(j->tags[i].name,tags[k]))mask|=UINT64_C(1)<<i;
    return mask;
}

static size_t bucket(uint32_t family,uint64_t ast,uint32_t permutation) {
    return (size_t)odr_mix(ast^((uint64_t)family<<32)^((uint64_t)permutation*UINT64_C(0x9e3779b97f4a7c15)))&(ODZ_RETAIN_BUCKETS-1);
}
static int merge_tags(OdzJob *j,Provenance *p,const char *const *tags,size_t count) {
    size_t i,k,add=0;
    for(i=0;i<count;i++) {
        for(k=0;k<p->count;k++)if(!strcmp(p->names[k],tags[i]))break;
        if(k==p->count){size_t x;for(x=0;x<i;x++)if(!strcmp(tags[x],tags[i]))break;if(x==i)++add;}
    }
    if(add) {
        const char **names=odz_host_resize(j,(void*)p->names,(p->count+add)*sizeof(*names),ODZ_MEM_RETAINED);
        if(!names)return odz_fail(j,"tag provenance allocation failed");
        p->names=names;
        for(i=0;i<count;i++) {
            for(k=0;k<p->count;k++)if(!strcmp(p->names[k],tags[i]))break;
            if(k==p->count)p->names[p->count++]=tags[i];
        }
    }
    p->tags|=tag_mask(j,tags,count);return 0;
}
static Provenance *provenance(OdzJob *j,uint32_t family,uint64_t ast) {
    size_t b=bucket(family,ast,0);Provenance *p=j->provenance[b];
    for(;p;p=p->next)if(p->family==family&&p->ast==ast)return p;
    p=odz_host_alloc(j,sizeof(*p),ODZ_MEM_RETAINED);
    if(!p){odz_fail(j,"tag provenance allocation failed");return NULL;}
    p->family=family;p->ast=ast;p->next=j->provenance[b];j->provenance[b]=p;return p;
}
static void provenance_free(OdzJob *j,Provenance *p) {
    Provenance **link=&j->provenance[bucket(p->family,p->ast,0)];
    while(*link&&*link!=p)link=&(*link)->next;
    if(*link)*link=p->next;
    odz_host_free(j,p->names);odz_host_free(j,p);
}
static void shared_unref(OdzJob *j,KeptShared *s) {
    if(s&&s->inline_storage)return;
    if(s&&!--s->refs) {
        KeptShared **link=&j->shared[bucket(s->family,s->ast,s->permutation)];
        Provenance *p=s->provenance;
        while(*link&&*link!=s)link=&(*link)->next;
        if(*link)*link=s->next;
        odz_host_free(j,s);
        if(p&&!--p->refs&&!p->pinned)provenance_free(j,p);
    }
}
/* A singleton without possible provenance has nothing to share. Pack its
 * metadata into the row allocation, avoiding hash entries/refcounts and extra
 * allocation work on structure-only screens. The temporary program capacities
 * are bounded by bytecode length, as in the original singleton representation. */
static int offer_singleton(OdzJob *j,const OdrCandidate *c,uint32_t permutation,uint64_t bank,float mse,const float *values,Kept **out) {
    const OdrProgram *programs[126]={0};size_t i,state,cap=0,bytes;int remap[256];unsigned next=0;
    Kept *k;KeptShared *s;Provenance *p;
    for(i=0;i<j->fixed->rhs_count;i++)programs[j->fixed->rhs[i].state_index]=&j->fixed->rhs[i].program;
    for(i=0;i<c->rhs_count;i++)programs[c->rhs[i].state_index]=&c->rhs[i].program;
    for(state=0;state<j->fixed->state_count;state++) {
        if(!programs[state]||programs[state]->byte_count>(SIZE_MAX-cap-5)/3)return odz_fail(j,"retained program overflow");
        cap+=5+3*programs[state]->byte_count;
    }
    if(cap>(SIZE_MAX-sizeof(*k)-sizeof(*s)-sizeof(*p)-c->slot_count*(sizeof(OdrAxis)+sizeof(float)))/2)
        return odz_fail(j,"retained record overflow");
    bytes=sizeof(*k)+sizeof(*s)+sizeof(*p)+c->slot_count*(sizeof(OdrAxis)+sizeof(float))+2*cap;
    k=odz_host_alloc(j,bytes,ODZ_MEM_RETAINED);if(!k)return odz_fail(j,"retention allocation failed");
    s=(KeptShared *)(k+1);p=(Provenance *)(s+1);s->inline_storage=1;s->provenance=p;
    k->refs=1;k->bytes=bytes;k->shared=s;k->family=c->family_index;k->ast=c->index;
    k->derivation=c->derivation_index;k->variant=c->variant_index;k->permutation=permutation;k->bank=bank;k->mse=mse;k->nslots=c->slot_count;
    k->slots=(OdrAxis *)(p+1);k->values=(float *)(k->slots+c->slot_count);
    k->numeric=(unsigned char *)(k->values+c->slot_count);k->structure=k->numeric+cap;
    if(c->slot_count){memcpy(k->slots,c->slots,c->slot_count*sizeof(OdrAxis));memcpy(k->values,values,c->slot_count*sizeof(float));}
    memset(remap,0xff,sizeof(remap));
    for(state=0;state<j->fixed->state_count;state++) {
        unsigned char *np=k->numeric+k->numeric_size,*sp=k->structure+k->structure_size;
        size_t n=resolve(programs[state],values,permutation,np+5,remap,&next,NULL),z=resolve(programs[state],NULL,permutation,sp+5,remap,&next,NULL);
        if(!n||!z){odz_keep_unref(j,k);return odz_fail(j,"invalid retained program");}
        np[0]=sp[0]=(unsigned char)state;append32(np+1,(uint32_t)n);append32(sp+1,(uint32_t)z);
        k->numeric_size+=5+n;k->structure_size+=5+z;
    }
    *out=k;return 0;
}

/* Constant-slot metadata and resolved structure are shared by every retained
 * bank row of an AST/permutation. Tag provenance is shared across permutations. */
int odz_offer(OdzJob *j,const OdrCandidate *c,uint32_t permutation,uint64_t bank,float mse,const float *values,Kept **out) {
    const OdrProgram *programs[126]={0};size_t i,state,numeric_size=0,structure_size=0,numeric_capacity=0,structure_capacity=0,b;
    int remap[256];unsigned next=0;Kept *k;KeptShared *shared;Provenance *p;
    *out=NULL;
    if(!j->provenance_possible&&!j->tag_count&&!c->tag_count&&j->local_k<=1&&j->row_k<=1)
        return offer_singleton(j,c,permutation,bank,mse,values,out);
    b=bucket(c->family_index,c->index,permutation);
    for(shared=j->shared[b];shared;shared=shared->next)
        if(shared->family==c->family_index&&shared->ast==c->index&&shared->permutation==permutation)break;
    for(i=0;i<j->fixed->rhs_count;i++)programs[j->fixed->rhs[i].state_index]=&j->fixed->rhs[i].program;
    for(i=0;i<c->rhs_count;i++)programs[c->rhs[i].state_index]=&c->rhs[i].program;
    memset(remap,0xff,sizeof(remap));
    for(state=0;state<j->fixed->state_count;state++) {
        size_t n,s,np,sp;
        if(!programs[state]||programs[state]->byte_count>(SIZE_MAX-5)/3)return odz_fail(j,"invalid retained program");
        n=resolve(programs[state],values,permutation,NULL,remap,&next,&np);
        s=resolve(programs[state],NULL,permutation,NULL,remap,&next,&sp);
        if(!n||!s||np>SIZE_MAX-5||sp>SIZE_MAX-5||numeric_size>SIZE_MAX-np-5||structure_size>SIZE_MAX-sp-5)return odz_fail(j,"retained program overflow");
        /* Toggle resolution first emits all state alternatives, then compacts
         * them in place. Reserve the high-water write position, not just the
         * final byte count. Hashes/reports still use only the compact payload. */
        if(numeric_size+np+5>numeric_capacity)numeric_capacity=numeric_size+np+5;
        if(structure_size+sp+5>structure_capacity)structure_capacity=structure_size+sp+5;
        numeric_size+=n+5;structure_size+=s+5;
    }
    if(!shared) {
        size_t overhead=sizeof(*shared)+c->slot_count*sizeof(OdrAxis),bytes;
        if(structure_capacity>SIZE_MAX-overhead)return odz_fail(j,"retained structure overflow");
        bytes=overhead+structure_capacity;
        shared=odz_host_alloc(j,bytes,ODZ_MEM_RETAINED);
        if(!shared)return odz_fail(j,"shared retention allocation failed");
        p=provenance(j,c->family_index,c->index);
        if(!p){odz_host_free(j,shared);return 1;}
        if(!p->initialized) {
            /* Preserve original tags before aliases even when a duplicate was
             * generated before the first score arrived. */
            Provenance initial={0};
            if(merge_tags(j,&initial,c->tags,c->tag_count)||merge_tags(j,&initial,p->names,p->count)) {
                odz_host_free(j,initial.names);odz_host_free(j,shared);
                if(!p->refs&&!p->pinned)provenance_free(j,p);
                return 1;
            }
            odz_host_free(j,p->names);p->names=initial.names;p->count=initial.count;p->tags|=initial.tags;p->initialized=1;
        }
        shared->family=c->family_index;shared->ast=c->index;shared->permutation=permutation;
        shared->nslots=c->slot_count;shared->provenance=p;++p->refs;
        shared->slots=(OdrAxis *)(shared+1);shared->structure=(unsigned char *)(shared->slots+c->slot_count);
        if(c->slot_count)memcpy(shared->slots,c->slots,c->slot_count*sizeof(OdrAxis));
        shared->structure_size=structure_size;structure_size=0;memset(remap,0xff,sizeof(remap));next=0;
        for(state=0;state<j->fixed->state_count;state++) {
            unsigned char *sp=shared->structure+structure_size;
            size_t n=resolve(programs[state],NULL,permutation,sp+5,remap,&next,NULL);
            sp[0]=(unsigned char)state;append32(sp+1,(uint32_t)n);structure_size+=5+n;
        }
        shared->next=j->shared[b];j->shared[b]=shared;
    }
    ++shared->refs;
    if(numeric_capacity>SIZE_MAX-sizeof(*k)-c->slot_count*sizeof(float)) {shared_unref(j,shared);return odz_fail(j,"retained numeric overflow");}
    k=odz_host_alloc(j,sizeof(*k)+c->slot_count*sizeof(float)+numeric_capacity,ODZ_MEM_RETAINED);
    if(!k){shared_unref(j,shared);return odz_fail(j,"retention allocation failed");}
    k->refs=1;k->bytes=sizeof(*k)+c->slot_count*sizeof(float)+numeric_capacity;k->shared=shared;
    k->family=c->family_index;k->ast=c->index;k->derivation=c->derivation_index;k->variant=c->variant_index;
    k->permutation=permutation;k->bank=bank;k->mse=mse;k->nslots=c->slot_count;
    k->slots=shared->slots;k->structure=shared->structure;k->structure_size=shared->structure_size;
    k->values=(float *)(k+1);if(c->slot_count)memcpy(k->values,values,c->slot_count*sizeof(float));
    k->numeric=(unsigned char *)(k->values+c->slot_count);
    memset(remap,0xff,sizeof(remap));next=0;
    for(state=0;state<j->fixed->state_count;state++) {
        unsigned char *np=k->numeric+k->numeric_size;
        size_t n=resolve(programs[state],values,permutation,np+5,remap,&next,NULL);
        np[0]=(unsigned char)state;append32(np+1,(uint32_t)n);k->numeric_size+=5+n;
    }
    k->tags=shared->provenance->tags;*out=k;return 0;
}

void odz_alias(void *user,uint64_t index,uint64_t derivation,uint64_t variant,const char *const *tags,size_t count) {
    void **pair=user;OdzJob *j=pair[0];size_t family=(size_t)(uintptr_t)pair[1],i;
    FamilyRun *f=&j->families[family];Kept *k;Provenance *p;uint64_t mask;
    (void)derivation;(void)variant;
    if(!count)return;
    pthread_mutex_lock(&j->mutex);
    p=provenance(j,(uint32_t)family,index);
    if(!p){pthread_mutex_unlock(&j->mutex);return;}
    p->pinned=1;
    if(merge_tags(j,p,tags,count)){pthread_mutex_unlock(&j->mutex);return;}
    mask=p->tags;
    for(k=f->archive?f->archive[index]:NULL;k;k=k->archive_next) {
        k->tags=mask;
        for(i=0;i<j->tag_count;i++)if(j->tags[i].ranking.unit!=3&&(mask&(UINT64_C(1)<<i)))odz_rank_offer(j,&j->tags[i].ranking,k);
    }
    for(k=f->row_archive?f->row_archive[index]:NULL;k;k=k->row_archive_next) {
        k->tags=mask;
        for(i=0;i<j->tag_count;i++)if(j->tags[i].ranking.unit==3&&(mask&(UINT64_C(1)<<i)))odz_rank_offer(j,&j->tags[i].ranking,k);
    }
    pthread_mutex_unlock(&j->mutex);
}
void odz_aliases_close(OdzJob *j,FamilyRun *f) {
    size_t b;uint32_t family=(uint32_t)(f-j->families);
    for(b=0;b<ODZ_RETAIN_BUCKETS;b++) {
        Provenance **link=&j->provenance[b];
        while(*link) {
            Provenance *p=*link;
            if(p->family!=family){link=&p->next;continue;}
            *link=p->next;p->next=NULL;p->pinned=0;
            if(!p->refs){odz_host_free(j,p->names);odz_host_free(j,p);}
        }
    }
}

void odz_retention_close(OdzJob *j) {
    odz_host_free(j,j->report_seen);j->report_seen=NULL;j->report_seen_capacity=0;
    size_t i,x;
    Ranking *r=&j->global;
    for(x=0;x<r->count;x++)odz_keep_unref(j,r->rows[x]);
    odz_host_free(j,r->rows);
    for(i=0;i<j->tag_count;i++) {
        r=&j->tags[i].ranking;
        for(x=0;x<r->count;x++)odz_keep_unref(j,r->rows[x]);
        odz_host_free(j,r->rows);
        odz_host_free(j,(void *)j->tags[i].name);
    }
    for(i=0;i<j->family_count;i++) {
        FamilyRun *f=&j->families[i];
        r=&f->ranking;
        for(x=0;x<r->count;x++)odz_keep_unref(j,r->rows[x]);
        odz_host_free(j,r->rows);
        if(f->archive) {
            for(x=0;x<j->grammar->families[i].max_variants;x++) {
                Kept *k=f->archive[x];
                while(k) {
                    Kept *next=k->archive_next;
                    odz_keep_unref(j,k);
                    k=next;
                }
            }
        }
        if(f->row_archive)for(x=0;x<j->grammar->families[i].max_variants;x++) {
            Kept *k=f->row_archive[x];
            while(k) { Kept *next=k->row_archive_next;odz_keep_unref(j,k);k=next; }
        }
        odz_host_free(j,f->row_archive);
        odz_host_free(j,f->archive);
        odz_aliases_close(j,f);
    }
}
