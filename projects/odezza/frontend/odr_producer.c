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
#include "odr_grammar_internal.h"
typedef struct Env {
    const Rule *rule;
    const Expanded *args[ODR_ARGS];
    const Alt *alt;
    uint32_t instance;
}

Env;

typedef struct Expand {
    OdrProducer *p;
    size_t choice;
    uint32_t instance;
    OdrResult result;
    OdrError *error;
}

Expand;

typedef struct Bytes {
    unsigned char *p;
    size_t n,capacity;
    OdrResult result;
}

Bytes;

static uint64_t mix(uint64_t x) {
    x+=UINT64_C(0x9e3779b97f4a7c15);
    x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);
    return x^(x>>31);
}

static uint64_t combinations(size_t n,unsigned k) {
    uint64_t v=1;
    unsigned i;
    if(k>n)return 0;
    for(i=1;i<=k;i++)v=v*(n-k+i)/i;
    return v;
}

static int choose(Expand *e,size_t count) {
    Choice *c;
    uint64_t x;
    if(!count||count>UINT32_MAX||e->choice>=e->p->choice_capacity) {
        e->result=ODR_CAPACITY;
        return 0;
    }
    c=&e->p->choices[e->choice];
    c->count=(uint32_t)count;
    if(e->p->grammar->sample) {
        x=0;
        c->value=(uint32_t)odr_random_below(e->p->grammar->seed^mix(e->p->attempts)^mix(e->choice),&x,count);
    }
    if(c->value>=count)c->value=0;
    ++e->choice;
    return (int)c->value;
}

static Expanded *node(Expand *e) {
    Expanded *n;
    if(e->p->node_count==e->p->node_capacity) {
        e->result=ODR_CAPACITY;
        return NULL;
    }
    n=&e->p->nodes[e->p->node_count++];
    memset(n,0,sizeof(*n));
    return n;
}

static void add_tags(Expand *e,const Alt *a) {
    size_t i,j;
    for(i=0;i<a->tag_count;i++) {
        int seen=0;
        for(j=0;j<e->p->tag_count;j++)if(!strcmp(e->p->tags[j],a->tags[i]))seen=1;
        if(!seen) {
            if(e->p->tag_count==e->p->tag_capacity) {
                e->result=ODR_CAPACITY;
                return;
            }
            e->p->tags[e->p->tag_count++]=a->tags[i];
        }
    }
}

static const Expanded *expand(Expand *e,const Node *n,const Env *env,unsigned depth,unsigned expansion_depth) {
    Expanded *out;
    const Rule *r=NULL;
    size_t i;
    OdrProducer *p=e->p;
    const OdrGrammar *g=p->grammar;
    const Family *f=p->family;
    if(e->result||!n)return NULL;
    if(depth>ODR_DEPTH) {
        e->result=ODR_CAPACITY;
        return NULL;
    }
    if(p->steps>=f->max_steps) {
        p->stop_reason=ODR_STOP_EXPANSION_STEPS;
        e->result=ODR_DONE;
        return NULL;
    }
    ++p->steps;
    if(n->binding_kind==2)return env->args[n->binding_index];
    if(n->binding_kind==1) {
        out=node(e);
        if(out) {
            out->op=0x81;
            out->state=n->binding_index;
        }
        return out;
    }
    if(n->binding_kind==3)r=n->binding;
    if(n->binding_kind==4) {
        size_t ri;
        r=n->binding;
        if(!r) {
            e->result=ODR_AST;
            return NULL;
        }
        ri=(size_t)(r-f->rules);
        if(p->shared[ri]==(const Expanded *)(uintptr_t)1) {
            e->result=ODR_AST;
            return NULL;
        }
        if(p->shared[ri])return p->shared[ri];
        if(expansion_depth>=g->max_expansion_depth) {
            e->result=ODR_DONE;
            return NULL;
        }
        p->shared[ri]=(const Expanded *)(uintptr_t)1;
        {
            int index=choose(e,r->count);
            const Alt *a=&r->alternatives[index];
            Env local;
            memset(&local,0,sizeof(local));
            local.alt=a;
            local.instance=++e->instance;
            add_tags(e,a);
            p->shared[ri]=expand(e,a->node,&local,depth+1,expansion_depth+1);
            return p->shared[ri];
        }
    }
    if(r) {
        Env local;
        int index;
        const Alt *a;
        if(expansion_depth>=g->max_expansion_depth) {
            e->result=ODR_DONE;
            return NULL;
        }
        memset(&local,0,sizeof(local));
        local.rule=r;
        for(i=0;i<n->argc;i++)local.args[i]=expand(e,n->args[i],env,depth+1,expansion_depth);
        if(e->result)return NULL;
        index=choose(e,r->count);
        a=&r->alternatives[index];
        local.alt=a;
        local.instance=++e->instance;
        add_tags(e,a);
        return expand(e,a->node,&local,depth+1,expansion_depth+1);
    }
    out=node(e);
    if(!out)return NULL;
    out->op=n->op;
    out->argc=n->argc;
    out->value=n->value;
    if(n->op==N_REF) {
        const Slot *slot=n->binding;
        if(n->binding_kind==6)out->instance=env->instance;
        if(!slot) {
            e->result=ODR_AST;
            return NULL;
        }
        out->slot=slot;
        out->op=slot->kind==4?0x84:0x82;
        out->argc=0;
        return out;
    }
    if(n->op==N_CALL)out->op=(uint32_t)odr_unary(n->name);
    if(n->op==N_NEG)out->op=0x94;
    for(i=0;i<n->argc;i++)out->args[i]=expand(e,n->args[i],env,depth+1,expansion_depth);
    return out;
}

static void advance(OdrProducer *p) {
    size_t i=p->choice_count;
    if(p->grammar->sample)return;
    while(i) {
        --i;
        if(++p->choices[i].value<p->choices[i].count) {
            size_t j;
            for(j=i+1;j<p->choice_count;j++)p->choices[j].value=0;
            return;
        }
        p->choices[i].value=0;
    }
    p->exhausted=1;
}

static OdrResult active_add(OdrProducer *p,const Slot *s,uint32_t instance) {
    size_t i;
    Active *a;
    for(i=0;i<p->active_count;i++)if(p->active[i].slot==s&&p->active[i].instance==instance)return ODR_OK;
    if(p->active_count==p->active_capacity)return ODR_CAPACITY;
    a=&p->active[p->active_count++];
    memset(a,0,sizeof(*a));
    a->slot=s;
    a->instance=instance;
    a->groups=s->kind==4?(s->coverage?s->group_count:combinations(s->state_count,s->arity)):1;
    return a->groups?ODR_OK:ODR_AST;
}

static OdrResult inspect_tree(OdrProducer *p,const Expanded *n,unsigned depth,size_t *count) {
    size_t i;
    OdrResult r;
    if(!n)return ODR_AST;
    if(depth>p->grammar->max_depth)return ODR_DONE;
    if(++*count>p->grammar->max_nodes)return ODR_DONE;
    if(n->slot) {
        r=active_add(p,n->slot,n->instance);
        if(r)return r;
    }
    for(i=0;i<n->argc;i++) {
        r=inspect_tree(p,n->args[i],depth+1,count);
        if(r)return r;
    }
    return ODR_OK;
}

static const Slot *dependency(const OdrProducer *p,const Active *active_slot,const char *name,uint32_t *instance) {
    const Slot *s=active_slot->slot;
    size_t i;
    *instance=0;
    for(i=0;i<s->scope_count;i++)if(!strcmp(s->scope[i].name,name)) {
        *instance=active_slot->instance;
        return &s->scope[i];
    }
    return gslot(p->family,name);
}

static int same_axis(const Active *a,const Active *b) {
    if(a->slot->axis.kind<2||b->slot->axis.kind<2)return 0;
    if(!a->slot->axis.axis||!b->slot->axis.axis||strcmp(a->slot->axis.axis,b->slot->axis.axis))return 0;
    if((a->slot->axis.axis==a->slot->name||b->slot->axis.axis==b->slot->name)&&a->instance!=b->instance)return 0;
    return 1;
}

static OdrResult prepare_tree(OdrProducer *p,OdrError *error) {
    Expand e;
    size_t i,j,count=0;
    OdrResult r;
    memset(&e,0,sizeof(e));
    e.p=p;
    e.error=error;
    p->node_count=p->active_count=p->tag_count=0;
    memset(p->shared,0,p->family->rule_count*sizeof(*p->shared));
    for(i=0;i<p->family->tag_count;i++)p->tags[p->tag_count++]=p->family->tags[i];
    for(i=0;i<p->grammar->state_count&&!e.result;i++)p->roots[i]=p->family->rhs[i]?expand(&e,p->family->rhs[i],NULL,0,0):NULL;
    p->choice_count=e.choice;
    p->derivation=p->attempts++;
    if(e.result) {
        advance(p);
        return e.result;
    }
    for(i=0;i<p->grammar->state_count;i++)if(p->roots[i]) {
        r=inspect_tree(p,p->roots[i],1,&count);
        if(r) {
            advance(p);
            return r;
        }
    }
    /* A transform can activate a constant absent from the RHS. */ for(i=0;i<p->active_count;i++) {
        const Slot *s=p->active[i].slot;
        const char *refs[2]= {
            s->scale_ref,s->shift_ref
        };
        unsigned k;
        for(k=0;k<2;k++)if(refs[k]) {
            uint32_t instance;
            const Slot *source=dependency(p,&p->active[i],refs[k],&instance);
            if(!source)return odr_error(error,ODR_SCHEMA,0,"unknown transform constant");
            r=active_add(p,source,instance);
            if(r)return r;
        }
    }
    for(i=1;i<p->active_count;i++) {
        Active x=p->active[i];
        j=i;
        while(j&&(strcmp(p->active[j-1].slot->name,x.slot->name)>0||(!strcmp(p->active[j-1].slot->name,x.slot->name)&&p->active[j-1].instance>x.instance))) {
            p->active[j]=p->active[j-1];
            --j;
        }
        p->active[j]=x;
    }
    {
        unsigned slot=0,bit=0;
        for(i=0;i<p->active_count;i++)if(p->active[i].slot->kind!=4)p->active[i].ordinal=slot++;
        for(i=p->active_count;i>0;i--)if(p->active[i-1].slot->kind==4) {
            p->active[i-1].bit=bit;
            bit+=p->active[i-1].slot->arity==2?1:2;
        }
        if(slot+2*p->grammar->state_count+1>=255||bit>32)return odr_error(error,ODR_CAPACITY,0,"native coefficient/toggle input layout exceeded");
    }
    p->numeric_count=1;
    for(i=p->active_count;i>0;i--) {
        Active *a=&p->active[i-1];
        int repeated=0;
        if(a->slot->kind==4||!a->slot->axis.kind)continue;
        for(j=0;j<i-1;j++)if(same_axis(a,&p->active[j])) {
            if(a->slot->axis.count!=p->active[j].slot->axis.count)return odr_error(error,ODR_SCHEMA,0,"shared RNG axes require equal counts");
            repeated=1;
        }
        if(repeated)continue;
        a->stride=p->numeric_count;
        for(j=i;j<p->active_count;j++)if(same_axis(a,&p->active[j]))p->active[j].stride=a->stride;
        if(p->numeric_count>UINT64_MAX/a->slot->axis.count)return ODR_OVERFLOW;
        p->numeric_count*=a->slot->axis.count;
    }
    p->variants=1;
    for(i=0;i<p->active_count;i++)if(p->active[i].slot->kind==4) {
        if(p->variants>UINT64_MAX/p->active[i].groups)return ODR_OVERFLOW;
        p->variants*=p->active[i].groups;
    }
    if(p->family->joint_count&&p->family->joint_count<p->variants) {
        odr_sample_ranks(p->variants,(size_t)p->family->joint_count,p->family->joint_seed,p->sample_ranks,p->sample_map);
        p->variants=p->family->joint_count;
    }
    p->variant=0;
    p->have_tree=1;
    p->tree_accepted=0;
    return ODR_OK;
}

static void put(Bytes *b,unsigned x) {
    if(b->n==b->capacity) {
        b->result=ODR_CAPACITY;
        return;
    }
    b->p[b->n++]=(unsigned char)x;
}

static void write_bytes(Bytes *b,const void *p,size_t size) {
    if(size>b->capacity-b->n) {
        b->result=ODR_CAPACITY;
        return;
    }
    memcpy(b->p+b->n,p,size);
    b->n+=size;
}

static void literal(Bytes *b,float x) {
    uint32_t bits;
    unsigned i;
    memcpy(&bits,&x,4);
    put(b,0x83);
    for(i=0;i<4;i++)put(b,bits>>(8*i));
}

static const Active *active(const OdrProducer *p,const Expanded *n) {
    size_t i;
    for(i=0;i<p->active_count;i++)if(p->active[i].slot==n->slot&&p->active[i].instance==n->instance)return &p->active[i];
    return NULL;
}

static void group(const Slot *s,uint64_t rank,uint32_t ids[4]) {
    unsigned i;
    size_t start=0;
    if(s->coverage==2)rank=s->sample_ranks[rank];
    if(s->coverage==1) {
        for(i=0;i<s->arity;i++)ids[i]=s->groups[rank*s->arity+i];
        return;
    }
    for(i=0;i<s->arity;i++) {
        size_t x;
        for(x=start;x<s->state_count;x++) {
            uint64_t count=combinations(s->state_count-x-1,s->arity-i-1);
            if(rank<count) {
                ids[i]=s->states[x];
                start=x+1;
                break;
            }
            rank-=count;
        }
    }
}

static void emit(Bytes *b,const OdrProducer *p,const Expanded *n,unsigned depth) {
    size_t i;
    if(b->result)return;
    if(!n||depth>ODR_DEPTH) {
        b->result=ODR_AST;
        return;
    }
    if(n->op==N_LITERAL)literal(b,n->value);
    else if(n->op==0x81) {
        put(b,0x81);
        put(b,n->state);
    }
    else if(n->slot) {
        const Active *a=active(p,n);
        if(!a) {
            b->result=ODR_AST;
            return;
        }
        if(n->slot->kind==4) {
            uint32_t ids[4];
            group(n->slot,a->group_index,ids);
            for(i=0;i<n->slot->arity;i++) {
                put(b,0x81);
                put(b,ids[i]);
            }
            put(b,n->slot->arity==2?0x84:0x85);
            put(b,a->bit);
            if(n->slot->arity==4)put(b,a->bit+1);
        }
        else {
            put(b,0x82);
            put(b,a->ordinal);
        }
    }
    else if(n->op==N_POW) {
        int exponent=(int)n->value;
        if(!exponent)literal(b,1);
        else {
            if(exponent<0)literal(b,1);
            for(i=0;i<(size_t)(exponent<0?-exponent:exponent);i++) {
                emit(b,p,n->args[0],depth+1);
                if(i)put(b,0x92);
            }
            if(exponent<0)put(b,0x93);
        }
    }
    else {
        for(i=0;i<n->argc;i++)emit(b,p,n->args[i],depth+1);
        put(b,n->op);
    }
}

static uint64_t hash_bytes(const void *data,size_t count) {
    const unsigned char *p=data;
    uint64_t h=UINT64_C(14695981039346656037);
    size_t i;
    for(i=0;i<count;i++)h=(h^p[i])*UINT64_C(1099511628211);
    return h?h:1;
}

static int dedup(OdrProducer *p,const unsigned char *key,size_t count,uint64_t *prior) {
    uint64_t h=hash_bytes(key,count);
    size_t cell=(size_t)(h%p->dedup_count);
    while(p->dedup[cell].hash) {
        Dedup *d=&p->dedup[cell];
        if(d->hash==h&&d->size==count&&!memcmp(p->keys+d->offset,key,count)){*prior=d->index;return 1;}
        cell=(cell+1)%p->dedup_count;
    }
    if(count>p->key_capacity-p->key_used)return -1;
    p->dedup[cell].hash=h;
    p->dedup[cell].index=p->next;
    p->dedup[cell].offset=p->key_used;
    p->dedup[cell].size=count;
    memcpy(p->keys+p->key_used,key,count);
    p->key_used+=count;
    return 0;
}

OdrResult odr_producer_create(const OdrGrammar *g,const OdrProducerOptions *options,void *arena,size_t capacity,size_t *required,OdrProducer **out,OdrError *error) {
    Arena a= {
        (unsigned char *)arena,capacity,0,ODR_OK
    };
    OdrProducer p,*stored;
    size_t base,table;
    if(out)*out=NULL;
    if(required)*required=0;
    if(!g||g->magic!=GRAMMAR_MAGIC||!required)return odr_error(error,ODR_SCHEMA,0,"invalid producer arguments");
    memset(&p,0,sizeof(p));
    p.grammar=g;
    if(options)p.options=*options;
    if(p.options.family_index>=g->family_count)return odr_error(error,ODR_SCHEMA,0,"invalid family index");
    p.family=&g->families[p.options.family_index];
    if(!p.options.max_attempts_per_batch)p.options.max_attempts_per_batch=4096;
    if(!p.options.max_asts)p.options.max_asts=p.family->max_variants;
    if(!p.options.max_attempts)p.options.max_attempts=p.family->max_attempts;
    if(p.options.max_asts>p.family->max_variants||p.options.max_attempts>p.family->max_attempts)return odr_error(error,ODR_SCHEMA,0,"producer exceeds family allocation");
    if(p.options.max_asts>(SIZE_MAX-1)/2)return ODR_OVERFLOW;
    table=(size_t)p.options.max_asts*2+1;
    base=(size_t)g->max_nodes*4+128;
    if(!p.options.dedup_capacity) {
        if(p.options.max_asts>SIZE_MAX/(base*4))return ODR_OVERFLOW;
        p.options.dedup_capacity=(size_t)p.options.max_asts*base*4;
    }
    if(arena&&(uintptr_t)arena%odr_arena_alignment())return odr_error(error,ODR_SCHEMA,0,"unaligned producer arena");
    stored=odr_take(&a,1,sizeof(p));
    if(p.family->joint_count>(SIZE_MAX/sizeof(uint64_t)-2)/4)return ODR_OVERFLOW;
    p.sample_capacity=(size_t)p.family->joint_count;
    p.sample_ranks=odr_take(&a,p.sample_capacity,sizeof(uint64_t));
    p.sample_map=odr_take(&a,4*p.sample_capacity+2,sizeof(uint64_t));
    p.choice_capacity=base;
    p.choices=odr_take(&a,base,sizeof(*p.choices));
    p.node_capacity=base;
    p.nodes=odr_take(&a,base,sizeof(*p.nodes));
    p.roots=odr_take(&a,g->state_count,sizeof(*p.roots));
    p.shared=odr_take(&a,p.family->rule_count,sizeof(*p.shared));
    p.active_capacity=base;
    p.active=odr_take(&a,base,sizeof(*p.active));
    p.tag_capacity=base+p.family->tag_count;
    p.tags=odr_take(&a,p.tag_capacity,sizeof(*p.tags));
    p.dedup_count=table;
    p.dedup=odr_take(&a,table,sizeof(*p.dedup));
    p.key_capacity=p.options.dedup_capacity;
    p.keys=odr_take(&a,p.key_capacity,1);
    p.scratch_capacity=(size_t)g->max_nodes*256+4096;
    p.scratch=odr_take(&a,p.scratch_capacity,1);
    *required=a.used;
    if(a.result)return a.result;
    if(arena&&a.used>capacity)return odr_error(error,ODR_BUFFER,0,"producer arena too small");
    if(stored)*stored=p;
    if(out)*out=stored;
    return ODR_OK;
}

OdrResult odr_batch_requirements(const OdrProducer *p,size_t count,size_t *required) {
    Arena a= {
        NULL,0,0,ODR_OK
    };
    Arena item= {
        NULL,0,0,ODR_OK
    };
    if(!p||!required)return ODR_SCHEMA;
    (void)odr_take(&a,count,sizeof(OdrCandidate));
    (void)odr_take(&item,p->grammar->state_count,sizeof(OdrRhs));
    (void)odr_take(&item,253,sizeof(OdrAxis));
    (void)odr_take(&item,p->tag_capacity,sizeof(char *));
    (void)odr_take(&item,p->scratch_capacity,1);
    (void)odr_take(&item,0,1);
    /* round each upper-bound record to arena alignment */ if(item.result)return item.result;
    (void)odr_take(&a,count,item.used);
    *required=a.used;
    return a.result;
}

OdrResult odr_producer_next(OdrProducer *p,uint64_t start,size_t count,void *arena,size_t capacity,size_t *required,OdrBatch *out,OdrError *error) {
    Arena a= {
        (unsigned char *)arena,capacity,0,ODR_OK
    };
    OdrCandidate *candidates;
    size_t produced=0;
    uint64_t first_attempt,first_visit;
    OdrResult r=ODR_OK;
    if(out)memset(out,0,sizeof(*out));
    if(!p||!required||!out||!count)return odr_error(error,ODR_SCHEMA,0,"invalid batch arguments");
    if(start!=p->next)return odr_error(error,ODR_INDEX,0,"start must equal the producer's next accepted index; no implicit prefix replay");
    r=odr_batch_requirements(p,count,required);
    if(r)return r;
    if(!arena)return ODR_OK;
    if((uintptr_t)arena%odr_arena_alignment())return odr_error(error,ODR_SCHEMA,0,"unaligned output arena");
    if(capacity<*required)return odr_error(error,ODR_BUFFER,0,"batch arena too small; cursor unchanged");
    candidates=odr_take(&a,count,sizeof(*candidates));
    first_attempt=p->attempts;
    first_visit=p->visits;
    while(produced<count&&p->next<p->options.max_asts&&!p->stop_reason) {
        size_t i,rhs_count=0,slot_count=0;
        uint64_t rank,prior=0;
        Bytes bytes= {
            p->scratch,0,p->scratch_capacity,ODR_OK
        };
        size_t offsets[ODR_STATES],sizes[ODR_STATES],program_end;
        int duplicate;
        OdrCandidate c;
        OdrRhs *rhs;
        OdrAxis *slots;
        const char **tag_list;
        unsigned char *programs;
        memset(&c,0,sizeof(c));
        if(p->visits-first_visit>=p->options.max_attempts_per_batch)break;
        if(!p->have_tree) {
            if(p->attempts-first_attempt>=p->options.max_attempts_per_batch)break;
            if(p->exhausted||p->attempts>=p->options.max_attempts||p->skeleton_count>=p->family->max_asts)break;
            p->tree_configs=0;p->tree_budget_reported=0;
            r=prepare_tree(p,error);
            if(r==ODR_DONE) {
                ++p->pruned;
                r=ODR_OK;
                continue;
            }
            if(r) {
                odr_error(error,r,0,"grammar expansion failed");
                break;
            }
        }
        rank=p->variant;
        if(p->family->joint_count) {
            uint64_t total=1;
            for(i=0;i<p->active_count;i++)if(p->active[i].slot->kind==4)total*=p->active[i].groups;
            if(p->family->joint_count<total)rank=p->sample_ranks[p->variant];
        }
        for(i=0;i<p->active_count;i++)if(p->active[i].slot->kind!=4)++slot_count;
        c.bank_count=p->numeric_count;
        {
            unsigned bits=0;
            uint64_t rows;
            for(i=0;i<p->active_count;i++)if(p->active[i].slot->kind==4)bits+=p->active[i].slot->arity==2?1:2;
            rows=(p->family->max_configs-p->configs)>>bits;
            if(!rows) {
                p->stop_reason=ODR_STOP_CONFIGURATIONS;
                break;
            }
            if(c.bank_count>rows)c.bank_count=rows;
            if(p->family->max_configs_per_skeleton) {
                rows=(p->family->max_configs_per_skeleton-p->tree_configs)>>bits;
                if(!rows) {
                    if(!p->tree_budget_reported)++p->configuration_limited_derivations;
                    p->have_tree=0;advance(p);continue;
                }
                if(c.bank_count>rows) {
                    c.bank_count=rows;
                    if(!p->tree_budget_reported)++p->configuration_limited_derivations;
                    p->tree_budget_reported=1;
                }
            }
        }
        ++p->visits;
        c.variant_index=rank;
        for(i=p->active_count;i>0;i--)if(p->active[i-1].slot->kind==4) {
            p->active[i-1].group_index=rank%p->active[i-1].groups;
            rank/=p->active[i-1].groups;
        }
        for(i=0;i<p->grammar->state_count;i++)if(p->roots[i]) {
            offsets[rhs_count]=bytes.n;
            emit(&bytes,p,p->roots[i],0);
            put(&bytes,0x80);
            sizes[rhs_count]=bytes.n-offsets[rhs_count];
            ++rhs_count;
        }
        if(bytes.result) {
            r=odr_error(error,bytes.result,0,"expanded postorder program exceeds producer capacity");
            break;
        }
        program_end=bytes.n;
        /* Exact collision resolution, with immutable slot identity and sharing. */ for(i=0;i<p->active_count;i++)if(p->active[i].slot->kind!=4) {
            uintptr_t identity=(uintptr_t)p->active[i].slot;
            write_bytes(&bytes,&identity,sizeof(identity));
            write_bytes(&bytes,&p->active[i].instance,sizeof(uint32_t));
        }
        if(bytes.result) {
            r=ODR_CAPACITY;
            break;
        }
        duplicate=dedup(p,bytes.p,bytes.n,&prior);
        if(duplicate<0) {
            r=odr_error(error,ODR_CAPACITY,0,"exact dedup arena full; completed prefix remains valid");
            break;
        }
        c.derivation_index=p->derivation;
        if(++p->variant==p->variants) {
            p->have_tree=0;
            advance(p);
        }
        if(duplicate) {
            ++p->duplicates;
            if(p->options.on_duplicate)p->options.on_duplicate(p->options.provenance_user,prior,c.derivation_index,c.variant_index,p->tags,p->tag_count);
            continue;
        }
        if(!p->tree_accepted) {
            ++p->skeleton_count;
            p->tree_accepted=1;
        }
        c.index=p->next++;
        c.family=p->family->name;
        c.family_index=p->options.family_index;
        c.rhs_count=(uint32_t)rhs_count;
        c.numeric_count=p->numeric_count;
        rhs=odr_take(&a,rhs_count,sizeof(*rhs));
        slots=odr_take(&a,slot_count,sizeof(*slots));
        tag_list=odr_take(&a,p->tag_count,sizeof(*tag_list));
        programs=odr_take(&a,program_end,1);
        memcpy(programs,bytes.p,program_end);
        rhs_count=0;
        for(i=0;i<p->grammar->state_count;i++)if(p->roots[i]) {
            rhs[rhs_count].state_index=(uint32_t)i;
            rhs[rhs_count].program.bytes=programs+offsets[rhs_count];
            rhs[rhs_count].program.byte_count=sizes[rhs_count];
            ++rhs_count;
        }
        for(i=0;i<p->active_count;i++) {
            Active *active_slot=&p->active[i];
            const Slot *s=active_slot->slot;
            if(s->kind==4) {
                c.toggle_bits+=s->arity==2?1:2;
                continue;
            }
            slots[c.slot_count]=s->axis;
            slots[c.slot_count].numeric_stride=active_slot->stride;
            slots[c.slot_count].instance=active_slot->instance;
            slots[c.slot_count].scale_slot=slots[c.slot_count].shift_slot=-1;
            {
                const char *refs[2]= {
                    s->scale_ref,s->shift_ref
                };
                unsigned k;
                for(k=0;k<2;k++)if(refs[k]) {
                    size_t j;
                    uint32_t instance;
                    const Slot *source=dependency(p,active_slot,refs[k],&instance);
                    for(j=0;j<p->active_count;j++)if(p->active[j].slot==source&&p->active[j].instance==instance) {
                        if(k)slots[c.slot_count].shift_slot=(int32_t)p->active[j].ordinal;
                        else slots[c.slot_count].scale_slot=(int32_t)p->active[j].ordinal;
                        break;
                    }
                }
            }
            ++c.slot_count;
        }
        for(i=0;i<p->tag_count;i++)tag_list[i]=p->tags[i];
        c.tags=tag_list;
        c.tag_count=p->tag_count;
        c.rhs=rhs;
        c.slots=slots;
        p->configs+=c.bank_count*(UINT64_C(1)<<c.toggle_bits);
        p->tree_configs+=c.bank_count*(UINT64_C(1)<<c.toggle_bits);
        candidates[produced++]=c;
    }
    if(!r) {
        if(p->configs==p->family->max_configs)p->stop_reason=ODR_STOP_CONFIGURATIONS;
        if(!p->stop_reason&&p->next>=p->options.max_asts)p->stop_reason=ODR_STOP_VARIANTS;
        if(!p->stop_reason&&!p->have_tree) {
            if(p->exhausted)p->stop_reason=ODR_STOP_EXHAUSTED;
            else if(p->attempts>=p->options.max_attempts)p->stop_reason=ODR_STOP_DERIVATIONS;
            else if(p->skeleton_count>=p->family->max_asts)p->stop_reason=ODR_STOP_SKELETONS;
        }
    }
    out->start=start;
    out->next_index=p->next;
    out->count=produced;
    out->candidates=candidates;
    out->attempts=p->attempts;
    out->duplicates=p->duplicates;
    out->pruned=p->pruned;
    out->yielded=p->attempts-first_attempt>=p->options.max_attempts_per_batch||p->visits-first_visit>=p->options.max_attempts_per_batch;
    out->variant_visits=p->visits;
    out->exhausted=p->stop_reason!=ODR_STOP_NONE;
    out->stop_reason=p->stop_reason;
    out->expansion_steps=p->steps;
    out->configurations_reserved=p->configs;
    out->configuration_limited_derivations=p->configuration_limited_derivations;
    out->unused_configurations=p->family->max_configs-p->configs;
    return r;
}
