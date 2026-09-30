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
#include <ctype.h>
static OdrResult bad(OdrError *e,const char *s) {
    return odr_error(e,ODR_SCHEMA,0,s);
}

static char *text_copy(const char *s,Arena *a) {
    char *p=odr_take(a,strlen(s)+1,1);
    if(p)strcpy(p,s);
    return p;
}

static int identifier(const char *s) {
    size_t i;
    if(!s[0]||!(isalpha((unsigned char)s[0])||s[0]=='_'))return 0;
    for(i=1;s[i];i++)if(!(isalnum((unsigned char)s[i])||s[i]=='_'))return 0;
    return 1;
}

static uint64_t limit(OdrJson j,const char *key,uint64_t fallback,OdrError *e) {
    OdrJson v;
    uint64_t n;
    if(!jget(j,key,&v))return fallback;
    if(!ju64(v,&n)||!n) {
        bad(e,"limits must be positive uint64 integers");
        return 0;
    }
    return n;
}

static OdrResult tags(OdrJson raw,Arena *a,const char *const **out,size_t *count,OdrError *e) {
    Ji it;
    OdrJson k,v;
    const char **p;
    size_t i=0;
    if(!raw.data) {
        *out=NULL;
        *count=0;
        return ODR_OK;
    }
    if(jtype(raw)=='"') {
        p=odr_take(a,1,sizeof(*p));
        {
            char *s=jcopy(raw,a);
            if(p)p[0]=s;
        }
        *out=p;
        *count=1;
        return a->result;
    }
    if(jtype(raw)!='[')return bad(e,"tags must be a string or string array");
    *count=jcount(raw);
    p=odr_take(a,*count,sizeof(*p));
    jiter(raw,&it);
    while(jnext(&it,&k,&v)) {
        size_t n;
        char *s;
        if(!jstring(v,NULL,0,&n)||!n)return bad(e,"empty or invalid tag");
        s=jcopy(v,a);
        if(p)p[i]=s;
        ++i;
    }
    *out=p;
    return a->result;
}

static int state_index(OdrJson states,OdrJson name,uint32_t *index) {
    Ji it;
    OdrJson k,v;
    char s[ODR_NAME];
    uint32_t n=0;
    if(!jstring(name,s,sizeof(s),NULL))return 0;
    jiter(states,&it);
    while(jnext(&it,&k,&v)) {
        if(jeq(v,s)) {
            *index=n;
            return 1;
        }
        ++n;
    }
    return 0;
}

static OdrResult operand(OdrJson j,const char *field,float fallback,Arena *a,float *value,const char **ref,OdrError *e) {
    OdrJson v;
    char name[ODR_NAME];
    *value=fallback;
    *ref=NULL;
    if(!jget(j,field,&v))return ODR_OK;
    if(jfloat(v,value))return ODR_OK;
    if(jstring(v,name,sizeof(name),NULL)&&!strncmp(name,"const.",6)) {
        *ref=jcopy(v,a);
        return a->result;
    }
    return bad(e,"transform operand must be finite FP32 or const.name");
}

static OdrResult parse_slot(OdrJson key,OdrJson def,unsigned kind,OdrJson root,OdrJson states,Arena *a,Slot *out,OdrError *e) {
    Slot s;
    char raw[ODR_NAME],name[ODR_NAME+8];
    OdrJson v,bank,section,k,item;
    Ji it;
    size_t n,i;
    uint64_t number;
    memset(&s,0,sizeof(s));
    s.kind=kind;
    s.axis.count=1;
    s.axis.scale_slot=s.axis.shift_slot=-1;
    if(!jstring(key,raw,sizeof(raw),NULL)||!identifier(raw))return bad(e,"invalid slot name");
    snprintf(name,sizeof(name),"%s.%s",kind==4?"leaf":kind==5?"theta":kind==2?"rng":"const",raw);
    s.name=text_copy(name,a);
    s.axis.name=s.name;
    if(kind==0||kind==5) {
        const char *fields=kind==5?"initial":"value,values,bank";
        if(!jkeys(def,fields,e)||jcount(def)!=1)return bad(e,"constant/parameter needs exactly one value source");
        if(jget(def,kind==5?"initial":"value",&v)) {
            float *values=odr_take(a,1,sizeof(float));
            float x;
            if(!jfloat(v,&x))return bad(e,"constant must be finite FP32");
            if(values)*values=x;
            s.axis.values=values;
            s.axis.kind=0;
        }
        else {
            if(jget(def,"bank",&v)) {
                char bankname[ODR_NAME];
                if(!jstring(v,bankname,sizeof(bankname),NULL)||!jget(root,"constant_banks",&section)||!jget(section,bankname,&v))return bad(e,"unknown constant bank");
            }
            else if(!jget(def,"values",&v))return bad(e,"missing constant values");
            if(jtype(v)!='['||!(n=jcount(v)))return bad(e,"constant values must be nonempty");
            {
                float *values=odr_take(a,n,sizeof(float));
                s.axis.values=values;
                s.axis.count=n;
                s.axis.kind=1;
                s.axis.axis=s.name;
                jiter(v,&it);
                i=0;
                while(jnext(&it,&k,&item)) {
                    float x;
                    if(!jfloat(item,&x))return bad(e,"constant bank must be finite FP32");
                    if(values)values[i]=x;
                    ++i;
                }
            }
        }
    }
    else if(kind==4) {
        if(!jkeys(def,"states,arity,coverage,groups,samples,seed",e))return ODR_SCHEMA;
        number=limit(def,"arity",2,e);
        if(number!=2&&number!=4)return bad(e,"toggle arity must be 2 or 4");
        s.arity=(unsigned)number;
        if(!jget(def,"states",&v)||jtype(v)!='['||(n=jcount(v))<s.arity)return bad(e,"too few toggle states");
        {
            uint32_t *ids=odr_take(a,n,sizeof(*ids));
            s.states=ids;
            s.state_count=n;
            jiter(v,&it);
            i=0;
            while(jnext(&it,&k,&item)) {
                uint32_t id;
                if(!state_index(states,item,&id))return bad(e,"unknown toggle state");
                if(ids)ids[i]=id;
                ++i;
            }
            if(ids) {
                size_t x,y;
                for(x=0;x<n;x++)for(y=x+1;y<n;y++) {
                    if(ids[x]==ids[y])return bad(e,"duplicate toggle state");
                    if(ids[x]>ids[y]) {
                        uint32_t t=ids[x];
                        ids[x]=ids[y];
                        ids[y]=t;
                    }
                }
            }
        }
        if(jget(def,"coverage",&v)&&!jeq(v,"all")) {
            if(jeq(v,"explicit")) {
                s.coverage=1;
                if(!jget(def,"groups",&v)||jtype(v)!='['||!(n=jcount(v)))return bad(e,"explicit toggles need groups");
                {
                    uint32_t *ids=odr_take(a,n*s.arity,sizeof(*ids));
                    s.groups=ids;
                    s.group_count=n;
                    jiter(v,&it);
                    i=0;
                    while(jnext(&it,&k,&item)) {
                        Ji gi;
                        OdrJson gk,gv;
                        unsigned x=0;
                        if(jtype(item)!='['||jcount(item)!=s.arity)return bad(e,"invalid toggle group");
                        jiter(item,&gi);
                        while(jnext(&gi,&gk,&gv)) {
                            uint32_t id;
                            if(!state_index(states,gv,&id))return bad(e,"unknown group state");
                            if(ids)ids[i*s.arity+x]=id;
                            ++x;
                        }
                        if(ids) {
                            unsigned x,y;
                            for(x=0;x<s.arity;x++) {
                                int member=0;
                                size_t z;
                                for(z=0;z<s.state_count;z++)if(ids[i*s.arity+x]==s.states[z])member=1;
                                if(!member)return bad(e,"explicit group state is outside leaf states");
                                for(y=x+1;y<s.arity;y++) {
                                    uint32_t *left=&ids[i*s.arity+x],*right=&ids[i*s.arity+y];
                                    if(*left==*right)return bad(e,"duplicate state in toggle group");
                                    if(*left>*right) {
                                        uint32_t swap=*left;
                                        *left=*right;
                                        *right=swap;
                                    }
                                }
                            }
                            {
                                size_t z;
                                for(z=0;z<i;z++)if(!memcmp(ids+z*s.arity,ids+i*s.arity,s.arity*sizeof(*ids)))return bad(e,"duplicate explicit toggle group");
                            }
                        }
                        ++i;
                    }
                }
            }
            else if (jeq(v,"sample")) {
                uint64_t total=1, samples, seed=0;
                unsigned x;
                uint64_t *ranks,*map;
                for(x=1;x<=s.arity;x++) total=total*(s.state_count-s.arity+x)/x;
                samples=limit(def,"samples",0,e);
                if(!samples)return bad(e,"sample coverage requires samples");
                if(samples>total)samples=total;
                if(samples>(SIZE_MAX/sizeof(uint64_t)-2)/4)return ODR_OVERFLOW;
                if(jget(def,"seed",&v)&&!ju64(v,&seed))return bad(e,"invalid toggle seed");
                s.coverage=2;
                s.group_count=(size_t)samples;
                ranks=odr_take(a,(size_t)samples,sizeof(*ranks));
                map=odr_take(a,4*(size_t)samples+2,sizeof(*map));
                s.sample_ranks=ranks;
                if(ranks&&map)odr_sample_ranks(total,(size_t)samples,seed,ranks,map);
            }
            else return bad(e,"unknown toggle coverage");
        }
        if(s.coverage!=2&&jget(def,"samples",&v))return bad(e,"samples requires sample coverage");
        if(s.coverage!=1&&jget(def,"groups",&v))return bad(e,"groups requires explicit coverage");
    }
    else {
        char bankname[ODR_NAME];
        OdrJson transform= {
            NULL,0
        };
        const char *first=NULL,*second=NULL;
        if(!jkeys(def,"bank,axis,stream,transform",e))return ODR_SCHEMA;
        if(!jget(def,"bank",&v)||!jstring(v,bankname,sizeof(bankname),NULL)||!jget(root,"rng_banks",&section)||!jget(section,bankname,&bank))return bad(e,"unknown RNG bank");
        s.axis.bank=jcopy(v,a);
        if(!jkeys(bank,"base,count,seed,scope",e))return ODR_SCHEMA;
        if(!jget(bank,"base",&v)||(!jeq(v,"uniform01")&&!jeq(v,"normal01")))return bad(e,"RNG base must be uniform01 or normal01");
        s.axis.kind=jeq(v,"normal01")?3:2;
        s.axis.count=limit(bank,"count",0,e);
        if(!s.axis.count)return bad(e,"RNG count must be positive");
        s.axis.seed=0;
        if(jget(bank,"seed",&v)&&!ju64(v,&s.axis.seed))return bad(e,"RNG seed must be uint64");
        if(jget(bank,"scope",&v)) {
            if(!jeq(v,"run")&&!jeq(v,"skeleton"))return bad(e,"RNG scope must be run or skeleton");
            s.axis.skeleton_scope=jeq(v,"skeleton");
        }
        s.axis.axis=s.name;
        if(jget(def,"axis",&v)) {
            if(jtype(v)!='"')return bad(e,"RNG axis must be a string");
            s.axis.axis=jcopy(v,a);
        }
        s.axis.stream=s.name;
        if(jget(def,"stream",&v)) {
            if(jtype(v)!='"')return bad(e,"RNG stream must be a string");
            s.axis.stream=jcopy(v,a);
        }
        if(jget(def,"transform",&transform)) {
            if(!jget(transform,"kind",&v))return bad(e,"transform requires kind");
            if(jeq(v,"identity")) {
                s.axis.transform=0;
                if(!jkeys(transform,"kind",e))return ODR_SCHEMA;
            }
            else if(jeq(v,"affine")) {
                s.axis.transform=1;
                first="scale";
                second="shift";
                if(!jkeys(transform,"kind,scale,shift",e))return ODR_SCHEMA;
            }
            else if(jeq(v,"uniform")||jeq(v,"log_uniform")) {
                s.axis.transform=jeq(v,"uniform")?2:4;
                first="low";
                second="high";
                if(s.axis.kind!=2||!jkeys(transform,"kind,low,high",e))return bad(e,"uniform transform requires uniform bank");
            }
            else if(jeq(v,"normal")) {
                s.axis.transform=3;
                first="std";
                second="mean";
                if(s.axis.kind!=3||!jkeys(transform,"kind,std,mean",e))return bad(e,"normal transform requires normal bank");
            }
            else return bad(e,"unknown RNG transform");
        }
        s.axis.scale=1;
        s.axis.shift=0;
        if(first) {
            OdrResult r=operand(transform,first,s.axis.transform==2||s.axis.transform==4?0:1,a,&s.axis.scale,&s.scale_ref,e);
            if(r)return r;
            r=operand(transform,second,s.axis.transform==2||s.axis.transform==4?1:0,a,&s.axis.shift,&s.shift_ref,e);
            if(r)return r;
        }
    }
    if(out)*out=s;
    return a->result;
}

/* Visit global definitions followed by overrides, with shadowed globals skipped. */ static size_t merged_count(OdrJson global,OdrJson local) {
    Ji it;
    OdrJson k,v,ignored;
    char name[ODR_NAME];
    size_t n=local.data?jcount(local):0;
    if(global.data) {
        jiter(global,&it);
        while(jnext(&it,&k,&v)) {
            if(!jstring(k,name,sizeof(name),NULL))continue;
            if(!jget(local,name,&ignored))++n;
        }
    }
    return n;
}

static OdrResult parse_slots(OdrJson root,OdrJson local,OdrJson states,Arena *a,const Slot **out,size_t *count,OdrError *e,int local_only) {
    static const char *sections[]= {
        "constants","rng","leaves","parameters"
    };
    static const unsigned kinds[]= {
        0,2,4,5
    };
    size_t n=0,index=0;
    unsigned x,pass;
    Slot *slots;
    for(x=0;x<4;x++) {
        OdrJson g= {
            NULL,0
        },l= {
            NULL,0
        };
        if(!local_only)(void)jget(root,sections[x],&g);
        (void)jget(local,sections[x],&l);
        if((g.data&&jtype(g)!='{')||(l.data&&jtype(l)!='{'))return bad(e,"slot definitions must be objects");
        n+=merged_count(g,l);
    }
    slots=odr_take(a,n,sizeof(*slots));
    for(x=0;x<4;x++) {
        OdrJson g= {
            NULL,0
        },l= {
            NULL,0
        };
        if(!local_only)(void)jget(root,sections[x],&g);
        (void)jget(local,sections[x],&l);
        for(pass=0;pass<2;pass++) {
            Ji it;
            OdrJson k,v,ignored;
            OdrJson selected=pass?l:g;
            if(!selected.data)continue;
            jiter(selected,&it);
            while(jnext(&it,&k,&v)) {
                char name[ODR_NAME];
                OdrResult r;
                if(!jstring(k,name,sizeof(name),NULL))return bad(e,"slot name too long");
                if(!pass&&jget(l,name,&ignored))continue;
                r=parse_slot(k,v,kinds[x],root,states,a,slots?&slots[index]:NULL,e);
                if(r)return r;
                ++index;
            }
        }
    }
    if(slots) {
        size_t i,j;
        for(i=1;i<n;i++) {
            Slot s=slots[i];
            j=i;
            while(j&&strcmp(slots[j-1].name,s.name)>0) {
                slots[j]=slots[j-1];
                --j;
            }
            slots[j]=s;
        }
    }
    if(slots) {
        size_t i;
        for(i=0;i<n;i++) {
            slots[i].scope=slots;
            slots[i].scope_count=n;
        }
    }
    *out=slots;
    *count=n;
    return a->result;
}

static OdrResult parse_alts(OdrJson raw,OdrJson root,OdrJson states,Arena *a,const Alt **out,size_t *count,OdrError *e) {
    OdrJson choices;
    size_t n,i=0;
    Alt *alts;
    Ji it;
    OdrJson k,v;
    if(jget(raw,"choices",&choices)) {
        if(!jkeys(raw,"choices",e))return ODR_SCHEMA;
        raw=choices;
    }
    n=jtype(raw)=='['?jcount(raw):1;
    if(!n)return bad(e,"empty rule alternatives");
    alts=odr_take(a,n,sizeof(*alts));
    if(jtype(raw)=='[')jiter(raw,&it);
    while(i<n) {
        OdrJson expr,t= {
            NULL,0
        },locals= {
            NULL,0
        };
        Alt alt;
        char text[32768];
        OdrResult r;
        memset(&alt,0,sizeof(alt));
        if(jtype(raw)=='[') {
            if(!jnext(&it,&k,&v))return ODR_SCHEMA;
        }
        else v=raw;
        expr=v;
        if(jtype(v)=='{') {
            if(!jkeys(v,"expr,tags,locals",e)||!jget(v,"expr",&expr))return bad(e,"production requires expr");
            (void)jget(v,"tags",&t);
            (void)jget(v,"locals",&locals);
        }
        if(!jstring(expr,text,sizeof(text),NULL))return bad(e,"production must be an expression string under 32768 bytes");
        r=odr_expr(text,a,&alt.node,e);
        if(r)return r;
        r=tags(t,a,&alt.tags,&alt.tag_count,e);
        if(r)return r;
        if(locals.data) {
            if(!jkeys(locals,"constants,rng,leaves,parameters",e))return ODR_SCHEMA;
            /* Bank definitions remain global, while local slots are isolated. */ r=parse_slots(root,locals,states,a,&alt.locals,&alt.local_count,e,1);
            if(r)return r;
        }
        if(alts)alts[i]=alt;
        ++i;
    }
    (void)root;
    *out=alts;
    *count=n;
    return a->result;
}

static OdrResult parse_rules(OdrJson root,OdrJson local,OdrJson states,Arena *a,const Rule **out,size_t *count,OdrError *e) {
    size_t n=0,index=0;
    unsigned shared,pass;
    Rule *rules;
    for(shared=0;shared<2;shared++) {
        OdrJson g= {
            NULL,0
        },l= {
            NULL,0
        };
        const char *field=shared?"shapes":"rules";
        (void)jget(root,field,&g);
        (void)jget(local,field,&l);
        if((g.data&&jtype(g)!='{')||(l.data&&jtype(l)!='{'))return bad(e,"rules/shapes must be objects");
        n+=merged_count(g,l);
    }
    rules=odr_take(a,n,sizeof(*rules));
    for(shared=0;shared<2;shared++) {
        OdrJson g= {
            NULL,0
        },l= {
            NULL,0
        };
        const char *field=shared?"shapes":"rules";
        (void)jget(root,field,&g);
        (void)jget(local,field,&l);
        for(pass=0;pass<2;pass++) {
            Ji it;
            OdrJson k,v,ignored;
            OdrJson selected=pass?l:g;
            if(!selected.data)continue;
            jiter(selected,&it);
            while(jnext(&it,&k,&v)) {
                char signature[1024],name[ODR_NAME],*p,*start;
                size_t len;
                Rule rule;
                OdrResult r;
                memset(&rule,0,sizeof(rule));
                rule.shared=(int)shared;
                if(!jstring(k,signature,sizeof(signature),NULL))return bad(e,"rule signature too long");
                if(!pass&&jget(l,signature,&ignored))continue;
                p=signature;
                while(isspace((unsigned char)*p))++p;
                start=p;
                while(isalnum((unsigned char)*p)||*p=='_')++p;
                len=(size_t)(p-start);
                if(!len||len>=sizeof(name))return bad(e,"invalid rule name");
                memcpy(name,start,len);
                name[len]=0;
                if(!identifier(name)||odr_unary(name)||!strcmp(name,"hole")||!strcmp(name,"pow"))return bad(e,"reserved rule name");
                rule.name=text_copy(name,a);
                while(isspace((unsigned char)*p))++p;
                if(*p=='('&&!shared) {
                    ++p;
                    while(*p&&*p!=')') {
                        while(isspace((unsigned char)*p))++p;
                        start=p;
                        while(isalnum((unsigned char)*p)||*p=='_')++p;
                        len=(size_t)(p-start);
                        if(!len||len>=sizeof(name)||rule.argc==ODR_ARGS)return bad(e,"invalid rule parameters");
                        memcpy(name,start,len);
                        name[len]=0;
                        rule.formals[rule.argc++]=text_copy(name,a);
                        while(isspace((unsigned char)*p))++p;
                        if(*p==',')++p;
                        else break;
                    }
                    if(*p++!=')')return bad(e,"invalid rule signature");
                }
                while(isspace((unsigned char)*p))++p;
                if(*p)return bad(e,"invalid rule signature");
                r=parse_alts(v,root,states,a,&rule.alternatives,&rule.count,e);
                if(r)return r;
                if(rules)rules[index]=rule;
                ++index;
            }
        }
    }
    *out=rules;
    *count=n;
    return a->result;
}

const Rule *grule(const Family *f,const char *name,int shared) {
    size_t i;
    for(i=0;i<f->rule_count;i++)if(f->rules[i].shared==shared&&!strcmp(f->rules[i].name,name))return &f->rules[i];
    return NULL;
}

const Slot *gslot(const Family *f,const char *name) {
    size_t i;
    char normalized[ODR_NAME+8];
    if(!strncmp(name,"param.",6)) {
        snprintf(normalized,sizeof(normalized),"theta.%s",name+6);
        name=normalized;
    }
    for(i=0;i<f->slot_count;i++)if(!strcmp(f->slots[i].name,name))return &f->slots[i];
    return NULL;
}

/* Resolve identifiers once. Expansion follows these pointers, never JSON names. */ static OdrResult validate_node(const OdrGrammar *g,const Family *f,const Node *n,const Rule *rule,const Alt *alt,unsigned depth,OdrError *e) {
    size_t i;
    Node *resolved=(Node *)n;
    const Rule *target;
    if(!n||depth>ODR_DEPTH)return odr_error(e,ODR_AST,0,"invalid/deep expression");
    if(n->op==N_REF) {
        if(!strncmp(n->name,"shape.",6)) {
            target=grule(f,n->name+6,1);
            if(!target)return bad(e,"unknown shared shape");
            resolved->binding=target;
            resolved->binding_kind=4;
        }
        else {
            const char *name=n->name;
            char normalized[ODR_NAME+8];
            if(!strncmp(name,"param.",6)) {
                snprintf(normalized,sizeof(normalized),"theta.%s",name+6);
                name=normalized;
            }
            resolved->binding=gslot(f,name);
            resolved->binding_kind=5;
            if(alt)for(i=0;i<alt->local_count;i++)if(!strcmp(alt->locals[i].name,name)) {
                resolved->binding=&alt->locals[i];
                resolved->binding_kind=6;
            }
            if(!resolved->binding)return bad(e,"unknown named slot");
        }
    }
    if(n->op==N_NAME) {
        target=grule(f,n->name,0);
        if(target&&target->argc==0) {
            resolved->binding=target;
            resolved->binding_kind=3;
        }
        for(i=0;i<g->state_count;i++)if(!strcmp(g->states[i],n->name)) {
            resolved->binding_index=(uint32_t)i;
            resolved->binding_kind=1;
        }
        if(rule)for(i=0;i<rule->argc;i++)if(!strcmp(rule->formals[i],n->name)) {
            resolved->binding_index=(uint32_t)i;
            resolved->binding_kind=2;
        }
        if(!resolved->binding_kind)return bad(e,"unknown state/formal/rule or unsupported time");
    }
    if(n->op==N_CALL&&!odr_unary(n->name)) {
        target=grule(f,n->name,0);
        if(!target||target->argc!=n->argc)return bad(e,"unknown rule or wrong argument count");
        resolved->binding=target;
        resolved->binding_kind=3;
    }
    for(i=0;i<n->argc;i++) {
        OdrResult r=validate_node(g,f,n->args[i],rule,alt,depth+1,e);
        if(r)return r;
    }
    return ODR_OK;
}

static OdrResult validate_slot(const Slot *s,const Family *f,OdrError *e) {
    const char *refs[2]= {
        s->scale_ref,s->shift_ref
    };
    unsigned k;
    size_t i;
    for(k=0;k<2;k++)if(refs[k]) {
        const Slot *source=gslot(f,refs[k]);
        for(i=0;i<s->scope_count;i++)if(!strcmp(s->scope[i].name,refs[k]))source=&s->scope[i];
        if(!source||source->kind!=0)return bad(e,"transform must reference a defined constant");
    }
    {
        float low[2]={s->axis.scale,s->axis.shift},high[2]={s->axis.scale,s->axis.shift};
        for(k=0;k<2;k++)if(refs[k]) {
            const Slot *source=gslot(f,refs[k]);
            for(i=0;i<s->scope_count;i++)if(!strcmp(s->scope[i].name,refs[k]))source=&s->scope[i];
            low[k]=high[k]=source->axis.values[0];
            for(i=1;i<source->axis.count;i++) {
                float x=source->axis.values[i];
                if(x<low[k])low[k]=x;
                if(x>high[k])high[k]=x;
            }
        }
        /* Conservative Cartesian bounds: every generated pair must be valid. */
        if((s->axis.transform==2||s->axis.transform==4)&&high[0]>low[1])return bad(e,"RNG low can exceed high");
        if(s->axis.transform==3&&low[0]<0)return bad(e,"RNG std must be nonnegative");
        if(s->axis.transform==4&&low[0]<=0)return bad(e,"log-uniform endpoints must be positive");
    }
    return ODR_OK;
}

static OdrResult build(OdrJson input,const OdrStatic *fixed,Arena *a,const OdrGrammar **out,OdrError *e) {
    OdrJson root=input,v,states,families= {
        NULL,0
    },expansion= {
        NULL,0
    },limits= {
        NULL,0
    },integration= {
        NULL,0
    };
    OdrGrammar data,*g;
    Family *f;
    Ji fit;
    size_t i=0;
    uint64_t number;
    OdrResult r;
    if(jget(input,"grammar",&v))root=v;
    memset(&data,0,sizeof(data));
    data.magic=GRAMMAR_MAGIC;
    data.fixed=fixed;
    if(!jkeys(root,"version,kind,states,problem_id,description,integration,rules,shapes,rhs,families,family_id,leaves,constant_banks,constants,rng_banks,rng,parameters,expansion,toggle_sampling,limits,retain,allocation",e))return ODR_SCHEMA;
    if(jget(root,"version",&v)&&(!ju64(v,&number)||number!=1))return bad(e,"grammar version must be 1");
    if(jget(root,"kind",&v)&&!jeq(v,"compile"))return odr_error(e,ODR_UNSUPPORTED,0,"scoring grammar only");
    if(jget(root,"integration",&integration)) {
        if(!jkeys(integration,"method,dt,max_steps,stiff,rtol,atol",e))return ODR_SCHEMA;
        if((jget(integration,"method",&v)&&!jeq(v,"rk4"))||(jget(integration,"stiff",&v)&&jtype(v)!='f')||jget(integration,"rtol",&v)||jget(integration,"atol",&v))return odr_error(e,ODR_UNSUPPORTED,0,"only fixed-step FP32 RK4 is supported");
    }
    if(jget(integration,"dt",&v)&&(!jfloat(v,&data.info.rk4_max_dt)||data.info.rk4_max_dt<=0))return bad(e,"integration.dt must be positive FP32");
    if(jget(integration,"max_steps",&v)&&(!ju64(v,&data.info.rk4_max_steps)||!data.info.rk4_max_steps))return bad(e,"integration.max_steps must be positive uint64");
    if(jget(root,"retain",&v)) {
        char *policy;
        if(jtype(v)!='{')return bad(e,"retain must be an object");
        policy=odr_take(a,v.size+1,1);
        if(policy) {
            memcpy(policy,v.data,v.size);
            policy[v.size]=0;
        }
        data.info.retention.data=policy;
        data.info.retention.size=v.size;
    }
    if(jget(root,"allocation",&v)) {
        OdrJson option;
        if(!jkeys(v,"mode,redistribute_unused",e))return ODR_SCHEMA;
        if((jget(v,"mode",&option)&&!jeq(option,"per_family"))||(jget(v,"redistribute_unused",&option)&&jtype(option)!='f'))return odr_error(e,ODR_UNSUPPORTED,0,"only reserved per-family allocation without redistribution is supported");
    }
    g=odr_take(a,1,sizeof(*g));
    r=odr_states(root,a,(const char ***)&data.states,&data.state_count,e);
    if(r)return r;
    (void)jget(root,"states",&states);
    if(fixed&&fixed->state_count!=data.state_count)return bad(e,"static and grammar state counts differ");
    if(data.states&&fixed)for(i=0;i<data.state_count;i++)if(strcmp(data.states[i],fixed->states[i]))return bad(e,"static and grammar state order differs");
    (void)jget(root,"expansion",&expansion);
    if(expansion.data&&!jkeys(expansion,"strategy,seed,max_nodes,max_depth,max_expansion_depth",e))return ODR_SCHEMA;
    if(jget(expansion,"strategy",&v)&&!jeq(v,"enumerate")) {
        if(jeq(v,"sample"))data.sample=1;
        else return bad(e,"unknown expansion strategy");
    }
    data.seed=0;
    if(jget(expansion,"seed",&v)&&!ju64(v,&data.seed))return bad(e,"expansion seed must be uint64");
    number=limit(expansion,"max_nodes",63,e);
    if(!number||number>4096)return bad(e,"max_nodes must be 1..4096");
    data.max_nodes=(uint32_t)number;
    number=limit(expansion,"max_depth",12,e);
    if(!number||number>ODR_DEPTH)return bad(e,"max_depth must be 1..128");
    data.max_depth=(uint32_t)number;
    number=limit(expansion,"max_expansion_depth",20,e);
    if(!number||number>64)return bad(e,"max_expansion_depth must be 1..64");
    data.max_expansion_depth=(uint32_t)number;
    (void)jget(root,"limits",&limits);
    if(limits.data&&!jkeys(limits,"max_skeletons,max_variants,max_configurations,max_configurations_per_skeleton,max_derivations,max_expansion_steps,max_seconds",e))return ODR_SCHEMA;
    if(jget(limits,"max_seconds",&v)&&(!jfloat(v,&data.info.max_seconds)||data.info.max_seconds<=0))return bad(e,"max_seconds must be positive");
    (void)jget(root,"families",&families);
    if(families.data&&(jtype(families)!='['||!jcount(families)))return bad(e,"families must be nonempty");
    data.family_count=families.data?jcount(families):1;
    f=odr_take(a,data.family_count,sizeof(*f));
    data.families=f;
    if(families.data)jiter(families,&fit);
    for(i=0;i<data.family_count;i++) {
        Family family;
        OdrJson local= {
            "{}",2
        },k,rhs= {
            NULL,0
        },t= {
            NULL,0
        },flimits= {
            NULL,0
        };
        const Node **roots;
        Ji sit;
        OdrJson sk,state;
        unsigned si=0;
        memset(&family,0,sizeof(family));
        if(families.data&&!jnext(&fit,&k,&local))return ODR_SCHEMA;
        if(!jkeys(local,"id,tags,description,rhs,rules,shapes,leaves,constants,rng,parameters,limits,toggle_sampling",e))return ODR_SCHEMA;
        if(jget(local,"toggle_sampling",&v)||jget(root,"toggle_sampling",&v)) {
            OdrJson seed;
            if(!jkeys(v,"count,seed",e))return ODR_SCHEMA;
            family.joint_count=limit(v,"count",0,e);
            if(!family.joint_count)return bad(e,"joint sample count required");
            family.joint_seed=data.seed;
            if(jget(v,"seed",&seed)&&!ju64(seed,&family.joint_seed))return bad(e,"invalid joint sample seed");
        }
        if(jget(local,"id",&v)||(!families.data&&jget(root,"family_id",&v))) {
            size_t length;
            if(!jstring(v,NULL,0,&length)||!length)return bad(e,"family id must be nonempty");
            family.name=jcopy(v,a);
        }
        else {
            if(families.data)return bad(e,"family id required");
            family.name=text_copy("default",a);
        }
        (void)jget(local,"tags",&t);
        r=tags(t,a,&family.tags,&family.tag_count,e);
        if(r)return r;
        (void)jget(local,"limits",&flimits);
        if(flimits.data&&!jkeys(flimits,"max_skeletons,max_variants,max_configurations,max_configurations_per_skeleton,max_derivations,max_expansion_steps",e))return ODR_SCHEMA;
        if(jget(flimits,"max_configurations_per_skeleton",&v)||jget(limits,"max_configurations_per_skeleton",&v)) {
            if(!ju64(v,&family.max_configs_per_skeleton)||!family.max_configs_per_skeleton)return bad(e,"max_configurations_per_skeleton must be positive");
        }
        r=odr_allocation(limits,families,i,"max_skeletons",10000,&family.max_asts,e);
        if(r)return r;
        r=odr_allocation(limits,families,i,"max_variants",100000,&family.max_variants,e);
        if(r)return r;
        r=odr_allocation(limits,families,i,"max_configurations",1000000000,&family.max_configs,e);
        if(r)return r;
        r=odr_allocation(limits,families,i,"max_derivations",1000000,&family.max_attempts,e);
        if(r)return r;
        r=odr_allocation(limits,families,i,"max_expansion_steps",100000000,&family.max_steps,e);
        if(r)return r;
        r=parse_slots(root,local,states,a,&family.slots,&family.slot_count,e,0);
        if(r)return r;
        r=parse_rules(root,local,states,a,&family.rules,&family.rule_count,e);
        if(r)return r;
        if(!jget(local,"rhs",&rhs))(void)jget(root,"rhs",&rhs);
        if(jtype(rhs)!='{')return bad(e,"family RHS must be an object");
        {
            Ji keys;
            OdrJson key,value;
            uint32_t ignored;
            jiter(rhs,&keys);
            while(jnext(&keys,&key,&value))if(!state_index(states,key,&ignored))return bad(e,"unknown RHS state");
        }
        roots=odr_take(a,data.state_count,sizeof(*roots));
        family.rhs=roots;
        jiter(states,&sit);
        while(jnext(&sit,&sk,&state)) {
            char name[ODR_NAME],text[32768];
            int is_fixed=0;
            size_t fi;
            const Node *node=NULL;
            (void)jstring(state,name,sizeof(name),NULL);
            if(fixed)for(fi=0;fi<fixed->rhs_count;fi++)if(fixed->rhs[fi].state_index==si)is_fixed=1;
            if(jget(rhs,name,&v)) {
                if(is_fixed)return bad(e,"omit fixed RHS from grammar; static phase owns it");
                if(!jstring(v,text,sizeof(text),NULL))return bad(e,"RHS must be expression string under 32768 bytes");
                r=odr_expr(text,a,&node,e);
                if(r)return r;
            }
            else if(!is_fixed)return bad(e,"missing variable RHS");
            if(roots)roots[si]=node;
            ++si;
        }
        if(f)f[i]=family;
    }
    if(g)*g=data;
    if(a->result)return a->result;
    if(g&&a->used<=a->capacity) {
        uint64_t allocated_asts=0,allocated_configs=0;
        for(i=0;i<data.family_count;i++) {
            size_t x,y;
            const Family *family=&f[i];
            if(allocated_asts>UINT64_MAX-family->max_asts||allocated_configs>UINT64_MAX-family->max_configs)return ODR_OVERFLOW;
            allocated_asts+=family->max_asts;
            allocated_configs+=family->max_configs;
            for(x=0;x<i;x++)if(!strcmp(f[x].name,family->name))return bad(e,"duplicate family id");
            for(x=0;x<family->slot_count;x++) {
                r=validate_slot(&family->slots[x],family,e);
                if(r)return r;
            }
            for(x=0;x<family->rule_count;x++) {
                const Rule *rule=&family->rules[x];
                for(y=0;y<x;y++)if(rule->shared==family->rules[y].shared&&!strcmp(rule->name,family->rules[y].name))return bad(e,"duplicate rule name/signature");
                for(y=0;y<data.state_count;y++)if(!strcmp(rule->name,data.states[y]))return bad(e,"rule name collides with state");
                for(y=0;y<rule->argc;y++) {
                    size_t z;
                    if(!identifier(rule->formals[y]))return bad(e,"invalid formal parameter");
                    for(z=0;z<y;z++)if(!strcmp(rule->formals[z],rule->formals[y]))return bad(e,"duplicate formal parameter");
                }
                for(y=0;y<rule->count;y++) {
                    size_t z;
                    for(z=0;z<rule->alternatives[y].local_count;z++) {
                        r=validate_slot(&rule->alternatives[y].locals[z],family,e);
                        if(r)return r;
                    }
                }
            }
            for(x=0;x<data.state_count;x++)if(family->rhs[x]) {
                r=validate_node(g,family,family->rhs[x],NULL,NULL,0,e);
                if(r)return r;
            }
            for(x=0;x<family->rule_count;x++)for(y=0;y<family->rules[x].count;y++) {
                r=validate_node(g,family,family->rules[x].alternatives[y].node,&family->rules[x],&family->rules[x].alternatives[y],0,e);
                if(r)return r;
            }
        }
        if(jget(limits,"max_skeletons",&v)&&ju64(v,&number)&&allocated_asts>number)return bad(e,"family structure allocations exceed request limit");
        if(jget(limits,"max_configurations",&v)&&ju64(v,&number)&&allocated_configs>number)return bad(e,"family configuration allocations exceed request limit");
    }
    if(out)*out=g;
    return ODR_OK;
}

OdrResult odr_grammar_parse(OdrJson input,const OdrStatic *fixed,void *arena,size_t capacity,size_t *required,const OdrGrammar **out,OdrError *error) {
    Arena measure= {
        NULL,0,0,ODR_OK
    },a= {
        (unsigned char *)arena,capacity,0,ODR_OK
    };
    OdrResult r;
    if(out)*out=NULL;
    if(required)*required=0;
    if(!required)return bad(error,"required_bytes is mandatory");
    if(arena&&(uintptr_t)arena%odr_arena_alignment())return bad(error,"unaligned arena");
    r=jvalidate(input,error);
    if(r)return r;
    r=build(input,fixed,&measure,NULL,error);
    *required=measure.used;
    if(r)return r;
    if(!arena)return ODR_OK;
    if(capacity<measure.used)return odr_error(error,ODR_BUFFER,0,"grammar arena too small");
    r=build(input,fixed,&a,out,error);
    if(r&&out)*out=NULL;
    return r;
}

size_t odr_grammar_family_count(const OdrGrammar *g) {
    return g&&g->magic==GRAMMAR_MAGIC?g->family_count:0;
}

const char *odr_grammar_family_name(const OdrGrammar *g,size_t i) {
    return g&&g->magic==GRAMMAR_MAGIC&&i<g->family_count?g->families[i].name:NULL;
}

OdrResult odr_grammar_info(const OdrGrammar *g,OdrGrammarInfo *out) {
    if(!g||g->magic!=GRAMMAR_MAGIC||!out)return ODR_SCHEMA;
    *out=g->info;
    return ODR_OK;
}

OdrResult odr_grammar_family_info(const OdrGrammar *g,size_t index,OdrFamilyInfo *out) {
    const Family *f;
    if(!g||g->magic!=GRAMMAR_MAGIC||!out||index>=g->family_count)return ODR_SCHEMA;
    f=&g->families[index];
    out->name=f->name;
    out->max_skeletons=f->max_asts;
    out->max_variants=f->max_variants;
    out->max_configurations=f->max_configs;
    out->max_derivations=f->max_attempts;
    out->max_expansion_steps=f->max_steps;
    out->max_configurations_per_skeleton=f->max_configs_per_skeleton;
    return ODR_OK;
}
