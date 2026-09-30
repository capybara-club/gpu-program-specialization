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
#include "odr_internal.h"
/* Reserve explicit allocations before dividing a request ceiling. No stealing. */ OdrResult odr_allocation(OdrJson limits,OdrJson families,size_t index,const char *key,uint64_t fallback,uint64_t *out,OdrError *e) {
    uint64_t ceiling=fallback,reserved=0,own=0,value;
    size_t missing=0,rank=0,i=0;
    int explicit_ceiling=0;
    Ji it;
    OdrJson k,f,l,v;
    if(jget(limits,key,&v)) {
        if(!ju64(v,&ceiling)||!ceiling)return odr_error(e,ODR_SCHEMA,0,"invalid request allocation");
        explicit_ceiling=1;
    }
    if(!families.data) {
        *out=ceiling;
        return ODR_OK;
    }
    jiter(families,&it);
    while(jnext(&it,&k,&f)) {
        l.data=NULL;
        l.size=0;
        (void)jget(f,"limits",&l);
        if(jget(l,key,&v)) {
            if(!ju64(v,&value)||!value)return odr_error(e,ODR_SCHEMA,0,"family allocations must be positive integers");
            if(reserved>UINT64_MAX-value)return odr_error(e,ODR_OVERFLOW,0,"family allocation sum overflows uint64");
            reserved+=value;
            if(i==index)own=value;
        }
        else {
            if(i<index)++rank;
            ++missing;
        }
        ++i;
    }
    if(!explicit_ceiling&&reserved&&missing)return odr_error(e,ODR_SCHEMA,0,"mixed allocations require an explicit request ceiling");
    if(explicit_ceiling&&reserved>ceiling)return odr_error(e,ODR_SCHEMA,0,"family allocations exceed request ceiling");
    if(index==SIZE_MAX) {
        if(missing&&ceiling-reserved<missing)return odr_error(e,ODR_SCHEMA,0,"family allocation would be zero");
        *out=missing?ceiling:reserved;return ODR_OK;
    }
    if(own) {
        *out=own;
        return ODR_OK;
    }
    if(!missing||reserved>ceiling)return odr_error(e,ODR_SCHEMA,0,"invalid family allocation");
    *out=(ceiling-reserved)/missing+(rank<(ceiling-reserved)%missing);
    return *out?ODR_OK:odr_error(e,ODR_SCHEMA,0,"family allocation would be zero");
}


OdrResult odr_request_allocations(OdrJson input,OdrAllocationInfo *out,OdrError *e) {
    OdrJson root,limits={0},families={0},v,k,f,l;Ji it;OdrResult r;uint64_t cap;
    static const char *keys[]={"max_skeletons","max_variants","max_configurations","max_derivations","max_expansion_steps"};
    static const uint64_t defaults[]={10000,100000,1000000000,1000000,100000000};
    uint64_t *fields[5];size_t i;
    if(!out)return odr_error(e,ODR_SCHEMA,0,"allocation output required");
    memset(out,0,sizeof(*out));r=jvalidate(input,e);if(r)return r;
    if(!jget(input,"grammar",&root))root=input;
    if(jtype(root)!='{')return odr_error(e,ODR_SCHEMA,0,"grammar must be an object");
    if(jget(root,"allocation",&v)) {
        OdrJson option;
        if(!jkeys(v,"mode,redistribute_unused",e))return ODR_SCHEMA;
        if((jget(v,"mode",&option)&&!jeq(option,"per_family"))||(jget(v,"redistribute_unused",&option)&&jtype(option)!='f'))return odr_error(e,ODR_UNSUPPORTED,0,"only reserved per-family allocation without redistribution is supported");
    }
    (void)jget(root,"limits",&limits);(void)jget(root,"families",&families);
    if(limits.data&&!jkeys(limits,"max_skeletons,max_variants,max_configurations,max_configurations_per_skeleton,max_derivations,max_expansion_steps,max_seconds",e))return ODR_SCHEMA;
    if(jget(limits,"max_configurations_per_skeleton",&v)&&(!ju64(v,&cap)||!cap))return odr_error(e,ODR_SCHEMA,0,"invalid skeleton configuration cap");
    if(families.data&&(jtype(families)!='['||!jcount(families)))return odr_error(e,ODR_SCHEMA,0,"families must be nonempty");
    out->families=families.data?jcount(families):1;
    if(families.data) {
        jiter(families,&it);
        while(jnext(&it,&k,&f)) {
            if(jtype(f)!='{')return odr_error(e,ODR_SCHEMA,0,"family must be an object");
            l.data=NULL;l.size=0;(void)jget(f,"limits",&l);
            if(l.data&&!jkeys(l,"max_skeletons,max_variants,max_configurations,max_configurations_per_skeleton,max_derivations,max_expansion_steps",e))return ODR_SCHEMA;
            if(jget(l,"max_configurations_per_skeleton",&v)&&(!ju64(v,&cap)||!cap))return odr_error(e,ODR_SCHEMA,0,"invalid skeleton configuration cap");
        }
    }
    fields[0]=&out->skeletons;fields[1]=&out->variants;fields[2]=&out->configurations;
    fields[3]=&out->derivations;fields[4]=&out->expansion_steps;
    for(i=0;i<5;i++) {r=odr_allocation(limits,families,SIZE_MAX,keys[i],defaults[i],fields[i],e);if(r)return r;}
    return ODR_OK;
}
