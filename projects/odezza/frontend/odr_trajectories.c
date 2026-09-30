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
typedef struct Rk4Work { float max_gap,min_gap; uint64_t intervals; uint32_t trajectories,points; } Rk4Work;
static OdrResult parse(OdrJson source,Arena *a,const OdrTrajectories **out,Rk4Work *work,OdrError *e) {
    OdrJson problem=jproblem(source),trajectories,k,v;
    OdrTrajectories *t=odr_take(a,1,sizeof(*t));
    const char **states=NULL;
    uint32_t ns=0,nt=0,np=0,ti=0,point=0;
    uint64_t observed=0;
    uint32_t *offsets;
    float *times,*values;
    OdrTrajectoryInfo *info;
    Ji tr;
    OdrResult r=odr_states(problem,a,&states,&ns,e);
    if(r)return r;
    if(!jkeys(problem,"states,trajectories,known_rhs",e))return ODR_SCHEMA;
    if(!jget(problem,"trajectories",&trajectories)||jtype(trajectories)!='['||!jcount(trajectories))return odr_error(e,ODR_SCHEMA,0,"expected nonempty trajectories");
    if(jcount(trajectories)>UINT32_MAX)return ODR_OVERFLOW;
    nt=(uint32_t)jcount(trajectories);
    jiter(trajectories,&tr);
    while(jnext(&tr,&k,&v)) {
        OdrJson ts;
        size_t n;
        if(!jget(v,"times",&ts)||jtype(ts)!='['||(n=jcount(ts))==0)return odr_error(e,ODR_SCHEMA,0,"trajectory times must be a nonempty array");
        if(n>UINT32_MAX-np)return odr_error(e,ODR_OVERFLOW,0,"too many trajectory points");
        np+=(uint32_t)n;
    }
    if(work) {work->trajectories=nt;work->points=np;}
    offsets=odr_take(a,(size_t)nt+1,sizeof(*offsets));
    times=odr_take(a,np,sizeof(*times));
    if(np>UINT32_MAX/ns)return odr_error(e,ODR_OVERFLOW,0,"state-major reference indexing exceeds native uint32 layout");
    values=odr_take(a,(size_t)ns*np,sizeof(*values));
    info=odr_take(a,nt,sizeof(*info));
    if(a->result)return a->result;
    if(t) {
        t->state_count=ns;
        t->trajectory_count=nt;
        t->point_count=np;
        t->states=states;
        t->offsets=offsets;
        t->times=times;
        t->values=values;
        t->info=info;
    }
    jiter(trajectories,&tr);
    while(jnext(&tr,&k,&v)) {
        OdrJson ts,vs,ic,mask= {
            NULL,0
        },tk,tv,row,mrow;
        Ji it,rows,ics,mrows;
        uint32_t j=0,s,n;
        float initial[ODR_STATES],previous=0,first=0,min_dt=FLT_MAX,max_dt=0,dt=0;
        uint64_t nobs=0;
        int uniform=1,masked;
        if(!jkeys(v,"initial,times,values,mask",e))return ODR_SCHEMA;
        if(!jget(v,"initial",&ic)||jtype(ic)!='['||jcount(ic)!=ns)return odr_error(e,ODR_SCHEMA,0,"each trajectory requires a complete initial vector");
        jiter(ic,&ics);
        s=0;
        while(jnext(&ics,&tk,&tv)) {
            if(!jfloat(tv,&initial[s++]))return odr_error(e,ODR_SCHEMA,0,"initial states must be finite FP32");
        }
        (void)jget(v,"times",&ts);
        n=(uint32_t)jcount(ts);
        if(!jget(v,"values",&vs)||jtype(vs)!='['||jcount(vs)!=n)return odr_error(e,ODR_SCHEMA,0,"values must have one row per time");
        masked=jget(v,"mask",&mask);
        if(masked&&(jtype(mask)!='['||jcount(mask)!=n))return odr_error(e,ODR_SCHEMA,0,"mask must have one row per time");
        jiter(ts,&it);
        jiter(vs,&rows);
        if(masked)jiter(mask,&mrows);
        if(offsets)offsets[ti]=point;
        while(jnext(&it,&tk,&tv)) {
            float time;
            Ji cells,mcells;
            if(!jfloat(tv,&time))return odr_error(e,ODR_SCHEMA,0,"times must be finite FP32");
            if(j) {
                float gap=time-previous;
                if(!(gap>0)||!isfinite(gap))return odr_error(e,ODR_SCHEMA,0,"times must strictly increase after FP32 conversion");
                if(j==1)dt=gap;
                else if(fabsf(gap-dt)>8*FLT_EPSILON*fmaxf(fabsf(gap),fabsf(dt)))uniform=0;
                if(gap<min_dt)min_dt=gap;
                if(gap>max_dt)max_dt=gap;
            }
            else first=time;
            previous=time;
            if(times)times[point]=time;
            if(!jnext(&rows,&tk,&row)||jtype(row)!='['||jcount(row)!=ns)return odr_error(e,ODR_SCHEMA,0,"each observation row must match state count");
            if(masked) {
                if(!jnext(&mrows,&tk,&mrow)||jtype(mrow)!='['||jcount(mrow)!=ns)return odr_error(e,ODR_SCHEMA,0,"each mask row must match state count");
                jiter(mrow,&mcells);
            }
            jiter(row,&cells);
            s=0;
            while(jnext(&cells,&tk,&tv)) {
                float value=NAN;
                int include=1,is_null=jtype(tv)=='n';
                if(masked) {
                    OdrJson mk,mv;
                    if(!jnext(&mcells,&mk,&mv)||(jtype(mv)!='t'&&jtype(mv)!='f'))return odr_error(e,ODR_SCHEMA,0,"mask values must be booleans");
                    include=jtype(mv)=='t';
                }
                if(!is_null&&!jfloat(tv,&value))return odr_error(e,ODR_SCHEMA,0,"observations must be finite FP32 or null");
                if(!j) {
                    if(!is_null&&value!=initial[s])return odr_error(e,ODR_SCHEMA,0,"initial observations must agree with the initial vector");
                    value=initial[s];
                }
                else if(is_null||!include) {
                    value=NAN;
                    if(t)t->allow_missing_observations=1;
                }
                else ++nobs;
                if(values)values[(size_t)s*np+point]=value;
                ++s;
            }
            ++point;
            ++j;
        }
        if(nobs>UINT32_MAX)return odr_error(e,ODR_OVERFLOW,0,"trajectory observation count overflow");
        if(info) {
            info[ti].first_point=point-n;
            info[ti].point_count=n;
            info[ti].observed_count=(uint32_t)nobs;
            info[ti].uniform_spacing=(uint32_t)uniform;
            info[ti].first_time=first;
            info[ti].last_time=previous;
            info[ti].min_dt=n>1?min_dt:0;
            info[ti].max_dt=max_dt;
            info[ti].dt=uniform?dt:NAN;
        }
        observed+=nobs;
        if(work&&n>1) {
            if(max_dt>work->max_gap)work->max_gap=max_dt;
            if(min_dt<work->min_gap)work->min_gap=min_dt;
            work->intervals+=n-1;
        }
        ++ti;
    }
    if(!observed)return odr_error(e,ODR_SCHEMA,0,"at least one noninitial observation is required");
    if(offsets)offsets[nt]=np;
    if(t)t->observed_count=observed;
    if(out)*out=t;
    return a->result;
}
OdrResult odr_trajectories_parse(OdrJson j,void *arena,size_t capacity,size_t *required,const OdrTrajectories **out,OdrError *e) {
    Arena a= {
        (unsigned char *)arena,capacity,0,ODR_OK
    };
    OdrResult r;
    if(out)*out=NULL;
    if(required)*required=0;
    if(!required)return odr_error(e,ODR_SCHEMA,0,"required_bytes is mandatory");
    if(arena&&(uintptr_t)arena%odr_arena_alignment())return odr_error(e,ODR_SCHEMA,0,"unaligned arena");
    r=jvalidate(j,e);
    if(r)return r;
    r=parse(j,&a,out,NULL,e);
    *required=a.used;
    if(r||a.result) {
        if(out)*out=NULL;
        return r?r:a.result;
    }
    if(arena&&capacity<a.used) {
        if(out)*out=NULL;
        return odr_error(e,ODR_BUFFER,0,"trajectory arena too small");
    }
    return ODR_OK;
}

static OdrResult rk4_layout(const Rk4Work *work,float max_dt,
    uint64_t max_steps,OdrRk4Layout *out,OdrError *error) {
    double steps;
    if(!out||!isfinite(max_dt)||max_dt<=0)return odr_error(error,ODR_SCHEMA,0,"positive RK4 max_dt required");
    memset(out,0,sizeof(*out));
    steps=ceil(work->max_gap/(double)max_dt);
    if(steps<1)steps=1;
    if(steps>UINT32_MAX)return odr_error(error,ODR_OVERFLOW,0,"RK4 subdivision count exceeds uint32");
    if(work->intervals&&!(work->min_gap/(float)steps>0))return odr_error(error,ODR_SCHEMA,0,"RK4 substep underflows FP32");
    out->steps_per_observation=(uint32_t)steps;
    out->steps_per_configuration=work->intervals*out->steps_per_observation;
    if(max_steps&&out->steps_per_configuration>max_steps)return odr_error(error,ODR_CAPACITY,0,"RK4 work exceeds max_steps per configuration");
    return ODR_OK;
}

OdrResult odr_trajectories_rk4(const OdrTrajectories *t,float max_dt,
    uint64_t max_steps,OdrRk4Layout *out,OdrError *error) {
    Rk4Work work={0,FLT_MAX,0,0,0};uint32_t i;
    if(!t)return odr_error(error,ODR_SCHEMA,0,"trajectories required");
    for(i=0;i<t->trajectory_count;i++)if(t->info[i].point_count>1) {
        if(t->info[i].max_dt>work.max_gap)work.max_gap=t->info[i].max_dt;
        if(t->info[i].min_dt<work.min_gap)work.min_gap=t->info[i].min_dt;
    }
    work.intervals=(uint64_t)t->point_count-t->trajectory_count;
    return rk4_layout(&work,max_dt,max_steps,out,error);
}

OdrResult odr_request_rk4_work(OdrJson request,OdrRk4Work *out,OdrError *error) {
    Arena a={NULL,0,0,ODR_OK};Rk4Work work={0,FLT_MAX,0,0,0};
    OdrJson g,integration,v;OdrResult result;float dt;uint64_t limit=0;
    if(!out)return odr_error(error,ODR_SCHEMA,0,"RK4 layout output required");
    memset(out,0,sizeof(*out));
    result=jvalidate(request,error);if(result)return result;
    if(!jget(request,"grammar",&g)||!jget(g,"integration",&integration)||
       !jget(integration,"dt",&v)||!jfloat(v,&dt)||dt<=0)
        return odr_error(error,ODR_SCHEMA,0,"integration.dt must be positive FP32");
    if(!jkeys(integration,"method,dt,max_steps,stiff,rtol,atol",error))return ODR_SCHEMA;
    if((jget(integration,"method",&v)&&!jeq(v,"rk4"))||
       (jget(integration,"stiff",&v)&&jtype(v)!='f')||jget(integration,"rtol",&v)||jget(integration,"atol",&v))
        return odr_error(error,ODR_UNSUPPORTED,0,"only fixed-step FP32 RK4 is supported");
    if(jget(integration,"max_steps",&v)&&(!ju64(v,&limit)||!limit))
        return odr_error(error,ODR_SCHEMA,0,"integration.max_steps must be positive uint64");
    result=parse(request,&a,NULL,&work,error);if(result)return result;
    {
        OdrRk4Layout layout={0};
        result=rk4_layout(&work,dt,limit,&layout,error);
        out->steps_per_observation=layout.steps_per_observation;
        out->steps_per_configuration=layout.steps_per_configuration;
        out->trajectory_count=work.trajectories;out->point_count=work.points;
        return result;
    }
}
