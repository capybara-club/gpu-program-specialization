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
#define _POSIX_C_SOURCE 200809L
#include "odezza_scoring_request.h"
#include "odr_cache.h"
#include <nvrtc.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
struct OdrScoringSession {
    OdezzaScoringPipeline *pipeline;
    OdrCache *artifacts;
    OdrScoringOptions options;
    OdezzaScoringRhs *fixed;
    char cache[1024],error[1024];
    OdrScoringStats stats;
};

static double now(void) {
    struct timespec t= {
        0,0
    };
    (void)clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec+(double)t.tv_nsec*1e-9;
}

static OdezzaScoringTemplateInfo shape(const OdrScoringSession *s) {
    OdezzaScoringTemplateInfo t;
    t.sm_version=s->options.shape.sm_version;
    t.state_capacity=(uint32_t)s->options.shape.state_capacity;
    t.constant_capacity=s->options.shape.constant_capacity;
    t.system_capacity=s->options.shape.system_capacity;
    t.shared_patch_capacity=s->options.shape.shared_patch_capacity;
    t.system_patch_capacity=s->options.shape.system_patch_capacity;
    return t;
}

static OdezzaResult cached_template(OdrScoringSession *s,OdezzaScoringTemplate **out) {
    OdezzaScoringTemplateInfo info=shape(s);
    OdezzaResult result=ODEZZA_SUCCESS;
    char key[256];void *data=NULL;size_t bytes=0;double start=now(),compile_seconds=0;
    uint64_t evictions=0,invalidations=0;int major=0,minor=0,hit=1,stored=0;
    *out=NULL;
    if(s->artifacts){evictions=odr_cache_evictions(s->artifacts);invalidations=odr_cache_invalidations(s->artifacts);}
    if(s->cache[0]&&!s->artifacts&&odr_cache_open(s->cache,&s->artifacts,s->error,sizeof(s->error)))return ODEZZA_ERROR_IO;
    if(nvrtcVersion(&major,&minor)!=NVRTC_SUCCESS)return ODEZZA_ERROR_COMPILER;
    /* Source identity and checksum are additionally validated by template_read.
     * Profile must change with compilation options or artifact compatibility. */
    snprintf(key,sizeof(key),"scoring-v1:c++17:nvrtc-%d.%d:sm%u:%u:%u:%u:%u:%u",major,minor,
        info.sm_version,info.state_capacity,info.constant_capacity,info.system_capacity,info.shared_patch_capacity,info.system_patch_capacity);
    if(s->artifacts) {
        if(odr_cache_lock(s->artifacts,key)){result=ODEZZA_ERROR_IO;goto done;}
        hit=odr_cache_get(s->artifacts,key,&data,&bytes);
        if(hit<0){result=ODEZZA_ERROR_IO;goto done;}
        if(!hit) {
            result=odezza_scoring_template_read(&info,data,bytes,out);free(data);data=NULL;
            if(!result){++s->stats.cache_hits;goto done;}
            if(*out){odezza_scoring_template_destroy(*out);*out=NULL;}
            if(result!=ODEZZA_ERROR_FORMAT)goto done;
            ++s->stats.cache_invalidations;
            if(odr_cache_remove(s->artifacts,key)){result=ODEZZA_ERROR_IO;goto done;}
        }
    }
    ++s->stats.cache_misses;
    result=odezza_scoring_template_create(&info,out);
    if(result) {
        if(*out){size_t n;(void)odezza_scoring_template_write_error(*out,s->error,sizeof(s->error),&n);}
        goto done;
    }
    (void)odezza_scoring_template_nvrtc_seconds(*out,&compile_seconds);
    s->stats.nvrtc_seconds+=compile_seconds;
    if(s->artifacts) {
        result=odezza_scoring_template_write(*out,NULL,0,&bytes);if(result)goto done;
        data=malloc(bytes);if(!data){result=ODEZZA_ERROR_ALLOCATION;goto done;}
        result=odezza_scoring_template_write(*out,data,bytes,&bytes);if(result)goto done;
        stored=odr_cache_put(s->artifacts,key,data,bytes,compile_seconds);
        if(stored<0)result=ODEZZA_ERROR_IO;
        if(stored>0)++s->stats.cache_oversized;
    }
done:
    free(data);
    if(s->artifacts) {
        s->stats.cache_evictions+=odr_cache_evictions(s->artifacts)-evictions;
        s->stats.cache_invalidations+=odr_cache_invalidations(s->artifacts)-invalidations;
        if(result==ODEZZA_ERROR_IO)snprintf(s->error,sizeof(s->error),"%s",odr_cache_error(s->artifacts));
        odr_cache_unlock(s->artifacts);
    }
    s->stats.template_prepare_seconds+=now()-start;
    return result;
}

static int grow(OdrScoringSession *s) {
    uint32_t max=s->options.maximum_patch_capacity,a=s->options.shape.shared_patch_capacity,b=s->options.shape.system_patch_capacity;
    if(a>=max&&b>=max)return 0;
    s->options.shape.shared_patch_capacity=a>max/2?max:a*2;
    s->options.shape.system_patch_capacity=b>max/2?max:b*2;
    ++s->stats.capacity_growths;
    return 1;
}

static OdezzaResult prepare(OdrScoringSession *s) {
    OdezzaResult r;
    for(;;) {
        OdezzaScoringTemplate *t=NULL;
        double start;
        r=cached_template(s,&t);
        if(r) {
            if(t)odezza_scoring_template_destroy(t);
            return r;
        }
        start=now();
        r=odezza_scoring_pipeline_create_with_template(&s->options.shape,t,&s->pipeline);
        s->stats.prespecialize_seconds+=now()-start;
        odezza_scoring_template_destroy(t);
        s->stats.actual_shape=shape(s);
        if(!r) {
            s->error[0]=0;
            return r;
        }
        if(s->pipeline) {
            size_t n;
            (void)odezza_scoring_pipeline_write_error(s->pipeline,s->error,sizeof(s->error),&n);
            {
                OdezzaResult cleanup=odezza_scoring_pipeline_destroy(s->pipeline);
                if(cleanup==ODEZZA_ERROR_CUDA_UNFENCED)return cleanup;
            }
            s->pipeline=NULL;
        }
        if(r!=ODEZZA_ERROR_SPECIALIZATION_CAPACITY||!grow(s))return r;
    }
}

OdezzaResult odr_scoring_create(const OdrStatic *fixed,const OdrScoringOptions *options,OdrScoringSession **out) {
    OdrScoringSession *s;
    size_t i;
    if(!out)return ODEZZA_ERROR_INVALID_ARGUMENT;
    *out=NULL;
    if(!fixed||!options||fixed->state_count!=options->shape.state_count||options->shape.fixed_rhs||options->shape.fixed_rhs_count)return ODEZZA_ERROR_INVALID_ARGUMENT;
    s=calloc(1,sizeof(*s));
    if(!s)return ODEZZA_ERROR_ALLOCATION;
    *out=s;
    s->options=*options;
    if(options->cache_directory) {
        if(strlen(options->cache_directory)>=sizeof(s->cache))return ODEZZA_ERROR_INVALID_ARGUMENT;
        strcpy(s->cache,options->cache_directory);
    }
    s->options.cache_directory=s->cache;
    if(!s->options.maximum_patch_capacity)s->options.maximum_patch_capacity=65536;
    if(!s->options.shape.shared_patch_capacity||s->options.shape.system_patch_capacity<2||s->options.shape.shared_patch_capacity>s->options.maximum_patch_capacity||s->options.shape.system_patch_capacity>s->options.maximum_patch_capacity)return ODEZZA_ERROR_INVALID_ARGUMENT;
    s->fixed=calloc(fixed->rhs_count?fixed->rhs_count:1,sizeof(*s->fixed));
    if(!s->fixed)return ODEZZA_ERROR_ALLOCATION;
    for(i=0;i<fixed->rhs_count;i++) {
        s->fixed[i].state_index=(uint8_t)fixed->rhs[i].state_index;
        s->fixed[i].program.bytes=fixed->rhs[i].program.bytes;
        s->fixed[i].program.byte_count=fixed->rhs[i].program.byte_count;
    }
    s->options.shape.fixed_rhs=s->fixed;
    s->options.shape.fixed_rhs_count=fixed->rhs_count;
    return prepare(s);
}

OdezzaResult odr_scoring_workspace(const OdrScoringSession *s,size_t *size,size_t *alignment) {
    if(!s||!s->pipeline)return ODEZZA_ERROR_INVALID_ARGUMENT;
    return odezza_scoring_pipeline_workspace_requirements(s->pipeline,size,alignment);
}

OdezzaResult odr_scoring_run(OdrScoringSession *s,const OdezzaScoringSystem *systems,size_t count,const OdezzaScoringLaunch *launch,void *workspace,size_t capacity,size_t *required,OdezzaScoringRunReport *report) {
    OdezzaResult r;
    size_t alignment;
    if(!s||!s->pipeline||!required||!report)return ODEZZA_ERROR_INVALID_ARGUMENT;
    memset(report,0,sizeof(*report));
    for(;;) {
        r=odr_scoring_workspace(s,required,&alignment);
        if(r)return r;
        if(!workspace)return ODEZZA_SUCCESS;
        if(capacity<*required)return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
        ++s->stats.run_attempts;
        r=odezza_scoring_pipeline_run(s->pipeline,systems,count,launch,workspace,capacity,report);
        if(r) {
            size_t n;
            (void)odezza_scoring_pipeline_write_error(s->pipeline,s->error,sizeof(s->error),&n);
        }
        else s->error[0]=0;
        if(r!=ODEZZA_ERROR_SPECIALIZATION_CAPACITY)return r;
        /* Native completion has fenced ordinary failures. Other modules may
         * already have run: preserve submitted-work accounting on retries. */
        s->stats.retry_requested_configurations+=report->configuration_count;
        if(!grow(s))return r;
        {
            OdezzaResult cleanup=odezza_scoring_pipeline_destroy(s->pipeline);
            if(cleanup==ODEZZA_ERROR_CUDA_UNFENCED)return cleanup;
        }
        s->pipeline=NULL;
        r=prepare(s);
        if(r)return r;
    }
}

OdezzaResult odr_scoring_stats(const OdrScoringSession *s,OdrScoringStats *out) {
    if(!s||!out)return ODEZZA_ERROR_INVALID_ARGUMENT;
    *out=s->stats;
    return ODEZZA_SUCCESS;
}

const char *odr_scoring_error(const OdrScoringSession *s) {
    return s?s->error:"invalid session";
}

OdezzaResult odr_scoring_destroy(OdrScoringSession *s) {
    OdezzaResult r=ODEZZA_SUCCESS;
    if(s) {
        if(s->pipeline)r=odezza_scoring_pipeline_destroy(s->pipeline);
        if(r==ODEZZA_ERROR_CUDA_UNFENCED)return r;
        free(s->fixed);
        odr_cache_close(s->artifacts);
        free(s);
    }
    return r;
}
