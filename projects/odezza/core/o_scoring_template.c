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
#include "o_odezza_internal.h"
#include "o_sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Explicit LE fields; no native pointers, padding or inspection layout on disk. */
#define HEADER 104u
static void put32(unsigned char *p, uint32_t x) {
    unsigned i; for (i=0; i<4; ++i) p[i]=(unsigned char)(x>>(8*i));
}
static uint32_t get32(const unsigned char *p) {
    return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;
}
static void hash(const void *p, size_t size, unsigned char out[32]) {
    OSha256 s; (void)o_sha256_init(&s); (void)o_sha256_update(&s,p,size); (void)o_sha256_final(&s,out);
}
static void shape_bytes(const OdezzaScoringTemplateInfo *i, unsigned char p[24]) {
    put32(p,i->sm_version); put32(p+4,i->state_capacity); put32(p+8,i->constant_capacity);
    put32(p+12,i->system_capacity); put32(p+16,i->shared_patch_capacity); put32(p+20,i->system_patch_capacity);
}
static OdezzaResult fail(OdezzaScoringTemplate *t, OdezzaResult r, const char *message) {
    if (t && !t->error[0]) (void)snprintf(t->error,sizeof(t->error),"%s (result %d)",message,(int)r);
    return r;
}
static OdezzaResult prepare(const OdezzaScoringTemplateInfo *info, OdezzaScoringTemplate **out) {
    OdezzaScoringTemplate *t;
    OdezzaResult r;
    if (!out) return ODEZZA_ERROR_INVALID_ARGUMENT;
    *out=NULL;
    t=calloc(1,sizeof(*t)); if (!t) return ODEZZA_ERROR_ALLOCATION;
    *out=t;
    if (!info || !info->state_capacity || info->state_capacity>128u ||
        !info->system_capacity || !info->shared_patch_capacity || !info->system_patch_capacity)
        return fail(t,ODEZZA_ERROR_INVALID_ARGUMENT,"invalid scoring template shape");
    if (info->sm_version!=89 && info->sm_version!=90 && info->sm_version!=120)
        return fail(t,ODEZZA_ERROR_UNSUPPORTED,"unsupported scoring architecture");
    t->info=*info;
    r=odezza_generate_scoring_cuda(info->state_capacity,info->constant_capacity,info->system_capacity,
        info->shared_patch_capacity,info->system_patch_capacity,NULL,0,&t->source_size);
    if (r || t->source_size==SIZE_MAX) return fail(t,r?r:ODEZZA_ERROR_OVERFLOW,"template source measurement failed");
    t->source=malloc(t->source_size+1);
    if (!t->source) return fail(t,ODEZZA_ERROR_ALLOCATION,"template source allocation failed");
    r=odezza_generate_scoring_cuda(info->state_capacity,info->constant_capacity,info->system_capacity,
        info->shared_patch_capacity,info->system_patch_capacity,t->source,t->source_size+1,&t->source_size);
    return r?fail(t,r,"template source generation failed"):r;
}
static OdezzaResult inspect(OdezzaScoringTemplate *t) {
    size_t size=0; OdezzaResult r; int patch_matches;
    r=odezza_inspect_scoring_cubin(t->cubin,t->cubin_size,NULL,0,&size,NULL);
    if (r) return fail(t,r,"template inspection measurement failed");
    t->arena=malloc(size?size:1);
    if (!t->arena) return fail(t,ODEZZA_ERROR_ALLOCATION,"template inspection allocation failed");
    r=odezza_inspect_scoring_cubin(t->cubin,t->cubin_size,t->arena,size,&size,&t->inspection);
    if (r) return fail(t,r,"template inspection failed");
    /* With one system, divisibility cannot distinguish an elided final branch.
     * Accept that one-instruction physical shortfall, but leave the inspected
     * capacity unchanged: specialization must honor the smaller real arena. */
    patch_matches=t->inspection->system_patch_capacity==t->info.system_patch_capacity ||
        (t->info.system_capacity==1 && t->inspection->system_capacity==1 &&
         t->info.system_patch_capacity>0 &&
         t->inspection->system_patch_capacity==t->info.system_patch_capacity-1u);
    if (t->inspection->architecture!=t->info.sm_version ||
        t->inspection->state_capacity!=t->info.state_capacity ||
        t->inspection->constant_capacity!=t->info.constant_capacity ||
        t->inspection->system_capacity!=t->info.system_capacity ||
        !patch_matches)
        return fail(t,ODEZZA_ERROR_FORMAT,"template inspection shape mismatch");
    t->valid=1; return ODEZZA_SUCCESS;
}
OdezzaResult odezza_scoring_template_create(const OdezzaScoringTemplateInfo *info, OdezzaScoringTemplate **out) {
    OdezzaResult r=prepare(info,out), compiled=ODEZZA_ERROR_COMPILER;
    OdezzaNvrtcCompilation *c=NULL; OdezzaScoringTemplate *t;
    char arch[64]; const char *options[2];
    if (r) return r;
    t=*out; (void)snprintf(arch,sizeof(arch),"--gpu-architecture=sm_%u",info->sm_version);
    options[0]="--std=c++17"; options[1]=arch;
    r=odezza_nvrtc_compilation_create(t->source,"odezza_scoring.cu",options,2,&c);
    if (c) (void)o_nvrtc_compilation_seconds(c, &t->nvrtc_seconds);
    if (!r) r=odezza_nvrtc_compilation_result(c,&compiled);
    if (!r && compiled) r=compiled;
    if (!r) r=odezza_nvrtc_compilation_cubin_size(c,&t->cubin_size);
    if (!r) {
        t->cubin=malloc(t->cubin_size);
        if (!t->cubin) r=ODEZZA_ERROR_ALLOCATION;
    }
    if (!r) r=odezza_nvrtc_compilation_write_cubin(c,t->cubin,t->cubin_size);
    if (r && c) {
        size_t size=0;
        if (odezza_nvrtc_compilation_log_size(c,&size)==ODEZZA_SUCCESS && size) {
            char *log=malloc(size);
            if (log && odezza_nvrtc_compilation_write_log(c,log,size)==ODEZZA_SUCCESS)
                (void)snprintf(t->error,sizeof(t->error),"NVRTC failed (result %d): %.900s",(int)r,log);
            free(log);
        }
    }
    if (c) (void)odezza_nvrtc_compilation_destroy(c);
    return r?fail(t,r,"template compilation failed"):inspect(t);
}
OdezzaResult odezza_scoring_template_write(const OdezzaScoringTemplate *t, void *buffer, size_t capacity, size_t *size_ret) {
    unsigned char *p=buffer;
    if (!t || !size_ret) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (!t->valid) return ODEZZA_ERROR_BUSY;
    if (t->cubin_size>SIZE_MAX-HEADER) return ODEZZA_ERROR_OVERFLOW;
    *size_ret=HEADER+t->cubin_size;
    if (!p) return ODEZZA_SUCCESS;
    if (capacity<*size_ret) return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
    memcpy(p,"ODEZT01\0",8); shape_bytes(&t->info,p+8);
    put32(p+32,(uint32_t)t->cubin_size); put32(p+36,(uint32_t)((uint64_t)t->cubin_size>>32));
    hash(t->source,t->source_size,p+40); hash(t->cubin,t->cubin_size,p+72);
    memcpy(p+HEADER,t->cubin,t->cubin_size); return ODEZZA_SUCCESS;
}
OdezzaResult odezza_scoring_template_nvrtc_seconds(const OdezzaScoringTemplate *t, double *seconds_ret) {
    if (!t || !seconds_ret) return ODEZZA_ERROR_INVALID_ARGUMENT;
    if (!t->valid) return ODEZZA_ERROR_BUSY;
    *seconds_ret = t->nvrtc_seconds;
    return ODEZZA_SUCCESS;
}
OdezzaResult odezza_scoring_template_read(const OdezzaScoringTemplateInfo *info,
    const void *buffer, size_t size, OdezzaScoringTemplate **out) {
    const unsigned char *p=buffer; unsigned char shape[24],digest[32];
    OdezzaResult r=prepare(info,out); OdezzaScoringTemplate *t; uint64_t bytes;
    if (r) return r;
    t=*out;
    if (!p || size<HEADER || memcmp(p,"ODEZT01\0",8)) return fail(t,ODEZZA_ERROR_FORMAT,"invalid template artifact header");
    shape_bytes(info,shape); bytes=(uint64_t)get32(p+32)|(uint64_t)get32(p+36)<<32;
    if (memcmp(shape,p+8,24) || bytes!=(uint64_t)(size-HEADER))
        return fail(t,ODEZZA_ERROR_FORMAT,"template artifact shape or size mismatch");
    hash(t->source,t->source_size,digest);
    if (memcmp(digest,p+40,32)) return fail(t,ODEZZA_ERROR_FORMAT,"template generator identity mismatch");
    hash(p+HEADER,size-HEADER,digest);
    if (memcmp(digest,p+72,32)) return fail(t,ODEZZA_ERROR_FORMAT,"template artifact checksum mismatch");
    t->cubin_size=size-HEADER; t->cubin=malloc(t->cubin_size?t->cubin_size:1);
    if (!t->cubin) return fail(t,ODEZZA_ERROR_ALLOCATION,"template artifact allocation failed");
    memcpy(t->cubin,p+HEADER,t->cubin_size);
    return inspect(t);
}
OdezzaResult odezza_scoring_template_write_error(const OdezzaScoringTemplate *t,
    char *buffer, size_t capacity, size_t *size_ret) {
    size_t n;
    if (!t || !size_ret) return ODEZZA_ERROR_INVALID_ARGUMENT;
    n=strlen(t->error); *size_ret=n;
    if (!buffer) return ODEZZA_SUCCESS;
    if (capacity<=n) return ODEZZA_ERROR_INSUFFICIENT_BUFFER;
    memcpy(buffer,t->error,n+1); return ODEZZA_SUCCESS;
}
OdezzaResult odezza_scoring_template_destroy(OdezzaScoringTemplate *t) {
    if (t) { free(t->source); free(t->cubin); free(t->arena); free(t); }
    return ODEZZA_SUCCESS;
}
