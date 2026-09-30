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
#include "odz_internal.h"
#include <assert.h>
#include <stdio.h>

int odz_fail(OdzJob *j,const char *format,...) {
    va_list args;
    if(!format)return 1;
    pthread_mutex_lock(&j->mutex);
    va_start(args,format);vsnprintf(j->error,sizeof(j->error),format,args);va_end(args);
    pthread_mutex_unlock(&j->mutex);
    return 1;
}
double odz_now(void) {return 0;}

typedef struct Race {OdzJob *job;pthread_barrier_t *barrier;void *p;} Race;
static void *race(void *arg) {
    Race *r=arg;
    pthread_barrier_wait(r->barrier);
    r->p=odz_host_alloc(r->job,48*1024*1024,ODZ_MEM_PIPELINE);
    pthread_barrier_wait(r->barrier);
    /* Both attempts finish before either reservation can be released. */
    odz_host_free(r->job,r->p);
    return NULL;
}
int main(void) {
    OdzJob j={0};HostMemory m;
    pthread_t threads[2];pthread_barrier_t barrier;Race races[2];
    unsigned char *p,*q;size_t i;
    pthread_mutex_init(&j.mutex,NULL);pthread_mutex_init(&j.memory_mutex,NULL);
    j.options.host_bytes=64*1024*1024;
    pthread_barrier_init(&barrier,NULL,2);
    for(i=0;i<2;i++) {races[i]=(Race){&j,&barrier,NULL};assert(!pthread_create(&threads[i],NULL,race,&races[i]));}
    for(i=0;i<2;i++)pthread_join(threads[i],NULL);
    m=odz_host_snapshot(&j);
    assert((races[0].p!=NULL)+(races[1].p!=NULL)==1);
    assert(m.total==0&&m.denied==1&&m.peak_total<=j.options.host_bytes);
    assert(m.peak[ODZ_MEM_PIPELINE]>48*1024*1024);
    /* Resize charges both live allocations; failure preserves the old data. */
    j.options.host_bytes=256;
    p=odz_host_alloc(&j,96,ODZ_MEM_RETAINED);assert(p);memset(p,0x5a,96);
    assert(!odz_host_resize(&j,p,160,ODZ_MEM_RETAINED));
    for(i=0;i<96;i++)assert(p[i]==0x5a);
    q=odz_host_resize(&j,p,64,ODZ_MEM_RETAINED);assert(q);
    for(i=0;i<64;i++)assert(q[i]==0x5a);
    odz_host_free(&j,q);assert(odz_host_snapshot(&j).total==0);
    assert(!odz_host_alloc(&j,SIZE_MAX,ODZ_MEM_TILE));
    assert(!odz_host_claim(&j,256,ODZ_MEM_TILE));
    assert(odz_host_claim(&j,1,ODZ_MEM_ARENA));
    odz_host_release(&j,256,ODZ_MEM_TILE);
    assert(odz_host_snapshot(&j).total==0);
    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&j.memory_mutex);pthread_mutex_destroy(&j.mutex);
    puts("shared reservations, concurrent rejection, resize preservation, overflow and release pass");
    return 0;
}
