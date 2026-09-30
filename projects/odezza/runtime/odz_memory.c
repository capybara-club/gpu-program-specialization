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
#include <sys/resource.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

/* Aligned headers charge actual requested storage, including our metadata.
 * This is a shared job reservation, never a per-device copy of the budget.
 * Opaque compiler/driver/allocator overhead is measured separately as RSS. */
typedef union HostHeader {
    struct { size_t bytes; unsigned kind; } data;
    long double alignment;
    void *pointer;
} HostHeader;

int odz_host_claim(OdzJob *j,size_t bytes,unsigned kind) {
    HostMemory *m=&j->memory;
    int failed;
    if(kind>=ODZ_MEM_CLASSES)return odz_fail(j,"invalid memory class");
    pthread_mutex_lock(&j->memory_mutex);
    failed=m->total>j->options.host_bytes||bytes>j->options.host_bytes-m->total;
    if(failed)++m->denied;
    else {
        m->total+=bytes;m->live[kind]+=bytes;
        if(m->total>m->peak_total)m->peak_total=m->total;
        if(m->live[kind]>m->peak[kind])m->peak[kind]=m->live[kind];
    }
    pthread_mutex_unlock(&j->memory_mutex);
    return failed?odz_fail(j,"memory budget: class %u needs %zu bytes; max_host_bytes=%llu",kind,bytes,(unsigned long long)j->options.host_bytes):0;
}

void odz_host_release(OdzJob *j,size_t bytes,unsigned kind) {
    pthread_mutex_lock(&j->memory_mutex);
    /* Internal ownership invariant: borrowed claims and owned allocations must
     * each be released exactly once, after all users have completed. */
    if(kind>=ODZ_MEM_CLASSES||bytes>j->memory.live[kind])abort();
    j->memory.live[kind]-=bytes;j->memory.total-=bytes;
    pthread_mutex_unlock(&j->memory_mutex);
}

HostMemory odz_host_snapshot(OdzJob *j) {
    HostMemory m;
    pthread_mutex_lock(&j->memory_mutex);m=j->memory;pthread_mutex_unlock(&j->memory_mutex);
    return m;
}

void *odz_host_alloc(OdzJob *j,size_t bytes,unsigned kind) {
    HostHeader *h;
    if(bytes>SIZE_MAX-sizeof(*h)) {odz_fail(j,"host allocation size overflow");return NULL;}
    if(odz_host_claim(j,bytes+sizeof(*h),kind))return NULL;
    h=calloc(1,bytes+sizeof(*h));
    if(!h) {
        odz_host_release(j,bytes+sizeof(*h),kind);
        odz_fail(j,"host allocation failed (%zu bytes)",bytes);
        return NULL;
    }
    h->data.bytes=bytes;h->data.kind=kind;
    return h+1;
}

void *odz_alloc(OdzJob *j,size_t bytes) {return odz_host_alloc(j,bytes,ODZ_MEM_ARENA);}

void odz_host_free(OdzJob *j,void *ptr) {
    HostHeader *h;
    size_t bytes;unsigned kind;
    if(!ptr)return;
    h=(HostHeader *)ptr-1;bytes=h->data.bytes+sizeof(*h);kind=h->data.kind;
    free(h);
    odz_host_release(j,bytes,kind);
}

void *odz_host_resize(OdzJob *j,void *ptr,size_t bytes,unsigned kind) {
    void *p=odz_host_alloc(j,bytes,kind);
    if(!p)return NULL;
    if(ptr) {
        HostHeader *h=(HostHeader *)ptr-1;
        memcpy(p,ptr,bytes<h->data.bytes?bytes:h->data.bytes);
        odz_host_free(j,ptr);
    }
    return p;
}

/* Sampled device-wide usage includes other processes and must not be summed
 * with our explicit allocation count. Each context-owning worker samples itself.
 * No CUDA stream synchronization is introduced for telemetry. */
void odz_memory_sample(OdzRuntime *r,OdzJob *j,int force) {
    size_t available=0,total=0;
    uint64_t rss=0,peak=0;
    int valid=0,gpu_valid=0;
    double now=odz_now();
    struct rusage usage;
    if(!force&&now-r->last_memory_sample<0.25)return;
    r->last_memory_sample=now;
#ifdef __linux__
    {
        FILE *f=fopen("/proc/self/statm","r");
        unsigned long pages,resident;
        long page_size=sysconf(_SC_PAGESIZE);
        if(f) {
            if(fscanf(f,"%lu %lu",&pages,&resident)==2&&page_size>0) {
                rss=(uint64_t)resident*(uint64_t)page_size;valid=1;
            }
            fclose(f);
        }
    }
#elif defined(__APPLE__)
    {
        mach_task_basic_info_data_t info;
        mach_msg_type_number_t count=MACH_TASK_BASIC_INFO_COUNT;
        if(task_info(mach_task_self(),MACH_TASK_BASIC_INFO,(task_info_t)&info,&count)==KERN_SUCCESS) {
            rss=info.resident_size;valid=1;
        }
    }
#endif
    if(!getrusage(RUSAGE_SELF,&usage)) {
        peak=(uint64_t)usage.ru_maxrss;
#ifndef __APPLE__
        peak*=1024;
#endif
    }
    if(cuMemGetInfo(&available,&total)==CUDA_SUCCESS&&available<=total)gpu_valid=1;
    pthread_mutex_lock(&j->mutex);
    {
        DeviceStats *d=&j->devices[r->worker_index];
        d->allocated_bytes=r->allocated_bytes;
        d->peak_allocated_bytes=r->peak_allocated_bytes;
        d->numeric_pool_bytes=r->pool_bytes;
        d->peak_numeric_pool_bytes=r->peak_pool_bytes;
        if(gpu_valid) {
            d->device_sample_valid=1;d->device_total_bytes=total;d->device_used_bytes=total-available;
            if(d->device_used_bytes>d->peak_device_used_bytes)d->peak_device_used_bytes=d->device_used_bytes;
        }
        if(valid) {j->rss_valid=1;j->rss_bytes=rss;if(rss>j->peak_sampled_rss_bytes)j->peak_sampled_rss_bytes=rss;}
        if(peak>j->process_peak_rss_bytes)j->process_peak_rss_bytes=peak;
    }
    pthread_mutex_unlock(&j->mutex);
}
