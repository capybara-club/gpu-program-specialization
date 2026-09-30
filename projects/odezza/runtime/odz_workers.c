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

struct OdzWorker {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    unsigned device;
    size_t index;
    const char *cache;
    OdzRuntime *runtime;
    OdzJob *job;
    int initialized,failed,quit,started;
    int reserve_pending,reserve_result;
    size_t reserve_trajectory,reserve_device,reserve_host;
};

int odz_runtime_reserve_buffers(OdzRuntime *r,size_t trajectories,size_t device,size_t host) {
    size_t i;
    if(!r||!trajectories||!device||!host||r->job_epoch||r->pooled_buffers||r->reservation_failed||r->fatal)return 1;
    if(odz_reserve_device_buffers(r,trajectories,device,host))return 1;
    for(i=1;i<r->device_count;i++) {
        OdzWorker *w=&r->workers[i-1];
        pthread_mutex_lock(&w->mutex);
        w->reserve_trajectory=trajectories;w->reserve_device=device;w->reserve_host=host;
        w->reserve_pending=1;
        pthread_cond_broadcast(&w->condition);
        while(w->reserve_pending)pthread_cond_wait(&w->condition,&w->mutex);
        if(w->reserve_result) {
            r->reservation_failed=1;
            snprintf(r->error,sizeof(r->error),"device %u pool reservation: %s",w->device,odz_runtime_error(w->runtime));
        }
        pthread_mutex_unlock(&w->mutex);
        if(r->reservation_failed)return 1;
    }
    return 0;
}

int odz_runtime_attempt_inputs(OdzRuntime *r) {
    size_t i;
    if(!r||r->job_epoch||r->fatal)return 1;
    r->attempt_inputs=1;
    for(i=1;i<r->device_count;i++) {
        OdzWorker *w=&r->workers[i-1];
        pthread_mutex_lock(&w->mutex);
        if(!w->initialized||w->failed||w->job) {
            pthread_mutex_unlock(&w->mutex);
            return 1;
        }
        w->runtime->attempt_inputs=1;
        pthread_mutex_unlock(&w->mutex);
    }
    return 0;
}

static void *worker_main(void *argument) {
    OdzWorker *w=argument;
    int failed=odz_runtime_create(w->device,w->cache,&w->runtime);
    pthread_mutex_lock(&w->mutex);
    w->failed=failed;
    w->initialized=1;
    if(w->runtime)w->runtime->worker_index=w->index;
    pthread_cond_broadcast(&w->condition);
    for(;;) {
        OdzJob *j;
        while(!w->job&&!w->quit&&!w->reserve_pending)pthread_cond_wait(&w->condition,&w->mutex);
        if(w->quit)break;
        if(w->reserve_pending) {
            pthread_mutex_unlock(&w->mutex);
            failed=odz_reserve_device_buffers(w->runtime,w->reserve_trajectory,w->reserve_device,w->reserve_host);
            pthread_mutex_lock(&w->mutex);
            w->reserve_result=failed;w->reserve_pending=0;
            pthread_cond_broadcast(&w->condition);
            continue;
        }
        j=w->job;
        pthread_mutex_unlock(&w->mutex);
        (void)odz_device_run(w->runtime,j);
        pthread_mutex_lock(&w->mutex);
        w->job=NULL;
        pthread_cond_broadcast(&w->condition);
    }
    pthread_mutex_unlock(&w->mutex);
    odz_runtime_destroy(w->runtime);
    return NULL;
}

int odz_runtime_create_devices(const unsigned *devices,size_t count,const char *cache,OdzRuntime **out) {
    OdzRuntime *r;
    size_t i,k;
    if(!out)return 1;
    *out=NULL;
    if(!devices||!count||count>ODZ_DEVICES)return 1;
    for(i=0;i<count;i++)for(k=0;k<i;k++)if(devices[i]==devices[k])return 1;
    if(odz_runtime_create(devices[0],cache,out))return 1;
    r=*out;
    if(count==1)return 0;
    r->workers=calloc(count-1,sizeof(*r->workers));
    if(!r->workers)return 1;
    for(i=1;i<count;i++) {
        OdzWorker *w=&r->workers[i-1];
        w->device=devices[i];
        w->index=i;
        w->cache=r->cache;
        pthread_mutex_init(&w->mutex,NULL);
        pthread_cond_init(&w->condition,NULL);
        r->device_count=i+1;
        if(pthread_create(&w->thread,NULL,worker_main,w)) {
            snprintf(r->error,sizeof(r->error),"device worker creation failed");
            return 1;
        }
        w->started=1;
    }
    for(i=1;i<count;i++) {
        OdzWorker *w=&r->workers[i-1];
        pthread_mutex_lock(&w->mutex);
        while(!w->initialized)pthread_cond_wait(&w->condition,&w->mutex);
        if(w->failed)snprintf(r->error,sizeof(r->error),"device %u initialization: %s",w->device,odz_runtime_error(w->runtime));
        pthread_mutex_unlock(&w->mutex);
        if(w->failed)return 1;
    }
    return 0;
}

int odz_workers_start(OdzRuntime *r,OdzJob *j) {
    size_t i;
    for(i=1;i<r->device_count;i++) {
        OdzWorker *w=&r->workers[i-1];
        pthread_mutex_lock(&j->mutex);
        j->devices[i].device=w->device;
        j->auxiliary_nvrtc_seconds+=w->runtime->auxiliary_nvrtc_seconds;
        j->reducer_setup_seconds+=w->runtime->reducer_setup_seconds;
        pthread_mutex_unlock(&j->mutex);
        pthread_mutex_lock(&w->mutex);
        w->job=j;
        pthread_cond_broadcast(&w->condition);
        pthread_mutex_unlock(&w->mutex);
    }
    return 0;
}

void odz_workers_wait(OdzRuntime *r,OdzJob *j) {
    size_t i;
    (void)j;
    for(i=1;i<r->device_count;i++) {
        OdzWorker *w=&r->workers[i-1];
        pthread_mutex_lock(&w->mutex);
        while(w->job)pthread_cond_wait(&w->condition,&w->mutex);
        pthread_mutex_unlock(&w->mutex);
    }
}

void odz_workers_close(OdzRuntime *r) {
    size_t i;
    if(!r->workers)return;
    for(i=1;i<r->device_count;i++) {
        OdzWorker *w=&r->workers[i-1];
        if(w->started) {
            pthread_mutex_lock(&w->mutex);
            w->quit=1;
            pthread_cond_broadcast(&w->condition);
            pthread_mutex_unlock(&w->mutex);
            pthread_join(w->thread,NULL);
        }
        pthread_cond_destroy(&w->condition);
        pthread_mutex_destroy(&w->mutex);
    }
    free(r->workers);
    r->workers=NULL;
}
