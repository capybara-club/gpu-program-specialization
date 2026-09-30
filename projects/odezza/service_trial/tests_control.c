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
/* Exercise the actual worker control loop with gated transport calls. CUDA is
 * linked for the normal worker entry point but no context/kernel is created. */
#define main unused_worker_main
#define t_rpc_timeout gated_rpc
#define natsConnection_Publish checked_publish
#include "worker.c"
#undef main
#undef t_rpc_timeout
#undef natsConnection_Publish
#include "odz_internal.h"
#include <assert.h>
static pthread_mutex_t gate_lock=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gate_condition=PTHREAD_COND_INITIALIZER;
static int entered,released,timeout_reply,published;
static const char *answer;
static Worker *current;
int gated_rpc(natsConnection *nc,const char *operation,const void *data,size_t size,natsMsg **reply,int64_t timeout) {
    (void)nc;(void)operation;(void)data;(void)size;assert(timeout==250);
    pthread_mutex_lock(&gate_lock);entered=1;pthread_cond_broadcast(&gate_condition);
    while(!released)pthread_cond_wait(&gate_condition,&gate_lock);
    *reply=NULL;
    if(!timeout_reply)assert(natsMsg_Create(reply,"reply",NULL,answer,(int)strlen(answer))==NATS_OK);
    pthread_mutex_unlock(&gate_lock);
    return timeout_reply;
}
natsStatus checked_publish(natsConnection *nc,const char *subject,const void *data,int size) {
    (void)nc;assert(!strcmp(subject,"$JS.ACK.test"));assert(size==4&&!memcmp(data,"+WPI",4));
    assert(!pthread_mutex_trylock(&current->lock));pthread_mutex_unlock(&current->lock);
    ++published;return NATS_OK;
}
static void initialize(Worker *w) {
    memset(w,0,sizeof(*w));pthread_mutex_init(&w->lock,NULL);
    w->active=w->connected=1;w->lease_deadline=t_now()+T_WORKER_LEASE;
    memset(w->generation,'a',32);memset(w->id,'b',32);memset(w->attempt,'c',32);
    strcpy(w->snapshot,"{}");
    assert(!odz_job_create("{}",2,&w->job));
    assert(natsMsg_Create(&w->delivery,"work","$JS.ACK.test",w->id,32)==NATS_OK);
    current=w;entered=released=timeout_reply=published=0;answer="OK\ncontinue";
}
static void finish(Worker *w,pthread_t thread) {
    pthread_mutex_lock(&gate_lock);released=1;pthread_cond_broadcast(&gate_condition);pthread_mutex_unlock(&gate_lock);
    pthread_join(thread,NULL);
    odz_job_destroy(w->job);natsMsg_Destroy(w->delivery);pthread_mutex_destroy(&w->lock);
}
int main(void) {
    int mode;double worst=0;
    assert(nats_Open(-1)==NATS_OK);
    for(mode=0;mode<4;mode++) {
        Worker w;pthread_t thread;double start,elapsed;OdzJob *job;
        initialize(&w);
        if(mode==0||mode==1)answer="OK\ncancel";
        if(mode==3)timeout_reply=1;
        assert(!pthread_create(&thread,NULL,control,&w));
        pthread_mutex_lock(&gate_lock);
        while(!entered)pthread_cond_wait(&gate_condition,&gate_lock);
        pthread_mutex_unlock(&gate_lock);
        start=t_now();assert(!pthread_mutex_trylock(&w.lock));elapsed=t_now()-start;
        if(elapsed>worst)worst=elapsed;
        /* Simulate real finalization while a heartbeat response is pending. */
        job=w.job;
        if(mode==0) {
            odz_job_destroy(w.job);w.job=NULL;
            assert(!odz_job_create("{}",2,&w.job));job=w.job;
            memset(w.id,'d',32);memset(w.attempt,'e',32);
        }
        natsMsg_Destroy(w.delivery);w.delivery=NULL;
        if(mode==2||mode==3)w.lease_deadline=t_now()-1;
        w.quit=1;
        pthread_mutex_unlock(&w.lock);
        pthread_mutex_lock(&gate_lock);released=1;pthread_cond_broadcast(&gate_condition);pthread_mutex_unlock(&gate_lock);
        pthread_join(thread,NULL);
        assert(job->cancel_signal==(mode!=0));
        assert(w.retire==(mode==2||mode==3));
        assert(!published); /* stale, cancelled, expired, or timed-out replies */
        odz_job_destroy(w.job);pthread_mutex_destroy(&w.lock);
    }
    {
        Worker w;pthread_t thread;initialize(&w);
        assert(!pthread_create(&thread,NULL,control,&w));
        pthread_mutex_lock(&gate_lock);while(!entered)pthread_cond_wait(&gate_condition,&gate_lock);pthread_mutex_unlock(&gate_lock);
        pthread_mutex_lock(&w.lock);w.quit=1;natsMsg_Destroy(w.delivery);w.delivery=NULL;pthread_mutex_unlock(&w.lock);
        finish(&w,thread);assert(published==1);
    }
    printf("{\"passed\":true,\"cases\":5,\"largest_attachment_lock_seconds\":%.9g,\"transport\":\"gated mock; actual worker control and runtime cancellation\"}\n",worst);
    assert(nats_CloseAndWait(5000)==NATS_OK);
    return 0;
}
