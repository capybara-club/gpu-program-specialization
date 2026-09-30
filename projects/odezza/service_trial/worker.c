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
#include "trial.h"
#include "odezza_runtime.h"
#include "odr_internal.h"
#include <signal.h>
typedef struct Worker {
    pthread_mutex_t lock;
    natsConnection *nc;
    OdzJob *job;
    natsMsg *delivery;
    char generation[33],id[33],attempt[33],name[33],snapshot[T_STATUS];
    int active,quit,retire,connected;
    double lease_deadline,cancelling;
} Worker;
static volatile sig_atomic_t stopping;
static void stop(int sig) {(void)sig;stopping=1;}
static size_t pool_bytes(const char *key,size_t fallback) {
    const char *s=getenv(key);char *end;unsigned long long mib;
    if(!s)return fallback*1024*1024;
    mib=strtoull(s,&end,10);
    if(end==s||*end||mib<1||mib>8192) {fprintf(stderr,"%s must be an integer MiB count in 1..8192\n",key);exit(2);}
    return (size_t)mib*1024*1024;
}
static int string_field(const char *p,size_t n,const char *key,char *out,size_t cap) {
    OdrJson root={p,n},v;size_t size;
    return !jget(root,key,&v)||!jstring(v,out,cap,&size);
}
static int call(Worker *w,const char *verb,size_t offset,int has_offset,const void *data,size_t n,natsMsg **m) {
    char op[220];
    if(has_offset)snprintf(op,sizeof(op),"%s.%s.%s.%s.%zu",verb,w->generation,w->id,w->attempt,offset);
    else snprintf(op,sizeof(op),"%s.%s.%s.%s",verb,w->generation,w->id,w->attempt);
    return t_rpc(w->nc,op,data,n,m);
}
/* Everything captured for network I/O is owned by this snapshot. No job or
 * delivery pointer survives an unlock. Responses are applied only to the same
 * attempt, while its local lease remains valid. */
typedef struct Heartbeat {
    char id[33],attempt[33],operation[220],snapshot[T_STATUS];
    char progress_subject[512];
} Heartbeat;
static void connection_changed(natsConnection *nc,void *arg) {
    Worker *w=arg;
    pthread_mutex_lock(&w->lock);
    /* Callback delivery may lag an actual reconnect; query current state. */
    w->connected=natsConnection_Status(nc)==NATS_CONN_STATUS_CONNECTED;
    pthread_mutex_unlock(&w->lock);
}
static void cancel_locked(Worker *w,double now,int retire) {
    if(retire)w->retire=1;
    if(!w->cancelling)w->cancelling=now;
    if(w->job)odz_job_cancel(w->job);
}
static void *control(void *arg) {
    Worker *w=arg;struct timespec tick={0,250000000};
    for(;;) {
        Heartbeat beat; natsMsg *reply=NULL;
        double sent,now;int send=0,renewed=0;
        memset(&beat,0,sizeof(beat));
        pthread_mutex_lock(&w->lock);
        if(w->quit) {pthread_mutex_unlock(&w->lock);break;}
        now=t_now();
        if(w->active) {
            if(now>=w->lease_deadline)cancel_locked(w,now,1);
            if(w->cancelling&&now-w->cancelling>T_CANCEL_GRACE) {
                fputs("worker retired after cancellation grace\n",stderr);_exit(3);
            }
            if(!w->retire&&w->connected) {
                if(w->job) {size_t size;odz_job_status(w->job,w->snapshot,sizeof(w->snapshot),&size);}
                memcpy(beat.id,w->id,sizeof(beat.id));memcpy(beat.attempt,w->attempt,sizeof(beat.attempt));
                memcpy(beat.snapshot,w->snapshot,sizeof(beat.snapshot));
                snprintf(beat.operation,sizeof(beat.operation),"beat.%s.%s.%s",w->generation,beat.id,beat.attempt);
                if(w->delivery) {
                    const char *subject=natsMsg_GetReply(w->delivery);
                    if(!subject||strlen(subject)>=sizeof(beat.progress_subject))cancel_locked(w,now,1);
                    else strcpy(beat.progress_subject,subject);
                }
                send=!w->retire;
            }
        }
        pthread_mutex_unlock(&w->lock);
        if(send) {
            sent=t_now();
            (void)t_rpc_timeout(w->nc,beat.operation,beat.snapshot,strlen(beat.snapshot),&reply,250);
            now=t_now();
            pthread_mutex_lock(&w->lock);
            if(w->active&&!strcmp(w->id,beat.id)&&!strcmp(w->attempt,beat.attempt)) {
                if(now>=w->lease_deadline)cancel_locked(w,now,1);
                else if(!w->retire&&reply) {
                    if(t_ok(reply)&&t_size(reply)==8&&!memcmp(t_body(reply),"continue",8)) {
                        /* Start at send time, never grant fresh time to an old reply. */
                        w->lease_deadline=sent+T_WORKER_LEASE;renewed=1;
                    } else if(t_ok(reply)&&t_size(reply)==6&&!memcmp(t_body(reply),"cancel",6)) {
                        cancel_locked(w,now,0);
                        w->lease_deadline=sent+T_WORKER_LEASE;
                    } else cancel_locked(w,now,1);
                }
            }
            pthread_mutex_unlock(&w->lock);
        }
        /* Both network operations are outside the execution/lifetime lock. */
        if(renewed&&beat.progress_subject[0])
            (void)natsConnection_Publish(w->nc,beat.progress_subject,"+WPI",4);
        natsMsg_Destroy(reply);
        nanosleep(&tick,NULL);
    }
    return NULL;
}

int main(int argc,char **argv) {
    Worker w;OdzRuntime *runtime=NULL;jsCtx *js=NULL;natsSubscription *sub=NULL;jsSubOptions opts;jsErrCode ec=0;
    char stream[64],subject[128],op[220],host[128],info[256],*request,*report;pthread_t thread;natsMsg *health=NULL;
    unsigned device,devices[8];size_t device_count=0;char *end,*start;int exit_code=0;
    size_t trajectory_bytes=pool_bytes("ODEZZA_TRAJECTORY_POOL_MIB",64);
    size_t tile_bytes=pool_bytes("ODEZZA_TILE_DEVICE_POOL_MIB",256);
    size_t result_bytes=pool_bytes("ODEZZA_TILE_HOST_POOL_MIB",64);
    size_t host_limit=pool_bytes("ODEZZA_JOB_HOST_LIMIT_MIB",2048);
    void *trajectory_arena;
    if(argc!=4) {fputs("usage: worker NATS_URL DEVICE CACHE_DIRECTORY\n",stderr);return 2;}
    start=argv[2];
    do {
        unsigned long d=strtoul(start,&end,10);
        if(end==start||d>7||device_count==8||(*end&&*end!=','))return 2;
        devices[device_count++]=(unsigned)d;
        if(!*end)break;
        start=end+1;
    } while(1);
    device=devices[0];
    if(gethostname(host,sizeof(host)))return 1;
    host[sizeof(host)-1]=0;
    if(strspn(host,"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_")!=strlen(host))return 1;
    if(snprintf(info,sizeof(info),"{\"host\":\"%s\",\"devices\":[%s],\"pid\":%ld,\"host_limit_bytes\":%zu}",host,argv[2],(long)getpid(),host_limit)>=(int)sizeof(info))return 2;
    memset(&w,0,sizeof(w));pthread_mutex_init(&w.lock,NULL);
    request=malloc(T_BYTES+1);report=malloc(T_BYTES+1);if(!request||!report)return 1;
    trajectory_arena=malloc(trajectory_bytes);if(!trajectory_arena)return 1;
    if(t_random(w.name)||t_connect_worker(&w.nc,argv[1],connection_changed,&w)||t_rpc(w.nc,"health",NULL,0,&health)||!t_ok(health))return 1;
    if(string_field(t_body(health),t_size(health),"generation",w.generation,sizeof(w.generation))||
       string_field(t_body(health),t_size(health),"stream",stream,sizeof(stream))||
       string_field(t_body(health),t_size(health),"subject",subject,sizeof(subject)))return 1;
    natsMsg_Destroy(health);
    connection_changed(w.nc,&w);
    if(odz_runtime_create_devices(devices,device_count,argv[3],&runtime)||odz_runtime_attempt_inputs(runtime)||odz_runtime_host_limit(runtime,host_limit)||
       odz_runtime_reserve_buffers(runtime,trajectory_bytes,tile_bytes,result_bytes)) {
        fprintf(stderr,"runtime startup: %s\n",odz_runtime_error(runtime));return 1;
    }
    if(natsConnection_JetStream(&js,w.nc,NULL))return 1;
    jsSubOptions_Init(&opts);opts.Stream=stream;opts.Consumer="executors";
    if(js_PullSubscribe(&sub,js,subject,"executors",NULL,&opts,&ec)) {fprintf(stderr,"pull subscription: %d\n",ec);return 1;}
    signal(SIGTERM,stop);signal(SIGINT,stop);
    if(pthread_create(&thread,NULL,control,&w))return 1;
    printf("ready device=%u worker=%s generation=%s\n",device,w.name,w.generation);fflush(stdout);
    while(!stopping&&!natsConnection_IsClosed(w.nc)) {
        natsMsgList list={0};natsMsg *reply=NULL;size_t bytes=0,n=0,required=0;int failed=0,quarantined=0,committed=0;double claimed_at;
        natsStatus fetch;
        if(natsConnection_Status(w.nc)!=NATS_CONN_STATUS_CONNECTED) {
            struct timespec retry={0,100000000};nanosleep(&retry,NULL);continue;
        }
        fetch=natsSubscription_Fetch(&list,sub,1,1000,&ec);
        if(stopping) {
            int k;
            for(k=0;k<list.Count;k++)(void)natsMsg_NakWithDelay(list.Msgs[k],1000,NULL);
            natsMsgList_Destroy(&list);
            break;
        }
        if(fetch==NATS_TIMEOUT)continue;
        if(fetch) {exit_code=1;break;}
        if(list.Count!=1) {natsMsgList_Destroy(&list);continue;}
        if(natsMsg_GetDataLength(list.Msgs[0])!=32) {(void)natsMsg_Term(list.Msgs[0],NULL);natsMsgList_Destroy(&list);continue;}
        pthread_mutex_lock(&w.lock);
        memcpy(w.id,natsMsg_GetData(list.Msgs[0]),32);w.id[32]=0;
        failed=!t_id(w.id,32)||t_random(w.attempt);
        pthread_mutex_unlock(&w.lock);
        if(failed) {natsMsgList_Destroy(&list);exit_code=1;break;}
        snprintf(op,sizeof(op),"claim.%s.%s.%s.%s",w.generation,w.id,w.attempt,w.name);
        claimed_at=t_now();
        if(t_rpc(w.nc,op,info,strlen(info),&reply)||!t_ok(reply)) {
            if(reply&&((t_size(reply)==8&&!memcmp(t_body(reply),"terminal",8))||(t_size(reply)==11&&!memcmp(t_body(reply),"unknown_job",11))))(void)natsMsg_Ack(list.Msgs[0],NULL);
            else (void)natsMsg_NakWithDelay(list.Msgs[0],1000,NULL);
            natsMsg_Destroy(reply);natsMsgList_Destroy(&list);continue;
        }
        natsMsg_Destroy(reply);reply=NULL;
        pthread_mutex_lock(&w.lock);
        w.delivery=list.Msgs[0];w.active=1;w.retire=0;w.lease_deadline=claimed_at+T_WORKER_LEASE;w.cancelling=0;
        strcpy(w.snapshot,"{\"status\":\"preparing\"}");
        pthread_mutex_unlock(&w.lock);
        do {
            if(call(&w,"request",bytes,1,NULL,0,&reply)||!t_ok(reply)) {failed=1;break;}
            n=t_size(reply);
            if(n>T_BYTES-bytes) {failed=1;break;}
            memcpy(request+bytes,t_body(reply),n);bytes+=n;natsMsg_Destroy(reply);reply=NULL;
        } while(n==T_CHUNK);
        natsMsg_Destroy(reply);reply=NULL;request[bytes]=0;
        if(!failed) {
            OdzJob *job=NULL;
            if(odz_job_create_borrowed(request,bytes,trajectory_arena,trajectory_bytes,&job))failed=1;
            else {
                pthread_mutex_lock(&w.lock);w.job=job;if(w.cancelling)odz_job_cancel(job);pthread_mutex_unlock(&w.lock);
                (void)odz_job_run(runtime,job);
                if(odz_job_report(job,NULL,0,&required)||required>T_BYTES+1||odz_job_report(job,report,T_BYTES+1,&required)) {
                    strcpy(report,"{\"status\":\"failed\",\"error\":\"report_limit_or_serialization_failure\"}");required=strlen(report)+1;
                }
                {OdrJson root={report,required-1},v;if(jget(root,"runtime_quarantined",&v)&&jeq(v,"true"))quarantined=1;}
                pthread_mutex_lock(&w.lock);
                {size_t z;odz_job_status(job,w.snapshot,sizeof(w.snapshot),&z);if(strstr(w.snapshot,"\"runtime_quarantined\":true"))quarantined=1;}
                w.job=NULL;pthread_mutex_unlock(&w.lock);
                odz_job_destroy(job);
            }
        }
        memset(request,0,bytes);
        if(failed) {strcpy(report,"{\"status\":\"failed\",\"error\":\"request_transfer_or_creation_failure\"}");required=strlen(report)+1;}
        for(bytes=0;bytes<required-1;) {
            n=required-1-bytes;if(n>T_CHUNK)n=T_CHUNK;
            if(call(&w,"report",bytes,1,report+bytes,n,&reply)||!t_ok(reply)) {natsMsg_Destroy(reply);reply=NULL;break;}
            natsMsg_Destroy(reply);reply=NULL;bytes+=n;
        }
        if(bytes==required-1&&!call(&w,"commit",bytes,1,NULL,0,&reply)&&t_ok(reply))committed=1;
        natsMsg_Destroy(reply);
        pthread_mutex_lock(&w.lock);w.active=0;w.delivery=NULL;
        if(w.retire)exit_code=1;
        pthread_mutex_unlock(&w.lock);
        if(committed)(void)natsMsg_AckSync(list.Msgs[0],NULL,&ec);
        natsMsgList_Destroy(&list);memset(report,0,required);
        printf("job=%s committed=%d quarantined=%d\n",w.id,committed,quarantined);fflush(stdout);
        if(quarantined||exit_code) {exit_code=1;break;}
    }
    pthread_mutex_lock(&w.lock);w.quit=1;pthread_mutex_unlock(&w.lock);pthread_join(thread,NULL);
    odz_runtime_destroy(runtime);
    natsConnection_Close(w.nc);
    return exit_code;
}
