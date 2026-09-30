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
#include "admission.h"
#include "odr_internal.h"
#include "odezza_runtime.h"
#include "odz_sizing.h"
#include <sys/file.h>
#include <fcntl.h>
#include <signal.h>
enum { EMPTY,RECEIVING,ADMITTING,QUEUED,RUNNING,TERMINAL };
typedef struct Job {
    char id[33],attempt[33],worker[33],worker_info[256],status[T_STATUS];
    char *request,*report;
    size_t expected,received,report_size;
    int state,cancel,attempts;
    uint64_t observed[T_ATTEMPTS];
    double admitted,started,lease,finished,budget;
    OdrRk4Work work;
    TrialAdmission allocation;
} Job;
typedef struct Server {
    pthread_mutex_t lock;
    natsConnection *nc;jsCtx *js;
    char generation[33],stream[64],subject[128];
    Job jobs[T_JOBS];
    TrialAdmissionPolicy policy;
    char admitted_ids[4096][33];size_t id_count;
} Server;
static volatile sig_atomic_t stopping;
static void stop(int sig) {(void)sig;stopping=1;}
static void error(natsConnection *nc,natsMsg *m,const char *s) {t_reply(nc,m,0,s,strlen(s));}
static void terminal(Job *j,const char *status,const char *reason) {
    j->state=TERMINAL;j->finished=t_now();
    snprintf(j->status,sizeof(j->status),"{\"status\":\"%s\",\"error\":\"%s\"}",status,reason);
    j->report_size=strlen(j->status);memcpy(j->report,j->status,j->report_size);
}
/* Envelope validation only. GPU host performs the full C grammar preparation. */
static int budget(OdrJson root,double *seconds) {
    OdrJson v,e,g,l;OdrError err;float f;
    *seconds=600;
    if(jvalidate(root,&err)||!jkeys(root,"problem,grammar,execution",&err))return 1;
    if(jget(root,"execution",&e)&&jget(e,"max_seconds",&v)) {
        if(!jfloat(v,&f)||f<=0)return 1;
        *seconds=f;
    }
    if(jget(root,"grammar",&g)&&jget(g,"limits",&l)&&jget(l,"max_seconds",&v)) {
        if(!jfloat(v,&f)||f<=0)return 1;
        if(f<*seconds)*seconds=f;
    }
    return *seconds>3600;
}
/* Error text may include JSON field names; always escape it before reporting. */
static void quote_detail(const char *in,char out[1155]) {
    size_t n=0,i;out[n++]='"';
    for(i=0;in[i]&&i<191;i++) {
        unsigned char c=(unsigned char)in[i];
        if(c<32) {snprintf(out+n,7,"\\u%04x",c);n+=6;}
        else {if(c=='"'||c=='\\')out[n++]='\\';out[n++]=(char)c;}
    }
    out[n++]='"';out[n]=0;
}
static void expire_lease(Job *j) {
    if(j->cancel)terminal(j,"cancelled","worker_lost_after_cancel");
    else if(j->attempts>=T_ATTEMPTS)terminal(j,"failed","worker_attempt_limit");
    else {j->state=QUEUED;j->attempt[0]=0;j->status[0]=0;j->report_size=0;}
}
static void observe_work(Job *j,const char *data,size_t n) {
    OdrJson root={data,n},counts,value;uint64_t count=0;
    if(j->attempts<1||j->attempts>T_ATTEMPTS)return;
    if(jget(root,"counts",&counts)&&jget(counts,"completed_configurations",&value)&&ju64(value,&count)&&count>j->observed[j->attempts-1])j->observed[j->attempts-1]=count;
}
static void status(Server *s,Job *j,natsMsg *m) {
    char b[8192],allocation[768];double now=t_now();
    if(!trial_admission_json(&j->allocation,allocation,sizeof(allocation)))strcpy(allocation,"null");
    int n=snprintf(b,sizeof(b),"{\"job_id\":\"%s\",\"generation\":\"%s\",\"state\":\"%s\","
        "\"attempts\":%d,\"observed_configurations_by_attempt\":[%llu,%llu,%llu],\"attempt_id\":\"%s\",\"worker\":\"%s\",\"cancel_requested\":%s,"
        "\"request_bytes\":%zu,\"report_bytes\":%zu,\"execution_budget_seconds\":%.9g,\"allocation\":%s,\"integration_work\":{\"steps_per_observation\":%u,\"steps_per_configuration\":%llu,\"trajectory_points\":%u,\"trajectory_count\":%u},\"worker_info\":%s,\"queue_seconds\":%.9g,\"service_seconds\":%.9g,\"native\":%s}",
        j->id,s->generation,j->state==TERMINAL?"terminal":j->state==RUNNING?"running":j->state==QUEUED?"queued":j->state==ADMITTING?"admitting":"receiving",
        j->attempts,(unsigned long long)j->observed[0],(unsigned long long)j->observed[1],(unsigned long long)j->observed[2],j->attempt,j->worker,j->cancel?"true":"false",j->expected,j->report_size,j->budget,allocation,
        j->work.steps_per_observation,(unsigned long long)j->work.steps_per_configuration,j->work.point_count,j->work.trajectory_count,j->worker_info[0]?j->worker_info:"null",
        (j->started?j->started:j->finished?j->finished:now)-j->admitted,(j->finished?j->finished:now)-j->admitted,j->status[0]?j->status:"null");
    t_reply(s->nc,m,1,b,(size_t)n);
}
static int number(const char *s,size_t *out) {
    char *end;unsigned long long n;
    if(!s||!*s||strspn(s,"0123456789")!=strlen(s))return 0;
    n=strtoull(s,&end,10);
    if(*end||n>T_BYTES)return 0;
    *out=(size_t)n;return 1;
}
static void rpc(natsConnection *nc,natsSubscription *sub,natsMsg *m,void *arg) {
    Server *s=arg;Job *j=NULL;char path[240],*v[6],*save=NULL,*p;
    int nv=0,i;size_t n=(size_t)natsMsg_GetDataLength(m),off=0;
    const char *data=natsMsg_GetData(m),*subject=natsMsg_GetSubject(m);
    (void)sub;
    if(strlen(subject)>=sizeof(path)||n>T_CHUNK) {error(nc,m,"message_limit");goto end;}
    strcpy(path,subject+strlen(T_PREFIX)+1);
    for(p=strtok_r(path,".",&save);p&&nv<6;p=strtok_r(NULL,".",&save))v[nv++]=p;
    if(p||!nv) {error(nc,m,"bad_operation");goto end;}
    pthread_mutex_lock(&s->lock);
    if(!strcmp(v[0],"health")&&nv==1) {
        char b[2048],policy[768];int active=0;
        if(!trial_admission_policy_json(&s->policy,policy,sizeof(policy))) {error(nc,m,"policy_report_failure");goto unlock;}
        for(i=0;i<T_JOBS;i++)if(s->jobs[i].state)active++;
        snprintf(b,sizeof(b),"{\"generation\":\"%s\",\"stream\":\"%s\",\"subject\":\"%s\",\"jobs\":%d,\"job_limit\":%d,\"request_limit\":%u,\"report_limit\":%u,\"admission_limits\":%s,\"integration_limits\":{\"points_per_configuration\":%u,\"steps_per_observation\":%u,\"steps_per_configuration\":%llu,\"base_work_units_per_tile\":%llu,\"maximum_parallel_target_configurations\":%llu,\"tile_policy\":\"" ODZ_TILE_SIZING_POLICY "\"}}",s->generation,s->stream,s->subject,active,T_JOBS,T_BYTES,T_BYTES,policy,ODZ_MAX_POINTS_PER_CONFIGURATION,ODZ_MAX_STEPS_PER_OBSERVATION,(unsigned long long)ODZ_MAX_STEPS_PER_CONFIGURATION,(unsigned long long)ODZ_BASE_TILE_WORK_UNITS,(unsigned long long)ODZ_MAX_PARALLEL_CONFIGURATIONS);
        t_reply(nc,m,1,b,strlen(b));goto unlock;
    }
    if(!strcmp(v[0],"jobs")&&nv==1) {
        char b[4096];size_t used=0;int first=1;
        used+=(size_t)snprintf(b+used,sizeof(b)-used,"{\"generation\":\"%s\",\"jobs\":[",s->generation);
        for(i=0;i<T_JOBS;i++)if(s->jobs[i].state) {
            Job *row=&s->jobs[i];
            used+=(size_t)snprintf(b+used,sizeof(b)-used,"%s{\"job_id\":\"%s\",\"terminal\":%s,\"attempts\":%d,\"worker_info\":%s}",first?"":",",row->id,row->state==TERMINAL?"true":"false",row->attempts,row->worker_info[0]?row->worker_info:"null");
            first=0;
        }
        used+=(size_t)snprintf(b+used,sizeof(b)-used,"]}");
        t_reply(nc,m,1,b,used);goto unlock;
    }
    if(nv<3||strcmp(v[1],s->generation)||!t_id(v[2],32)) {error(nc,m,"unknown_generation_or_job");goto unlock;}
    for(i=0;i<T_JOBS;i++)if(s->jobs[i].state&&!strcmp(s->jobs[i].id,v[2])) {j=&s->jobs[i];break;}
    if(!strcmp(v[0],"begin")&&nv==4) {
        if(!number(v[3],&off)||!off) {error(nc,m,"request_limit");goto unlock;}
        if(j) {
            if(j->expected!=off)error(nc,m,"idempotency_conflict");
            else status(s,j,m);
            goto unlock;
        }
        for(i=0;i<(int)s->id_count;i++)if(!strcmp(s->admitted_ids[i],v[2])) {error(nc,m,"released_job");goto unlock;}
        if(s->id_count==4096) {error(nc,m,"generation_id_limit");goto unlock;}
        for(i=0;i<T_JOBS;i++)if(!s->jobs[i].state) {j=&s->jobs[i];break;}
        if(!j) {error(nc,m,"admission_full");goto unlock;}
        {char *req=j->request,*rep=j->report;memset(j,0,sizeof(*j));j->request=req;j->report=rep;}
        strcpy(s->admitted_ids[s->id_count++],v[2]);
        strcpy(j->id,v[2]);j->expected=off;j->state=RECEIVING;j->admitted=t_now();status(s,j,m);goto unlock;
    }
    if(!j) {error(nc,m,"unknown_job");goto unlock;}
    if(j->state==RUNNING&&t_now()-j->lease>T_LEASE)expire_lease(j);
    if(!strcmp(v[0],"put")&&nv==4) {
        if(!number(v[3],&off)||off>j->expected||n>j->expected-off||off>j->received) {error(nc,m,"invalid_range");goto unlock;}
        if(off<j->received||j->state!=RECEIVING) {
            if(off+n>j->received||memcmp(j->request+off,data,n))error(nc,m,"idempotency_conflict");
            else t_reply(nc,m,1,NULL,0);
        } else {memcpy(j->request+off,data,n);j->received+=n;t_reply(nc,m,1,NULL,0);}
    } else if(!strcmp(v[0],"submit")&&nv==3) {
        if(j->state!=RECEIVING) {status(s,j,m);goto unlock;}
        if(j->received!=j->expected) {error(nc,m,"incomplete_request");goto unlock;}
        j->state=ADMITTING;
        /* Slot/request lifetime is pinned by ADMITTING; cancellation only sets
         * a flag, release is refused. Parsing must not hold the control lock. */
        pthread_mutex_unlock(&s->lock);
        {
            OdrJson request={j->request,j->expected};OdrRk4Work work={0};
            TrialAdmission allocation={0};
            OdrError detail={0};double seconds=600;const char *reason=NULL;
            if(budget(request,&seconds))reason="invalid_envelope_or_budget_above_3600s";
            else {
                OdrResult r=odr_request_rk4_work(request,&work,&detail);
                if(r==ODR_CAPACITY||r==ODR_OVERFLOW||work.steps_per_observation>ODZ_MAX_STEPS_PER_OBSERVATION||work.steps_per_configuration>ODZ_MAX_STEPS_PER_CONFIGURATION||work.point_count>ODZ_MAX_POINTS_PER_CONFIGURATION)reason="integration_work_limit";
                else if(r)reason="invalid_trajectory_or_integration";
            }
            if(!reason)reason=trial_admission_check(request,&work,&s->policy,&allocation,&detail);
            pthread_mutex_lock(&s->lock);
            j->budget=seconds;j->work=work;j->allocation=allocation;
            if(reason||j->cancel) {
                terminal(j,j->cancel?"cancelled":"failed",j->cancel?"cancelled_during_admission":reason);
                if(!j->cancel&&reason) {
                    /* Stable error plus numerical bounds; no unescaped input. */
                    char quoted[1155],allocated[768],policy[768];quote_detail(detail.message,quoted);
                    trial_admission_json(&allocation,allocated,sizeof(allocated));
                    trial_admission_policy_json(&s->policy,policy,sizeof(policy));
                    j->report_size=(size_t)snprintf(j->report,T_BYTES,
                        "{\"status\":\"failed\",\"error\":\"%s\",\"detail\":%s,\"allocation\":%s,\"admission_limits\":%s,\"work_counts_available\":%s,\"trajectory_points\":%u,\"max_trajectory_points\":%u,\"steps_per_observation\":%u,\"steps_per_configuration\":%llu,\"max_steps_per_observation\":%u,\"max_steps_per_configuration\":%llu,\"completed_configurations\":0}",
                        reason,quoted,allocated,policy,work.steps_per_observation?"true":"false",work.point_count,ODZ_MAX_POINTS_PER_CONFIGURATION,work.steps_per_observation,(unsigned long long)work.steps_per_configuration,
                        ODZ_MAX_STEPS_PER_OBSERVATION,(unsigned long long)ODZ_MAX_STEPS_PER_CONFIGURATION);
                }
                status(s,j,m);goto unlock;
            }
        }
        /* Broker I/O must never hold the control mutex. */
        pthread_mutex_unlock(&s->lock);
        {jsPubOptions opts;jsErrCode ec=0;natsStatus r;
         jsPubOptions_Init(&opts);opts.MsgId=j->id;opts.MaxWait=3000;
         r=js_Publish(NULL,s->js,s->subject,j->id,32,&opts,&ec);
         pthread_mutex_lock(&s->lock);
         /* A worker may already have claimed it before the publish ACK. */
         if(j->state==ADMITTING) {
             if(r)terminal(j,"failed","queue_admission_unconfirmed");
             else if(j->cancel)terminal(j,"cancelled","cancelled_during_admission");
             else j->state=QUEUED;
         }
        }
        status(s,j,m);
    } else if(!strcmp(v[0],"status")&&nv==3)status(s,j,m);
    else if(!strcmp(v[0],"cancel")&&nv==3) {
        j->cancel=1;
        if(j->state==RECEIVING||j->state==QUEUED)terminal(j,"cancelled","cancelled_before_execution");
        status(s,j,m);
    } else if(!strcmp(v[0],"release")&&nv==3) {
        if(j->state!=TERMINAL)error(nc,m,"job_active");
        else {j->state=EMPTY;memset(j->request,0,j->received);memset(j->report,0,j->report_size);t_reply(nc,m,1,NULL,0);}
    } else if(!strcmp(v[0],"result")&&nv==4) {
        if(j->state!=TERMINAL)error(nc,m,"job_active");
        else if(!number(v[3],&off)||off>j->report_size)error(nc,m,"invalid_range");
        else {size_t z=j->report_size-off;if(z>T_CHUNK)z=T_CHUNK;t_reply(nc,m,1,j->report+off,z);}
    } else if(!strcmp(v[0],"claim")&&nv==5&&t_id(v[3],32)&&t_id(v[4],32)) {
        OdrJson info={data,n};OdrError err;
        if(n&&(n>=sizeof(j->worker_info)||jvalidate(info,&err))) {error(nc,m,"invalid_worker_info");goto unlock;}
        if(j->state==TERMINAL)error(nc,m,"terminal");
        else if(j->state==RUNNING&&strcmp(j->attempt,v[3]))error(nc,m,"attempt_busy");
        else if(j->state==RECEIVING||j->state==ADMITTING)error(nc,m,"not_admitted");
        else {
            if(j->state!=RUNNING) {j->attempts++;j->report_size=0;strcpy(j->attempt,v[3]);strcpy(j->worker,v[4]);j->state=RUNNING;if(!j->started)j->started=t_now();}
            if(n) {memcpy(j->worker_info,data,n);j->worker_info[n]=0;}
            j->lease=t_now();t_reply(nc,m,1,NULL,0);
        }
    } else if(nv>=4&&t_id(v[3],32)&&!strcmp(j->attempt,v[3])) {
        if(!strcmp(v[0],"request")&&nv==5&&j->state==RUNNING) {
            if(!number(v[4],&off)||off>j->expected)error(nc,m,"invalid_range");
            else {size_t z=j->expected-off;if(z>T_CHUNK)z=T_CHUNK;t_reply(nc,m,1,j->request+off,z);}
        } else if(!strcmp(v[0],"beat")&&nv==4&&j->state==TERMINAL) {
            t_reply(nc,m,1,"cancel",6);
        } else if(!strcmp(v[0],"beat")&&nv==4&&j->state==RUNNING) {
            OdrJson snapshot={data,n};OdrError e;
            if(n>=sizeof(j->status)||jvalidate(snapshot,&e))error(nc,m,"invalid_status");
            else {observe_work(j,data,n);memcpy(j->status,data,n);j->status[n]=0;j->lease=t_now();t_reply(nc,m,1,j->cancel?"cancel":"continue",j->cancel?6:8);}
        } else if(!strcmp(v[0],"report")&&nv==5&&j->state==RUNNING) {
            if(!number(v[4],&off)||off>j->report_size||n>T_BYTES-off)error(nc,m,"invalid_range");
            else if(off<j->report_size) {
                if(off+n>j->report_size||memcmp(j->report+off,data,n))error(nc,m,"report_conflict");
                else t_reply(nc,m,1,NULL,0);
            } else {memcpy(j->report+off,data,n);j->report_size+=n;t_reply(nc,m,1,NULL,0);}
        } else if(!strcmp(v[0],"commit")&&nv==5) {
            if(j->state==TERMINAL)t_reply(nc,m,1,NULL,0);
            else if(j->state!=RUNNING||!number(v[4],&off)||off!=j->report_size||!off)error(nc,m,"incomplete_report");
            else {
                OdrJson report={j->report,j->report_size},vstatus;OdrError e;char state[24];size_t len;
                if(jvalidate(report,&e)||!jget(report,"status",&vstatus)||!jstring(vstatus,state,sizeof(state),&len)||
                   (strcmp(state,"complete")&&strcmp(state,"failed")&&strcmp(state,"cancelled")&&strcmp(state,"timeout")))error(nc,m,"invalid_terminal_report");
                else {observe_work(j,j->report,j->report_size);j->state=TERMINAL;j->finished=t_now();snprintf(j->status,sizeof(j->status),"{\"status\":\"%s\"}",state);t_reply(nc,m,1,NULL,0);}
            }
        } else error(nc,m,"stale_attempt");
    } else error(nc,m,"stale_attempt_or_bad_operation");
unlock:
    pthread_mutex_unlock(&s->lock);
end:
    natsMsg_Destroy(m);
}
int main(int argc,char **argv) {
    static const char *operations[]={"health","jobs","begin","put","submit","status","cancel","release","result","claim","request","beat","report","commit"};
    Server *s;jsStreamConfig sc;jsConsumerConfig cc;jsErrCode ec=0;natsSubscription *subs[14]={0};
    const char *subjects[1];int i,fd;struct timespec tick={0,100000000};
    if(argc!=3) {fprintf(stderr,"usage: orchestrator NATS_URL LOCKFILE\n");return 2;}
    fd=open(argv[2],O_CREAT|O_RDWR,0600);
    if(fd<0||flock(fd,LOCK_EX|LOCK_NB)) {fputs("another orchestrator owns the lock\n",stderr);return 1;}
    s=calloc(1,sizeof(*s));if(!s)return 1;pthread_mutex_init(&s->lock,NULL);
    {OdrError e={0};trial_admission_defaults(&s->policy);
     if(trial_admission_environment(&s->policy,&e)) {fprintf(stderr,"%s\n",e.message);return 1;}}
    for(i=0;i<T_JOBS;i++) {s->jobs[i].request=calloc(1,T_BYTES);s->jobs[i].report=calloc(1,T_BYTES);if(!s->jobs[i].request||!s->jobs[i].report)return 1;}
    if(t_random(s->generation)||t_connect(&s->nc,argv[1])||natsConnection_JetStream(&s->js,s->nc,NULL))return 1;
    snprintf(s->stream,sizeof(s->stream),"ODZ_%s",s->generation);
    snprintf(s->subject,sizeof(s->subject),"odezza.work.v1.%s.score",s->generation);subjects[0]=s->subject;
    jsStreamConfig_Init(&sc);sc.Name=s->stream;sc.Subjects=subjects;sc.SubjectsLen=1;
    sc.Retention=js_WorkQueuePolicy;sc.Storage=js_MemoryStorage;sc.MaxMsgs=32;sc.MaxBytes=65536;sc.MaxAge=INT64_C(86400000000000);sc.Discard=js_DiscardNew;sc.Replicas=1;
    if(js_AddStream(NULL,s->js,&sc,NULL,&ec)) {fprintf(stderr,"stream creation: %d\n",ec);return 1;}
    jsConsumerConfig_Init(&cc);cc.Durable="executors";cc.AckPolicy=js_AckExplicit;cc.AckWait=INT64_C(5000000000);cc.MaxAckPending=8;cc.MaxRequestBatch=1;cc.MaxWaiting=16;cc.MaxDeliver=-1;
    if(js_AddConsumer(NULL,s->js,s->stream,&cc,NULL,&ec)) {fprintf(stderr,"consumer creation: %d\n",ec);return 1;}
    /* Separate subscriptions/dispatch threads for uploads and fast control. */
    for(i=0;i<14;i++) {
        char topic[128];snprintf(topic,sizeof(topic),T_PREFIX".%s%s",operations[i],i>1?".>":"");
        if(natsConnection_Subscribe(&subs[i],s->nc,topic,rpc,s))return 1;
        natsSubscription_SetPendingLimits(subs[i],32,2*1024*1024);
    }
    if(natsConnection_FlushTimeout(s->nc,3000))return 1;
    printf("ready generation=%s\n",s->generation);fflush(stdout);
    signal(SIGINT,stop);signal(SIGTERM,stop);
    while(!stopping&&!natsConnection_IsClosed(s->nc)) {
        pthread_mutex_lock(&s->lock);
        for(i=0;i<T_JOBS;i++) {
            Job *j=&s->jobs[i];double now=t_now();
            if(j->state==TERMINAL&&now-j->finished>3600) {memset(j->request,0,j->received);memset(j->report,0,j->report_size);j->state=EMPTY;continue;}
            if(j->state==RUNNING&&now-j->started>j->budget+10)terminal(j,"timeout","service_execution_deadline");
            else if(j->state==RUNNING&&now-j->lease>T_LEASE) {
                expire_lease(j);
            } else if((j->state==RECEIVING||j->state==QUEUED)&&now-j->admitted>600)terminal(j,"timeout","admission_wait_expired");
        }
        pthread_mutex_unlock(&s->lock);nanosleep(&tick,NULL);
    }
    /* Stop admission before process exit; all accepted state was memory-only. */
    natsConnection_Close(s->nc);
    return 0;
}
