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
/* Measures existing C frontend preparation, not wire encoding or GPU execution.
 * Each thread reuses separate caller-owned arenas. No dependency on a broker. */
#define _POSIX_C_SOURCE 200809L
#include "odezza_request.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct Gate {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int start;
} Gate;
typedef struct Worker {
    Gate *gate;
    OdrJson json;
    void *ta, *sa, *ga;
    size_t tn, sn, gn, count, completed;
    OdrError error;
    int failed;
} Worker;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec+t.tv_nsec*1e-9;
}

static void *run(void *arg) {
    Worker *w=arg;
    size_t i,n;
    const OdrTrajectories *t;
    const OdrStatic *s;
    const OdrGrammar *g;
    pthread_mutex_lock(&w->gate->mutex);
    while(!w->gate->start)pthread_cond_wait(&w->gate->condition,&w->gate->mutex);
    pthread_mutex_unlock(&w->gate->mutex);
    for(i=0;i<w->count;i++) {
        if(odr_trajectories_parse(w->json,NULL,0,&n,&t,&w->error)||n>w->tn ||
           odr_trajectories_parse(w->json,w->ta,w->tn,&n,&t,&w->error)||
           odr_static_parse(w->json,NULL,0,&n,&s,&w->error)||n>w->sn ||
           odr_static_parse(w->json,w->sa,w->sn,&n,&s,&w->error)||
           odr_grammar_parse(w->json,s,NULL,0,&n,&g,&w->error)||n>w->gn ||
           odr_grammar_parse(w->json,s,w->ga,w->gn,&n,&g,&w->error)) {
            w->failed=1;
            break;
        }
        if(t->state_count!=s->state_count||!odr_grammar_family_count(g)) {
            w->failed=1;
            break;
        }
        ++w->completed;
    }
    return NULL;
}

int main(int argc,char **argv) {
    unsigned threads,i,started=0;
    size_t count,bytes,tn,sn,gn,needed,completed=0;
    FILE *f;
    long length;
    char *text;
    void *scratch;
    const OdrTrajectories *t;
    const OdrStatic *s;
    const OdrGrammar *g;
    OdrError error={0};
    OdrJson json;
    Worker workers[64]={{0}};
    pthread_t ids[64];
    Gate gate={PTHREAD_MUTEX_INITIALIZER,PTHREAD_COND_INITIALIZER,0};
    int failed=0;
    double begin,elapsed;
    if(argc!=4) {
        fprintf(stderr,"usage: prepare_benchmark REQUEST.json THREADS TOTAL_REQUESTS\n");
        return 2;
    }
    threads=(unsigned)strtoul(argv[2],NULL,10);
    count=(size_t)strtoull(argv[3],NULL,10);
    if(!threads||threads>64||!count||count>10000000)return 2;
    f=fopen(argv[1],"rb");
    if(!f)return 2;
    if(fseek(f,0,SEEK_END)||(length=ftell(f))<=0||length>64*1024*1024) {
        fclose(f);return 2;
    }
    rewind(f);bytes=(size_t)length;text=malloc(bytes);
    if(!text) {fclose(f);return 2;}
    if(fread(text,1,bytes,f)!=bytes) {fclose(f);free(text);return 2;}
    fclose(f);json.data=text;json.size=bytes;
    if(odr_trajectories_parse(json,NULL,0,&tn,&t,&error)||
       odr_static_parse(json,NULL,0,&sn,&s,&error))goto invalid;
    scratch=malloc(sn?sn:1);
    if(!scratch) {free(text);return 2;}
    if(odr_static_parse(json,scratch,sn,&needed,&s,&error)||
       odr_grammar_parse(json,s,NULL,0,&gn,&g,&error)) {
        free(scratch);goto invalid;
    }
    free(scratch);
    if(tn>512u*1024u*1024u/threads||sn>512u*1024u*1024u/threads-tn||
       gn>512u*1024u*1024u/threads-tn-sn) {free(text);return 2;}
    for(i=0;i<threads;i++) {
        Worker *w=&workers[i];
        w->gate=&gate;w->json=json;w->tn=tn;w->sn=sn;w->gn=gn;
        w->count=count/threads+(i<count%threads);
        w->ta=malloc(tn?tn:1);w->sa=malloc(sn?sn:1);w->ga=malloc(gn?gn:1);
        if(!w->ta||!w->sa||!w->ga||pthread_create(&ids[i],NULL,run,w)) {
            failed=1;break;
        }
        ++started;
    }
    begin=now();
    pthread_mutex_lock(&gate.mutex);gate.start=1;
    pthread_cond_broadcast(&gate.condition);pthread_mutex_unlock(&gate.mutex);
    for(i=0;i<started;i++) {
        pthread_join(ids[i],NULL);
        completed+=workers[i].completed;
        if(workers[i].failed) {
            fprintf(stderr,"worker %u: %s\n",i,workers[i].error.message);
            failed=1;
        }
    }
    elapsed=now()-begin;
    printf("{\"threads\":%u,\"requests\":%zu,\"completed\":%zu,\"input_bytes\":%zu,"
           "\"arena_bytes_per_worker\":%zu,\"seconds\":%.9g,\"requests_per_second\":%.9g,"
           "\"scope\":\"C trajectory/static/grammar measure+build; warm reusable arenas; no wire encoding or AST generation\"}\n",
           threads,count,completed,bytes,tn+sn+gn,elapsed,completed/elapsed);
    for(i=0;i<threads;i++) {free(workers[i].ta);free(workers[i].sa);free(workers[i].ga);}
    pthread_mutex_destroy(&gate.mutex);pthread_cond_destroy(&gate.condition);
    free(text);
    return failed||completed!=count;
invalid:
    fprintf(stderr,"prepare: %s\n",error.message);free(text);return 1;
}
