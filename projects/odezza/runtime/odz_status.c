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

void odz_publish(OdzJob *j) {
    Snapshot s={0};
    size_t i;
    HostMemory m=odz_host_snapshot(j);
    s.state=j->state;
    s.quarantined=j->quarantined;
    s.retention_incomplete=j->retention_incomplete;
    s.configurations=j->configurations;
    s.valid=j->valid;
    s.invalid=j->invalid;
    s.chunks=j->chunks;
    s.host_bytes=m.live[ODZ_MEM_ARENA];
    s.retention_bytes=m.live[ODZ_MEM_RETAINED];
    s.charged_host_bytes=m.total;s.peak_host_bytes=m.peak_total;
    s.host_limit_bytes=j->options.host_bytes;s.denied_reservations=m.denied;
    s.rss_bytes=j->rss_bytes;s.peak_sampled_rss_bytes=j->peak_sampled_rss_bytes;
    s.process_peak_rss_bytes=j->process_peak_rss_bytes;s.rss_valid=j->rss_valid;
    s.started=j->started;
    s.finished=j->finished;
    if(j->families)for(i=0;i<j->family_count;i++)s.generated+=j->families[i].generated;
    memcpy(s.error,j->error,sizeof(s.error));
    pthread_mutex_lock(&j->snapshot_mutex);
    j->snapshot=s;
    pthread_mutex_unlock(&j->snapshot_mutex);
}

int odz_job_status(OdzJob *j,char *output,size_t capacity,size_t *required) {
    static const char *states[]={"queued","preparing","running","complete","cancelled","failed","timeout"};
    Snapshot s;
    char escaped[3073];
    size_t i,n=0;
    int bytes;
    double elapsed;
    if(!j||!required)return -1;
    pthread_mutex_lock(&j->snapshot_mutex);
    s=j->snapshot;
    pthread_mutex_unlock(&j->snapshot_mutex);
    /* Bounded JSON escaping, without allocations or reading candidate memory. */
    for(i=0;s.error[i];i++) {
        unsigned c=(unsigned char)s.error[i];
        if(c<32) {snprintf(escaped+n,7,"\\u%04x",c);n+=6;}
        else {if(c=='"'||c=='\\')escaped[n++]='\\';escaped[n++]=(char)c;}
    }
    escaped[n]=0;
    elapsed=s.started?(s.finished?s.finished:odz_now())-s.started:0;
    bytes=snprintf(output,output?capacity:0,
        "{\"status\":\"%s\",\"error\":%s%s%s,\"runtime_quarantined\":%s,\"retention_complete\":%s,"
        "\"counts\":{\"completed_configurations\":%llu,\"valid\":%llu,\"invalid\":%llu,\"completed_chunks\":%llu,\"generated_asts\":%llu},"
        "\"timing\":{\"total_seconds\":%.9g},\"memory\":{\"job_arena_bytes\":%llu,\"retained_bytes\":%llu,\"charged_live_bytes\":%llu,\"charged_peak_bytes\":%llu,\"host_budget_bytes\":%llu,\"denied_reservations\":%llu,\"process_rss_bytes\":%llu,\"rss_sample_valid\":%s}}",
        states[s.state],n?"\"":"null",escaped,n?"\"":"",s.quarantined?"true":"false",s.retention_incomplete?"false":"true",
        (unsigned long long)s.configurations,(unsigned long long)s.valid,(unsigned long long)s.invalid,
        (unsigned long long)s.chunks,(unsigned long long)s.generated,elapsed,
        (unsigned long long)s.host_bytes,(unsigned long long)s.retention_bytes,
        (unsigned long long)s.charged_host_bytes,(unsigned long long)s.peak_host_bytes,
        (unsigned long long)s.host_limit_bytes,(unsigned long long)s.denied_reservations,
        (unsigned long long)s.rss_bytes,s.rss_valid?"true":"false");
    if(bytes<0)return -1;
    *required=(size_t)bytes+1;
    return output&&capacity<*required?1:0;
}
