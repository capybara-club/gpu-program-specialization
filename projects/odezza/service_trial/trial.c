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
#include <fcntl.h>
#include <errno.h>
double t_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+1e-9*t.tv_nsec;
}
int t_id(const char *s,size_t n) {
    size_t i;
    if(!s||strlen(s)!=n)return 0;
    for(i=0;i<n;i++)if(!((s[i]>='0'&&s[i]<='9')||(s[i]>='a'&&s[i]<='f')))return 0;
    return 1;
}
int t_random(char out[33]) {
    unsigned char b[16];size_t n=0;int fd=open("/dev/urandom",O_RDONLY);
    if(fd<0)return 1;
    while(n<sizeof(b)) {
        ssize_t r=read(fd,b+n,sizeof(b)-n);
        if(r<0&&errno==EINTR)continue;
        if(r<=0) {close(fd);return 1;}
        n+=(size_t)r;
    }
    close(fd);
    for(n=0;n<16;n++)snprintf(out+2*n,3,"%02x",b[n]);
    return 0;
}
static int connect_options(natsConnection **out,const char *url,natsConnectionHandler changed,void *closure) {
    natsOptions *o=NULL;natsStatus s=natsOptions_Create(&o);
    if(!s)s=natsOptions_SetURL(o,url);
    if(!s)s=natsOptions_SetTimeout(o,3000);
    /* The orchestrator still fails closed. Workers may reconnect within their
     * attempt lease; generation/attempt fencing remains authoritative. */
    if(!s)s=natsOptions_SetAllowReconnect(o,changed!=NULL);
    if(changed) {
        if(!s)s=natsOptions_SetMaxReconnect(o,50);
        if(!s)s=natsOptions_SetReconnectWait(o,100);
        if(!s)s=natsOptions_SetReconnectBufSize(o,0);
        if(!s)s=natsOptions_SetDisconnectedCB(o,changed,closure);
        if(!s)s=natsOptions_SetReconnectedCB(o,changed,closure);
        if(!s)s=natsOptions_SetClosedCB(o,changed,closure);
    }
    if(!s)s=natsConnection_Connect(out,o);
    natsOptions_Destroy(o);
    if(s)fprintf(stderr,"NATS connection: %s\n",natsStatus_GetText(s));
    return s!=NATS_OK;
}
int t_connect(natsConnection **out,const char *url) {return connect_options(out,url,NULL,NULL);}
int t_connect_worker(natsConnection **out,const char *url,natsConnectionHandler changed,void *closure) {
    return connect_options(out,url,changed,closure);
}
void t_reply(natsConnection *nc,natsMsg *m,int ok,const void *data,size_t size) {
    const char *subject=natsMsg_GetReply(m);char *p;
    size_t prefix=ok?3:4;
    if(!subject||!*subject||size>T_CHUNK+8192)return;
    p=malloc(prefix+size);
    if(!p)return;
    memcpy(p,ok?"OK\n":"ERR\n",prefix);
    if(size)memcpy(p+prefix,data,size);
    (void)natsConnection_Publish(nc,subject,p,(int)(prefix+size));
    free(p);
}
int t_rpc_timeout(natsConnection *nc,const char *op,const void *data,size_t size,natsMsg **reply,int64_t timeout_ms) {
    char subject[256];
    if(snprintf(subject,sizeof(subject),T_PREFIX".%s",op)>=(int)sizeof(subject)||size>T_CHUNK)return 1;
    *reply=NULL;
    return natsConnection_Request(reply,nc,subject,data,(int)size,timeout_ms)!=NATS_OK;
}
int t_rpc(natsConnection *nc,const char *op,const void *data,size_t size,natsMsg **reply) {
    return t_rpc_timeout(nc,op,data,size,reply,3000);
}
int t_ok(natsMsg *m) {return m&&natsMsg_GetDataLength(m)>=3&&!memcmp(natsMsg_GetData(m),"OK\n",3);}
const char *t_body(natsMsg *m) {return natsMsg_GetData(m)+(t_ok(m)?3:4);}
size_t t_size(natsMsg *m) {
    int n=natsMsg_GetDataLength(m),p=t_ok(m)?3:4;
    return n>=p?(size_t)(n-p):0;
}
