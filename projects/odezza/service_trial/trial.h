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
#ifndef ODEZZA_TRIAL_H
#define ODEZZA_TRIAL_H
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif
#include <nats/nats.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#define T_PREFIX "odezza.trial.v1"
#define T_JOBS 8
#define T_BYTES (8u*1024u*1024u)
#define T_CHUNK (256u*1024u)
#define T_STATUS 4096
#define T_LEASE 15.0
#define T_ATTEMPTS 3
#define T_WORKER_LEASE 5.0 /* conservative relative to the server's 15 s lease */
#define T_CANCEL_GRACE 10.0
double t_now(void);
int t_id(const char *s, size_t n);
int t_random(char out[33]);
int t_connect(natsConnection **out, const char *url);
int t_connect_worker(natsConnection **out,const char *url,natsConnectionHandler changed,void *closure);
/* Replies are OK\n + bytes, or ERR\n + a stable ASCII error code. */
void t_reply(natsConnection *nc,natsMsg *msg,int ok,const void *data,size_t size);
int t_rpc(natsConnection *nc,const char *operation,const void *data,size_t size,natsMsg **reply);
int t_rpc_timeout(natsConnection *nc,const char *operation,const void *data,size_t size,natsMsg **reply,int64_t timeout_ms);
const char *t_body(natsMsg *m);
size_t t_size(natsMsg *m);
int t_ok(natsMsg *m);
#endif
