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
#ifndef ODZ_RETENTION_TEST_SUPPORT_H
#define ODZ_RETENTION_TEST_SUPPORT_H
/* Pure retention test harness: no CUDA runtime or driver calls. Allocation
 * counts include every shared/provenance/row record; ASan checks lifetimes. */
typedef union H {size_t bytes;long double align;} H;
void *odz_host_alloc(OdzJob *j,size_t n,unsigned kind){H *h=calloc(1,sizeof(*h)+n);(void)kind;assert(h);h->bytes=n;j->memory.total+=n;if(j->memory.total>j->memory.peak_total)j->memory.peak_total=j->memory.total;return h+1;}
void *odz_alloc(OdzJob *j,size_t n){return odz_host_alloc(j,n,0);}
void odz_host_free(OdzJob *j,void *p){if(p){H *h=(H*)p-1;j->memory.total-=h->bytes;free(h);}}
void *odz_host_resize(OdzJob *j,void *p,size_t n,unsigned kind){void *q=odz_host_alloc(j,n,kind);if(p){H *h=(H*)p-1;memcpy(q,p,n<h->bytes?n:h->bytes);odz_host_free(j,p);}return q;}
int odz_fail(OdzJob *j,const char *fmt,...){va_list v;va_start(v,fmt);vsnprintf(j->error,sizeof(j->error),fmt,v);va_end(v);return 1;}

#endif
