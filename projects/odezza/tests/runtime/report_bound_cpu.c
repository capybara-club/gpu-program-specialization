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
#include <assert.h>
#include "retention_support.h"
int main(int argc,char **argv) {
    FILE *f;long length;char *json;OdrJson root;OdrError error;size_t n;OdzJob *j=calloc(1,sizeof(*j));int result;
    assert(argc==2&&j);f=fopen(argv[1],"rb");assert(f);assert(!fseek(f,0,SEEK_END));length=ftell(f);rewind(f);assert(length>0);
    json=malloc((size_t)length+1);assert(json&&fread(json,1,(size_t)length,f)==(size_t)length);fclose(f);json[length]=0;
    root.data=json;root.size=(size_t)length;
#define PARSE(fn,field,arena) do{assert(!fn(root,NULL,0,&n,&j->field,&error));j->arena=malloc(n);assert(j->arena);assert(!fn(root,j->arena,n,&n,&j->field,&error));}while(0)
    PARSE(odr_static_parse,fixed,sa);PARSE(odr_trajectories_parse,trajectories,ta);
    assert(!odr_grammar_parse(root,j->fixed,NULL,0,&n,&j->grammar,&error));j->ga=malloc(n);assert(j->ga);
    assert(!odr_grammar_parse(root,j->fixed,j->ga,n,&n,&j->grammar,&error));
    j->family_count=j->grammar->family_count;j->families=calloc(j->family_count,sizeof(*j->families));assert(j->families);
    j->options.report_bytes=ODZ_MAX_REPORT_BYTES;assert(!odz_retention_parse(j));
    result=odz_report_preflight(j);
    printf("{\"request\":\"%s\",\"rejected\":%s,\"bound\":%llu,\"error\":\"%s\"}\n",argv[1],result?"true":"false",(unsigned long long)j->report_bound,j->error);
    odz_retention_close(j);assert(j->memory.total==0);free(j->families);free(j->ga);free(j->sa);free(j->ta);free(json);free(j);
    return 0;
}
