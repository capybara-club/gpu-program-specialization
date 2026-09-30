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
#include "odr_internal.h"
#include <ctype.h>
size_t odr_arena_alignment(void) {
    return sizeof(OdrAlign);
}
const char *odr_result_string(OdrResult r) {
    static const char *names[]= {
        "success","insufficient arena","invalid JSON","invalid schema",         "unsupported","overflow","invalid AST","invalid index","capacity exhausted","exhausted"
    };
    return (unsigned)r<sizeof(names)/sizeof(*names)?names[r]:"unknown result";
}
OdrResult odr_error(OdrError *e,OdrResult r,size_t at,const char *message) {
    if(e) {
        e->code=r;
        e->offset=at;
        snprintf(e->message,sizeof(e->message),"%s",message);
    }
    return r;
}
void *odr_take(Arena *a,size_t n,size_t z) {
    size_t aligned,bytes;
    void *p=NULL;
    if(a->result)return NULL;
    if((z && n>SIZE_MAX/z)||a->used>SIZE_MAX-(sizeof(OdrAlign)-1)) {
        a->result=ODR_OVERFLOW;
        return NULL;
    }
    bytes=n*z;
    aligned=(a->used+sizeof(OdrAlign)-1)/sizeof(OdrAlign)*sizeof(OdrAlign);
    if(bytes>SIZE_MAX-aligned) {
        a->result=ODR_OVERFLOW;
        return NULL;
    }
    a->used=aligned+bytes;
    if(a->base && a->used<=a->capacity) {
        p=a->base+aligned;
        memset(p,0,bytes);
    }
    return p;
}
static void ws(const char **p,const char *end) {
    while(*p<end && (**p==' '||**p=='\n'||**p=='\r'||**p=='\t'))++*p;
}
static int hex4(const char **p,const char *end,uint32_t *v) {
    unsigned i;
    *v=0;
    for(i=0;i<4;i++) {
        int c;
        if(*p==end)return 0;
        c=(unsigned char)*(*p)++;
        if(c>='0'&&c<='9')c-='0';
        else if(c>='a'&&c<='f')c=c-'a'+10;
        else if(c>='A'&&c<='F')c=c-'A'+10;
        else return 0;
        *v=(*v<<4)|(unsigned)c;
    }
    return 1;
}
/* Read one decoded Unicode scalar. Reject embedded NUL: C identifiers/strings
 * cannot represent it losslessly. Validate UTF-8 rather than copying invalid bytes. */ static int scalar(const char **p,const char *end,uint32_t *v) {
    unsigned c,n,i,min;
    uint32_t lo;
    if(*p>=end)return 0;
    c=(unsigned char)*(*p)++;
    if(c=='\\') {
        if(*p>=end)return 0;
        c=(unsigned char)*(*p)++;
        if(c=='u') {
            if(!hex4(p,end,v))return 0;
            if(*v>=0xd800 && *v<=0xdbff) {
                if(end-*p<2||(*p)[0]!='\\'||(*p)[1]!='u')return 0;
                *p+=2;
                if(!hex4(p,end,&lo)||lo<0xdc00||lo>0xdfff)return 0;
                *v=0x10000+((*v-0xd800)<<10)+(lo-0xdc00);
            }
            else if(*v>=0xdc00 && *v<=0xdfff)return 0;
            return *v!=0;
        }
        if(c=='b')c=8;
        else if(c=='f')c=12;
        else if(c=='n')c=10;
        else if(c=='r')c=13;
        else if(c=='t')c=9;
        else if(c!='"'&&c!='\\'&&c!='/')return 0;
        *v=c;
        return 1;
    }
    if(c<32||c=='"')return 0;
    if(c<128) {
        *v=c;
        return 1;
    }
    if(c>=0xc2&&c<=0xdf) {
        n=1;
        *v=c&31;
        min=0x80;
    }
    else if(c>=0xe0&&c<=0xef) {
        n=2;
        *v=c&15;
        min=0x800;
    }
    else if(c>=0xf0&&c<=0xf4) {
        n=3;
        *v=c&7;
        min=0x10000;
    }
    else return 0;
    for(i=0;i<n;i++) {
        if(*p==end)return 0;
        c=(unsigned char)*(*p)++;
        if((c&0xc0)!=0x80)return 0;
        *v=(*v<<6)|(c&63);
    }
    return *v>=min&&*v<=0x10ffff&&!(*v>=0xd800&&*v<=0xdfff);
}
static int string_end(const char **p,const char *end) {
    uint32_t c;
    if(*p==end||*(*p)++!='"')return 0;
    while(*p<end&&**p!='"')if(!scalar(p,end,&c))return 0;
    if(*p==end)return 0;
    ++*p;
    return 1;
}
static int same_string(OdrJson a,OdrJson b) {
    const char *p=a.data+1,*q=b.data+1,*pe=a.data+a.size-1,*qe=b.data+b.size-1;
    uint32_t x,y;
    while(p<pe&&q<qe) {
        if(!scalar(&p,pe,&x)||!scalar(&q,qe,&y)||x!=y)return 0;
    }
    return p==pe&&q==qe;
}
static int number_end(const char **p,const char *end) {
    if(*p<end&&**p=='-')++*p;
    if(*p==end)return 0;
    if(**p=='0')++*p;
    else {
        if(**p<'1'||**p>'9')return 0;
        do {
            ++*p;
        }
        while(*p<end&&isdigit((unsigned char)**p));
    }
    if(*p<end&&**p=='.') {
        ++*p;
        if(*p==end||!isdigit((unsigned char)**p))return 0;
        do {
            ++*p;
        }
        while(*p<end&&isdigit((unsigned char)**p));
    }
    if(*p<end&&(**p=='e'||**p=='E')) {
        ++*p;
        if(*p<end&&(**p=='+'||**p=='-'))++*p;
        if(*p==end||!isdigit((unsigned char)**p))return 0;
        do {
            ++*p;
        }
        while(*p<end&&isdigit((unsigned char)**p));
    }
    return 1;
}
static int skip(const char **p,const char *end,unsigned depth,int duplicates) {
    int object;
    char close;
    const char *start;
    ws(p,end);
    if(*p==end||depth>ODR_DEPTH)return 0;
    if(**p=='"')return string_end(p,end);
    if(**p=='-'||isdigit((unsigned char)**p))return number_end(p,end);
    if(**p!='{'&&**p!='[') {
        const char *s=(**p=='t'?"true":**p=='f'?"false":**p=='n'?"null":"");
        size_t n=strlen(s);
        if(!n||(size_t)(end-*p)<n||memcmp(*p,s,n))return 0;
        *p+=n;
        return 1;
    }
    object=**p=='{';
    close=object?'}':']';
    ++*p;
    start=*p;
    ws(p,end);
    if(*p<end&&**p==close) {
        ++*p;
        return 1;
    }
    for(;;) {
        if(object) {
            OdrJson key;
            const char *begin=*p;
            key.data=begin;
            if(!string_end(p,end))return 0;
            key.size=(size_t)(*p-begin);
            if(duplicates) {
                const char *q=start;
                ws(&q,end);
                while(q<begin) {
                    OdrJson old;
                    old.data=q;
                    if(!string_end(&q,end))return 0;
                    old.size=(size_t)(q-old.data);
                    if(same_string(key,old))return 0;
                    ws(&q,end);
                    ++q;
                    if(!skip(&q,end,depth+1,0))return 0;
                    ws(&q,end);
                    if(q<begin&&*q==',')++q;
                    ws(&q,end);
                }
            }
            ws(p,end);
            if(*p==end||*(*p)++!=':')return 0;
        }
        if(!skip(p,end,depth+1,duplicates))return 0;
        ws(p,end);
        if(*p==end)return 0;
        if(**p==close) {
            ++*p;
            return 1;
        }
        if(*(*p)++!=',')return 0;
        ws(p,end);
    }
}
OdrResult jvalidate(OdrJson j,OdrError *e) {
    const char *p=j.data;
    if(!p||!j.size)return odr_error(e,ODR_JSON,0,"empty JSON");
    if(!skip(&p,j.data+j.size,0,1))return odr_error(e,ODR_JSON,(size_t)(p-j.data),"invalid JSON, duplicate key, invalid string, or nesting limit");
    ws(&p,j.data+j.size);
    if(p!=j.data+j.size)return odr_error(e,ODR_JSON,(size_t)(p-j.data),"trailing JSON data");
    return ODR_OK;
}
int jtype(OdrJson j) {
    const char *p=j.data;
    if(!p)return 0;
    ws(&p,j.data+j.size);
    return p<j.data+j.size?*p:0;
}
void jiter(OdrJson j,Ji *it) {
    it->value=j;
    it->p=j.data;
    it->object=jtype(j)=='{';
    if(it->p) {
        ws(&it->p,j.data+j.size);
        ++it->p;
    }
}
int jnext(Ji *it,OdrJson *key,OdrJson *value) {
    const char *end=it->value.data+it->value.size,*p=it->p,*s;
    ws(&p,end);
    if(p==end||*p==']'||*p=='}')return 0;
    if(*p==',') {
        ++p;
        ws(&p,end);
    }
    if(key) {
        key->data=NULL;
        key->size=0;
    }
    if(it->object) {
        s=p;
        if(!string_end(&p,end))return 0;
        if(key) {
            key->data=s;
            key->size=(size_t)(p-s);
        }
        ws(&p,end);
        if(p==end||*p++!=':')return 0;
        ws(&p,end);
    }
    s=p;
    if(!skip(&p,end,0,0))return 0;
    value->data=s;
    value->size=(size_t)(p-s);
    it->p=p;
    return 1;
}
int jstring(OdrJson j,char *out,size_t capacity,size_t *length) {
    const char *p=j.data,*end;
    size_t n=0;
    uint32_t c;
    if(!p||j.size<2||*p!='"')return 0;
    end=p+j.size-1;
    if(*end!='"')return 0;
    ++p;
    while(p<end) {
        unsigned char b[4];
        unsigned k,i;
        if(!scalar(&p,end,&c))return 0;
        if(c<0x80) {
            b[0]=(unsigned char)c;
            k=1;
        }
        else if(c<0x800) {
            b[0]=0xc0|(c>>6);
            b[1]=0x80|(c&63);
            k=2;
        }
        else if(c<0x10000) {
            b[0]=0xe0|(c>>12);
            b[1]=0x80|((c>>6)&63);
            b[2]=0x80|(c&63);
            k=3;
        }
        else {
            b[0]=0xf0|(c>>18);
            b[1]=0x80|((c>>12)&63);
            b[2]=0x80|((c>>6)&63);
            b[3]=0x80|(c&63);
            k=4;
        }
        for(i=0;i<k;i++) {
            if(out&&n<capacity)out[n]=b[i];
            ++n;
        }
    }
    if(out&&n<capacity)out[n]=0;
    if(length)*length=n;
    return !out||n<capacity;
}
int jeq(OdrJson j,const char *s) {
    const char *p=j.data,*end;
    uint32_t c;
    if(!p||j.size<2||*p!='"')return 0;
    end=p+j.size-1;
    ++p;
    while(p<end) {
        if(!scalar(&p,end,&c)||c>127||!(*s)||(unsigned char)*s++!=c)return 0;
    }
    return !*s;
}
int jget(OdrJson j,const char *name,OdrJson *out) {
    Ji it;
    OdrJson k,v;
    if(jtype(j)!='{')return 0;
    jiter(j,&it);
    while(jnext(&it,&k,&v))if(jeq(k,name)) {
        *out=v;
        return 1;
    }
    return 0;
}
size_t jcount(OdrJson j) {
    Ji it;
    OdrJson k,v;
    size_t n=0;
    jiter(j,&it);
    while(jnext(&it,&k,&v))++n;
    return n;
}
char *jcopy(OdrJson j,Arena *a) {
    size_t n=0;
    char *p;
    if(!jstring(j,NULL,0,&n)) {
        a->result=ODR_SCHEMA;
        return NULL;
    }
    p=odr_take(a,n+1,1);
    if(p)jstring(j,p,n+1,NULL);
    return p;
}
int ju64(OdrJson j,uint64_t *out) {
    uint64_t n=0;
    size_t i;
    if(!j.size)return 0;
    for(i=0;i<j.size;i++) {
        unsigned c=(unsigned char)j.data[i];
        if(c<'0'||c>'9'||n>(UINT64_MAX-(c-'0'))/10)return 0;
        n=n*10+c-'0';
    }
    *out=n;
    return 1;
}
int jfloat(OdrJson j,float *out) {
    const char *p=j.data,*end=p+j.size;
    long double n=0;
    int sign=1,exp=0,es=1,frac=0;
    if(!j.size)return 0;
    if(*p=='-') {
        sign=-1;
        ++p;
    }
    while(p<end&&isdigit((unsigned char)*p)) {
        n=n*10+(*p++-'0');
    }
    if(p<end&&*p=='.') {
        ++p;
        while(p<end&&isdigit((unsigned char)*p)) {
            n=n*10+(*p++-'0');
            ++frac;
        }
    }
    if(p<end&&(*p=='e'||*p=='E')) {
        ++p;
        if(p<end&&(*p=='-'||*p=='+')) {
            if(*p=='-')es=-1;
            ++p;
        }
        while(p<end&&isdigit((unsigned char)*p)) {
            if(exp<100000)exp=exp*10+(*p-'0');
            ++p;
        }
    }
    if(p!=end||jtype(j)=='n'||jtype(j)=='t'||jtype(j)=='f')return 0;
    if(n==0){*out=sign<0?-0.0f:0.0f;return 1;}
    n=sign*n*powl(10.0L,(long double)(es*exp-frac));
    if(!isfinite(n)||fabsl(n)>FLT_MAX)return 0;
    *out=(float)n;
    return isfinite(*out);
}
int jkeys(OdrJson j,const char *allowed,OdrError *e) {
    Ji it;
    OdrJson k,v;
    char name[ODR_NAME];
    if(jtype(j)!='{') {
        odr_error(e,ODR_SCHEMA,0,"expected object");
        return 0;
    }
    jiter(j,&it);
    while(jnext(&it,&k,&v)) {
        const char *p=allowed;
        int found=0;
        if(!jstring(k,name,sizeof(name),NULL)) {
            odr_error(e,ODR_SCHEMA,0,"field name too long");
            return 0;
        }
        while(*p) {
            const char *end=strchr(p,',');
            size_t n=end?(size_t)(end-p):strlen(p);
            if(strlen(name)==n&&!memcmp(name,p,n)) {
                found=1;
                break;
            }
            if(!end)break;
            p=end+1;
        }
        if(!found) {
            if(e) {
                e->code=ODR_SCHEMA;
                e->offset=0;
                snprintf(e->message,sizeof(e->message),"unknown field: %.127s",name);
            }
            return 0;
        }
    }
    return 1;
}
OdrJson jproblem(OdrJson j) {
    OdrJson p;
    return jget(j,"problem",&p)?p:j;
}
OdrResult odr_section(OdrJson j,const char *key,OdrJson *out,OdrError *e) {
    OdrResult r;
    if(!key||!out)return odr_error(e,ODR_SCHEMA,0,"missing section arguments");
    out->data=NULL;
    out->size=0;
    r=jvalidate(j,e);
    if(r)return r;
    if(!jget(j,key,out))return odr_error(e,ODR_SCHEMA,0,"missing section");
    return ODR_OK;
}
OdrResult odr_states(OdrJson j,Arena *a,const char ***names,uint32_t *count,OdrError *e) {
    Ji it;
    OdrJson k,v,states;
    const char **rows;
    uint32_t n=0;
    char name[ODR_NAME];
    if(!jget(j,"states",&states)||jtype(states)!='['||!jcount(states)||jcount(states)>ODR_STATES)return odr_error(e,ODR_SCHEMA,0,"states must contain 1..126 names");
    *count=(uint32_t)jcount(states);
    rows=odr_take(a,*count,sizeof(*rows));
    jiter(states,&it);
    while(jnext(&it,&k,&v)) {
        Ji prev;
        OdrJson pk,pv;
        size_t i;
        if(!jstring(v,name,sizeof(name),NULL)||!name[0]||!strcmp(name,"t")||!(isalpha((unsigned char)name[0])||name[0]=='_'))return odr_error(e,ODR_SCHEMA,0,"invalid state name");
        for(i=1;name[i];i++)if(!(isalnum((unsigned char)name[i])||name[i]=='_'))return odr_error(e,ODR_SCHEMA,0,"invalid state identifier");
        jiter(states,&prev);
        while(jnext(&prev,&pk,&pv)&&pv.data<v.data)if(jeq(pv,name))return odr_error(e,ODR_SCHEMA,0,"duplicate state name");
        {
            char *s=jcopy(v,a);
            if(rows)rows[n]=s;
        }
        ++n;
    }
    if(names)*names=rows;
    return a->result;
}
