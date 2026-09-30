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
typedef struct Writer {
    unsigned char *p;
    size_t n,cap;
    OdrResult result;
    OdrJson states;
}
Writer;
static void byte(Writer *w,unsigned b) {
    if(w->n==SIZE_MAX) {
        w->result=ODR_OVERFLOW;
        return;
    }
    if(w->p&&w->n<w->cap)w->p[w->n]=(unsigned char)b;
    ++w->n;
}
static void emit(Writer *w,const Node *n,unsigned depth) {
    unsigned i;
    int exponent,code;
    uint32_t bits;
    if(w->result)return;
    if(!n||depth>ODR_DEPTH) {
        w->result=ODR_AST;
        return;
    }
    if(n->op==N_LITERAL) {
        memcpy(&bits,&n->value,4);
        byte(w,0x83);
        for(i=0;i<4;i++)byte(w,bits>>(8*i));
    }
    else if(n->op==N_NAME) {
        Ji it;
        OdrJson k,v;
        unsigned state=0;
        int found=0;
        jiter(w->states,&it);
        while(jnext(&it,&k,&v)) {
            if(jeq(v,n->name)) {
                found=1;
                break;
            }
            ++state;
        }
        if(!found) {
            w->result=ODR_AST;
            return;
        }
        byte(w,0x81);
        byte(w,state);
    }
    else if(n->op==N_POW) {
        exponent=(int)n->value;
        if(exponent==0) {
            byte(w,0x83);
            byte(w,0);
            byte(w,0);
            byte(w,0x80);
            byte(w,0x3f);
        }
        else {
            if(exponent<0) {
                byte(w,0x83);
                byte(w,0);
                byte(w,0);
                byte(w,0x80);
                byte(w,0x3f);
            }
            for(i=0;i<(unsigned)(exponent<0?-exponent:exponent);i++) {
                emit(w,n->args[0],depth+1);
                if(i)byte(w,0x92);
            }
            if(exponent<0)byte(w,0x93);
        }
    }
    else if(n->op==N_REF) {
        w->result=ODR_AST;
    }
    else if(n->op==N_CALL) {
        code=odr_unary(n->name);
        if(!code) {
            w->result=ODR_AST;
            return;
        }
        emit(w,n->args[0],depth+1);
        byte(w,(unsigned)code);
    }
    else {
        for(i=0;i<n->argc;i++)emit(w,n->args[i],depth+1);
        byte(w,n->op==N_NEG?0x94:n->op);
    }
    if(w->n>16u*1024u*1024u)w->result=ODR_CAPACITY;
}
OdrResult odr_static_parse(OdrJson input,void *arena,size_t capacity,size_t *required,const OdrStatic **out,OdrError *error) {
    union {
        OdrAlign align;
        unsigned char data[262144];
    }
    scratch;
    Arena a= {
        (unsigned char *)arena,capacity,0,ODR_OK
    };
    OdrStatic *s;
    OdrJson j=jproblem(input),rhs= {
        NULL,0
    },states,k,v;
    OdrRhs *rows;
    const char **names;
    uint32_t ns=0,nrhs=0,index=0;
    Ji it;
    OdrResult r;
    if(out)*out=NULL;
    if(required)*required=0;
    if(!required)return odr_error(error,ODR_SCHEMA,0,"required_bytes is mandatory");
    if(arena&&(uintptr_t)arena%odr_arena_alignment())return odr_error(error,ODR_SCHEMA,0,"unaligned arena");
    r=jvalidate(input,error);
    if(r)return r;
    if(!jkeys(j,"states,known_rhs,trajectories",error))return ODR_SCHEMA;
    s=odr_take(&a,1,sizeof(*s));
    r=odr_states(j,&a,&names,&ns,error);
    if(r)return r;
    (void)jget(j,"states",&states);
    if(jget(j,"known_rhs",&rhs)) {
        if(jtype(rhs)!='{')return odr_error(error,ODR_SCHEMA,0,"known_rhs must be an object");
        nrhs=(uint32_t)jcount(rhs);
        if(nrhs>ns)return ODR_SCHEMA;
    }
    rows=odr_take(&a,nrhs,sizeof(*rows));
    if(rhs.data) {
        jiter(rhs,&it);
        while(jnext(&it,&k,&v)) {
            char name[ODR_NAME],text[32768];
            Arena tmp= {
                scratch.data,sizeof(scratch.data),0,ODR_OK
            };
            const Node *root=NULL;
            Writer w;
            unsigned state=0;
            Ji sit;
            OdrJson sk,sv;
            int found=0;
            unsigned char *code;
            if(!jstring(k,name,sizeof(name),NULL)||!jstring(v,text,sizeof(text),NULL))return odr_error(error,ODR_SCHEMA,0,"known RHS must be expression strings under 32768 bytes");
            jiter(states,&sit);
            while(jnext(&sit,&sk,&sv)) {
                if(jeq(sv,name)) {
                    found=1;
                    break;
                }
                ++state;
            }
            if(!found)return odr_error(error,ODR_SCHEMA,0,"known RHS names an undeclared state");
            r=odr_expr(text,&tmp,&root,error);
            if(r)return r;
            if(tmp.used>tmp.capacity)return odr_error(error,ODR_CAPACITY,0,"static expression parser capacity exceeded");
            memset(&w,0,sizeof(w));
            w.states=states;
            emit(&w,root,0);
            byte(&w,0x80);
            if(w.result)return odr_error(error,w.result,0,"static RHS must contain only states, literals and scoring operators");
            code=odr_take(&a,w.n,1);
            if(rows) {
                rows[index].state_index=state;
                rows[index].program.bytes=code;
                rows[index].program.byte_count=w.n;
            }
            if(code) {
                w.p=code;
                w.cap=w.n;
                w.n=0;
                emit(&w,root,0);
                byte(&w,0x80);
            }
            ++index;
        }
    }
    *required=a.used;
    if(a.result)return a.result;
    if(arena&&capacity<a.used)return odr_error(error,ODR_BUFFER,0,"static arena too small");
    if(s) {
        s->state_count=ns;
        s->rhs_count=nrhs;
        s->states=names;
        s->rhs=rows;
    }
    if(out)*out=s;
    return ODR_OK;
}
