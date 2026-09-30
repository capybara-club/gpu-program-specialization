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
typedef struct Expr {
    const char *start,*p;
    Arena *arena;
    OdrError *error;
    OdrResult result;
    unsigned depth;
    size_t nodes;
}
Expr;
typedef struct Value {
    const Node *node;
    float number;
    int literal;
    char name[ODR_NAME];
}
Value;
static void spaces(Expr *p) {
    while(isspace((unsigned char)*p->p))++p->p;
}
static void failure(Expr *p,const char *message) {
    if(!p->result)p->result=odr_error(p->error,ODR_AST,(size_t)(p->p-p->start),message);
}
static Value expression(Expr *p);
static Value unary(Expr *p);
static Value build(Expr *p,unsigned op,const char *name,Value *args,unsigned argc,float number) {
    Value v;
    Node *n;
    unsigned i;
    memset(&v,0,sizeof(v));
    v.literal=op==N_LITERAL;
    v.number=number;
    if(name) {
        size_t len=strlen(name);
        if(len>=sizeof(v.name)) {
            failure(p,"name too long");
            return v;
        }
        memcpy(v.name,name,len+1);
    }
    n=odr_take(p->arena,1,sizeof(*n));
    ++p->nodes;
    if(n) {
        n->op=op;
        n->argc=argc;
        n->value=number;
        for(i=0;i<argc;i++)n->args[i]=args[i].node;
    }
    if(name) {
        char *s=odr_take(p->arena,strlen(name)+1,1);
        if(s)strcpy(s,name);
        if(n)n->name=s;
    }
    v.node=n;
    return v;
}
static Value atom(Expr *p) {
    Value v,args[ODR_ARGS];
    char name[ODR_NAME];
    size_t len=0;
    unsigned argc=0;
    memset(&v,0,sizeof(v));
    spaces(p);
    if(++p->depth>ODR_DEPTH) {
        failure(p,"expression nesting limit");
        --p->depth;
        return v;
    }
    if(*p->p=='(') {
        ++p->p;
        v=expression(p);
        spaces(p);
        if(*p->p!=')')failure(p,"expected closing parenthesis");
        else ++p->p;
    }
    else if(isdigit((unsigned char)*p->p)||(*p->p=='.'&&isdigit((unsigned char)p->p[1]))) {
        const char *s=p->p;
        OdrJson j;
        float x;
        while(isdigit((unsigned char)*p->p))++p->p;
        if(*p->p=='.') {
            ++p->p;
            while(isdigit((unsigned char)*p->p))++p->p;
        }
        if(*p->p=='e'||*p->p=='E') {
            ++p->p;
            if(*p->p=='+'||*p->p=='-')++p->p;
            if(!isdigit((unsigned char)*p->p))failure(p,"invalid exponent");
            while(isdigit((unsigned char)*p->p))++p->p;
        }
        j.data=s;
        j.size=(size_t)(p->p-s);
        if(!jfloat(j,&x))failure(p,"literal is not finite FP32");
        else v=build(p,N_LITERAL,NULL,NULL,0,x);
    }
    else if(isalpha((unsigned char)*p->p)||*p->p=='_') {
        while(isalnum((unsigned char)*p->p)||*p->p=='_'||*p->p=='.') {
            if(len+1<sizeof(name))name[len++]=*p->p;
            else failure(p,"name too long");
            ++p->p;
        }
        name[len]=0;
        spaces(p);
        if(*p->p=='(') {
            ++p->p;
            spaces(p);
            if(*p->p!=')')for(;;) {
                if(argc==ODR_ARGS) {
                    failure(p,"too many arguments");
                    break;
                }
                args[argc++]=expression(p);
                spaces(p);
                if(*p->p!=',')break;
                ++p->p;
            }
            if(*p->p!=')')failure(p,"expected function closing parenthesis");
            else ++p->p;
            if(!strcmp(name,"pow")) {
                if(argc!=2||!args[1].literal||args[1].number!=truncf(args[1].number)||fabsf(args[1].number)>16)failure(p,"pow requires a literal integer exponent in [-16,16]");
                else v=build(p,N_POW,NULL,args,1,args[1].number);
            }
            else if(!strcmp(name,"hole")) {
                if(!argc||!args[0].name[0]||strchr(args[0].name,'.'))failure(p,"hole requires a rule name");
                else v=build(p,N_CALL,args[0].name,args+1,argc-1,0);
            }
            else if(odr_unary(name)) {
                if(argc!=1)failure(p,"unary function requires one argument");
                else v=build(p,N_CALL,name,args,argc,0);
            }
            else v=build(p,N_CALL,name,args,argc,0);
        }
        else v=build(p,strchr(name,'.')?N_REF:N_NAME,name,NULL,0,0);
    }
    else failure(p,"expected expression");
    --p->depth;
    return v;
}
static Value power(Expr *p) {
    Value v=atom(p);
    spaces(p);
    if(p->p[0]=='*'&&p->p[1]=='*') {
        Value exp;
        p->p+=2;
        exp=unary(p);
        if(!exp.literal||exp.number!=truncf(exp.number)||fabsf(exp.number)>16)failure(p,"power requires literal integer exponent in [-16,16]");
        else v=build(p,N_POW,NULL,&v,1,exp.number);
    }
    return v;
}
static Value unary(Expr *p) {
    Value v;
    spaces(p);
    if(*p->p=='-'||*p->p=='+') {
        int negative=*p->p++=='-';
        if(++p->depth>ODR_DEPTH) {
            memset(&v,0,sizeof(v));
            failure(p,"unary nesting limit");
            --p->depth;
            return v;
        }
        v=unary(p);
        --p->depth;
        if(negative) {
            if(v.literal)v=build(p,N_LITERAL,NULL,NULL,0,-v.number);
            else v=build(p,N_NEG,NULL,&v,1,0);
        }
        return v;
    }
    return power(p);
}
static Value multiply(Expr *p) {
    Value v=unary(p);
    spaces(p);
    while(!p->result&&(*p->p=='*'||*p->p=='/')) {
        unsigned op=*p->p++=='*'?N_MUL:N_DIV;
        Value args[2];
        args[0]=v;
        args[1]=unary(p);
        v=build(p,op,NULL,args,2,0);
        spaces(p);
    }
    return v;
}
static Value expression(Expr *p) {
    Value v=multiply(p);
    spaces(p);
    while(!p->result&&(*p->p=='+'||*p->p=='-')) {
        unsigned op=*p->p++=='+'?N_ADD:N_SUB;
        Value args[2];
        args[0]=v;
        args[1]=multiply(p);
        v=build(p,op,NULL,args,2,0);
        spaces(p);
    }
    return v;
}
int odr_unary(const char *name) {
    static const char *names[]= {
        "sqrt","abs","sin","cos","tanh","exp","log"
    };
    static const int codes[]= {
        0x95,0x97,0x9b,0x9c,0xa0,0xa1,0xa2
    };
    size_t i;
    for(i=0;i<sizeof(names)/sizeof(*names);i++)if(!strcmp(name,names[i]))return codes[i];
    return 0;
}
OdrResult odr_expr(const char *text,Arena *a,const Node **out,OdrError *error) {
    Expr p;
    Value v;
    memset(&p,0,sizeof(p));
    p.start=p.p=text;
    p.arena=a;
    p.error=error;
    v=expression(&p);
    spaces(&p);
    if(*p.p)failure(&p,"trailing expression input");
    if(out)*out=v.node;
    return p.result?p.result:a->result;
}
OdrResult odr_expr_measure(const char *text,size_t *nodes,OdrError *error) {
    Arena a= {
        NULL,0,0,ODR_OK
    };
    Expr p;
    Value v;
    memset(&p,0,sizeof(p));
    p.start=p.p=text;
    p.arena=&a;
    p.error=error;
    v=expression(&p);
    (void)v;
    spaces(&p);
    if(*p.p)failure(&p,"trailing expression input");
    *nodes=p.nodes;
    return p.result?p.result:a.result;
}
