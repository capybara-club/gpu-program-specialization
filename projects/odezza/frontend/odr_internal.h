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
#ifndef ODR_INTERNAL_H
#define ODR_INTERNAL_H
#include "odezza_request.h"
#include <math.h>
#include <float.h>
#include <limits.h>
#include <string.h>
#include <stdio.h>
#define ODR_DEPTH 128u
#define ODR_STATES 126u
#define ODR_ARGS 16u
#define ODR_NAME 128u
typedef union OdrAlign { long double d; void *p; uint64_t u; } OdrAlign;
typedef struct Arena { unsigned char *base; size_t capacity, used; OdrResult result; } Arena;
void *odr_take(Arena *a, size_t count, size_t size);
OdrResult odr_error(OdrError *e, OdrResult r, size_t offset, const char *message);
typedef struct Ji { OdrJson value; const char *p; int object; } Ji;
OdrResult jvalidate(OdrJson j, OdrError *e);
int jeq(OdrJson j, const char *s);
int jtype(OdrJson j);
int jget(OdrJson j, const char *key, OdrJson *out);
size_t jcount(OdrJson j);
void jiter(OdrJson j, Ji *it);
int jnext(Ji *it, OdrJson *key, OdrJson *value);
int jfloat(OdrJson j, float *out);
int ju64(OdrJson j, uint64_t *out);
int jstring(OdrJson j, char *out, size_t capacity, size_t *length);
char *jcopy(OdrJson j, Arena *a);
int jkeys(OdrJson j, const char *allowed, OdrError *e);
OdrJson jproblem(OdrJson j);
OdrResult odr_states(OdrJson j, Arena *a, const char ***names, uint32_t *count, OdrError *e);
/* Prepared expression nodes refer to names; resolved once into rule/slot IDs. */
enum { N_LITERAL=1, N_NAME, N_REF, N_CALL, N_POW, N_NEG, N_POS,
       N_ADD=0x90, N_SUB, N_MUL, N_DIV };
typedef struct Node { uint32_t op, argc; float value; const char *name;
    const void *binding; uint32_t binding_kind, binding_index;
    const struct Node *args[ODR_ARGS]; } Node;
OdrResult odr_expr(const char *text, Arena *a, const Node **out, OdrError *error);
OdrResult odr_expr_measure(const char *text, size_t *nodes, OdrError *error);
int odr_unary(const char *name);
OdrResult odr_allocation(OdrJson limits,OdrJson families,size_t index,const char *key,uint64_t fallback,uint64_t *out,OdrError *error);
#endif
