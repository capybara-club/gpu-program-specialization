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
#ifndef BENCH_LOWER_H
#define BENCH_LOWER_H
#include "secant.h"
#define B_CODE 4096u
#define B_NODES 256u
typedef struct BLeaf { unsigned kind,slot; float scale,offset; } BLeaf;
typedef struct BChoice { unsigned count,lo,hi; BLeaf leaf[4]; } BChoice;
typedef struct BNode { unsigned op,arity,child[2]; BChoice choice; } BNode;
typedef struct BTree { BNode node[B_NODES]; unsigned count,root; size_t bytes; } BTree;
/* Only the GP's leaf selectors are accepted. Arithmetic inside a selector is
 * rejected explicitly, not resolved or silently expanded. */
int b_parse(const uint8_t *,size_t,unsigned,unsigned,unsigned,BTree *);
int b_lower(const BTree *,uint8_t *,size_t,BChoice *,size_t *,size_t);
BLeaf b_choice(const BChoice *,unsigned);
float b_constant(BLeaf,const float *);
int b_resolve(const BTree *,unsigned,const float *,uint8_t *,size_t);
int b_tables(const BChoice *,size_t,size_t,const float *,size_t,size_t,unsigned,uint32_t *,uint32_t *);
#endif
