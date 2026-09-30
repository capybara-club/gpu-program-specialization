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
#include "lower.h"
#include "secant_instructions.h"
#include <math.h>
#include <string.h>
static unsigned
arity(unsigned op)
{
  switch (op) {
  case SECANT_AST_INSTRUCTION_TYPE_ADD_F32:
  case SECANT_AST_INSTRUCTION_TYPE_SUB_F32:
  case SECANT_AST_INSTRUCTION_TYPE_MUL_F32:
  case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
  case SECANT_AST_INSTRUCTION_TYPE_MIN_F32:
  case SECANT_AST_INSTRUCTION_TYPE_MAX_F32:
    return 2;
  case SECANT_AST_INSTRUCTION_TYPE_NEG_F32:
  case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32:
  case SECANT_AST_INSTRUCTION_TYPE_ABS_F32:
  case SECANT_AST_INSTRUCTION_TYPE_TANH_F32:
  case SECANT_AST_INSTRUCTION_TYPE_SIN_F32:
  case SECANT_AST_INSTRUCTION_TYPE_COS_F32:
  case SECANT_AST_INSTRUCTION_TYPE_EXP_F32:
  case SECANT_AST_INSTRUCTION_TYPE_LOG_F32:
    return 1;
  default:
    return 0;
  }
}
int
b_parse(const uint8_t * p, size_t length, unsigned cols, unsigned constants, unsigned bits, BTree *t)
{
  unsigned stack[B_NODES], sp = 0;
  size_t at = 0;
  memset(t, 0, sizeof(*t));
  if (!p || !length || bits > 16)
    return 0;
  while (at < length && t->count < B_NODES) {
    unsigned op = p[at], a = arity(op), width = 1, i;
    BNode *n = t->node + t->count;
    if (op == SECANT_AST_INSTRUCTION_TYPE_RETURN_F32) {
      if (sp != 1)
        return 0;
      t->root = stack[0];
      t->bytes = at + 1;
      return 1;
    }
    if (op == SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32 || op == SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32) {
      unsigned choices = op == SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32 ? 2 : 4;
      width = choices == 2 ? 2 : 3;
      if (at + width > length || sp < choices)
        return 0;
      n->choice.count = choices;
      n->choice.lo = p[at + 1];
      n->choice.hi = choices == 4 ? p[at + 2] : 0;
      if (n->choice.lo >= bits || (choices == 4 && (n->choice.hi >= bits || n->choice.hi == n->choice.lo)))
        return 0;
      for (i = 0; i < choices; ++i) {
        BNode *v = t->node + stack[sp - choices + i];
        if (v->op || v->choice.count != 1)
          return 0;
        n->choice.leaf[i] = v->choice.leaf[0];
      }
      sp -= choices;
    } else if (op == SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32 || op == SECANT_AST_INSTRUCTION_TYPE_BANK_CONSTANT_F32 || op == SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32) {
      BLeaf *l = n->choice.leaf;
      width = op == SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32 ? 10 : 2;
      if (at + width > length)
        return 0;
      n->choice.count = 1;
      l->slot = p[at + 1];
      l->kind = op == SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32 ? 0 : 1;
      l->scale = 1;
      if (l->slot >= (l->kind ? constants : cols))
        return 0;
      if (width == 10) {
        l->kind = 3;
        memcpy(&l->scale, p + at + 2, 4);
        memcpy(&l->offset, p + at + 6, 4);
        if (!isfinite(l->scale) || !isfinite(l->offset))
          return 0;
      }
    } else if (op == SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32) {
      width = 5;
      if (at + width > length)
        return 0;
      n->choice.count = 1;
      n->choice.leaf[0].kind = 2;
      memcpy(&n->choice.leaf[0].offset, p + at + 1, 4);
      if (!isfinite(n->choice.leaf[0].offset))
        return 0;
    } else if (a) {
      if (sp < a)
        return 0;
      n->op = op;
      n->arity = a;
      for (i = 0; i < a; ++i)
        n->child[i] = stack[sp - a + i];
      sp -= a;
    } else
      return 0;
    stack[sp++] = t->count++;
    at += width;
  } return 0;
}
BLeaf
b_choice(const BChoice *c, unsigned p)
{
  unsigned i = c->count > 1 ? (p >> c->lo) & 1u : 0;
  if (c->count == 4)
    i |= ((p >> c->hi) & 1u) << 1;
  return c->leaf[i];
}
float
b_constant(BLeaf l, const float *bank)
{
  volatile float v;
  if (l.kind == 2)
    return l.offset;
  if (l.kind == 1)
    return bank[l.slot];
  v = l.scale * bank[l.slot];
  return v + l.offset;
}
static int
emit(const BTree *t, unsigned id, uint8_t * p, size_t cap, size_t * at, BChoice *choices, size_t * slots, size_t limit, int resolve, unsigned permutation, const float *bank)
{
  const BNode *n = t->node + id;
  unsigned i;
  if (n->op) {
    for (i = 0; i < n->arity; ++i)
      if (!emit(t, n->child[i], p, cap, at, choices, slots, limit, resolve, permutation, bank))
        return 0;
    if (*at >= cap)
      return 0;
    p[(*at)++] = (uint8_t) n->op;
  } else if (resolve || (n->choice.count == 1 && n->choice.leaf[0].kind == 2)) {
    BLeaf l = b_choice(&n->choice, permutation);
    float v;
    if (*at + (l.kind == 0 ? 2 : 5) > cap)
      return 0;
    if (l.kind == 0) {
      p[(*at)++] = SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32;
      p[(*at)++] = (uint8_t) l.slot;
    } else {
      v = b_constant(l, bank);
      p[(*at)++] = SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32;
      memcpy(p + *at, &v, 4);
      *at += 4;
    }
  } else {
    if (*slots >= limit || *at + 2 > cap)
      return 0;
    choices[*slots] = n->choice;
    p[(*at)++] = 0xb9;
    p[(*at)++] = (uint8_t) (*slots);
    ++*slots;
  }
  return 1;
}
int
b_lower(const BTree *t, uint8_t * p, size_t cap, BChoice *choices, size_t * slots, size_t limit)
{
  size_t at = 0;
  if (!t->count || !emit(t, t->root, p, cap, &at, choices, slots, limit, 0, 0, NULL) || at >= cap)
    return 0;
  p[at++] = 0x83;
  return (int)at;
}
int
b_resolve(const BTree *t, unsigned perm, const float *bank, uint8_t * p, size_t cap)
{
  size_t at = 0, n = 0;
  if (!t->count || !emit(t, t->root, p, cap, &at, NULL, &n, 0, 1, perm, bank) || at >= cap)
    return 0;
  p[at++] = 0x83;
  return (int)at;
}
int
b_tables(const BChoice *choices, size_t slots, size_t stride, const float *banks, size_t nb, size_t nc, unsigned bits, uint32_t * masks, uint32_t * words)
{
  size_t b, p, s, perms = (size_t) 1 << bits;
  if (slots > stride || stride > 32 || bits > 16)
    return 0;
  memset(masks, 0, nb * perms * 4);
  memset(words, 0, nb * perms * stride * 4);
  for (p = 0; p < perms; ++p)
    for (s = 0; s < slots; ++s) {
      BLeaf l = b_choice(choices + s, (unsigned)p);
      for (b = 0; b < nb; ++b) {
        size_t q = b * perms + p;
        if (l.kind == 0) {
          masks[q] |= 1u << s;
          words[q * stride + s] = l.slot;
        } else {
          float v = b_constant(l, banks + b * nc);
          memcpy(words + q * stride + s, &v, 4);
        }
      }
    }
  return 1;
}
