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
#include "internal.h"
#include <stdarg.h>
#include <stdio.h>
unsigned sr_arity(uint8_t op) {
  switch (op) {
  case 0:
    return 0;
  case SECANT_AST_INSTRUCTION_TYPE_ADD_F32:
  case SECANT_AST_INSTRUCTION_TYPE_SUB_F32:
  case SECANT_AST_INSTRUCTION_TYPE_MUL_F32:
  case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
  case SECANT_AST_INSTRUCTION_TYPE_MIN_F32:
  case SECANT_AST_INSTRUCTION_TYPE_MAX_F32:
    return 2;
  case SECANT_AST_INSTRUCTION_TYPE_NEG_F32:
  case SECANT_AST_INSTRUCTION_TYPE_SIN_F32:
  case SECANT_AST_INSTRUCTION_TYPE_COS_F32:
  case SECANT_AST_INSTRUCTION_TYPE_TANH_F32:
  case SECANT_AST_INSTRUCTION_TYPE_ABS_F32:
  case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32:
  case SECANT_AST_INSTRUCTION_TYPE_EXP_F32:
  case SECANT_AST_INSTRUCTION_TYPE_LOG_F32:
    return 1;
  default:
    return UINT_MAX;
  }
}
SecantResult sr_validate_config(const SRConfig *c) {
  size_t i, n;
  if (!c || !c->population || c->population > UINT32_MAX || !c->num_inputs ||
      c->num_inputs > 128 || c->num_constants > 128 - c->num_inputs ||
      !c->num_banks || c->toggle_bits > SECANT_SR_MAX_BITS || !c->max_nodes ||
      c->max_nodes > SECANT_SR_MAX_NODES || c->max_depth > 32 ||
      c->initial_depth > c->max_depth || !c->tournament ||
      c->elites >= c->population || !c->operators || !c->num_operators ||
      c->num_operators > 32 || !isfinite(c->parsimony) || c->parsimony < 0 ||
      !isfinite(c->crossover_probability) || c->crossover_probability < 0 ||
      !isfinite(c->mutation_probability) || c->mutation_probability < 0 ||
      c->crossover_probability + c->mutation_probability > 1 ||
      !isfinite(c->toggle_probability) || c->toggle_probability < 0 ||
      c->toggle_probability > 1 || !isfinite(c->four_way_probability) ||
      c->four_way_probability < 0 || c->four_way_probability > 1 ||
      !isfinite(c->coefficient_probability) || c->coefficient_probability < 0 ||
      c->coefficient_probability > 1 || c->align_crossover_bits > 1 ||
      !isfinite(c->toggle_mutation_probability) || c->toggle_mutation_probability < 0 ||
      c->toggle_mutation_probability > 1 || !isfinite(c->leaf_mix_probability) ||
      c->leaf_mix_probability < 0 || c->leaf_mix_probability > 1 ||
      !isfinite(c->power_mutation_probability) || c->power_mutation_probability < 0 ||
      c->power_mutation_probability > 1)
    return SECANT_ERROR_INVALID_VALUE;
  if (!sr_mul(c->num_banks, (size_t)1 << c->toggle_bits, &n) ||
      !sr_mul(c->num_banks, c->num_constants, &n))
    return SECANT_ERROR_OVERFLOW;
  for (i = 0; i < c->num_operators; ++i)
    if (!sr_arity(c->operators[i]) || sr_arity(c->operators[i]) > 2)
      return SECANT_ERROR_UNSUPPORTED_OP;
  return SECANT_SUCCESS;
}
SecantResult sr_validate(const SRConfig *c, const SRNode *nodes, size_t n,
                         uint16_t *first, uint16_t *depth) {
  uint16_t stack[SECANT_SR_MAX_NODES], starts[SECANT_SR_MAX_NODES],
      depths[SECANT_SR_MAX_NODES];
  size_t i, j, sp = 0;
  if (!c || !nodes || !n || n > c->max_nodes || n > SECANT_SR_MAX_NODES)
    return SECANT_ERROR_BAD_PROGRAM;
  for (i = 0; i < n; ++i) {
    unsigned a = sr_arity(nodes[i].op);
    uint16_t d = 0;
    if (a > sp)
      return SECANT_ERROR_BAD_PROGRAM;
    starts[i] = (uint16_t)i;
    if (!a) {
      unsigned choices = nodes[i].choices;
      if (choices != 1 && choices != 2 && choices != 4)
        return SECANT_ERROR_BAD_PROGRAM;
      if (choices > 1 && nodes[i].low_bit >= c->toggle_bits)
        return SECANT_ERROR_BAD_PROGRAM;
      if (choices == 4 && (nodes[i].high_bit >= c->toggle_bits ||
                           nodes[i].high_bit == nodes[i].low_bit))
        return SECANT_ERROR_BAD_PROGRAM;
      for (j = 0; j < choices; ++j) {
        const SRLeaf *l = &nodes[i].leaf[j];
        if ((l->kind == SR_COLUMN && l->slot >= c->num_inputs) ||
            (l->kind == SR_COEFFICIENT && l->slot >= c->num_constants) ||
            (l->kind == SR_FITTED_COEFFICIENT && !c->num_constants) ||
            ((l->kind == SR_LITERAL || l->kind == SR_FITTED_COEFFICIENT) && !isfinite(l->value)) ||
            l->kind < SR_COLUMN || l->kind > SR_FITTED_COEFFICIENT)
          return SECANT_ERROR_BAD_PROGRAM;
      }
    } else {
      if (nodes[i].choices)
        return SECANT_ERROR_BAD_PROGRAM;
      starts[i] = starts[stack[sp - a]];
      for (j = sp - a; j < sp; ++j)
        if (depths[stack[j]] >= d)
          d = depths[stack[j]] + 1;
      sp -= a;
    }
    if (d > c->max_depth)
      return SECANT_ERROR_BAD_PROGRAM;
    depths[i] = d;
    stack[sp++] = (uint16_t)i;
  }
  if (sp != 1)
    return SECANT_ERROR_BAD_PROGRAM;
  if (first)
    memcpy(first, starts, n * sizeof(*first));
  if (depth)
    *depth = depths[n - 1];
  return SECANT_SUCCESS;
}
uint64_t sr_fingerprint(const SRNode *nodes, size_t count) {
  size_t i, j;
  uint64_t h = UINT64_C(1469598103934665603);
#define HASH(v)                                                                \
  do {                                                                         \
    h ^= (uint64_t)(v);                                                        \
    h *= UINT64_C(1099511628211);                                              \
  } while (0)
  for (i = 0; i < count; ++i) {
    HASH(nodes[i].op);
    HASH(nodes[i].choices);
    if (nodes[i].choices > 1)
      HASH(nodes[i].low_bit);
    if (nodes[i].choices == 4)
      HASH(nodes[i].high_bit);
    for (j = 0; j < nodes[i].choices; ++j) {
      uint32_t bits = 0;
      HASH(nodes[i].leaf[j].kind);
      if (nodes[i].leaf[j].kind == SR_LITERAL || nodes[i].leaf[j].kind == SR_FITTED_COEFFICIENT)
        memcpy(&bits, &nodes[i].leaf[j].value, 4);
      else
        bits = nodes[i].leaf[j].slot;
      HASH(bits);
      if (nodes[i].leaf[j].kind == SR_FITTED_COEFFICIENT) HASH(nodes[i].leaf[j].slot);
    }
  }
#undef HASH
  return h;
}
static size_t sr_leaf_bytes(const SRLeaf *l, uint8_t *p) {
  if (l->kind == SR_FITTED_COEFFICIENT) {
    /* Frozen centers do not index their host parameter identity on the GPU. */
    if (p) secant_ast_affine_bank_write(p, 0, 0.0f, l->value);
    return 10;
  }
  if (l->kind == SR_LITERAL) {
    if (p) {
      p[0] = SECANT_AST_INSTRUCTION_TYPE_CONSTANT_BITS_F32;
      memcpy(p + 1, &l->value, 4);
    }
    return 5;
  }
  if (p) {
    p[0] = l->kind == SR_COLUMN ? SECANT_AST_INSTRUCTION_TYPE_COLUMN_F32
                                : SECANT_AST_INSTRUCTION_TYPE_BANK_CONSTANT_F32;
    p[1] = (uint8_t)l->slot;
  }
  return 2;
}
static SRLeaf sr_selected(const SRNode *node, uint32_t perm,
                          const float *bank) {
  unsigned k = node->choices > 1 ? ((perm >> node->low_bit) & 1) : 0;
  SRLeaf leaf;
  if (node->choices == 4)
    k |= ((perm >> node->high_bit) & 1) << 1;
  leaf = node->leaf[k];
  if (leaf.kind == SR_FITTED_COEFFICIENT) {
    leaf.kind = SR_LITERAL; leaf.slot = 0;
  } else if (leaf.kind == SR_COEFFICIENT) {
    leaf.value = bank[leaf.slot];
    leaf.kind = SR_LITERAL;
    leaf.slot = 0;
  }
  return leaf;
}
static SecantResult sr_encode(const SRConfig *c, const SRNode *nodes, size_t n,
                              const float *bank, uint32_t perm, int resolve,
                              uint8_t *output, size_t cap, size_t *needed) {
  size_t i, j, size = 1, at = 0;
  SecantResult r = sr_validate(c, nodes, n, NULL, NULL);
  if (r != SECANT_SUCCESS || !needed)
    return r == SECANT_SUCCESS ? SECANT_ERROR_INVALID_VALUE : r;
  for (i = 0; i < n; ++i) {
    if (nodes[i].op)
      ++size;
    else if (resolve) {
      SRLeaf l = sr_selected(nodes + i, perm, bank);
      size += sr_leaf_bytes(&l, NULL);
    } else {
      for (j = 0; j < nodes[i].choices; ++j)
        size += sr_leaf_bytes(nodes[i].leaf + j, NULL);
      if (nodes[i].choices > 1)
        size += nodes[i].choices == 2 ? 2 : 3;
    }
  }
  *needed = size;
  if (!output)
    return SECANT_SUCCESS;
  if (cap < size)
    return SECANT_ERROR_INSUFFICIENT_BUFFER;
  for (i = 0; i < n; ++i) {
    if (nodes[i].op)
      output[at++] = nodes[i].op;
    else if (resolve) {
      SRLeaf l = sr_selected(nodes + i, perm, bank);
      at += sr_leaf_bytes(&l, output + at);
    } else {
      for (j = 0; j < nodes[i].choices; ++j)
        at += sr_leaf_bytes(nodes[i].leaf + j, output + at);
      if (nodes[i].choices > 1) {
        output[at++] = nodes[i].choices == 2
                           ? SECANT_AST_INSTRUCTION_TYPE_TOGGLE2_F32
                           : SECANT_AST_INSTRUCTION_TYPE_TOGGLE4_F32;
        output[at++] = nodes[i].low_bit;
        if (nodes[i].choices == 4)
          output[at++] = nodes[i].high_bit;
      }
    }
  }
  output[at] = SECANT_AST_INSTRUCTION_TYPE_RETURN_F32;
  return SECANT_SUCCESS;
}
SecantResult secant_sr_program_write(const SRConfig *c, const SRNode *n,
                                     size_t count, uint8_t *out, size_t cap,
                                     size_t *size) {
  if (sr_validate_config(c) != SECANT_SUCCESS)
    return SECANT_ERROR_INVALID_VALUE;
  return sr_encode(c, n, count, NULL, 0, 0, out, cap, size);
}
static SecantResult sr_bank(const SRConfig *c, const float *banks,
                            size_t elements, uint64_t index,
                            const float **out) {
  size_t count, configs, i;
  if (sr_validate_config(c) != SECANT_SUCCESS)
    return SECANT_ERROR_INVALID_VALUE;
  if (!sr_mul(c->num_banks, c->num_constants, &count) ||
      !sr_mul(c->num_banks, (size_t)1 << c->toggle_bits, &configs))
    return SECANT_ERROR_OVERFLOW;
  if (index >= configs || elements < count || (count && !banks))
    return SECANT_ERROR_INVALID_VALUE;
  *out = count ? banks + (index >> c->toggle_bits) * c->num_constants : NULL;
  for (i = 0; i < c->num_constants; ++i)
    if (!isfinite((*out)[i]))
      return SECANT_ERROR_INVALID_VALUE;
  return SECANT_SUCCESS;
}
SecantResult secant_sr_resolve(const SRConfig *c, const SRNode *n, size_t count,
                               const float *banks, size_t elements,
                               uint64_t index, uint8_t *out, size_t cap,
                               size_t *size) {
  const float *bank;
  SecantResult r = sr_bank(c, banks, elements, index, &bank);
  if (r != SECANT_SUCCESS)
    return r;
  return sr_encode(c, n, count, bank,
                   (uint32_t)(index & (((uint64_t)1 << c->toggle_bits) - 1)), 1,
                   out, cap, size);
}
typedef struct SRText {
  char *out;
  size_t cap, used;
} SRText;
static void sr_text(SRText *t, const char *fmt, ...) {
  va_list ap;
  int n;
  va_start(ap, fmt);
  n = vsnprintf(t->out && t->used < t->cap ? t->out + t->used : NULL,
                t->out && t->used < t->cap ? t->cap - t->used : 0, fmt, ap);
  va_end(ap);
  if (n > 0)
    t->used += (size_t)n;
}
static const char *sr_op(uint8_t op) {
  switch (op) {
  case SECANT_AST_INSTRUCTION_TYPE_ADD_F32:
    return "+";
  case SECANT_AST_INSTRUCTION_TYPE_SUB_F32:
    return "-";
  case SECANT_AST_INSTRUCTION_TYPE_MUL_F32:
    return "*";
  case SECANT_AST_INSTRUCTION_TYPE_DIV_F32:
    return "/";
  case SECANT_AST_INSTRUCTION_TYPE_MIN_F32:
    return "min";
  case SECANT_AST_INSTRUCTION_TYPE_MAX_F32:
    return "max";
  case SECANT_AST_INSTRUCTION_TYPE_NEG_F32:
    return "-";
  case SECANT_AST_INSTRUCTION_TYPE_SIN_F32:
    return "sin";
  case SECANT_AST_INSTRUCTION_TYPE_COS_F32:
    return "cos";
  case SECANT_AST_INSTRUCTION_TYPE_TANH_F32:
    return "tanh";
  case SECANT_AST_INSTRUCTION_TYPE_ABS_F32:
    return "abs";
  case SECANT_AST_INSTRUCTION_TYPE_SQRT_F32:
    return "sqrt";
  case SECANT_AST_INSTRUCTION_TYPE_EXP_F32:
    return "exp";
  case SECANT_AST_INSTRUCTION_TYPE_LOG_F32:
    return "log";
  default:
    return "?";
  }
}
static void sr_format_node(SRText *t, const SRNode *n, const uint16_t *first,
                           size_t i, const float *bank, uint32_t perm) {
  unsigned a = sr_arity(n[i].op);
  if (!a) {
    SRLeaf l = sr_selected(n + i, perm, bank);
    if (l.kind == SR_COLUMN)
      sr_text(t, "x%u", l.slot);
    else
      sr_text(t, "%.9g", l.value);
  } else if (a == 1) {
    sr_text(t, "%s(", sr_op(n[i].op));
    sr_format_node(t, n, first, i - 1, bank, perm);
    sr_text(t, ")");
  } else {
    int infix = n[i].op != SECANT_AST_INSTRUCTION_TYPE_MIN_F32 &&
                n[i].op != SECANT_AST_INSTRUCTION_TYPE_MAX_F32;
    sr_text(t, infix ? "(" : "%s(", sr_op(n[i].op));
    sr_format_node(t, n, first, first[i - 1] - 1, bank, perm);
    if (infix)
      sr_text(t, " %s ", sr_op(n[i].op));
    else
      sr_text(t, ", ");
    sr_format_node(t, n, first, i - 1, bank, perm);
    sr_text(t, ")");
  }
}
SecantResult secant_sr_format(const SRConfig *c, const SRNode *n, size_t count,
                              const float *banks, size_t elements,
                              uint64_t index, char *out, size_t cap,
                              size_t *needed) {
  const float *bank;
  uint16_t first[SECANT_SR_MAX_NODES];
  SRText text = {out, cap, 0};
  SecantResult r = sr_bank(c, banks, elements, index, &bank);
  if (r != SECANT_SUCCESS || !needed)
    return r == SECANT_SUCCESS ? SECANT_ERROR_INVALID_VALUE : r;
  r = sr_validate(c, n, count, first, NULL);
  if (r != SECANT_SUCCESS)
    return r;
  sr_format_node(&text, n, first, count - 1, bank,
                 (uint32_t)(index & (((uint64_t)1 << c->toggle_bits) - 1)));
  *needed = text.used + 1;
  return out && cap < *needed ? SECANT_ERROR_INSUFFICIENT_BUFFER
                              : SECANT_SUCCESS;
}
