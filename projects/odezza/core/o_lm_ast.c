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
#include "o_odezza_internal.h"
#include <math.h>
#include <string.h>
#define L ODEZZA_AST_LITERAL_F32
#define ADD ODEZZA_AST_ADD_F32
#define SUB ODEZZA_AST_SUB_F32
#define MUL ODEZZA_AST_MUL_F32
#define DIV ODEZZA_AST_DIV_F32
#define NEG ODEZZA_AST_NEG_F32
static uint32_t node(OLmExpressions *c, uint8_t op, uint32_t v, const uint32_t *a, uint8_t arity) {
    OLmNode n;
    uint32_t i;
    uint16_t depth = 0;
    if (c->result)
        return 0;
    memset(&n, 0, sizeof(n));
    n.opcode = op;
    n.value = v;
    n.arity = arity;
    for (i = 0; i < arity; ++i) {
        n.child[i] = a[i];
        if (c->nodes[a[i]].depth >= depth)
            depth = c->nodes[a[i]].depth + 1u;
    }
    if (depth > 128u) {
        c->result = ODEZZA_ERROR_AST;
        return 0;
    }
    n.depth = depth;
    for (i = 0; i < c->count; ++i)
        if (!memcmp(&n, &c->nodes[i], sizeof(n)))
            return i;
    if (c->count == O_LM_MAX_NODES) {
        c->result = ODEZZA_ERROR_SPECIALIZATION_CAPACITY;
        return 0;
    }
    c->nodes[c->count] = n;
    return c->count++;
}
static float number(const OLmExpressions *c, uint32_t x) {
    float v;
    memcpy(&v, &c->nodes[x].value, 4);
    return v;
}
static int literal(const OLmExpressions *c, uint32_t x, float v) {
    return c->nodes[x].opcode == L && number(c, x) == v;
}
static uint32_t lit(OLmExpressions *c, float f) {
    uint32_t v;
    if (!isfinite(f)) {
        c->result = ODEZZA_ERROR_AST;
        return 0;
    }
    memcpy(&v, &f, 4);
    return node(c, L, v, NULL, 0);
}
static uint32_t unary(OLmExpressions *c, uint8_t op, uint32_t a) {
    if (op == NEG) {
        if (c->nodes[a].opcode == L)
            return lit(c, -number(c, a));
        if (c->nodes[a].opcode == NEG)
            return c->nodes[a].child[0];
    }
    return node(c, op, 0, &a, 1);
}
static uint32_t binary(OLmExpressions *c, uint8_t op, uint32_t a, uint32_t b) {
    uint32_t args[2] = {a, b};
    if (op == ADD) {
        if (literal(c, a, 0))
            return b;
        if (literal(c, b, 0))
            return a;
    }
    if (op == SUB) {
        if (literal(c, b, 0))
            return a;
        if (literal(c, a, 0))
            return unary(c, NEG, b);
    }
    if (op == MUL) {
        if (literal(c, a, 0) || literal(c, b, 0))
            return 0;
        if (literal(c, a, 1))
            return b;
        if (literal(c, b, 1))
            return a;
        if (literal(c, a, -1))
            return unary(c, NEG, b);
        if (literal(c, b, -1))
            return unary(c, NEG, a);
    }
    if (op == DIV) {
        if (literal(c, a, 0))
            return 0;
        if (literal(c, b, 1))
            return a;
    }
    if (c->nodes[a].opcode == L && c->nodes[b].opcode == L) {
        float x = number(c, a), y = number(c, b);
        if (op == ADD)
            return lit(c, x + y);
        if (op == SUB)
            return lit(c, x - y);
        if (op == MUL)
            return lit(c, x * y);
        if (op == DIV)
            return lit(c, x / y);
    }
    return node(c, op, 0, args, 2);
}
static OdezzaResult parse(OLmExpressions *c, OdezzaAstProgram p, const OdezzaLmShape *s, uint32_t bits,
                          uint32_t *root) {
    uint32_t stack[129], n = 0;
    size_t at = 0;
    OdezzaAstAnalysis analysis;
    O_RETURN_IF_ERROR(odezza_validate_ast_program(p, s->state_count, s->parameter_count, bits, 1, &analysis));
    while (at < p.byte_count) {
        uint8_t op = p.bytes[at++], arity = 0;
        uint32_t v = 0, x;
        if (op == ODEZZA_AST_RETURN_F32) {
            *root = stack[0];
            return c->result;
        }
        if (op == ODEZZA_AST_ABS_F32 || op == ODEZZA_AST_MIN_F32 || op == ODEZZA_AST_MAX_F32)
            return ODEZZA_ERROR_UNSUPPORTED;
        if (op == ODEZZA_AST_STATE_F32 || op == ODEZZA_AST_CONSTANT_F32)
            v = p.bytes[at++];
        else if (op == L) {
            unsigned int j;
            float f;
            for (j = 0; j < 4; ++j)
                v |= (uint32_t)p.bytes[at++] << (8u * j);
            memcpy(&f, &v, 4);
            if (!isfinite(f))
                return ODEZZA_ERROR_AST;
        } else if (op == ODEZZA_AST_TOGGLE2_F32) {
            v = p.bytes[at++];
            arity = 2;
        } else if (op == ODEZZA_AST_TOGGLE4_F32) {
            v = p.bytes[at];
            v |= (uint32_t)p.bytes[at + 1] << 8;
            at += 2;
            arity = 4;
        } else if (op == ADD || op == SUB || op == MUL || op == DIV)
            arity = 2;
        else if (op == ODEZZA_AST_FMA_F32)
            arity = 3;
        else
            arity = 1;
        if (n < arity || n - arity >= 129)
            return ODEZZA_ERROR_AST;
        x = node(c, op, v, stack + n - arity, arity);
        n -= arity;
        stack[n++] = x;
        if (c->result)
            return c->result;
    }
    return ODEZZA_ERROR_AST;
}
#define B(op, a, b) binary(c, op, a, b)
#define U(op, a) unary(c, op, a)
static uint32_t derivative(OLmExpressions *c, uint32_t x, uint32_t target, uint32_t states,
                           const uint32_t *d) {
    OLmNode n = c->nodes[x];
    uint32_t a = n.child[0], b = n.child[1], da = d[a], db = d[b], v, args[4], j;
    switch (n.opcode) {
    case L:
        return 0;
    case ODEZZA_AST_STATE_F32:
        return n.value == target ? 1 : 0;
    case ODEZZA_AST_CONSTANT_F32:
        return n.value + states == target ? 1 : 0;
    case ADD:
        return B(ADD, da, db);
    case SUB:
        return B(SUB, da, db);
    case MUL:
        return B(ADD, B(MUL, da, b), B(MUL, a, db));
    case DIV:
        return B(DIV, B(SUB, da, B(MUL, x, db)), b);
    case NEG:
        return U(NEG, da);
    case ODEZZA_AST_FMA_F32:
        return B(ADD, B(ADD, B(MUL, da, b), B(MUL, a, db)), d[n.child[2]]);
    case ODEZZA_AST_SQRT_F32:
        return B(DIV, da, B(MUL, lit(c, 2), x));
    case ODEZZA_AST_RCP_F32:
        return U(NEG, B(DIV, da, B(MUL, a, a)));
    case ODEZZA_AST_SIN_F32:
        return B(MUL, da, U(ODEZZA_AST_COS_F32, a));
    case ODEZZA_AST_COS_F32:
        return U(NEG, B(MUL, da, U(ODEZZA_AST_SIN_F32, a)));
    case ODEZZA_AST_EX2_F32:
        return B(MUL, B(MUL, da, lit(c, 0.6931471805599453f)), x);
    case ODEZZA_AST_LG2_F32:
        return B(DIV, da, B(MUL, a, lit(c, 0.6931471805599453f)));
    case ODEZZA_AST_RSQRT_F32:
        return B(MUL, B(MUL, lit(c, -0.5f), da), B(MUL, x, B(MUL, x, x)));
    case ODEZZA_AST_TANH_F32:
        return B(MUL, da, B(SUB, 1, B(MUL, x, x)));
    case ODEZZA_AST_EXP_F32:
        return B(MUL, da, x);
    case ODEZZA_AST_LOG_F32:
        return B(DIV, da, a);
    case ODEZZA_AST_TOGGLE2_F32:
    case ODEZZA_AST_TOGGLE4_F32:
        v = d[n.child[0]];
        for (j = 0; j < n.arity; ++j)
            args[j] = d[n.child[j]];
        for (j = 1; j < n.arity; ++j)
            if (args[j] != v)
                break;
        return j == n.arity ? v : node(c, n.opcode, n.value, args, n.arity);
    default:
        c->result = ODEZZA_ERROR_UNSUPPORTED;
        return 0;
    }
}
OdezzaResult o_lm_expressions(const OdezzaAstProgram *rhs, const OdezzaLmShape *s, uint32_t bits,
                              OLmExpressions *c) {
    uint32_t i, t, base, d[O_LM_MAX_NODES];
    if (!rhs || !c)
        return ODEZZA_ERROR_INVALID_ARGUMENT;
    O_RETURN_IF_ERROR(o_lm_validate_shape(s));
    memset(c, 0, sizeof(*c));
    lit(c, 0);
    lit(c, 1);
    for (i = 0; i < s->state_count; ++i)
        O_RETURN_IF_ERROR(parse(c, rhs[i], s, bits, &c->roots[i]));
    base = c->count;
    for (t = 0; t < s->state_count + s->parameter_count; ++t) {
        memset(d, 0, sizeof(d));
        for (i = 0; i < base; ++i) {
            d[i] = derivative(c, i, t, s->state_count, d);
            if (c->result)
                return c->result;
        }
        for (i = 0; i < s->state_count; ++i)
            c->roots[s->state_count + t * s->state_count + i] = d[c->roots[i]];
    }
    return c->result;
}
