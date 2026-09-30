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
#include "o_constrained_search.h"
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>

#ifdef COMPARE_BASELINE
OCgStatus baseline_enumerate(const OCgConfig *, OCgSubmit, void *, OCgReport *);
#endif

typedef struct Digest { uint64_t hash, count; } Digest;
static int collect(const OCgCandidate *batch, size_t count, void *context) {
    Digest *d = context;
    size_t i, j;
    for (i = 0; i < count; ++i) {
        assert(batch[i].ordinal == d->count++);
        for (j = 0; j < batch[i].byte_count; ++j)
            d->hash = (d->hash ^ batch[i].bytes[j]) * UINT64_C(1099511628211);
    }
    return 1;
}

int main(void) {
    OCgProduction rules[40];
    OCgGrammar grammar = {rules, 0, 4, 3};
    const unsigned ops[] = {O_AST_SIN_F32, O_AST_COS_F32,
        O_AST_ADD_F32, O_AST_SUB_F32, O_AST_MUL_F32, O_AST_DIV_F32};
    OCgConfig c = o_cg_default_config();
    unsigned depth, i, nodes;
    for (depth = 0; depth < 4; ++depth) {
        for (i = 0; i < 3; ++i)
            rules[grammar.production_count++] = (OCgProduction){depth,O_AST_STATE_F32,i,0,0};
        rules[grammar.production_count++] = (OCgProduction){depth,O_AST_CONSTANT_F32,0,0,0};
        if (depth) for (i = 0; i < 6; ++i)
            rules[grammar.production_count++] = (OCgProduction){depth,ops[i],0,depth-1,depth-1};
    }
    c.grammar = &grammar; c.state_count = 3; c.constant_capacity = c.maximum_parameters = 1;
    c.canonical_parameter_names = 0; c.literal_count = 0;
    c.required_state_mask = 0; c.maximum_function_depth = 3;
    c.forbid_trig_in_denominators = c.forbid_same_state_product_outside_denominators = 0;
    for (nodes = 1; nodes <= 15; ++nodes) {
        OCgReport r;
        Digest d = {UINT64_C(14695981039346656037),0};
        c.minimum_nodes = c.maximum_nodes = nodes;
        c.target_accepted = 1336; c.maximum_branch_visits = 1336000;
        assert(o_cg_enumerate(&c,collect,&d,&r) != O_CG_WORK_LIMIT);
        if (nodes >= 5) assert(r.accepted == 1336);
        printf("nodes=%u accepted=%" PRIu64 " visits=%" PRIu64 " hash=%" PRIu64 "\n",
               nodes,r.accepted,r.branch_visits,d.hash);
#ifdef COMPARE_BASELINE
        {
            OCgReport old; Digest before = {UINT64_C(14695981039346656037),0};
            /* Small strata compare complete output or equal prefixes. Large
             * strata reproduce the original starvation under the same cap. */
            baseline_enumerate(&c,collect,&before,&old);
            if (old.status != O_CG_WORK_LIMIT) {
                assert(before.count == d.count && before.hash == d.hash);
                assert(old.status == r.status);
            } else {
                printf("baseline nodes=%u accepted=%" PRIu64 " visits=%" PRIu64 " work_limit\n",
                       nodes,old.accepted,old.branch_visits);
                /* Compare exactly the accepted baseline prefix, even when it
                 * was truncated by work limits. */
                if (old.accepted) {
                    OCgReport prefix; Digest pd = {UINT64_C(14695981039346656037),0};
                    c.target_accepted = old.accepted;
                    o_cg_enumerate(&c,collect,&pd,&prefix);
                    assert(pd.count == before.count && pd.hash == before.hash);
                }
            }
        }
#endif
    }
    /* Impossible target must exhaust quickly; recursive rules must retain
     * conservative bounds rather than being mistaken for finite DAGs. */
    c.minimum_nodes = c.maximum_nodes = 16;
    {
        OCgReport r; Digest d = {0,0}; c.target_accepted = 1;
        assert(o_cg_enumerate(&c,collect,&d,&r) == O_CG_EXHAUSTED);
        assert(r.accepted == 0 && r.branch_visits == 0);
    }
    puts("Depth-3 quota and ordered-output checks passed");
    return 0;
}
