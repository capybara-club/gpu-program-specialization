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
#ifndef ODR_GRAMMAR_INTERNAL_H
#define ODR_GRAMMAR_INTERNAL_H
#include "odr_internal.h"
#define GRAMMAR_MAGIC UINT64_C(0x4f44524752414d31)
typedef struct Slot {
    const char *name;
    unsigned kind,arity,coverage;
    const uint32_t *states;size_t state_count;
    const uint32_t *groups;size_t group_count;
    const uint64_t *sample_ranks;
    OdrAxis axis;
    const char *scale_ref,*shift_ref;
    const struct Slot *scope;size_t scope_count;
} Slot;
typedef struct Alt {const Node *node;const char *const *tags;size_t tag_count;const Slot *locals;size_t local_count;} Alt;
typedef struct Rule {const char *name;const char *formals[ODR_ARGS];size_t argc,count;const Alt *alternatives;int shared;} Rule;
typedef struct Family {
    const char *name;const Node *const *rhs;const Rule *rules;size_t rule_count;
    const Slot *slots;size_t slot_count;const char *const *tags;size_t tag_count;
    uint64_t max_asts,max_variants,max_configs,max_configs_per_skeleton,max_attempts,max_steps,joint_count,joint_seed;
} Family;
struct OdrGrammar {
    uint64_t magic,seed;uint32_t state_count,max_nodes,max_depth,max_expansion_depth,sample;
    const char *const *states;const OdrStatic *fixed;
    size_t family_count;const Family *families;
    OdrGrammarInfo info;
};
typedef struct Choice {uint32_t value,count;} Choice;
typedef struct Expanded {uint32_t op,argc;float value;const struct Expanded *args[ODR_ARGS];const Slot *slot;uint32_t instance,state;} Expanded;
typedef struct Active {const Slot *slot;uint32_t instance,ordinal,bit;uint64_t group_index,groups,stride;} Active;
typedef struct Dedup {uint64_t hash,index;size_t offset,size;} Dedup;
struct OdrProducer {
    const OdrGrammar *grammar;const Family *family;OdrProducerOptions options;
    uint64_t next,attempts,duplicates,pruned,derivation;
    uint64_t steps,configs,visits,tree_configs,configuration_limited_derivations;
    int tree_budget_reported;
    OdrStopReason stop_reason;
    int exhausted,have_tree,tree_accepted;
    uint64_t skeleton_count;
    Choice *choices;size_t choice_count,choice_capacity;
    Expanded *nodes;size_t node_count,node_capacity;
    const Expanded **roots,**shared;
    Active *active;size_t active_count,active_capacity;
    const char **tags;size_t tag_count,tag_capacity;
    uint64_t variant,variants;
    uint64_t numeric_count;
    uint64_t *sample_ranks,*sample_map;size_t sample_capacity;
    Dedup *dedup;size_t dedup_count;
    unsigned char *keys;size_t key_used,key_capacity;
    unsigned char *scratch;size_t scratch_capacity;
};
const Rule *grule(const Family *f,const char *name,int shared);
const Slot *gslot(const Family *f,const char *name);
uint64_t odr_mix(uint64_t x);
uint64_t odr_random_below(uint64_t seed,uint64_t *counter,uint64_t bound);
void odr_sample_ranks(uint64_t total,size_t count,uint64_t seed,uint64_t *ranks,uint64_t *map);
#endif
