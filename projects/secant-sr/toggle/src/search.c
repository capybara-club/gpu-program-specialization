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
static const uint8_t defaults[] = {
    SECANT_AST_INSTRUCTION_TYPE_ADD_F32, SECANT_AST_INSTRUCTION_TYPE_SUB_F32,
    SECANT_AST_INSTRUCTION_TYPE_MUL_F32, SECANT_AST_INSTRUCTION_TYPE_SIN_F32,
    SECANT_AST_INSTRUCTION_TYPE_COS_F32};
SRConfig secant_sr_config_default(void) {
  SRConfig c;
  memset(&c, 0, sizeof(c));
  c.population = 1024;
  c.num_inputs = 1;
  c.num_constants = 4;
  c.num_banks = 64;
  c.toggle_bits = 6;
  c.max_nodes = 31;
  c.max_depth = 7;
  c.initial_depth = 3;
  c.tournament = 4;
  c.elites = 8;
  c.seed = 42;
  c.parsimony = 1e-6;
  c.crossover_probability = .45;
  c.mutation_probability = .45;
  c.toggle_probability = .8;
  c.four_way_probability = .3;
  c.coefficient_probability = .25;
  c.operators = defaults;
  c.num_operators = sizeof(defaults);
  return c;
}
SecantResult secant_sr_bank_generate(float *out, size_t banks, size_t slots,
                                     uint64_t seed, int normal, float a,
                                     float b) {
  size_t n, i;
  uint64_t rng = seed;
  if (!banks || !isfinite(a) || !isfinite(b) || (normal != 0 && normal != 1) ||
      (normal ? b < 0 : b < a))
    return SECANT_ERROR_INVALID_VALUE;
  if (!sr_mul(banks, slots, &n))
    return SECANT_ERROR_OVERFLOW;
  if (n && !out)
    return SECANT_ERROR_INVALID_VALUE;
  for (i = 0; i < n; ++i) {
    double u = sr_unit(&rng), value;
    if (normal) {
      double v = sr_unit(&rng);
      value = a + b * sqrt(-2 * log(1 - u)) * cos(6.283185307179586 * v);
    } else
      value = (1 - u) * (double)a + u * (double)b;
    out[i] = (float)value;
    if (!isfinite(out[i]))
      return SECANT_ERROR_OVERFLOW;
  }
  return SECANT_SUCCESS;
}
static size_t pick(SRSearch s, size_t n) {
  return (size_t)(sr_random(&s->rng) % n);
}
static SRLeaf leaf(SRSearch s) {
  SRLeaf l;
  double v = sr_unit(&s->rng);
  memset(&l, 0, sizeof(l));
  if (v < s->config.coefficient_probability && s->config.num_constants) {
    l.kind = SR_COEFFICIENT;
    l.slot = (uint32_t)pick(s, s->config.num_constants);
  } else if (v > .92) {
    static const float values[] = {-2, -1, -.5f, 0, .5f, 1, 2};
    l.kind = SR_LITERAL;
    l.value = values[pick(s, sizeof(values) / sizeof(*values))];
  } else {
    l.kind = SR_COLUMN;
    l.slot = (uint32_t)pick(s, s->config.num_inputs);
  }
  return l;
}
static SRNode leaf_node(SRSearch s) {
  SRNode n;
  size_t i;
  memset(&n, 0, sizeof(n));
  n.choices = 1;
  if (s->config.toggle_bits && sr_unit(&s->rng) < s->config.toggle_probability)
    n.choices = s->config.toggle_bits >= 2 &&
                        sr_unit(&s->rng) < s->config.four_way_probability
                    ? 4
                    : 2;
  if (n.choices > 1)
    n.low_bit = (uint8_t)pick(s, s->config.toggle_bits);
  if (n.choices == 4)
    n.high_bit =
        (uint8_t)((n.low_bit + 1 + pick(s, s->config.toggle_bits - 1)) %
                  s->config.toggle_bits);
  for (i = 0; i < n.choices; ++i)
    n.leaf[i] = leaf(s);
  return n;
}
static size_t tree(SRSearch s, SRNode *out, unsigned depth, size_t budget) {
  size_t used = 0, left_budget;
  uint8_t op;
  unsigned arity;
  if (!depth || budget < 2 || sr_unit(&s->rng) < .2) {
    *out = leaf_node(s);
    return 1;
  }
  op = s->operators[pick(s, s->config.num_operators)];
  arity = sr_arity(op);
  if (budget < 1 + arity) {
    *out = leaf_node(s);
    return 1;
  }
  left_budget = arity == 2 ? 1 + pick(s, budget - 2) : budget - 1;
  used = tree(s, out, depth - 1, left_budget);
  if (arity == 2)
    used += tree(s, out + used, depth - 1, budget - 1 - used);
  memset(out + used, 0, sizeof(*out));
  out[used].op = op;
  return used + 1;
}
static void finish(SRCandidate *p) {
  p->fingerprint = sr_fingerprint(p->nodes, p->count);
  p->score.sse = INFINITY;
  p->score.reserved = 0;
  p->score.configuration = UINT64_MAX;
  p->score.valid_configurations = 0;
}
SecantResult secant_sr_create(const SRConfig *c, const float *banks,
                              size_t elements, SRSearch *out) {
  SRSearch s;
  size_t i, bank_count, bytes;
  SecantResult r;
  if (!out)
    return SECANT_ERROR_INVALID_VALUE;
  *out = NULL;
  r = sr_validate_config(c);
  if (r != SECANT_SUCCESS)
    return r;
  if (!sr_mul(c->population, sizeof(SRCandidate), &bytes) ||
      !sr_mul(c->population, SR_PROGRAM_BYTES, &bytes) ||
      !sr_mul(c->num_banks, c->num_constants, &bank_count) ||
      !sr_mul(bank_count, sizeof(float), &bytes))
    return SECANT_ERROR_OVERFLOW;
  if (elements < bank_count || (bank_count && !banks))
    return SECANT_ERROR_INVALID_VALUE;
  for (i = 0; i < bank_count; ++i)
    if (!isfinite(banks[i]))
      return SECANT_ERROR_INVALID_VALUE;
  s = (SRSearch)calloc(1, sizeof(*s));
  if (!s)
    return SECANT_ERROR_ALLOCATION_FAILED;
  s->config = *c;
  memcpy(s->operators, c->operators, c->num_operators);
  s->config.operators = s->operators;
  s->rng = c->seed;
  s->population[0] = (SRCandidate *)calloc(c->population, sizeof(SRCandidate));
  s->population[1] = (SRCandidate *)calloc(c->population, sizeof(SRCandidate));
  s->programs = (uint8_t *)malloc(c->population * SR_PROGRAM_BYTES);
  s->asts = (const uint8_t **)calloc(c->population, sizeof(*s->asts));
  s->banks = bank_count ? (float *)malloc(bank_count * sizeof(float)) : NULL;
  if (!s->population[0] || !s->population[1] || !s->programs || !s->asts ||
      (bank_count && !s->banks)) {
    secant_sr_destroy(s);
    return SECANT_ERROR_ALLOCATION_FAILED;
  }
  if (bank_count)
    memcpy(s->banks, banks, bank_count * sizeof(float));
  for (i = 0; i < c->population; ++i) {
    SRCandidate *p = s->population[0] + i;
    p->count = (uint16_t)tree(
        s, p->nodes, (unsigned)(i % (c->initial_depth + 1)), c->max_nodes);
    r = sr_validate(c, p->nodes, p->count, NULL, &p->depth);
    if (r != SECANT_SUCCESS) {
      secant_sr_destroy(s);
      return r;
    }
    finish(p);
  }
  s->progress.best_mse = INFINITY;
  s->progress.best_r2 = -INFINITY;
  *out = s;
  return SECANT_SUCCESS;
}
void secant_sr_destroy(SRSearch s) {
  if (!s)
    return;
  free(s->population[0]);
  free(s->population[1]);
  free(s->programs);
  free(s->asts);
  free(s->banks);
  free(s);
}
const SRConfig *secant_sr_config(SRSearch s) { return s ? &s->config : NULL; }
const float *secant_sr_banks(SRSearch s) { return s ? s->banks : NULL; }
SecantResult secant_sr_ask(SRSearch s, SecantAstProgramSet *programs) {
  size_t i, needed;
  SecantResult r;
  if (!s || !programs)
    return SECANT_ERROR_INVALID_VALUE;
  for (i = 0; !s->encoded && i < s->config.population; ++i) {
    SRCandidate *p = s->population[s->current] + i;
    uint8_t *output = s->programs + i * SR_PROGRAM_BYTES;
    r = secant_sr_program_write(&s->config, p->nodes, p->count, output,
                                SR_PROGRAM_BYTES, &needed);
    if (r != SECANT_SUCCESS)
      return r;
    s->asts[i] = output;
  }
  s->encoded = 1;
  memset(programs, 0, sizeof(*programs));
  programs->asts.items = s->asts;
  programs->asts.count = s->config.population;
  return SECANT_SUCCESS;
}
static int better(const SRCandidate *a, const SRCandidate *b) {
  if (a->score.sse != b->score.sse)
    return a->score.sse < b->score.sse;
  if (a->count != b->count)
    return a->count < b->count;
  return a->fingerprint < b->fingerprint;
}
SecantResult secant_sr_tell(SRSearch s, const SRScore *scores, size_t count,
                            size_t rows, double ssd) {
  size_t i, configs;
  uint64_t work, finite = 0;
  if (!s || !scores || count != s->config.population || !rows ||
      !isfinite(ssd) || ssd < 0)
    return SECANT_ERROR_INVALID_VALUE;
  if (s->scored || !s->encoded)
    return SECANT_ERROR_INVALID_STATE;
  if (s->rows && (s->rows != rows || s->target_ssd != ssd))
    return SECANT_ERROR_INVALID_VALUE;
  if (!sr_mul(s->config.num_banks, (size_t)1 << s->config.toggle_bits,
              &configs) ||
      (uint64_t)count > UINT64_MAX / configs)
    return SECANT_ERROR_OVERFLOW;
  work = (uint64_t)count * configs;
  if (work > UINT64_MAX - s->progress.configurations ||
      count > SIZE_MAX - s->progress.structures)
    return SECANT_ERROR_OVERFLOW;
  /* Validate every result before changing any fitness or archive state. */
  for (i = 0; i < count; ++i) {
    const SRScore *v = scores + i;
    if (v->reserved || v->valid_configurations > configs)
      return SECANT_ERROR_INVALID_VALUE;
    if (v->valid_configurations) {
      if (!isfinite(v->sse) || v->sse < 0 || v->configuration >= configs)
        return SECANT_ERROR_INVALID_VALUE;
    } else if (v->sse != INFINITY || v->configuration != UINT64_MAX)
      return SECANT_ERROR_INVALID_VALUE;
    finite += v->valid_configurations;
  }
  for (i = 0; i < count; ++i) {
    SRCandidate *p = s->population[s->current] + i;
    p->score = scores[i];
    if (p->score.valid_configurations &&
        (!s->has_best || better(p, &s->best))) {
      s->best = *p;
      s->best_generation = s->generation;
      s->has_best = 1;
      s->progress.best_mse = (double)p->score.sse / rows;
      s->progress.best_r2 =
          ssd > 0 ? 1 - p->score.sse / ssd : (p->score.sse == 0 ? 1 : 0);
    }
  }
  s->progress.generation = s->generation;
  s->progress.structures += count;
  s->progress.configurations += work;
  s->progress.finite_configurations += finite;
  s->progress.rejected_variations = s->rejected;
  s->rows = rows;
  s->target_ssd = ssd;
  s->scored = 1;
  return SECANT_SUCCESS;
}
static const SRCandidate *parent(SRSearch s) {
  const SRCandidate *best = NULL;
  size_t i;
  double scale = s->has_best ? fmax(s->best.score.sse, 1.0) : 1.0;
  /* Selection rewards data fit plus a small structural cost; archive reports
   * raw SSE, never the penalized score. */
  for (i = 0; i < s->config.tournament; ++i) {
    const SRCandidate *p =
        s->population[s->current] + pick(s, s->config.population);
    if (!best ||
        p->score.sse / scale + s->config.parsimony * p->count <
            best->score.sse / scale + s->config.parsimony * best->count)
      best = p;
  }
  return best;
}
static size_t pick_leaf(SRSearch s, const SRCandidate *p) {
  size_t i, count = 0, selected;
  for (i = 0; i < p->count; ++i) count += !p->nodes[i].op;
  selected = pick(s, count);
  for (i = 0; i < p->count; ++i) if (!p->nodes[i].op && !selected--) return i;
  return 0; /* Every validated postorder expression has at least one leaf. */
}
static void grow_leaf(SRSearch s, SRNode *n, uint32_t permutation, unsigned choices) {
  SRLeaf anchor = n->leaf[sr_selector_choice(n, permutation)];
  unsigned i;
  memset(n, 0, sizeof(*n));
  n->choices = (uint8_t)choices;
  if (choices > 1) n->low_bit = (uint8_t)pick(s, s->config.toggle_bits);
  if (choices == 4)
    n->high_bit = (uint8_t)((n->low_bit+1+pick(s, s->config.toggle_bits-1)) % s->config.toggle_bits);
  for (i = 0; i < choices; ++i) n->leaf[i] = leaf(s);
  n->leaf[sr_selector_choice(n, permutation)] = anchor;
}
static int mutate_toggle(SRSearch s, SRCandidate *child, const SRCandidate *parent) {
  SRNode *n;
  unsigned active, action;
  uint32_t permutation = (uint32_t)parent->score.configuration;
  if (!s->config.toggle_bits || !parent->score.valid_configurations) return 0;
  *child = *parent;
  n = child->nodes + pick_leaf(s, parent);
  action = (unsigned)pick(s, 3);
  if (n->choices == 1 || action == 1) {
    unsigned choices[] = {1, 2, 4}, available[2], count = 0, i;
    for (i = 0; i < (s->config.toggle_bits > 1 ? 3u : 2u); ++i)
      if (choices[i] != n->choices) available[count++] = choices[i];
    grow_leaf(s, n, permutation, available[pick(s, count)]);
  } else if (action == 0) {
    unsigned replace;
    active = sr_selector_choice(n, permutation);
    replace = (active+1+(unsigned)pick(s, n->choices-1)) % n->choices;
    n->leaf[replace] = leaf(s); /* Keep the winning alternative available. */
  } else {
    unsigned low = (unsigned)pick(s, s->config.toggle_bits), high = 0;
    if (n->choices == 4) high = (low+1+(unsigned)pick(s, s->config.toggle_bits-1)) % s->config.toggle_bits;
    sr_selector_rebind(n, low, high, permutation, permutation);
  }
  return 1;
}
static int mix_leaf(SRSearch s, SRCandidate *child, const SRCandidate *a, const SRCandidate *b) {
  SRNode *target;
  const SRNode *donor;
  unsigned active, replace;
  if (!s->config.toggle_bits || !a->score.valid_configurations || !b->score.valid_configurations) return 0;
  *child = *a;
  target = child->nodes + pick_leaf(s, a);
  donor = b->nodes + pick_leaf(s, b);
  if (target->choices == 1) grow_leaf(s, target, (uint32_t)a->score.configuration, 2);
  active = sr_selector_choice(target, (uint32_t)a->score.configuration);
  replace = (active+1+(unsigned)pick(s, target->choices-1)) % target->choices;
  target->leaf[replace] = donor->leaf[sr_selector_choice(donor, (uint32_t)b->score.configuration)];
  return 1;
}
static int offspring(SRSearch s, SRCandidate *child) {
  const SRCandidate *a = parent(s);
  uint16_t first[SECANT_SR_MAX_NODES];
  size_t at, len, newlen, count;
  unsigned aligned = 0, mutated = 0, mixed = 0, reused = 0, powered = 0;
  double choice = sr_unit(&s->rng);
  if (sr_validate(&s->config, a->nodes, a->count, first, NULL) !=
      SECANT_SUCCESS)
    return 0;
  at = pick(s, a->count);
  len = at + 1 - first[at];
  if (choice < s->config.crossover_probability) {
    const SRCandidate *b = parent(s);
    uint16_t bf[SECANT_SR_MAX_NODES];
    size_t bp = pick(s, b->count);
    if (s->config.leaf_mix_probability > 0 && sr_unit(&s->rng) < s->config.leaf_mix_probability &&
        mix_leaf(s, child, a, b)) {
      mixed = 1;
      goto validate_child;
    }
    if (sr_validate(&s->config, b->nodes, b->count, bf, NULL) != SECANT_SUCCESS)
      return 0;
    newlen = bp + 1 - bf[bp];
    count = a->count - len + newlen;
    if (count > s->config.max_nodes)
      return 0;
    memcpy(child->nodes, a->nodes, first[at] * sizeof(SRNode));
    memcpy(child->nodes + first[at], b->nodes + bf[bp],
           newlen * sizeof(SRNode));
    memcpy(child->nodes + first[at] + newlen, a->nodes + at + 1,
           (a->count - at - 1) * sizeof(SRNode));
    child->count = (uint16_t)count;
    if (s->config.align_crossover_bits && a->score.valid_configurations && b->score.valid_configurations) {
      reused = sr_align_subtree(child->nodes, count, first[at], newlen, s->config.toggle_bits,
                                (uint32_t)b->score.configuration, (uint32_t)a->score.configuration);
      aligned = 1;
    }
  } else if (choice <
             s->config.crossover_probability + s->config.mutation_probability) {
    *child = *a;
    if (s->config.power_mutation_probability > 0 &&
        sr_unit(&s->rng) < s->config.power_mutation_probability) {
      size_t j, at_out=first[at], exponent=2+pick(s,2);
      int allowed=0;
      for(j=0;j<s->config.num_operators;++j)
        allowed|=s->operators[j]==SECANT_AST_INSTRUCTION_TYPE_MUL_F32;
      count=a->count+(exponent-1)*(len+1);
      if(allowed && count<=s->config.max_nodes) {
        for(j=0;j<exponent;++j) {
          memcpy(child->nodes+at_out,a->nodes+first[at],len*sizeof(SRNode));at_out+=len;
          if(j) { memset(child->nodes+at_out,0,sizeof(SRNode));child->nodes[at_out++].op=SECANT_AST_INSTRUCTION_TYPE_MUL_F32; }
        }
        memcpy(child->nodes+at_out,a->nodes+at+1,(a->count-at-1)*sizeof(SRNode));
        child->count=(uint16_t)count;powered=1;
        goto validate_child;
      }
    }
    if (s->config.toggle_mutation_probability > 0 && sr_unit(&s->rng) < s->config.toggle_mutation_probability &&
        mutate_toggle(s, child, a)) {
      mutated = 1;
      goto validate_child;
    }
    if (sr_unit(&s->rng) < .6) {
      SRNode *n = child->nodes + at;
      if (!n->op) {
        if (n->choices > 1 && sr_unit(&s->rng) < .3) {
          n->low_bit = (uint8_t)pick(s, s->config.toggle_bits);
          if (n->choices == 4)
            n->high_bit = (uint8_t)((n->low_bit + 1 +
                                     pick(s, s->config.toggle_bits - 1)) %
                                    s->config.toggle_bits);
        } else
          n->leaf[pick(s, n->choices)] = leaf(s);
      } else {
        size_t j;
        for (j = 0; j < 16; ++j) {
          uint8_t op = s->operators[pick(s, s->config.num_operators)];
          if (sr_arity(op) == sr_arity(n->op)) {
            n->op = op;
            break;
          }
        }
      }
    } else {
      SRNode replace[SECANT_SR_MAX_NODES];
      newlen = tree(s, replace, (unsigned)pick(s, s->config.initial_depth + 1),
                    s->config.max_nodes - (a->count - len));
      count = a->count - len + newlen;
      memcpy(child->nodes + first[at], replace, newlen * sizeof(SRNode));
      memcpy(child->nodes + first[at] + newlen, a->nodes + at + 1,
             (a->count - at - 1) * sizeof(SRNode));
      child->count = (uint16_t)count;
    }
  } else
    child->count = (uint16_t)tree(
        s, child->nodes, (unsigned)pick(s, s->config.initial_depth + 1),
        s->config.max_nodes);
validate_child:
  if (sr_validate(&s->config, child->nodes, child->count, NULL,
                  &child->depth) != SECANT_SUCCESS)
    return 0;
  finish(child);
  s->progress.aligned_crossovers += aligned;
  s->progress.toggle_mutations += mutated;
  s->progress.leaf_mixes += mixed;
  s->progress.reused_toggle_bits += reused;
  s->progress.power_mutations += powered;
  return 1;
}
SecantResult secant_sr_advance(SRSearch s) {
  size_t i, try;
  SRCandidate *next;
  if (!s)
    return SECANT_ERROR_INVALID_VALUE;
  if (!s->scored)
    return SECANT_ERROR_INVALID_STATE;
  if (s->generation == SIZE_MAX)
    return SECANT_ERROR_OVERFLOW;
  next = s->population[1 - s->current];
  for (i = 0; i < s->config.population; ++i) {
    if (i < s->config.elites && s->has_best) {
      next[i] = i == 0 ? s->best : *parent(s);
      finish(next + i);
      continue;
    }
    for (try = 0; try < 32; ++try) {
      if (offspring(s, next + i))
        break;
      ++s->rejected;
    }
    if (try == 32) {
      next[i].count = 1;
      next[i].nodes[0] = leaf_node(s);
      next[i].depth = 0;
      finish(next + i);
    }
  }
  s->current = 1 - s->current;
  ++s->generation;
  s->scored = 0;
  s->encoded = 0;
  return SECANT_SUCCESS;
}
SecantResult secant_sr_best(SRSearch s, SRModel *out) {
  if (!s || !out)
    return SECANT_ERROR_INVALID_VALUE;
  if (!s->has_best)
    return SECANT_ERROR_INVALID_STATE;
  out->nodes = s->best.nodes;
  out->num_nodes = s->best.count;
  out->score = s->best.score;
  out->generation = s->best_generation;
  out->fingerprint = s->best.fingerprint;
  return SECANT_SUCCESS;
}
SRProgress secant_sr_progress(SRSearch s) {
  SRProgress p;
  memset(&p, 0, sizeof(p));
  return s ? s->progress : p;
}
SecantResult secant_sr_seed(SRSearch s, size_t i, const SRNode *nodes,
                            size_t count) {
  SRCandidate *p;
  uint16_t depth;
  SecantResult r;
  if (!s || i >= s->config.population)
    return SECANT_ERROR_INVALID_VALUE;
  if (s->generation || s->scored)
    return SECANT_ERROR_INVALID_STATE;
  r = sr_validate(&s->config, nodes, count, NULL, &depth);
  if (r != SECANT_SUCCESS)
    return r;
  p = s->population[0] + i;
  memcpy(p->nodes, nodes, count * sizeof(*nodes));
  p->count = (uint16_t)count;
  p->depth = depth;
  finish(p);
  s->encoded = 0;
  return SECANT_SUCCESS;
}

SecantResult secant_sr_candidate(SRSearch s, size_t i, SRModel *out) {
  SRCandidate *p;
  if (!s || !out || i >= s->config.population) return SECANT_ERROR_INVALID_VALUE;
  if (!s->scored) return SECANT_ERROR_INVALID_STATE;
  p = s->population[s->current] + i;
  out->nodes = p->nodes; out->num_nodes = p->count; out->score = p->score;
  out->generation = s->generation; out->fingerprint = p->fingerprint;
  return SECANT_SUCCESS;
}
SecantResult secant_sr_accept_refined(SRSearch s, size_t index, const SRModel *model) {
  SRCandidate *p;
  const SRLeaf *old_parameters[SECANT_SR_MAX_NODES * 4];
  const SRLeaf *new_parameters[SECANT_SR_MAX_NODES * 4];
  size_t parameter_count = 0;
  size_t i, j;
  SecantResult r;
  if (!s || !model || index >= s->config.population) return SECANT_ERROR_INVALID_VALUE;
  if (!s->scored) return SECANT_ERROR_INVALID_STATE;
  p = s->population[s->current] + index;
  r = sr_validate(&s->config, model->nodes, model->num_nodes, NULL, NULL);
  if (r != SECANT_SUCCESS) return r;
  if (model->num_nodes != p->count || model->score.reserved ||
      !isfinite(model->score.sse) || model->score.sse < 0 ||
      model->score.configuration >= (UINT64_C(1) << s->config.toggle_bits) ||
      model->score.valid_configurations != 1 || model->score.sse > p->score.sse)
    return SECANT_ERROR_INVALID_VALUE;
  /* Refinement may change parameters, never operators, fixed literals, states,
   * or the complete selector groups that GP retains. */
  for (i = 0; i < p->count; ++i) {
    const SRNode *a = p->nodes+i, *b = model->nodes+i;
    if (a->op != b->op || a->choices != b->choices ||
        (a->choices > 1 && a->low_bit != b->low_bit) ||
        (a->choices == 4 && a->high_bit != b->high_bit)) return SECANT_ERROR_BAD_PROGRAM;
    for (j = 0; j < a->choices; ++j) {
      const SRLeaf *x = a->leaf+j, *y = b->leaf+j;
      if (x->kind == SR_COEFFICIENT || x->kind == SR_FITTED_COEFFICIENT) {
        size_t k;
        if (y->kind != SR_FITTED_COEFFICIENT) return SECANT_ERROR_BAD_PROGRAM;
        /* Preserve parameter sharing in both directions. Canonical slot IDs may
         * change, but refinement must not split or merge the model's parameters. */
        for (k = 0; k < parameter_count; ++k) {
          const SRLeaf *a = old_parameters[k], *b = new_parameters[k];
          int same_before = x->kind == a->kind && x->slot == a->slot &&
              (x->kind == SR_COEFFICIENT || !memcmp(&x->value, &a->value, sizeof(float)));
          int same_after = y->slot == b->slot && !memcmp(&y->value, &b->value, sizeof(float));
          if (same_before != same_after) return SECANT_ERROR_BAD_PROGRAM;
        }
        old_parameters[parameter_count] = x;
        new_parameters[parameter_count++] = y;
      } else if (x->kind != y->kind ||
          (x->kind == SR_COLUMN ? x->slot != y->slot : memcmp(&x->value, &y->value, 4)))
        return SECANT_ERROR_BAD_PROGRAM;
    }
  }
  memcpy(p->nodes, model->nodes, p->count * sizeof(SRNode));
  p->fingerprint = sr_fingerprint(p->nodes, p->count); p->score = model->score;
  s->encoded = 0;
  if (!s->has_best || better(p, &s->best)) {
    s->best = *p; s->best_generation = s->generation; s->has_best = 1;
    s->progress.best_mse = p->score.sse / (double)s->rows;
    s->progress.best_r2 = s->target_ssd > 0 ? 1-p->score.sse/s->target_ssd : (p->score.sse == 0 ? 1 : 0);
  }
  return SECANT_SUCCESS;
}
