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
#include "secant_sr_refine.h"

typedef struct SRParameterKey { SRLeafKind kind; uint32_t slot, bits; } SRParameterKey;
#define SR_MAX_PARAMETERS (SECANT_SR_MAX_NODES * 4u)
struct SRRefinerImpl {
  SRConfig config;
  uint8_t operators[32];
  SRRefineOptions options;
  SRCandidate *models;
  float *centers, *scales, *banks;
  size_t *parameters, *generations;
  size_t *selected, *selected_count;
  uint64_t sequence;
  uint8_t *storage;
  const uint8_t **asts;
  size_t count, bank_elements;
  int pending;
  SRRefineStats stats;
};
static void philox(uint32_t v[4], uint64_t seed) {
  uint32_t k0 = (uint32_t)seed, k1 = (uint32_t)(seed >> 32);
  unsigned i;
  for (i = 0; i < 10; ++i) {
    uint64_t a = UINT64_C(0xd2511f53) * v[0], b = UINT64_C(0xcd9e8d57) * v[2];
    uint32_t x = (uint32_t)(b >> 32) ^ v[1] ^ k0;
    uint32_t z = (uint32_t)(a >> 32) ^ v[3] ^ k1;
    v[0] = x; v[1] = (uint32_t)b; v[2] = z; v[3] = (uint32_t)a;
    k0 += UINT32_C(0x9e3779b9); k1 += UINT32_C(0xbb67ae85);
  }
}
SecantResult secant_sr_refinement_bank(float *out, size_t trials, size_t slots,
                                       uint64_t seed, uint32_t round) {
  size_t i, j, n;
  if (!out || !trials || !slots || slots > SECANT_AST_MAX_INPUTS)
    return SECANT_ERROR_INVALID_VALUE;
  if (!sr_mul(trials, slots, &n) || n > SIZE_MAX / sizeof(float))
    return SECANT_ERROR_OVERFLOW;
  for (i = 0; i < trials; ++i)
    for (j = 0; j < slots; j += 4) {
      uint32_t v[4] = {(uint32_t)i, (uint32_t)((uint64_t)i >> 32), round, (uint32_t)(j / 4)};
      size_t k;
      philox(v, seed);
      for (k = 0; k < 4 && j + k < slots; ++k)
        out[i * slots + j + k] = i ? (float)(v[k] >> 8) * 0x1p-23f - 1.0f : 0.0f;
    }
  return SECANT_SUCCESS;
}
SRRefineOptions secant_sr_refine_options_default(void) {
  SRRefineOptions o = {128, 64, 42, 1.f, .25f, .5f, 1e-6f, 1e6f, SR_REFINE_ALL_PARAMETERS};
  return o;
}
static int parameter(const SRLeaf *l) {
  return l->kind == SR_COEFFICIENT || l->kind == SR_FITTED_COEFFICIENT;
}
static SecantResult map_parameters(const SRConfig *c, const SRModel *m,
                                    SRCandidate *out, size_t *count,
                                    const float *bank, float *centers, int active_only) {
  SRParameterKey keys[SECANT_SR_MAX_NODES * 4];
  size_t i, j, n = 0;
  SecantResult r;
  if (!m || !count) return SECANT_ERROR_INVALID_VALUE;
  r = sr_validate(c, m->nodes, m->num_nodes, NULL, NULL);
  if (r != SECANT_SUCCESS) return r;
  if (out) {
    memset(out, 0, sizeof(*out));
    memcpy(out->nodes, m->nodes, m->num_nodes * sizeof(SRNode));
    out->count = (uint16_t)m->num_nodes;
    out->score = m->score;
    out->score.configuration &= ((UINT64_C(1) << c->toggle_bits) - 1);
    out->score.valid_configurations = 1;
  }
  for (i = 0; i < m->num_nodes; ++i) for (j = 0; j < m->nodes[i].choices; ++j) {
    const SRLeaf *l = &m->nodes[i].leaf[j];
    SRParameterKey key;
    size_t p;
    if (active_only && j != sr_selector_choice(m->nodes+i, (uint32_t)m->score.configuration)) continue;
    if (!parameter(l)) continue;
    key.kind = l->kind; key.slot = l->slot; key.bits = 0;
    if (l->kind == SR_FITTED_COEFFICIENT) memcpy(&key.bits, &l->value, 4);
    for (p = 0; p < n; ++p)
      if (keys[p].kind == key.kind && keys[p].slot == key.slot && keys[p].bits == key.bits) break;
    if (p == n) keys[n++] = key;
    if (out) {
      float v;
      v = l->kind == SR_COEFFICIENT ? bank[l->slot] : l->value;
      out->nodes[i].leaf[j].kind = SR_FITTED_COEFFICIENT;
      out->nodes[i].leaf[j].slot = (uint32_t)p;
      out->nodes[i].leaf[j].value = v;
      centers[p] = v;
    }
  }
  *count = n;
  if (out) out->fingerprint = sr_fingerprint(out->nodes, out->count);
  return SECANT_SUCCESS;
}
SecantResult secant_sr_refine_parameter_count(const SRConfig *c, const SRModel *m, size_t *count) {
  SecantResult r = sr_validate_config(c);
  return r == SECANT_SUCCESS ? map_parameters(c, m, NULL, count, NULL, NULL, 0) : r;
}
SecantResult secant_sr_refine_active_parameter_count(const SRConfig *c, const SRModel *m, size_t *count) {
  SecantResult r = sr_validate_config(c);
  return r == SECANT_SUCCESS ? map_parameters(c, m, NULL, count, NULL, NULL, 1) : r;
}
SecantResult secant_sr_refiner_create(const SRConfig *c, const SRRefineOptions *o, SRRefiner *out) {
  SRRefiner s;
  size_t nc, nb, bytes, configurations;
  SecantResult r;
  if (!out) return SECANT_ERROR_INVALID_VALUE;
  *out = NULL;
  r = sr_validate_config(c);
  if (r != SECANT_SUCCESS) return r;
  if (!o || !o->capacity || !c->num_constants || o->trials < 2 ||
      (o->parameters != SR_REFINE_ALL_PARAMETERS && o->parameters != SR_REFINE_ACTIVE_BLOCK) ||
      !isfinite(o->initial_scale) || !isfinite(o->scale_learning_rate) ||
      !isfinite(o->failure_decay) || !isfinite(o->minimum_scale) || !isfinite(o->maximum_scale) ||
      o->minimum_scale <= 0 || o->maximum_scale < o->minimum_scale ||
      o->initial_scale < o->minimum_scale || o->initial_scale > o->maximum_scale ||
      o->scale_learning_rate < 0 || o->scale_learning_rate > 1 ||
      o->failure_decay <= 0 || o->failure_decay > 1)
    return SECANT_ERROR_INVALID_VALUE;
  if (!sr_mul(o->trials, (size_t)1 << c->toggle_bits, &configurations) ||
      !sr_mul(o->capacity, SR_MAX_PARAMETERS, &nc) || !sr_mul(o->trials, c->num_constants, &nb) ||
      nc > SIZE_MAX / sizeof(float) || nc > SIZE_MAX / sizeof(size_t) || nb > SIZE_MAX / sizeof(float) ||
      o->capacity > SIZE_MAX / sizeof(SRCandidate) || !sr_mul(o->capacity, SR_PROGRAM_BYTES, &bytes))
    return SECANT_ERROR_OVERFLOW;
  s = calloc(1, sizeof(*s));
  if (!s) return SECANT_ERROR_ALLOCATION_FAILED;
  s->config = *c; memcpy(s->operators, c->operators, c->num_operators);
  s->config.operators=s->operators; s->options = *o; s->bank_elements = nb;
  s->models = calloc(o->capacity, sizeof(*s->models));
  s->centers = calloc(nc, sizeof(float)); s->scales = calloc(nc, sizeof(float));
  s->banks = malloc(nb * sizeof(float)); s->parameters = calloc(o->capacity, sizeof(size_t));
  s->generations = calloc(o->capacity, sizeof(size_t));
  s->selected = calloc(nc, sizeof(size_t));
  s->selected_count = calloc(o->capacity, sizeof(size_t));
  s->storage = malloc(bytes); s->asts = calloc(o->capacity, sizeof(*s->asts));
  if (!s->models || !s->centers || !s->scales || !s->banks || !s->parameters || !s->generations || !s->selected || !s->selected_count || !s->storage || !s->asts) {
    secant_sr_refiner_destroy(s); return SECANT_ERROR_ALLOCATION_FAILED;
  }
  *out = s; return SECANT_SUCCESS;
}
void secant_sr_refiner_destroy(SRRefiner s) {
  if (!s) return;
  free(s->models); free(s->centers); free(s->scales); free(s->banks);
  free(s->selected); free(s->selected_count);
  free(s->parameters); free(s->generations); free(s->storage); free(s->asts); free(s);
}
SecantResult secant_sr_refiner_seed(SRRefiner s, const SRModel *m, size_t count,
                                    const float *banks, size_t elements, uint64_t sequence) {
  size_t i, k, required, configs;
  if (!s) return SECANT_ERROR_INVALID_VALUE;
  s->count = 0; s->pending = 0; memset(&s->stats, 0, sizeof(s->stats));
  s->sequence = sequence;
  if (!m || !count || count > s->options.capacity || !banks ||
      !sr_mul(s->config.num_banks, s->config.num_constants, &required) || elements < required ||
      !sr_mul(s->config.num_banks, (size_t)1 << s->config.toggle_bits, &configs))
    return SECANT_ERROR_INVALID_VALUE;
  s->stats.bank_seed = s->options.seed ^ (sequence * UINT64_C(0x9e3779b97f4a7c15));
  for (i = 0; i < required; ++i) if (!isfinite(banks[i])) return SECANT_ERROR_INVALID_VALUE;
  for (i = 0; i < count; ++i) {
    SecantResult r;
    size_t off = i * SR_MAX_PARAMETERS;
    if (!isfinite(m[i].score.sse) || m[i].score.sse < 0 || !m[i].score.valid_configurations ||
        m[i].score.reserved || m[i].score.configuration >= configs || m[i].score.valid_configurations > configs)
      return SECANT_ERROR_INVALID_VALUE;
    if ((uintptr_t)m[i].nodes >= (uintptr_t)s->models &&
        (uintptr_t)m[i].nodes < (uintptr_t)s->models + s->options.capacity*sizeof(*s->models))
      return SECANT_ERROR_INVALID_VALUE;
    s->generations[i] = m[i].generation;
    r = map_parameters(&s->config, m + i, s->models + i, s->parameters + i,
        banks + (m[i].score.configuration >> s->config.toggle_bits) * s->config.num_constants, s->centers + off, 0);
    if (r != SECANT_SUCCESS) return r;
    if (s->options.parameters == SR_REFINE_ALL_PARAMETERS && s->parameters[i] > s->config.num_constants)
      return SECANT_ERROR_INSUFFICIENT_BUFFER;
    if (!s->parameters[i]) return SECANT_ERROR_INVALID_VALUE;
    for (k = 0; k < s->parameters[i]; ++k) s->scales[off+k] = s->options.initial_scale;
  }
  s->count = count; return SECANT_SUCCESS;
}
SecantResult secant_sr_refiner_ask(SRRefiner s, SecantAstProgramSet *out, const float **banks, size_t *elements) {
  size_t i;
  SecantResult r;
  if (!s || !out || !banks || !elements) return SECANT_ERROR_INVALID_VALUE;
  if (!s->count || s->pending) return SECANT_ERROR_INVALID_STATE;
  if (s->stats.rounds > UINT32_MAX) return SECANT_ERROR_OVERFLOW;
  r = secant_sr_refinement_bank(s->banks, s->options.trials, s->config.num_constants,
      s->stats.bank_seed, (uint32_t)s->stats.rounds);
  if (r != SECANT_SUCCESS) return r;
  for (i = 0; i < s->count; ++i) {
    size_t n, off = 0, k = i * SR_MAX_PARAMETERS, j, choice;
    size_t order[SR_MAX_PARAMETERS], count = 0, first = 0;
    unsigned char active[SR_MAX_PARAMETERS] = {0};
    SRCandidate *model = s->models+i;
    uint8_t *p = s->storage + i * SR_PROGRAM_BYTES;
    if (s->options.parameters == SR_REFINE_ACTIVE_BLOCK) {
      for (j=0;j<model->count;++j) if (!model->nodes[j].op) {
        SRNode *node=model->nodes+j;
        SRLeaf *l=node->leaf+sr_selector_choice(node,(uint32_t)model->score.configuration);
        if (parameter(l)) active[l->slot]=1;
      }
      for (j=0;j<s->parameters[i];++j) if (active[j]) order[count++]=j;
    }
    if (!count) for (j=0;j<s->parameters[i];++j) order[count++]=j;
    if (count>s->config.num_constants) {
      size_t blocks=(count+s->config.num_constants-1)/s->config.num_constants;
      first=(size_t)((s->sequence%blocks+s->stats.rounds%blocks)%blocks)*s->config.num_constants;
    }
    s->selected_count[i]=count-first;
    if (s->selected_count[i]>s->config.num_constants) s->selected_count[i]=s->config.num_constants;
    for (j=0;j<s->selected_count[i];++j) s->selected[k+j]=order[first+j];
    r = secant_sr_program_write(&s->config, s->models[i].nodes, s->models[i].count, p, SR_PROGRAM_BYTES, &n);
    if (r != SECANT_SUCCESS) return r;
    /* Map host identities to this round's bounded GPU slots. Unselected
     * coefficients retain zero-scale centers, including inactive alternatives. */
    for (j=0;j<model->count;++j) {
      SRNode *node=model->nodes+j;
      if (node->op) { ++off; continue; }
      for (choice=0;choice<node->choices;++choice) {
        SRLeaf *l=node->leaf+choice;
        if (parameter(l)) {
          size_t slot;
          for (slot=0;slot<s->selected_count[i];++slot) if (s->selected[k+slot]==l->slot) break;
          if (slot<s->selected_count[i])
            secant_ast_affine_bank_write(p+off,(uint8_t)slot,s->scales[k+l->slot],s->centers[k+l->slot]);
        }
        off+=secant_ast_instruction_size_get(p+off);
      }
      if (node->choices>1) off+=node->choices==2?2:3;
    }
    if (off+1!=n) return SECANT_ERROR_BAD_PROGRAM;
    s->asts[i] = p;
  }
  memset(out, 0, sizeof(*out)); out->asts.items = s->asts; out->asts.count = s->count;
  *banks = s->banks; *elements = s->bank_elements; s->pending = 1;
  return SECANT_SUCCESS;
}
SecantResult secant_sr_refiner_tell(SRRefiner s, const SRScore *scores, size_t count) {
  size_t i, j, k, configs;
  uint64_t work, finite = 0;
  if (!s || !scores || count != s->count) return SECANT_ERROR_INVALID_VALUE;
  if (!s->pending) return SECANT_ERROR_INVALID_STATE;
  if (!sr_mul(s->options.trials, (size_t)1 << s->config.toggle_bits, &configs) || count > UINT64_MAX / configs)
    return SECANT_ERROR_OVERFLOW;
  work = (uint64_t)count * configs;
  if (work > UINT64_MAX - s->stats.configurations) return SECANT_ERROR_OVERFLOW;
  for (i = 0; i < count; ++i) {
    const SRScore *v = scores+i;
    if (v->reserved || v->valid_configurations > configs ||
        (v->valid_configurations ? (!isfinite(v->sse) || v->sse < 0 || v->configuration >= configs) :
          (v->sse != INFINITY || v->configuration != UINT64_MAX))) return SECANT_ERROR_INVALID_VALUE;
    /* A finite final score can mask an overflowing coefficient (for example,
     * behind min/max). Never import such a parameter or partially update a batch. */
    if (v->valid_configurations && v->sse < s->models[i].score.sse) {
      size_t trial = (size_t)(v->configuration >> s->config.toggle_bits);
      uint32_t perm = (uint32_t)v->configuration;
      for (j = 0; j < s->models[i].count; ++j) {
        const SRNode *node = s->models[i].nodes+j;
        unsigned choice = node->choices > 1 ? (perm >> node->low_bit) & 1u : 0;
        if (!node->choices) continue;
        if (node->choices == 4) choice |= ((perm >> node->high_bit) & 1u) << 1;
        if (parameter(node->leaf+choice)) {
          size_t parameter_id = node->leaf[choice].slot, base=i*SR_MAX_PARAMETERS, slot;
          for(slot=0;slot<s->selected_count[i];++slot) if(s->selected[base+slot]==parameter_id) break;
          if(slot<s->selected_count[i]) {
            volatile float step = s->scales[base+parameter_id]*s->banks[trial*s->config.num_constants+slot];
            if (!isfinite(s->centers[base+parameter_id]+step)) return SECANT_ERROR_INVALID_VALUE;
          }
        }
      }
    }
    finite += v->valid_configurations;
  }
  for (i = 0; i < count; ++i) {
    SRCandidate *m = s->models+i;
    size_t base = i * SR_MAX_PARAMETERS;
    uint8_t active[SR_MAX_PARAMETERS] = {0};
    int improved = scores[i].valid_configurations && scores[i].sse < m->score.sse;
    uint32_t perm = (uint32_t)(improved ? scores[i].configuration : m->score.configuration);
    size_t trial = improved ? (size_t)(scores[i].configuration >> s->config.toggle_bits) : 0;
    for (j = 0; j < m->count; ++j) if (!m->nodes[j].op) {
      SRNode *n = m->nodes+j;
      unsigned choice = n->choices > 1 ? (perm >> n->low_bit) & 1u : 0;
      if (n->choices == 4) choice |= ((perm >> n->high_bit) & 1u) << 1;
      if (parameter(n->leaf+choice)) active[n->leaf[choice].slot] = 1;
    }
    for (k = 0; k < s->selected_count[i]; ++k) if (active[s->selected[base+k]]) {
      size_t id=s->selected[base+k];
      float scale = s->scales[base+id];
      if (improved) {
        volatile float step = scale * s->banks[trial*s->config.num_constants+k];
        s->centers[base+id] += step;
        scale += s->options.scale_learning_rate * (2.f*fabsf(step)-scale);
      } else scale *= s->options.failure_decay;
      s->scales[base+id] = fminf(s->options.maximum_scale, fmaxf(s->options.minimum_scale, scale));
    }
    if (improved) {
      for (j = 0; j < m->count; ++j) for (k = 0; k < m->nodes[j].choices; ++k) {
        SRLeaf *l = &m->nodes[j].leaf[k];
        if (parameter(l)) l->value = s->centers[base+l->slot];
      }
      m->score = scores[i]; m->score.configuration &= ((UINT64_C(1) << s->config.toggle_bits)-1);
      m->score.valid_configurations = 1;
      m->fingerprint = sr_fingerprint(m->nodes, m->count); ++s->stats.accepted;
    }
  }
  s->stats.configurations += work; s->stats.finite_configurations += finite;
  ++s->stats.rounds; s->pending = 0; return SECANT_SUCCESS;
}
SecantResult secant_sr_refiner_model(SRRefiner s, size_t i, SRModel *out) {
  SRCandidate *m;
  if (!s || !out || i >= s->count) return SECANT_ERROR_INVALID_VALUE;
  m=s->models+i; out->nodes=m->nodes; out->num_nodes=m->count; out->score=m->score;
  out->generation=s->generations[i]; out->fingerprint=m->fingerprint;
  return SECANT_SUCCESS;
}
SRRefineStats secant_sr_refiner_stats(SRRefiner s) {
  SRRefineStats empty={0}; return s ? s->stats : empty;
}
