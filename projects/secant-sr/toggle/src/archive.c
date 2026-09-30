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
#include "secant_sr_archive.h"
typedef struct SRArchiveEntry {
  SRCandidate model;
  uint8_t key[SECANT_SR_MAX_NODES * 9u];
  size_t key_bytes, generation;
} SRArchiveEntry;
struct SRArchiveImpl {
  SRConfig config;
  uint8_t operators[32];
  SRArchiveEntry *entries;
  size_t count, capacity;
  SRArchiveStats stats;
};
SecantResult secant_sr_archive_create(const SRConfig *c, size_t capacity, SRArchive *out) {
  SRArchive a;SecantResult r;
  if(!out)return SECANT_ERROR_INVALID_VALUE;
  *out=NULL;r=sr_validate_config(c);if(r!=SECANT_SUCCESS)return r;
  if(!capacity)return SECANT_ERROR_INVALID_VALUE;
  if(capacity>SIZE_MAX/sizeof(SRArchiveEntry))return SECANT_ERROR_OVERFLOW;
  a=calloc(1,sizeof(*a));if(!a)return SECANT_ERROR_ALLOCATION_FAILED;
  a->entries=calloc(capacity,sizeof(*a->entries));
  if(!a->entries){free(a);return SECANT_ERROR_ALLOCATION_FAILED;}
  a->config=*c;memcpy(a->operators,c->operators,c->num_operators);a->config.operators=a->operators;
  a->capacity=capacity;*out=a;return SECANT_SUCCESS;
}
void secant_sr_archive_destroy(SRArchive a){if(a){free(a->entries);free(a);}}
static void key_u32(SRArchiveEntry *entry,uint32_t value) {
  unsigned i;for(i=0;i<4;++i)entry->key[entry->key_bytes++]=(uint8_t)(value>>(8*i));
}
SecantResult secant_sr_archive_offer(SRArchive a,const SRModel *m,const float *banks,size_t elements) {
  SRArchiveEntry entry;SRLeaf identities[SECANT_SR_MAX_NODES];size_t count=0,i,n,configs,position;
  SecantResult r;
  if(!a||!m)return SECANT_ERROR_INVALID_VALUE;
  r=sr_validate(&a->config,m->nodes,m->num_nodes,NULL,NULL);if(r!=SECANT_SUCCESS)return r;
  if(!sr_mul(a->config.num_banks,a->config.num_constants,&n)||elements<n||(n&&!banks)||
     !sr_mul(a->config.num_banks,(size_t)1<<a->config.toggle_bits,&configs))return SECANT_ERROR_INVALID_VALUE;
  if(m->score.reserved || m->score.valid_configurations>configs)return SECANT_ERROR_INVALID_VALUE;
  if(!m->score.valid_configurations) {
    if(m->score.sse!=INFINITY||m->score.configuration!=UINT64_MAX)return SECANT_ERROR_INVALID_VALUE;
    ++a->stats.visited;return SECANT_SUCCESS;
  }
  if(!isfinite(m->score.sse)||m->score.sse<0||m->score.configuration>=configs)return SECANT_ERROR_INVALID_VALUE;
  ++a->stats.visited;
  if(a->count==a->capacity&&m->score.sse>=a->entries[a->count-1].model.score.sse){++a->stats.score_pruned;return SECANT_SUCCESS;}
  memset(&entry,0,sizeof(entry));entry.model.count=(uint16_t)m->num_nodes;entry.generation=m->generation;
  entry.model.score=m->score;entry.model.score.configuration=0;entry.model.score.valid_configurations=1;
  for(i=0;i<m->num_nodes;++i){
    const SRNode *source=m->nodes+i;SRNode *dest=entry.model.nodes+i;
    dest->op=source->op;entry.key[entry.key_bytes++]=source->op;
    if(!source->op){
      size_t j;uint32_t bits=0;const SRLeaf *l=source->leaf+sr_selector_choice(source,(uint32_t)m->score.configuration);
      dest->choices=1;dest->leaf[0]=*l;
      if(l->kind==SR_COEFFICIENT||l->kind==SR_FITTED_COEFFICIENT){
        for(j=0;j<count;++j)if(identities[j].kind==l->kind&&identities[j].slot==l->slot&&
          (l->kind==SR_COEFFICIENT||!memcmp(&identities[j].value,&l->value,4)))break;
        if(j==count)identities[count++]=*l;
        dest->leaf[0].kind=SR_FITTED_COEFFICIENT;dest->leaf[0].slot=(uint32_t)j;
        if(l->kind==SR_COEFFICIENT)dest->leaf[0].value=banks[(m->score.configuration>>a->config.toggle_bits)*a->config.num_constants+l->slot];
        if(!isfinite(dest->leaf[0].value))return SECANT_ERROR_INVALID_VALUE;
        entry.key[entry.key_bytes++]=2;key_u32(&entry,(uint32_t)j);
      }else if(l->kind==SR_COLUMN){entry.key[entry.key_bytes++]=1;key_u32(&entry,l->slot);}
      else{entry.key[entry.key_bytes++]=3;memcpy(&bits,&l->value,4);key_u32(&entry,bits);}
    }
  }
  if(!count)return SECANT_SUCCESS;
  ++a->stats.eligible;entry.model.fingerprint=sr_fingerprint(entry.model.nodes,entry.model.count);
  for(position=0;position<a->count;++position)if(a->entries[position].key_bytes==entry.key_bytes&&
      !memcmp(a->entries[position].key,entry.key,entry.key_bytes))break;
  if(position<a->count){
    ++a->stats.duplicates;
    if(entry.model.score.sse>=a->entries[position].model.score.sse)return SECANT_SUCCESS;
  }else if(a->count<a->capacity)++a->count;
  else position=a->count-1;
  /* Remove/reinsert to keep training-score order; model ownership stays local. */
  for(i=position;i+1<a->count;++i)a->entries[i]=a->entries[i+1];
  position=a->count-1;
  while(position&&a->entries[position-1].model.score.sse>entry.model.score.sse){a->entries[position]=a->entries[position-1];--position;}
  a->entries[position]=entry;++a->stats.retained;return SECANT_SUCCESS;
}
size_t secant_sr_archive_count(SRArchive a){return a?a->count:0;}
SecantResult secant_sr_archive_model(SRArchive a,size_t index,SRModel *out){
  SRArchiveEntry *e;if(!a||!out||index>=a->count)return SECANT_ERROR_INVALID_VALUE;e=a->entries+index;
  out->nodes=e->model.nodes;out->num_nodes=e->model.count;out->score=e->model.score;
  out->fingerprint=e->model.fingerprint;out->generation=e->generation;return SECANT_SUCCESS;
}
SRArchiveStats secant_sr_archive_stats(SRArchive a){SRArchiveStats empty={0};return a?a->stats:empty;}
