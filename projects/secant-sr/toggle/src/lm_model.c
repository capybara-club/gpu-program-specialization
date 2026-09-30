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
#include "lm_internal.h"
#include "secant_sr_refine.h"
#include <float.h>

typedef char sr_lm_instruction_size[(sizeof(SRLMInstruction)==16)?1:-1];
typedef char sr_lm_program_size[(sizeof(SRLMProgram)==1088)?1:-1];
typedef char sr_lm_state_size[(sizeof(SRLMState)==88)?1:-1];
SRLMOptions secant_sr_lm_options_default(void) {
  SRLMOptions o={128,32,4,8,8,64,0,42,1.f,.001};return o;
}
SecantResult sr_lm_batch_create(const SRConfig *c,const SRLMOptions *o,SRLMBatch *b) {
  size_t programs,states,bytes;SecantResult r;
  if(!b)return SECANT_ERROR_INVALID_VALUE;
  memset(b,0,sizeof(*b));r=sr_validate_config(c);if(r)return r;
  if(!o||!o->capacity||o->capacity>c->population||!o->bindings||o->bindings>65536||
     !o->starts||o->starts>128||!o->iterations||o->iterations>1000||
     !o->parameters||o->parameters>8||(o->threads!=32&&o->threads!=64&&o->threads!=128)||
     !isfinite(o->initial_scale)||o->initial_scale<0||!isfinite(o->initial_damping)||
     o->initial_damping<=0||o->device<0)return SECANT_ERROR_INVALID_VALUE;
  if(!sr_mul(o->capacity,o->bindings,&programs)||!sr_mul(programs,o->starts,&states)||
     programs>UINT_MAX||states>UINT_MAX||!sr_mul(o->capacity,SR_PROGRAM_BYTES,&bytes)||
     programs>SIZE_MAX/sizeof(SRLMProgram)||states>SIZE_MAX/sizeof(SRLMState))return SECANT_ERROR_OVERFLOW;
  b->config=*c;memcpy(b->operators,c->operators,c->num_operators);b->config.operators=b->operators;b->options=*o;
  b->models=calloc(o->capacity,sizeof(*b->models));b->generations=calloc(o->capacity,sizeof(size_t));
  b->first=calloc(o->capacity,sizeof(size_t));b->length=calloc(o->capacity,sizeof(size_t));
  b->programs=calloc(programs,sizeof(*b->programs));b->states=calloc(states,sizeof(*b->states));
  b->storage=malloc(bytes);b->asts=calloc(o->capacity,sizeof(*b->asts));
  b->jitter=malloc(o->starts*SR_LM_PARAMETERS*sizeof(float));
  if(!b->models||!b->generations||!b->first||!b->length||!b->programs||!b->states||!b->storage||!b->asts||!b->jitter){sr_lm_batch_free(b);return SECANT_ERROR_ALLOCATION_FAILED;}
  return SECANT_SUCCESS;
}
void sr_lm_batch_free(SRLMBatch *b) {
  if(!b)return;
  free(b->models);free(b->generations);free(b->first);free(b->length);
  free(b->programs);free(b->states);free(b->storage);free(b->asts);free(b->jitter);memset(b,0,sizeof(*b));
}
static uint32_t bits(float v){uint32_t u;memcpy(&u,&v,4);return u;}
SecantResult sr_lm_batch_prepare(SRLMBatch *b,const SRModel *models,size_t count,
    const float *banks,size_t elements,uint64_t sequence) {
  size_t m,required,configs;uint64_t rng;
  if(!b)return SECANT_ERROR_INVALID_VALUE;
  b->count=b->program_count=b->state_count=0;memset(&b->stats,0,sizeof(b->stats));
  if(!models||!count||count>b->options.capacity||!sr_mul(b->config.num_banks,b->config.num_constants,&required)||
     elements<required||(required&&!banks)||!sr_mul(b->config.num_banks,(size_t)1<<b->config.toggle_bits,&configs))return SECANT_ERROR_INVALID_VALUE;
  for(m=0;m<required;++m)if(!isfinite(banks[m]))return SECANT_ERROR_INVALID_VALUE;
  /* Borrowed results cannot be fed back while this batch is being overwritten. */
  for(m=0;m<count;++m){
    uintptr_t p=(uintptr_t)models[m].nodes,base=(uintptr_t)b->models;
    if(p>=base && p-base<b->options.capacity*sizeof(*b->models))return SECANT_ERROR_INVALID_VALUE;
  }
  rng=b->options.seed^(sequence*UINT64_C(0x9e3779b97f4a7c15));
  {SecantResult r=secant_sr_refinement_bank(b->jitter,b->options.starts,SR_LM_PARAMETERS,rng,(uint32_t)(sequence>>32));if(r)return r;}
  for(m=0;m<count;++m){
    const SRModel *source=models+m;SRCandidate *dest=b->models+m;
    SRLeaf identities[SECANT_SR_MAX_NODES*4];float centers[SECANT_SR_MAX_NODES*4];
    size_t parameters=0,i,j,k,n;unsigned used[16],nbits=0,mask=0,incumbent;
    uint64_t offset=sr_random(&rng),stride=sr_random(&rng)|1;
    SecantResult r=sr_validate(&b->config,source->nodes,source->num_nodes,NULL,NULL);if(r)return r;
    if(!isfinite(source->score.sse)||source->score.sse<0||!source->score.valid_configurations||
       source->score.reserved||source->score.configuration>=configs||source->score.valid_configurations>configs)return SECANT_ERROR_INVALID_VALUE;
    memset(dest,0,sizeof(*dest));dest->count=(uint16_t)source->num_nodes;dest->score=source->score;
    memcpy(dest->nodes,source->nodes,source->num_nodes*sizeof(SRNode));b->generations[m]=source->generation;
    incumbent=(unsigned)(source->score.configuration&(((uint64_t)1<<b->config.toggle_bits)-1));
    dest->score.configuration=incumbent;dest->score.valid_configurations=1;
    for(i=0;i<source->num_nodes;++i){
      const SRNode *node=source->nodes+i;
      if(node->choices>1)mask|=1u<<node->low_bit;
      if(node->choices==4)mask|=1u<<node->high_bit;
      for(j=0;j<node->choices;++j){
        const SRLeaf *leaf=node->leaf+j;SRLeaf *target=dest->nodes[i].leaf+j;
        if(leaf->kind!=SR_COEFFICIENT&&leaf->kind!=SR_FITTED_COEFFICIENT)continue;
        for(k=0;k<parameters;++k)if(identities[k].kind==leaf->kind&&identities[k].slot==leaf->slot&&
          (leaf->kind==SR_COEFFICIENT||bits(identities[k].value)==bits(leaf->value)))break;
        if(k==parameters){identities[parameters++]=*leaf;centers[k]=leaf->kind==SR_COEFFICIENT?
          banks[(source->score.configuration>>b->config.toggle_bits)*b->config.num_constants+leaf->slot]:leaf->value;}
        *target=(SRLeaf){SR_FITTED_COEFFICIENT,(uint32_t)k,centers[k]};
      }
    }
    for(i=0;i<b->config.toggle_bits;++i)if(mask&(1u<<i))used[nbits++]=(unsigned)i;
    n=(size_t)1<<nbits;if(n>b->options.bindings)n=b->options.bindings;
    b->first[m]=b->program_count;b->length[m]=n;
    for(j=0;j<n;++j){
      SRLMProgram *p=b->programs+b->program_count;unsigned active[SECANT_SR_MAX_NODES],na=0;
      unsigned stack[SECANT_SR_MAX_NODES],sp=0,permutation=incumbent;
      size_t block,first;
      if(j){
        unsigned compact;
        /* An odd-stride permutation visits each nonzero mask once. */
        do{compact=(unsigned)(offset&(((uint64_t)1<<nbits)-1));offset+=stride;}while(!compact);
        for(k=0;k<nbits;++k)if(compact&(1u<<k))permutation^=1u<<used[k];
      }
      memset(p,0,sizeof(*p));p->count=dest->count;p->model=(unsigned)m;p->permutation=permutation;
      for(i=0;i<dest->count;++i)if(!dest->nodes[i].op){
        const SRLeaf *l=dest->nodes[i].leaf+sr_selector_choice(dest->nodes+i,permutation);
        if(l->kind!=SR_FITTED_COEFFICIENT)continue;
        for(k=0;k<na&&active[k]!=l->slot;++k){}
        if(k==na)active[na++]=l->slot;
      }
      block=(na+b->options.parameters-1)/b->options.parameters;
      first=block?(sequence%block)*b->options.parameters:0;
      p->parameters=na-(unsigned)first;if(p->parameters>b->options.parameters)p->parameters=b->options.parameters;
      if(na>b->options.parameters)++b->stats.blocked_bindings;
      if(!na)++b->stats.inactive_bindings;
      for(k=0;k<p->parameters;++k){p->parameter_ids[k]=active[first+k];p->center[k]=centers[active[first+k]];}
      for(i=0;i<dest->count;++i){
        const SRNode *node=dest->nodes+i;SRLMInstruction *out=p->code+i;unsigned arity=sr_arity(node->op);
        out->op=node->op;
        if(arity){out->a=stack[sp-arity];if(arity==2)out->b=stack[sp-1];sp-=arity;}
        else{
          const SRLeaf *l=node->leaf+sr_selector_choice(node,permutation);
          if(l->kind==SR_COLUMN){out->op=1;out->a=l->slot;}
          else{out->value=l->value;for(k=0;k<p->parameters&&p->parameter_ids[k]!=l->slot;++k){}
            if(l->kind==SR_FITTED_COEFFICIENT&&k<p->parameters){out->op=2;out->a=(unsigned)k;}}
        }
        stack[sp++]=(unsigned)i;
      }
      for(k=0;k<b->options.starts;++k){
        SRLMState *s=b->states+b->state_count++;memset(s,0,sizeof(*s));s->loss=INFINITY;s->damping=b->options.initial_damping;
        for(i=0;i<p->parameters;++i){
          float v=p->center[i];
          if(k){double scale=b->options.initial_scale*(k%3==1?.1:k%3==2?1.:3.)*fmax(1.,fabs(v));double proposal=v+b->jitter[k*SR_LM_PARAMETERS+i]*scale;if(isfinite(proposal)&&fabs(proposal)<=FLT_MAX)v=(float)proposal;}
          s->best[i]=s->trial[i]=v;
        }
      }
      ++b->program_count;
    }
    dest->fingerprint=sr_fingerprint(dest->nodes,dest->count);
  }
  b->count=count;b->stats.bindings=b->program_count;b->stats.states=b->state_count;return SECANT_SUCCESS;
}
SecantResult sr_lm_batch_finish(SRLMBatch *b,SecantAstProgramSet *out) {
  size_t m;if(!b||!out||!b->count)return SECANT_ERROR_INVALID_STATE;
  for(m=0;m<b->count;++m){
    size_t p,s,best=SIZE_MAX,bytes;double loss=INFINITY;SRCandidate *model=b->models+m;
    for(p=b->first[m];p<b->first[m]+b->length[m];++p)for(s=0;s<b->options.starts;++s){
      size_t at=p*b->options.starts+s;const SRLMState *state=b->states+at;
      b->stats.accepted_steps+=state->accepted;
      b->stats.invalid_evaluations+=state->invalid;
      if(isfinite(state->loss)&&state->loss>=0&&state->loss<loss){best=at;loss=state->loss;}
    }
    if(best!=SIZE_MAX&&loss<=FLT_MAX){
      const SRLMState *state=b->states+best;const SRLMProgram *program=b->programs+best/b->options.starts;
      size_t i,j,k;
      for(i=0;i<model->count;++i)for(j=0;j<model->nodes[i].choices;++j){
        SRLeaf *leaf=model->nodes[i].leaf+j;
        if(leaf->kind!=SR_FITTED_COEFFICIENT)continue;
        for(k=0;k<program->parameters;++k)if(leaf->slot==program->parameter_ids[k]){
          if(!isfinite(state->best[k]))return SECANT_ERROR_INVALID_VALUE;
          leaf->value=state->best[k];break;
        }
      }
      model->score.sse=(float)loss;model->score.configuration=program->permutation;
      model->fingerprint=sr_fingerprint(model->nodes,model->count);
    }
    b->asts[m]=b->storage+m*SR_PROGRAM_BYTES;
    {SecantResult r=secant_sr_program_write(&b->config,model->nodes,model->count,b->storage+m*SR_PROGRAM_BYTES,SR_PROGRAM_BYTES,&bytes);if(r)return r;}
  }
  memset(out,0,sizeof(*out));out->asts.items=b->asts;out->asts.count=b->count;return SECANT_SUCCESS;
}
