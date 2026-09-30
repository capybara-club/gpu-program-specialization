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
#include "secant_sr_refine.h"
#include "secant_instructions.h"
#ifdef SR_HAS_CUDA
#include "secant_sr_cuda.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); exit(1); } } while(0)
#define SEC(x) CHECK((x)==SECANT_SUCCESS)
static SRLeaf col(unsigned n) { SRLeaf l={SR_COLUMN,n,0}; return l; }
static SRLeaf coef(unsigned n) { SRLeaf l={SR_COEFFICIENT,n,0}; return l; }
static void check_block_rotation(void) {
  SRConfig c=secant_sr_config_default();SRRefineOptions o=secant_sr_refine_options_default();
  SRRefiner fit=NULL;SRNode nodes[9]={{0}};SRModel m={0},result;
  SecantAstProgramSet p;float banks[4]={0};const float *jitter;
  size_t i,round,n;SRScore center={0,0,0,1};
  c.population=4;c.elites=1;c.num_inputs=1;c.num_constants=2;c.num_banks=2;c.toggle_bits=0;
  o.capacity=1;o.trials=2;o.parameters=SR_REFINE_ACTIVE_BLOCK;
  /* Five independent centers, including two equal values, exceed two GPU slots. */
  for(i=0;i<5;++i){nodes[i].choices=1;nodes[i].leaf[0]=(SRLeaf){SR_FITTED_COEFFICIENT,(uint32_t)(100+i),(float)(i/2)};}
  for(i=5;i<9;++i)nodes[i].op=SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
  m.nodes=nodes;m.num_nodes=9;m.score=center;
  SEC(secant_sr_refiner_create(&c,&o,&fit));SEC(secant_sr_refiner_seed(fit,&m,1,banks,4,2));
  for(round=0;round<6;++round){
    const uint8_t *code;size_t first=((2+round)%3)*2;
    SEC(secant_sr_refiner_ask(fit,&p,&jitter,&n));code=p.asts.items[0];
    CHECK(n==4 && jitter[0]==0 && jitter[1]==0);
    for(i=0;i<5;++i){
      int selected=i>=first && i<first+2;
      CHECK(*code==SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32);
      CHECK((secant_ast_affine_bank_scale_get(code)>0)==selected);
      CHECK(secant_ast_affine_bank_offset_get(code)==nodes[i].leaf[0].value);
      if(selected) CHECK(code[1]==i-first);
      code+=secant_ast_instruction_size_get(code);
    }
    SEC(secant_sr_refiner_tell(fit,&center,1));
  }
  SEC(secant_sr_refiner_model(fit,0,&result));
  SEC(secant_sr_refine_parameter_count(&c,&result,&n));CHECK(n==5);
  for(i=0;i<5;++i)CHECK(result.nodes[i].leaf[0].value==nodes[i].leaf[0].value);
  secant_sr_refiner_destroy(fit);
}
static void check_active_blocks(void) {
  enum { ROWS=23, BANKS=64, CONFIGS=BANKS*16 };
  SRConfig c=secant_sr_config_default();SRRefineOptions o=secant_sr_refine_options_default();
  SRRefiner fit=NULL;SRSearch search=NULL;SRNode nodes[5]={{0}};SRModel m={0},done;
  uint8_t code[256];const uint8_t *asts[]={code};SecantAstProgramSet p={0};
  float banks[BANKS*2]={0},x[ROWS],y[ROWS],before[CONFIGS],grid[CONFIGS];
  SRScore score,initial[4];size_t i,j,n,total,active;const float *jitter;
#ifdef SR_HAS_CUDA
  SRCudaScorer gpu=NULL;SRCudaOptions options=secant_sr_cuda_options_default();SRCudaStats timing;
#endif
  c.population=4;c.elites=1;c.num_inputs=1;c.num_constants=2;c.num_banks=BANKS;c.toggle_bits=4;
  o.capacity=1;o.trials=BANKS;o.parameters=SR_REFINE_ACTIVE_BLOCK;
  nodes[0].choices=nodes[3].choices=4;nodes[0].high_bit=1;nodes[3].low_bit=2;nodes[3].high_bit=3;
  for(i=0;i<4;++i) {
    nodes[0].leaf[i]=(SRLeaf){SR_FITTED_COEFFICIENT,(uint32_t)(100+i),i?(float)(10+i):2.7f};
    nodes[3].leaf[i]=(SRLeaf){SR_FITTED_COEFFICIENT,(uint32_t)(200+i),i?(float)(20+i):-.1f};
  }
  nodes[1].choices=1;nodes[1].leaf[0]=col(0);nodes[2].op=SECANT_AST_INSTRUCTION_TYPE_MUL_F32;nodes[4].op=SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
  for(i=0;i<ROWS;++i){x[i]=(float)i*.2f-2;y[i]=2.75f*x[i]-.125f;}
  SEC(secant_sr_program_write(&c,nodes,5,code,sizeof(code),&n));p.asts.items=asts;p.asts.count=1;
  SEC(secant_sr_cpu_score(&c,&p,x,y,ROWS,banks,before,CONFIGS,&score));
  m.nodes=nodes;m.num_nodes=5;m.score=score;
  SEC(secant_sr_refine_parameter_count(&c,&m,&total));SEC(secant_sr_refine_active_parameter_count(&c,&m,&active));
  CHECK(total==8 && active==2 && score.configuration==0);
  SEC(secant_sr_create(&c,banks,BANKS*2,&search));
  for(i=0;i<4;++i){SEC(secant_sr_seed(search,i,nodes,5));initial[i]=score;}
  {SecantAstProgramSet population;SEC(secant_sr_ask(search,&population));SEC(secant_sr_tell(search,initial,4,ROWS,1));}
  SEC(secant_sr_refiner_create(&c,&o,&fit));SEC(secant_sr_refiner_seed(fit,&m,1,banks,BANKS*2,0));
#ifdef SR_HAS_CUDA
  options.asts_per_kernel=2;options.kernels_per_module=1;
  SEC(secant_sr_cuda_create(&c,x,y,ROWS,banks,&options,&gpu,&timing));
#endif
  for(i=0;i<32;++i) {
    SEC(secant_sr_refiner_ask(fit,&p,&jitter,&n));
    SEC(secant_sr_cpu_score(&c,&p,x,y,ROWS,jitter,grid,CONFIGS,&score));
    if(!i) for(j=0;j<16;++j) CHECK(grid[j]==before[j]); /* Every incumbent toggle alternative is retained. */
#ifdef SR_HAS_CUDA
    {SRScore actual;SEC(secant_sr_cuda_score_banks(gpu,&p,jitter,BANKS,n,&actual,1,&timing));
      CHECK(fabsf(actual.sse-score.sse)<1e-4f*fmaxf(1,score.sse));score=actual;}
#endif
    SEC(secant_sr_refiner_tell(fit,&score,1));
  }
  SEC(secant_sr_refiner_model(fit,0,&done));CHECK(done.score.sse<1e-5f);
  for(i=1;i<4;++i){CHECK(done.nodes[0].leaf[i].value==nodes[0].leaf[i].value);CHECK(done.nodes[3].leaf[i].value==nodes[3].leaf[i].value);}
  SEC(secant_sr_accept_refined(search,0,&done));
  SEC(secant_sr_refine_parameter_count(&c,&done,&total));CHECK(total==8);
  CHECK(secant_sr_refiner_seed(fit,&done,1,banks,BANKS*2,1)==SECANT_ERROR_INVALID_VALUE);
  SEC(secant_sr_candidate(search,0,&done));
  SEC(secant_sr_refiner_seed(fit,&done,1,banks,BANKS*2,1));
  secant_sr_refiner_destroy(fit);secant_sr_destroy(search);
#ifdef SR_HAS_CUDA
  SEC(secant_sr_cuda_destroy(gpu));
#endif
}
static void check_parameter_contract(void) {
  SRConfig c=secant_sr_config_default();
  SRRefineOptions o=secant_sr_refine_options_default();
  SRRefiner fit=NULL; SRSearch search=NULL;
  SRNode nodes[3]={{0}}, revised[3]; SRModel model={0};
  SRScore scores[4]; SecantAstProgramSet programs;
  float banks[4]={1,2,3,4}; size_t i,count;
  c.population=4;c.elites=1;c.num_inputs=1;c.num_constants=2;c.num_banks=2;c.toggle_bits=0;
  nodes[0].choices=nodes[1].choices=1;
  nodes[0].leaf[0]=coef(0);nodes[1].leaf[0]=coef(0);
  nodes[2].op=SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
  SEC(secant_sr_create(&c,banks,4,&search));
  for(i=0;i<4;++i) {
    SEC(secant_sr_seed(search,i,nodes,3));
    scores[i]=(SRScore){10,0,0,2};
  }
  SEC(secant_sr_ask(search,&programs));
  SEC(secant_sr_tell(search,scores,4,2,1));
  memcpy(revised,nodes,sizeof(nodes));
  revised[0].leaf[0]=(SRLeaf){SR_FITTED_COEFFICIENT,1,.4f};
  revised[1].leaf[0]=(SRLeaf){SR_FITTED_COEFFICIENT,1,.5f};
  model.nodes=revised;model.num_nodes=3;model.score=(SRScore){1,0,0,1};
  CHECK(secant_sr_accept_refined(search,0,&model)==SECANT_ERROR_BAD_PROGRAM);
  revised[1].leaf[0].value=.4f;
  SEC(secant_sr_accept_refined(search,0,&model));
  SEC(secant_sr_advance(search));
  SEC(secant_sr_ask(search,&programs));
  CHECK(programs.asts.items[0][0]==SECANT_AST_INSTRUCTION_TYPE_AFFINE_BANK_F32);
  secant_sr_destroy(search);
  /* Crossover can bring different fitted values with the same former slot ID.
   * Count them separately; reject rather than silently tie over-capacity values. */
  nodes[0].choices=2;nodes[0].low_bit=0;c.toggle_bits=1;
  nodes[0].leaf[0]=(SRLeaf){SR_FITTED_COEFFICIENT,0,1};
  nodes[0].leaf[1]=(SRLeaf){SR_FITTED_COEFFICIENT,0,2};
  nodes[1].leaf[0]=coef(0);model.nodes=nodes;
  SEC(secant_sr_refine_parameter_count(&c,&model,&count));CHECK(count==3);
  o.capacity=1;o.trials=2;
  SEC(secant_sr_refiner_create(&c,&o,&fit));
  CHECK(secant_sr_refiner_seed(fit,&model,1,banks,4,0)==SECANT_ERROR_INSUFFICIENT_BUFFER);
  { const float *jitter;size_t elements;
    CHECK(secant_sr_refiner_ask(fit,&programs,&jitter,&elements)==SECANT_ERROR_INVALID_STATE); }
  secant_sr_refiner_destroy(fit);
}
int main(void) {
  enum { N=3, ROWS=31, BANKS=64, CONFIGS=BANKS*8, ROUNDS=24 };
  SRConfig c=secant_sr_config_default();
  SRRefineOptions o=secant_sr_refine_options_default();
  SRRefiner fit=NULL;
  SRNode nodes[N][5]; SRModel models[N];
  uint8_t code[N][256], resolved[512]; const uint8_t *asts[N];
  float x[ROWS*2], y[ROWS], banks[BANKS*2], grid[N*CONFIGS];
  SRScore scores[N], cpu_scores[N], baseline[N];
  SecantAstProgramSet p={0};
  size_t i,j,bytes,n;
#ifdef SR_HAS_CUDA
  SRCudaOptions options=secant_sr_cuda_options_default();
  SRCudaScorer gpu=NULL; SRCudaStats timing;
  double pipeline=0, device=0, transfers=0;
#endif
  c.population=16;c.elites=1;c.num_inputs=2;c.num_constants=2;c.num_banks=BANKS;c.toggle_bits=3;
  o.capacity=N;o.trials=BANKS;o.seed=123;o.initial_scale=2;
  for(i=0;i<ROWS;++i){x[i]=(float)i*.1f-1.5f;x[ROWS+i]=sinf((float)i*.8f);y[i]=2.75f*x[i]-.125f;}
  SEC(secant_sr_bank_generate(banks,BANKS,2,321,0,-4,4));
  /* A known local basin tests numerical convergence, not global discovery. */
  banks[0]=2.7f; banks[1]=-.1f;
  memset(nodes,0,sizeof(nodes));memset(models,0,sizeof(models));
  for(i=0;i<N;++i){
    nodes[i][0].choices=4;nodes[i][0].low_bit=0;nodes[i][0].high_bit=1;
    nodes[i][0].leaf[0]=col(1);nodes[i][0].leaf[1]=(SRLeaf){SR_LITERAL,0,1};
    nodes[i][0].leaf[2]=coef(0);nodes[i][0].leaf[3]=col(0);
    nodes[i][1].choices=1;nodes[i][1].leaf[0]=col(0);
    nodes[i][2].op=SECANT_AST_INSTRUCTION_TYPE_MUL_F32;
    nodes[i][3].choices=2;nodes[i][3].low_bit=2;
    nodes[i][3].leaf[0]=col(i%2);nodes[i][3].leaf[1]=coef(1);
    nodes[i][4].op=SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
    SEC(secant_sr_program_write(&c,nodes[i],5,code[i],sizeof(code[i]),&bytes));asts[i]=code[i];
  }
  p.asts.items=asts;p.asts.count=N;
  SEC(secant_sr_cpu_score(&c,&p,x,y,ROWS,banks,grid,N*CONFIGS,scores));
  memcpy(baseline,scores,sizeof(baseline));
#ifdef SR_HAS_CUDA
  options.asts_per_kernel=2;options.kernels_per_module=1;options.ast_batch=3;
  SEC(secant_sr_cuda_create(&c,x,y,ROWS,banks,&options,&gpu,&timing));
#endif
  for(i=0;i<N;++i){models[i].nodes=nodes[i];models[i].num_nodes=5;models[i].score=scores[i];}
  SEC(secant_sr_refiner_create(&c,&o,&fit));
  SEC(secant_sr_refiner_seed(fit,models,N,banks,BANKS*2,0));
  for(j=0;j<ROUNDS;++j){
    const float *jitter;
    SRScore bad[N];
    SEC(secant_sr_refiner_ask(fit,&p,&jitter,&n));
    CHECK(secant_sr_refiner_ask(fit,&p,&jitter,&n)==SECANT_ERROR_INVALID_STATE);
    CHECK(jitter[0]==0 && jitter[1]==0);
    SEC(secant_sr_cpu_score(&c,&p,x,y,ROWS,jitter,grid,N*CONFIGS,cpu_scores));
#ifdef SR_HAS_CUDA
    SEC(secant_sr_cuda_score_banks(gpu,&p,jitter,BANKS,n,scores,N,&timing));
    pipeline+=timing.pipeline_seconds;device+=timing.device_seconds;transfers+=timing.transfer_seconds;
    for(i=0;i<N;++i){
      float expected=grid[i*CONFIGS+scores[i].configuration];
      CHECK(fabsf(scores[i].sse-expected)<1e-4f*fmaxf(1,expected));
      CHECK(fabsf(scores[i].sse-cpu_scores[i].sse)<1e-4f*fmaxf(1,cpu_scores[i].sse));
    }
#else
    memcpy(scores,cpu_scores,sizeof(scores));
#endif
    memcpy(bad,scores,sizeof(bad));bad[N-1].configuration=UINT64_MAX;
    CHECK(secant_sr_refiner_tell(fit,bad,N)==SECANT_ERROR_INVALID_VALUE);
    CHECK(secant_sr_refiner_stats(fit).rounds==j);
    SEC(secant_sr_refiner_tell(fit,scores,N));
    CHECK(secant_sr_refiner_tell(fit,scores,N)==SECANT_ERROR_INVALID_STATE);
  }
  for(i=0;i<N;++i){
    SRModel model;
    SEC(secant_sr_refiner_model(fit,i,&model));
    printf("model %zu: SSE %.9g permutation %llu slope %.9g offset %.9g\n",i,model.score.sse,(unsigned long long)model.score.configuration,model.nodes[0].leaf[2].value,model.nodes[3].leaf[1].value);
    CHECK(model.score.sse<1e-4f && model.score.sse<=baseline[i].sse); CHECK(model.nodes[0].choices==4 && model.nodes[3].choices==2);
    CHECK(model.nodes[0].leaf[1].value==1 && model.nodes[0].leaf[1].kind==SR_LITERAL);
    CHECK(model.nodes[0].leaf[2].kind==SR_FITTED_COEFFICIENT);
    SEC(secant_sr_resolve(&c,model.nodes,model.num_nodes,banks,BANKS*2,model.score.configuration,resolved,sizeof(resolved),&bytes));
    SEC(secant_sr_program_write(&c,model.nodes,model.num_nodes,code[i],sizeof(code[i]),&bytes));asts[i]=code[i];
  }
  p.asts.items=asts;p.asts.count=N;
  SEC(secant_sr_cpu_score(&c,&p,x,y,ROWS,banks,grid,N*CONFIGS,cpu_scores));
#ifdef SR_HAS_CUDA
  /* Ordinary scoring must restore the base bank after a refinement upload. */
  SEC(secant_sr_cuda_score(gpu,&p,scores,N,&timing));
  for(i=0;i<N;++i) CHECK(fabsf(scores[i].sse-cpu_scores[i].sse)<1e-4f);
  printf("CUDA refinement: pipeline=%.6f device=%.6f transfer=%.6f\n",pipeline,device,transfers);
  for(i=0;i<N;++i) {
    SEC(secant_sr_program_write(&c,nodes[i],5,code[i],sizeof(code[i]),&bytes));
    asts[i]=code[i];
  }
  p.asts.items=asts;
  SEC(secant_sr_cuda_score(gpu,&p,scores,N,&timing));
  for(i=0;i<N;++i) CHECK(fabsf(scores[i].sse-baseline[i].sse)<1e-4f*fmaxf(1,baseline[i].sse));
  SEC(secant_sr_cuda_destroy(gpu));
#endif
  for(i=0;i<N;++i) CHECK(cpu_scores[i].sse<1e-4f);
  CHECK(secant_sr_refiner_stats(fit).configurations==(uint64_t)ROUNDS*N*CONFIGS);
  /* Check replay determinism, then parameter capacity and inheritance. */
  { float a[16],b[16]; SEC(secant_sr_refinement_bank(a,8,2,9,0));SEC(secant_sr_refinement_bank(b,8,2,9,0));CHECK(!memcmp(a,b,sizeof(a)));
    SEC(secant_sr_refinement_bank(b,8,2,9,1));CHECK(memcmp(a,b,sizeof(a))); }
  secant_sr_refiner_destroy(fit);
  check_parameter_contract();check_active_blocks();check_block_rotation();
  puts("native toggle refinement: convergence, replay, fixed literals, partial packs and transactional tell passed");
  return 0;
}
