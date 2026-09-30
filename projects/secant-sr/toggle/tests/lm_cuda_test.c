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
#include "secant_sr_lm.h"
#include "secant_sr_cuda.h"
#include "secant_instructions.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);exit(1);} } while(0)
#define SEC(x) do {SecantResult r=(x);if(r){fprintf(stderr,"line %d: %s: %s\n",__LINE__,#x,secant_result_to_string(r));exit(1);}} while(0)
static void run(int nonlinear) {
  enum {ROWS=37,N=3,BANKS=4,CONFIGS=BANKS*16};
  SRConfig c=secant_sr_config_default();SRLMOptions o=secant_sr_lm_options_default();
  SRLMCuda lm=NULL;SRCudaScorer scorer=NULL;SRCudaOptions so=secant_sr_cuda_options_default();
  SRLMStats init,stats;SRCudaStats timing;SRSearch search=NULL;
  SRNode nodes[N][63]={{{0}}};SRModel models[N],fitted;SecantAstProgramSet p={0};
  uint8_t code[N][1024];const uint8_t *asts[N];size_t i,j,bytes,count=nonlinear?6:5;
  float x[ROWS*2],y[ROWS],banks[BANKS*2]={1,.1f,.5f,-.2f,1.5f,.3f,-2,-1},grid[N*CONFIGS];
  SRScore baseline[N],cpu[N],gpu[N],population[4];
  c.population=4;c.elites=1;c.num_inputs=2;c.num_constants=2;c.num_banks=BANKS;c.toggle_bits=4;
  o.capacity=N;o.bindings=16;o.starts=4;o.iterations=20;
  so.ast_batch=N;so.asts_per_kernel=2;so.kernels_per_module=1;
  for(i=0;i<ROWS;++i){x[i]=(float)i*.1f-1.8f;x[ROWS+i]=cosf((float)i);y[i]=nonlinear?sinf(1.3f*x[i])+.2f:2.75f*x[i]-.125f;}
  for(i=0;i<N;++i){
    nodes[i][0].choices=4;nodes[i][0].low_bit=0;nodes[i][0].high_bit=1;
    nodes[i][0].leaf[0]=(SRLeaf){SR_COLUMN,1,0};nodes[i][0].leaf[1]=(SRLeaf){SR_LITERAL,0,1};
    nodes[i][0].leaf[2]=(SRLeaf){SR_COEFFICIENT,0,0};nodes[i][0].leaf[3]=(SRLeaf){SR_FITTED_COEFFICIENT,5,-.4f};
    nodes[i][1].choices=2;nodes[i][1].low_bit=2;
    nodes[i][1].leaf[0]=(SRLeaf){SR_COLUMN,1,0};nodes[i][1].leaf[1]=(SRLeaf){SR_COLUMN,0,0};
    nodes[i][2].op=SECANT_AST_INSTRUCTION_TYPE_MUL_F32;
    if(nonlinear) nodes[i][3].op=SECANT_AST_INSTRUCTION_TYPE_SIN_F32;
    nodes[i][count-2].choices=2;nodes[i][count-2].low_bit=3;
    nodes[i][count-2].leaf[0]=(SRLeaf){SR_COLUMN,1,0};nodes[i][count-2].leaf[1]=(SRLeaf){SR_COEFFICIENT,1,0};
    nodes[i][count-1].op=SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
    SEC(secant_sr_program_write(&c,nodes[i],count,code[i],sizeof(code[i]),&bytes));asts[i]=code[i];
  }
  p.asts.items=asts;p.asts.count=N;
  SEC(secant_sr_cpu_score(&c,&p,x,y,ROWS,banks,grid,N*CONFIGS,baseline));
  for(i=0;i<N;++i)models[i]=(SRModel){nodes[i],count,baseline[i],0,0};
  SEC(secant_sr_create(&c,banks,BANKS*2,&search));
  for(i=0;i<4;++i){SEC(secant_sr_seed(search,i,nodes[i%N],count));population[i]=baseline[i%N];}
  {SecantAstProgramSet full;SEC(secant_sr_ask(search,&full));SEC(secant_sr_tell(search,population,4,ROWS,1));}
  SEC(secant_sr_cuda_create(&c,x,y,ROWS,banks,&so,&scorer,&timing));
  SEC(secant_sr_lm_cuda_create(&c,&o,x,y,ROWS,&lm,&init));
  SEC(secant_sr_lm_cuda_fit(lm,models,N,banks,BANKS*2,0,60,&p,&stats));
  CHECK(stats.statistics_evaluations==N*16*4*21 && stats.row_evaluations==stats.statistics_evaluations*ROWS);
  SEC(secant_sr_cpu_score(&c,&p,x,y,ROWS,banks,grid,N*CONFIGS,cpu));
  SEC(secant_sr_cuda_score_banks(scorer,&p,banks,1,2,gpu,N,&timing));
  for(i=0;i<N;++i){
    SEC(secant_sr_lm_cuda_model(lm,i,&fitted));
    CHECK(fitted.score.sse<1e-8 && cpu[i].sse<1e-8 && gpu[i].sse<1e-8);
    CHECK(fitted.nodes[0].choices==4 && fitted.nodes[1].choices==2);
    CHECK(fitted.nodes[0].leaf[1].kind==SR_LITERAL && fitted.nodes[0].leaf[1].value==1);
    fitted.score=gpu[i];fitted.score.valid_configurations=1;
    SEC(secant_sr_accept_refined(search,i,&fitted));
  }
  /* A different target with a shared coefficient checks reverse accumulation. */
  SEC(secant_sr_lm_cuda_destroy(lm));lm=NULL;
  for(j=0;j<ROWS;++j)y[j]=2.75f*(x[j]+1);
  memset(nodes,0,sizeof(nodes));nodes[0][0].choices=nodes[0][1].choices=nodes[0][3].choices=1;
  nodes[0][0].leaf[0]=nodes[0][3].leaf[0]=(SRLeaf){SR_COEFFICIENT,0,0};
  nodes[0][1].leaf[0]=(SRLeaf){SR_COLUMN,0,0};
  nodes[0][2].op=SECANT_AST_INSTRUCTION_TYPE_MUL_F32;nodes[0][4].op=SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
  models[0]=(SRModel){nodes[0],5,{1e9f,0,0,1},0,0};
  SEC(secant_sr_lm_cuda_create(&c,&o,x,y,ROWS,&lm,&init));
  SEC(secant_sr_lm_cuda_fit(lm,models,1,banks,BANKS*2,1,60,&p,&stats));
  SEC(secant_sr_lm_cuda_model(lm,0,&fitted));
  CHECK(fitted.score.sse<1e-8 && fabsf(fitted.nodes[0].leaf[0].value-2.75f)<1e-5);
  CHECK(fitted.nodes[0].leaf[0].slot==fitted.nodes[3].leaf[0].slot && fitted.nodes[0].leaf[0].value==fitted.nodes[3].leaf[0].value);
  printf("LM %s: SSE %.9g, setup %.4fs, fit %.4fs, device %.4fs, regs %d local %d\n",nonlinear?"nonlinear":"linear",fitted.score.sse,init.setup_seconds,stats.seconds,stats.device_seconds,init.registers,init.local_bytes);
  SEC(secant_sr_lm_cuda_destroy(lm));lm=NULL;
  /* Maximum postfix stack: 32 leaves followed by 31 additions. */
  c.max_nodes=63;c.max_depth=32;
  memset(nodes,0,sizeof(nodes));
  for(i=0;i<32;++i){nodes[0][i].choices=1;nodes[0][i].leaf[0]=(SRLeaf){i?SR_COLUMN:SR_COEFFICIENT,0,0};}
  for(i=32;i<63;++i)nodes[0][i].op=SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
  for(i=0;i<ROWS;++i)y[i]=2.75f+31*x[i];
  models[0]=(SRModel){nodes[0],63,{1e9f,0,0,1},0,0};
  SEC(secant_sr_lm_cuda_create(&c,&o,x,y,ROWS,&lm,&init));CHECK(init.local_bytes==0);
  SEC(secant_sr_lm_cuda_fit(lm,models,1,banks,BANKS*2,1,60,&p,&stats));
  SEC(secant_sr_lm_cuda_model(lm,0,&fitted));CHECK(fitted.score.sse<1e-7);
  SEC(secant_sr_lm_cuda_destroy(lm));lm=NULL;
  /* Eight distinct coefficients with a rank-one Jacobian exercise damping. */
  memset(nodes,0,sizeof(nodes));
  for(i=0;i<8;++i){nodes[0][i].choices=1;nodes[0][i].leaf[0]=(SRLeaf){SR_FITTED_COEFFICIENT,(unsigned)i,.01f*(float)i};}
  for(i=8;i<15;++i)nodes[0][i].op=SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
  for(i=0;i<ROWS;++i)y[i]=1.25f;
  models[0]=(SRModel){nodes[0],15,{1e9f,0,0,1},0,0};
  SEC(secant_sr_lm_cuda_create(&c,&o,x,y,ROWS,&lm,&init));
  SEC(secant_sr_lm_cuda_fit(lm,models,1,banks,BANKS*2,2,60,&p,&stats));
  SEC(secant_sr_lm_cuda_model(lm,0,&fitted));CHECK(fitted.score.sse<1e-8);
  SEC(secant_sr_lm_cuda_destroy(lm));SEC(secant_sr_cuda_destroy(scorer));secant_sr_destroy(search);
}
int main(void){run(0);run(1);puts("GPU LM: convergence, mixed toggles, shared derivatives, native-SASS rescore and GP acceptance passed");return 0;}
