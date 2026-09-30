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
#include "../src/lm_internal.h"
#include <stdio.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"line %d: %s\n",__LINE__,#x);exit(1);} } while(0)
#define SEC(x) CHECK((x)==SECANT_SUCCESS)
int main(void) {
  SRConfig c=secant_sr_config_default();
  SRLMOptions o=secant_sr_lm_options_default();SRLMBatch b;
  SRNode nodes[5]={{0}},wide[19]={{0}};SRModel m={0};SecantAstProgramSet programs;
  float banks[]={1,2,3,4};size_t i,j;
  c.population=4;c.elites=1;c.num_inputs=2;c.num_constants=2;c.num_banks=2;c.toggle_bits=4;c.max_depth=16;
  o.capacity=2;o.bindings=16;o.starts=4;
  nodes[0].choices=4;nodes[0].low_bit=0;nodes[0].high_bit=2;
  nodes[0].leaf[0]=(SRLeaf){SR_COLUMN,0,0};nodes[0].leaf[1]=(SRLeaf){SR_COEFFICIENT,0,0};
  nodes[0].leaf[2]=(SRLeaf){SR_FITTED_COEFFICIENT,0,3};nodes[0].leaf[3]=(SRLeaf){SR_LITERAL,0,7};
  nodes[1].choices=1;nodes[1].leaf[0]=(SRLeaf){SR_COEFFICIENT,0,0};nodes[2].op=SECANT_AST_INSTRUCTION_TYPE_MUL_F32;
  nodes[3].choices=2;nodes[3].low_bit=3;
  nodes[3].leaf[0]=(SRLeaf){SR_FITTED_COEFFICIENT,0,3};nodes[3].leaf[1]=(SRLeaf){SR_FITTED_COEFFICIENT,0,4};
  nodes[4].op=SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
  m.nodes=nodes;m.num_nodes=5;m.score=(SRScore){10,0,16+3,32};
  SEC(sr_lm_batch_create(&c,&o,&b));
  SEC(sr_lm_batch_prepare(&b,&m,1,banks,4,0));
  CHECK(b.program_count==8 && b.state_count==32);
  CHECK(b.programs[0].permutation==3);
  for(i=0;i<8;++i) {
    CHECK((b.programs[i].permutation&2)==2); /* unused bit stays fixed */
    for(j=0;j<i;++j) CHECK(b.programs[i].permutation!=b.programs[j].permutation);
    for(j=0;j<b.programs[i].parameters;++j) CHECK(b.states[i*4].trial[j]==b.programs[i].center[j]);
  }
  CHECK(b.models[0].nodes[0].leaf[1].slot==b.models[0].nodes[1].leaf[0].slot);
  CHECK(b.models[0].nodes[0].leaf[2].slot==b.models[0].nodes[3].leaf[0].slot);
  CHECK(b.models[0].nodes[0].leaf[1].slot!=b.models[0].nodes[0].leaf[2].slot); /* equal centers, independent */
  CHECK(b.models[0].nodes[3].leaf[0].slot!=b.models[0].nodes[3].leaf[1].slot);
  CHECK(b.models[0].nodes[0].leaf[3].kind==SR_LITERAL && b.models[0].nodes[0].leaf[3].value==7);
  {
    SRLMProgram saved[8];SRLMState states[32];
    memcpy(saved,b.programs,sizeof(saved));memcpy(states,b.states,sizeof(states));
    SEC(sr_lm_batch_prepare(&b,&m,1,banks,4,0));
    CHECK(!memcmp(saved,b.programs,sizeof(saved)) && !memcmp(states,b.states,sizeof(states)));
  }
  b.states[0].loss=0;
  for(j=0;j<b.programs[0].parameters;++j)b.states[0].best[j]=10+(float)j;
  SEC(sr_lm_batch_finish(&b,&programs));
  CHECK(b.models[0].nodes[0].leaf[1].value==b.models[0].nodes[1].leaf[0].value);
  CHECK(b.models[0].nodes[0].leaf[2].value==b.models[0].nodes[3].leaf[0].value);
  {
    SRModel alias=m;alias.nodes=b.models[0].nodes;
    CHECK(sr_lm_batch_prepare(&b,&alias,1,banks,4,0)==SECANT_ERROR_INVALID_VALUE);
    CHECK(sr_lm_batch_finish(&b,&programs)==SECANT_ERROR_INVALID_STATE);
  }
  /* Wider models retain all identities and explicitly rotate active blocks. */
  for(i=0;i<10;++i){wide[i].choices=1;wide[i].leaf[0]=(SRLeaf){SR_FITTED_COEFFICIENT,(unsigned)i,(float)i};}
  for(i=10;i<19;++i)wide[i].op=SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
  m.nodes=wide;m.num_nodes=19;
  for(i=0;i<2;++i){
    SEC(sr_lm_batch_prepare(&b,&m,1,banks,4,i));
    CHECK(b.stats.blocked_bindings==1 && b.programs[0].parameters==(i?2:8));
    CHECK(b.programs[0].parameter_ids[0]==i*8);
  }
  sr_lm_batch_free(&b);
  puts("LM models: toggle coverage, sharing, literal preservation, deterministic starts, alias rejection and parameter blocks passed");
  return 0;
}
