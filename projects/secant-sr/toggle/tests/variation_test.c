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
#include "../src/internal.h"
#include <stdio.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); exit(1); } } while (0)
#define SEC(x) CHECK((x)==SECANT_SUCCESS)
static SRNode selector(unsigned choices, unsigned low, unsigned high) {
  SRNode n={0}; unsigned i;
  n.choices=(uint8_t)choices;n.low_bit=(uint8_t)low;n.high_bit=(uint8_t)high;
  for(i=0;i<choices;++i) n.leaf[i]=(SRLeaf){SR_LITERAL,0,(float)(i+3)};
  return n;
}
static void alignment(void) {
  unsigned bits,old,new;
  /* Test both sufficient free bits and exhaustion. Recipient and donor may
   * select contradictory values for the same bit before crossover. */
  for(bits=2;bits<=3;++bits) for(old=0;old<(1u<<bits);++old) for(new=0;new<(1u<<bits);++new) {
    SRNode tree[5]={{0}},before[5];unsigned reused;
    tree[0]=selector(2,0,0);tree[1]=selector(4,0,1);tree[2]=selector(2,0,0);
    tree[3].op=SECANT_AST_INSTRUCTION_TYPE_MUL_F32;tree[4].op=SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
    memcpy(before,tree,sizeof(tree));
    reused=sr_align_subtree(tree,5,1,3,bits,old,new);
    CHECK(reused==(bits==2 ? 1u : 0u));
    CHECK(!memcmp(tree,before,sizeof(SRNode)));
    CHECK(tree[1].low_bit!=tree[1].high_bit);
    CHECK(tree[1].low_bit==tree[2].low_bit); /* Deliberate donor coupling remains. */
    if(bits==3) CHECK(tree[1].low_bit!=0 && tree[1].high_bit!=0);
    CHECK(tree[1].leaf[sr_selector_choice(tree+1,new)].value==before[1].leaf[sr_selector_choice(before+1,old)].value);
    CHECK(tree[2].leaf[sr_selector_choice(tree+2,new)].value==before[2].leaf[sr_selector_choice(before+2,old)].value);
    { unsigned p,old_set=0,new_set=0;
      for(p=0;p<(1u<<bits);++p) {
        unsigned a=(unsigned)before[1].leaf[sr_selector_choice(before+1,p)].value-3;
        unsigned b=(unsigned)before[2].leaf[sr_selector_choice(before+2,p)].value-3;
        unsigned x=(unsigned)tree[1].leaf[sr_selector_choice(tree+1,p)].value-3;
        unsigned y=(unsigned)tree[2].leaf[sr_selector_choice(tree+2,p)].value-3;
        old_set|=1u<<(a+4*b);new_set|=1u<<(x+4*y);
      }
      CHECK(old_set==new_set); /* No donor combinations silently removed. */
    }
  }
}
static void preserve_winner(void) {
  enum { P=256, ROWS=9, CONFIGS=32 };
  SRConfig c=secant_sr_config_default();
  SRNode nodes[3]={{0}};
  float banks[4]={.4f,2.f,-.9f,3.f},x[ROWS*2],y[ROWS],*grid;
  SRScore scores[P];size_t i,mode;
  c.population=P;c.elites=1;c.num_inputs=2;c.num_constants=2;c.num_banks=2;c.toggle_bits=4;
  c.max_nodes=7;c.initial_depth=2;c.max_depth=3;
  nodes[0]=selector(4,0,1);nodes[1]=selector(2,0,0);
  nodes[0].leaf[0]=(SRLeaf){SR_COLUMN,0,0};nodes[0].leaf[1]=(SRLeaf){SR_COEFFICIENT,0,0};
  nodes[0].leaf[2]=(SRLeaf){SR_COLUMN,1,0};nodes[1].leaf[0]=(SRLeaf){SR_COLUMN,1,0};
  nodes[1].leaf[1]=(SRLeaf){SR_COEFFICIENT,1,0};nodes[2].op=SECANT_AST_INSTRUCTION_TYPE_ADD_F32;
  for(i=0;i<ROWS;++i){x[i]=(float)i*.2f-1;x[i+ROWS]=cosf((float)i);y[i]=x[i]+x[i+ROWS]+.03f;}
  grid=malloc(P*CONFIGS*sizeof(float));CHECK(grid);
  for(mode=0;mode<2;++mode) {
    SRSearch s;SecantAstProgramSet p;uint64_t anchor;float expected;
    unsigned changed=0;SRProgress stats;
    c.crossover_probability=mode==0?0:1;c.mutation_probability=mode==0?1:0;
    c.toggle_mutation_probability=mode==0?1:0;c.leaf_mix_probability=mode==0?0:1;
    SEC(secant_sr_create(&c,banks,4,&s));
    for(i=0;i<P;++i) SEC(secant_sr_seed(s,i,nodes,3));
    SEC(secant_sr_ask(s,&p));SEC(secant_sr_cpu_score(&c,&p,x,y,ROWS,banks,grid,P*CONFIGS,scores));
    anchor=scores[0].configuration;expected=grid[anchor];
    SEC(secant_sr_tell(s,scores,P,ROWS,1));SEC(secant_sr_advance(s));
    for(i=1;i<P;++i) {
      SRCandidate *child=s->population[s->current]+i;
      CHECK(child->count==3);SEC(sr_validate(&c,child->nodes,3,NULL,NULL));
      changed+=memcmp(nodes,child->nodes,sizeof(nodes))!=0;
    }
    CHECK(changed>0);
    SEC(secant_sr_ask(s,&p));SEC(secant_sr_cpu_score(&c,&p,x,y,ROWS,banks,grid,P*CONFIGS,scores));
    for(i=0;i<P;++i) CHECK(grid[i*CONFIGS+anchor]==expected);
    stats=secant_sr_progress(s);
    CHECK((mode==0?stats.toggle_mutations:stats.leaf_mixes)==P-1);
    secant_sr_destroy(s);
  }
  free(grid);
}
static void powers(void) {
  enum { P=64, ROWS=7, CONFIGS=8 };
  SRConfig c=secant_sr_config_default();SRSearch s;SRNode node=selector(4,0,1);
  float banks[2]={.3f,.9f},x[ROWS],y[ROWS]={0},grid[P*CONFIGS];SRScore scores[P];SecantAstProgramSet p;
  size_t i,j,q;
  c.population=P;c.elites=1;c.num_inputs=1;c.num_constants=1;c.num_banks=2;c.toggle_bits=2;c.max_nodes=7;
  c.crossover_probability=0;c.mutation_probability=1;c.power_mutation_probability=1;
  node.leaf[0]=(SRLeaf){SR_COLUMN,0,0};node.leaf[1]=(SRLeaf){SR_COEFFICIENT,0,0};
  for(i=0;i<ROWS;++i)x[i]=(float)i*.2f-1;
  SEC(secant_sr_create(&c,banks,2,&s));for(i=0;i<P;++i)SEC(secant_sr_seed(s,i,&node,1));
  SEC(secant_sr_ask(s,&p));SEC(secant_sr_cpu_score(&c,&p,x,y,ROWS,banks,grid,P*CONFIGS,scores));
  SEC(secant_sr_tell(s,scores,P,ROWS,1));SEC(secant_sr_advance(s));
  SEC(secant_sr_ask(s,&p));SEC(secant_sr_cpu_score(&c,&p,x,y,ROWS,banks,grid,P*CONFIGS,scores));
  for(i=1;i<P;++i) {
    SRCandidate *candidate=s->population[s->current]+i;
    CHECK(candidate->count==3 || candidate->count==5);
    for(q=0;q<CONFIGS;++q) {
      float sum=0;
      for(j=0;j<ROWS;++j) {
        unsigned choice=(unsigned)q&3;float v=choice==0?x[j]:choice==1?banks[q/4]:node.leaf[choice].value;
        float power=v*v;if(candidate->count==5)power*=v;sum+=power*power;
      }
      CHECK(fabsf(grid[i*CONFIGS+q]-sum)<1e-5f*fmaxf(1,sum));
    }
  }
  CHECK(secant_sr_progress(s).power_mutations==P-1);
  secant_sr_destroy(s);
}
int main(void) {
  alignment();preserve_winner();powers();
  puts("toggle alignment, bit exhaustion, donor coverage and retained-configuration scoring passed");
  return 0;
}
