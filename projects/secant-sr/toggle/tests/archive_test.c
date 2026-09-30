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
#include "secant_sr_archive.h"
#include "secant_instructions.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define CHECK(v) do{if(!(v)){fprintf(stderr,"line %d: %s\n",__LINE__,#v);exit(1);}}while(0)
#define SEC(v) CHECK((v)==SECANT_SUCCESS)
int main(void){
  SRConfig c=secant_sr_config_default();SRArchive a=NULL;SRNode nodes[3]={{0}};SRModel m={0},out;
  float banks[4]={1,2,3,4};uint8_t before[256],after[256];size_t n,z;
  c.population=4;c.elites=1;c.num_inputs=2;c.num_constants=2;c.num_banks=2;c.toggle_bits=1;
  nodes[0].choices=1;nodes[0].leaf[0]=(SRLeaf){SR_COEFFICIENT,0,0};
  nodes[1].choices=2;nodes[1].leaf[0]=(SRLeaf){SR_COLUMN,0,0};nodes[1].leaf[1]=(SRLeaf){SR_COLUMN,1,0};
  nodes[2].op=SECANT_AST_INSTRUCTION_TYPE_MUL_F32;
  m.nodes=nodes;m.num_nodes=3;m.score=(SRScore){10,0,2,4};m.generation=5;
  SEC(secant_sr_archive_create(&c,2,&a));SEC(secant_sr_archive_offer(a,&m,banks,4));
  SEC(secant_sr_archive_model(a,0,&out));CHECK(out.generation==5&&out.score.configuration==0);
  SEC(secant_sr_resolve(&c,m.nodes,3,banks,4,2,before,sizeof(before),&n));
  SEC(secant_sr_resolve(&c,out.nodes,3,banks,4,0,after,sizeof(after),&z));CHECK(n==z&&!memcmp(before,after,n));
  CHECK(out.nodes[0].leaf[0].value==3&&out.nodes[1].choices==1);
  /* A different coefficient value/name does not buy another archive slot. */
  nodes[0].leaf[0].slot=1;m.score.sse=9;m.score.configuration=0;
  SEC(secant_sr_archive_offer(a,&m,banks,4));CHECK(secant_sr_archive_count(a)==1);
  SEC(secant_sr_archive_model(a,0,&out));CHECK(out.nodes[0].leaf[0].value==2);
  m.score.sse=8;m.score.configuration=1;SEC(secant_sr_archive_offer(a,&m,banks,4));CHECK(secant_sr_archive_count(a)==2);
  nodes[2].op=SECANT_AST_INSTRUCTION_TYPE_ADD_F32;m.score.sse=7;
  SEC(secant_sr_archive_offer(a,&m,banks,4));SEC(secant_sr_archive_model(a,0,&out));CHECK(out.score.sse==7);
  SEC(secant_sr_archive_model(a,1,&out));CHECK(out.score.sse==8);
  nodes[0].leaf[0]=(SRLeaf){SR_LITERAL,0,3};m.score.sse=0;
  SEC(secant_sr_archive_offer(a,&m,banks,4));CHECK(secant_sr_archive_count(a)==2);
  secant_sr_archive_destroy(a);
  /* Parameter ties distinguish structures even when current centers coincide. */
  SEC(secant_sr_archive_create(&c,4,&a));nodes[0].leaf[0]=(SRLeaf){SR_FITTED_COEFFICIENT,100,1};
  nodes[1].choices=1;nodes[1].leaf[0]=(SRLeaf){SR_FITTED_COEFFICIENT,100,1};m.score.configuration=0;m.score.sse=2;
  SEC(secant_sr_archive_offer(a,&m,banks,4));nodes[1].leaf[0].slot=101;
  SEC(secant_sr_archive_offer(a,&m,banks,4));CHECK(secant_sr_archive_count(a)==2);
  SEC(secant_sr_archive_model(a,0,&out));CHECK(out.nodes[0].leaf[0].slot==out.nodes[1].leaf[0].slot);
  SEC(secant_sr_archive_model(a,1,&out));CHECK(out.nodes[0].leaf[0].slot!=out.nodes[1].leaf[0].slot);
  CHECK(secant_sr_archive_offer(a,&m,banks,3)==SECANT_ERROR_INVALID_VALUE);
  CHECK(secant_sr_archive_model(a,2,&out)==SECANT_ERROR_INVALID_VALUE);
  secant_sr_archive_destroy(a);puts("archive ownership, replay, binding diversity, capacity and parameter sharing passed");return 0;
}
