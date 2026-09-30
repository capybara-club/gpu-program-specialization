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
#include "secant.h"
#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static double score(const uint8_t *ast,size_t inputs,size_t rows,const float *x,const float *y){
  const uint8_t *programs[]={ast};float out=0;SecantCpuSSERun r=secant_cpu_sse_run_init();
  r.programs.asts.items=programs;r.programs.asts.count=1;r.num_inputs=inputs;r.num_targets=1;r.num_rows=rows;
  r.input=(SecantConstHostMatrixF32){x,inputs*rows,rows};r.targets=(SecantConstHostMatrixF32){y,rows,rows};
  r.output=(SecantHostMatrixF32){&out,1,1};
  if(secant_cpu_run_sse(&r)!=SECANT_SUCCESS)return INFINITY;
  return out/(double)rows;
}
int main(int argc,char **argv){
  size_t inputs,rows,test,n,i;double variance,test_variance,a,b;float *x,*y,*vx,*vy;
  uint8_t ast[SECANT_AST_MAX_PROGRAM_BYTES]={0};
  if(argc!=3||strlen(argv[2])%2)return 2;
  n=strlen(argv[2])/2;if(!n||n>=sizeof(ast))return 2;
  for(i=0;i<n;++i){unsigned byte;if(sscanf(argv[2]+2*i,"%2x",&byte)!=1)return 2;ast[i]=(uint8_t)byte;}
  if(!secant_sr_dataset_binary_info(argv[1],&inputs,&rows,&test))return 2;
  if(inputs>128||!rows||!test||rows>SIZE_MAX/4/inputs||test>SIZE_MAX/4/inputs)return 2;
  x=malloc(inputs*rows*4);y=malloc(rows*4);vx=malloc(inputs*test*4);vy=malloc(test*4);
  if(!x||!y||!vx||!vy)return 2;
  if(!secant_sr_dataset_binary_load(argv[1],inputs,x,y,rows,vx,vy,test,&variance,&test_variance))return 2;
  a=score(ast,inputs,rows,x,y);b=score(ast,inputs,test,vx,vy);
  if(!isfinite(a)||!isfinite(b))return 3;
  printf("{\"train_mse\":%.12g,\"validation_mse\":%.12g}\n",a,b);
  free(x);free(y);free(vx);free(vy);return 0;
}
