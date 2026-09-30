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
#include "lm_layout.h"

// One CTA owns one (AST, toggle binding, coefficient start). All lanes execute
// the same operators over different rows; bindings remain fixed during LM.
// Program interpretation is intentional in this first quality-oriented backend.
static __device__ bool evaluate(const SRLMProgram &p, const float *theta,
    const float *input, unsigned long long rows, unsigned long long row,
    float &prediction, float *gradient) {
  float value[SR_LM_NODES], adjoint[SR_LM_NODES];
  for(unsigned i=0;i<p.count;++i){
    const SRLMInstruction n=p.code[i];float a=0,b=0,v=0;
    if(n.op>=0x80){a=value[n.a];if(n.op==0x85||n.op==0x86||n.op==0x87||n.op==0x88||n.op==0x8d||n.op==0x8e)b=value[n.b];}
    switch(n.op){
    case 0:v=n.value;break;
    case 1:v=input[(unsigned long long)n.a*rows+row];break;
    case 2:v=theta[n.a];break;
    case 0x85:v=a+b;break;case 0x86:v=a-b;break;case 0x87:v=a*b;break;case 0x88:v=a/b;break;
    case 0x89:v=-a;break;case 0x8a:v=sqrtf(a);break;case 0x8c:v=fabsf(a);break;
    case 0x8d:v=fminf(a,b);break;case 0x8e:v=fmaxf(a,b);break;
    case 0x90:v=sinf(a);break;case 0x91:v=cosf(a);break;case 0x95:v=tanhf(a);break;
    case 0xba:v=expf(a);break;case 0xbb:v=logf(a);break;default:return false;
    }
    if(!isfinite(v))return false;
    value[i]=v;adjoint[i]=0;
  }
  for(unsigned j=0;j<SR_LM_PARAMETERS;++j)gradient[j]=0;
  prediction=value[p.count-1];adjoint[p.count-1]=1;
  for(int i=(int)p.count-1;i>=0;--i){
    const SRLMInstruction n=p.code[i];const float g=adjoint[i];float a=0,b=0,da=0,db=0;
    if(g==0)continue;
    if(n.op==2){gradient[n.a]+=g;continue;}
    if(n.op<0x80)continue;
    a=value[n.a];
    if(n.op==0x85||n.op==0x86||n.op==0x87||n.op==0x88||n.op==0x8d||n.op==0x8e)b=value[n.b];
    switch(n.op){
    case 0x85:da=g;db=g;break;case 0x86:da=g;db=-g;break;
    case 0x87:da=g*b;db=g*a;break;case 0x88:da=g/b;db=-(g*a/b)/b;break;
    case 0x89:da=-g;break;case 0x8a:da=g*.5f/value[i];break;
    case 0x8c:da=a>0?g:a<0?-g:0;break;
    case 0x8d:da=a<=b?g:0;db=a<=b?0:g;break;
    case 0x8e:da=a>=b?g:0;db=a>=b?0:g;break;
    case 0x90:da=g*cosf(a);break;case 0x91:da=-g*sinf(a);break;
    case 0x95:da=g*(1-value[i]*value[i]);break;
    case 0xba:da=g*value[i];break;case 0xbb:da=g/a;break;default:return false;
    }
    if(!isfinite(da)||!isfinite(db))return false;
    adjoint[n.a]+=da;
    if(n.op==0x85||n.op==0x86||n.op==0x87||n.op==0x88||n.op==0x8d||n.op==0x8e)adjoint[n.b]+=db;
  }
  for(unsigned j=0;j<p.parameters;++j)if(!isfinite(gradient[j]))return false;
  return true;
}

extern "C" __global__ void lm_statistics(const SRLMProgram *programs,
    const SRLMState *states, unsigned starts, const float *input,
    const float *target, unsigned long long rows, double *output) {
  const unsigned state=blockIdx.x,t=threadIdx.x;
  const SRLMProgram &p=programs[state/starts];
  const unsigned n=p.parameters, size=1+n+n*(n+1)/2;
  double sums[SR_LM_STATISTICS];bool valid=true;
  for(unsigned j=0;j<size;++j)sums[j]=0;
  for(unsigned long long row=t;row<rows;row+=blockDim.x){
    float v,g[SR_LM_PARAMETERS];
    if(!evaluate(p,states[state].trial,input,rows,row,v,g)){valid=false;continue;}
    const double residual=(double)v-(double)target[row];
    sums[0]+=residual*residual;
    for(unsigned j=0;j<n;++j)sums[1+j]+=(double)g[j]*residual;
    unsigned at=1+n;
    for(unsigned j=0;j<n;++j)for(unsigned k=0;k<=j;++k)sums[at++]+=(double)g[j]*g[k];
  }
  extern __shared__ double shared[];
  if(!valid)sums[0]=__longlong_as_double(0x7ff0000000000000ull);
  for(unsigned j=0;j<size;++j){
    double v=sums[j];
    for(unsigned offset=16;offset;offset>>=1)v+=__shfl_down_sync(0xffffffff,v,offset);
    if(!(t&31))shared[(t>>5)*size+j]=v;
  }
  __syncthreads();
  if(!t)for(unsigned j=0;j<size;++j){
    double v=0;for(unsigned w=0;w<blockDim.x/32;++w)v+=shared[w*size+j];
    output[(unsigned long long)state*SR_LM_STATISTICS+j]=v;
  }
}

extern "C" __global__ void lm_step(const SRLMProgram *programs, SRLMState *states,
    unsigned starts, unsigned count, const double *trial_stats, double *best_stats,
    unsigned iteration, unsigned propose) {
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  SRLMState &s=states[i];const unsigned n=programs[i/starts].parameters;
  const unsigned size=1+n+n*(n+1)/2;
  const double *trial=trial_stats+(unsigned long long)i*SR_LM_STATISTICS;
  double *best=best_stats+(unsigned long long)i*SR_LM_STATISTICS;
  bool valid=true;for(unsigned j=0;j<size;++j)if(!isfinite(trial[j]))valid=false;
  if(valid&&trial[0]<s.loss){
    for(unsigned j=0;j<size;++j)best[j]=trial[j];
    for(unsigned j=0;j<n;++j)s.best[j]=s.trial[j];
    s.loss=trial[0];
    if(iteration){s.accepted++;s.damping=fmax(1e-12,s.damping*.3);}
  }else if(iteration)s.damping=fmin(1e12,s.damping*10);
  if(!valid)s.invalid++;
  if(!propose||!isfinite(s.loss)||!n)return;
  double matrix[SR_LM_PARAMETERS*SR_LM_PARAMETERS],delta[SR_LM_PARAMETERS],max_diag=0;
  unsigned at=1+n;
  for(unsigned j=0;j<n;++j)for(unsigned k=0;k<=j;++k){
    double v=best[at++];matrix[j*n+k]=matrix[k*n+j]=v;
    if(j==k)max_diag=fmax(max_diag,v);
  }
  for(unsigned j=0;j<n;++j){
    matrix[j*n+j]+=s.damping*fmax(matrix[j*n+j],fmax(max_diag*1e-12,1e-30));
    delta[j]=-best[1+j];
  }
  valid=true;
  for(unsigned j=0;j<n&&valid;++j)for(unsigned k=0;k<=j;++k){
    double v=matrix[j*n+k];for(unsigned l=0;l<k;++l)v-=matrix[j*n+l]*matrix[k*n+l];
    if(j==k){if(!(v>0)||!isfinite(v)){valid=false;break;}matrix[j*n+k]=sqrt(v);}
    else matrix[j*n+k]=v/matrix[k*n+k];
  }
  if(valid){
    for(unsigned j=0;j<n;++j){for(unsigned k=0;k<j;++k)delta[j]-=matrix[j*n+k]*delta[k];delta[j]/=matrix[j*n+j];}
    for(int j=(int)n-1;j>=0;--j){for(unsigned k=j+1;k<n;++k)delta[j]-=matrix[k*n+j]*delta[k];delta[j]/=matrix[j*n+j];}
    for(unsigned j=0;j<n;++j)if(!isfinite((float)(s.best[j]+delta[j])))valid=false;
  }
  for(unsigned j=0;j<n;++j)s.trial[j]=valid?(float)(s.best[j]+delta[j]):s.best[j];
  if(!valid)s.damping=fmin(1e12,s.damping*10);
}
