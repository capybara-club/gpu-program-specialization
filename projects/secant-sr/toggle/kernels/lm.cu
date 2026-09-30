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
#ifndef SR_LM_STACK
#define SR_LM_STACK 32
#endif

// Fixed-position postfix stacks: every index is a compile-time constant after
// unrolling. A push/pop shifts registers rather than dynamically indexing a
// private array. A 63-node binary tree needs at most 32 stack entries.
static __device__ __forceinline__ bool directional(const SRLMProgram &p,
    const float *theta, const float *input, unsigned long long rows,
    unsigned long long row, unsigned parameter, float &prediction, float &gradient) {
  float value[SR_LM_STACK], derivative[SR_LM_STACK];
  #pragma unroll
  for(unsigned j=0;j<SR_LM_STACK;++j){value[j]=0;derivative[j]=0;}
  for(unsigned i=0;i<p.count;++i){
    const SRLMInstruction n=p.code[i];
    const bool binary=n.op==0x85||n.op==0x86||n.op==0x87||n.op==0x88||n.op==0x8d||n.op==0x8e;
    float a=binary?value[1]:value[0],b=value[0];
    float da=binary?derivative[1]:derivative[0],db=derivative[0],v=0,d=0;
    switch(n.op){
    case 0:v=n.value;break;
    case 1:v=input[(unsigned long long)n.a*rows+row];break;
    case 2:v=theta[n.a];d=n.a==parameter;break;
    case 0x85:v=a+b;d=da+db;break;
    case 0x86:v=a-b;d=da-db;break;
    case 0x87:v=a*b;d=da*b+a*db;break;
    case 0x88:v=a/b;d=(da-v*db)/b;break;
    case 0x89:v=-a;d=-da;break;
    case 0x8a:v=sqrtf(a);d=da==0?0:da*.5f/v;break;
    case 0x8c:v=fabsf(a);d=a>0?da:a<0?-da:0;break;
    case 0x8d:v=fminf(a,b);d=a<=b?da:db;break;
    case 0x8e:v=fmaxf(a,b);d=a>=b?da:db;break;
    case 0x90:v=sinf(a);d=da*cosf(a);break;
    case 0x91:v=cosf(a);d=-da*sinf(a);break;
    case 0x95:v=tanhf(a);d=da*(1-v*v);break;
    case 0xba:v=expf(a);d=da*v;break;
    case 0xbb:v=logf(a);d=da==0?0:da/a;break;
    default:return false;
    }
    if(!isfinite(v)||!isfinite(d))return false;
    if(n.op<0x80){
      #pragma unroll
      for(int j=SR_LM_STACK-1;j>0;--j){value[j]=value[j-1];derivative[j]=derivative[j-1];}
    }else if(binary){
      #pragma unroll
      for(unsigned j=1;j<SR_LM_STACK-1;++j){value[j]=value[j+1];derivative[j]=derivative[j+1];}
    }
    value[0]=v;derivative[0]=d;
  }
  prediction=value[0];gradient=derivative[0];return true;
}
template<unsigned P> static __device__ __forceinline__ bool evaluate_n(const SRLMProgram &p,
    const float *theta, const float *input, unsigned long long rows,
    unsigned long long row, float &prediction, float *gradient) {
  // Sequential directional sweeps bound live registers independently of the
  // parameter count; each sweep reuses the same scalarized postfix stack.
  #pragma unroll
  for(unsigned j=0;j<P;++j){
    gradient[j]=0;
    if(j<p.parameters || j==0)
      if(!directional(p,theta,input,rows,row,j,prediction,gradient[j]))return false;
  }
  return true;
}
// A small parameter set uses one lane per row with exactly the required
// accumulator count. This avoids forcing every fit through the eight-parameter
// register footprint. Programs are grouped by 1/2/4/8 parameters on the host.
template<unsigned P> static __device__ __forceinline__ void statistics_single(
    const SRLMProgram *programs, const SRLMState *states, unsigned starts,
    const float *input, const float *target, unsigned long long rows,
    double *output, const unsigned *indices, unsigned offset) {
  const unsigned state=indices[offset+blockIdx.x],t=threadIdx.x;
  const SRLMProgram &p=programs[state/starts];
  constexpr unsigned size=1+P+P*(P+1)/2;
  double sums[size];bool valid=true;
  #pragma unroll
  for(unsigned j=0;j<size;++j)sums[j]=0;
  for(unsigned long long row=t;row<rows;row+=blockDim.x){
    float prediction,g[P];
    if(!evaluate_n<P>(p,states[state].trial,input,rows,row,prediction,g)){valid=false;continue;}
    const double residual=(double)prediction-target[row];
    sums[0]+=residual*residual;
    #pragma unroll
    for(unsigned j=0;j<P;++j){
      sums[1+j]+=(double)g[j]*residual;
      #pragma unroll
      for(unsigned k=0;k<=j;++k)sums[1+P+j*(j+1)/2+k]+=(double)g[j]*g[k];
    }
  }
  extern __shared__ double shared[];
  if(!valid)sums[0]=__longlong_as_double(0x7ff0000000000000ull);
  #pragma unroll
  for(unsigned j=0;j<size;++j){
    double v=sums[j];
    for(unsigned shift=16;shift;shift>>=1)v+=__shfl_down_sync(0xffffffff,v,shift);
    if(!(t&31))shared[(t>>5)*SR_LM_STATISTICS+j]=v;
  }
  __syncthreads();
  if(!t){
    #pragma unroll
    for(unsigned j=0;j<SR_LM_STATISTICS;++j)output[(unsigned long long)state*SR_LM_STATISTICS+j]=0;
    #pragma unroll
    for(unsigned j=0;j<size;++j){
      const unsigned index=j<=P?j:j+SR_LM_PARAMETERS-P;
      double v=0;for(unsigned w=0;w<blockDim.x/32;++w)v+=shared[w*SR_LM_STATISTICS+j];
      output[(unsigned long long)state*SR_LM_STATISTICS+index]=v;
    }
  }
}
// Two lanes share a row. Each computes half the directional derivatives and
// owns alternating statistics. This bounds per-thread accumulators at 23
// doubles while preserving double-precision SSE/Jtr/JtJ accumulation.
static __device__ __forceinline__ void statistics_pair(const SRLMProgram *programs,
    const SRLMState *states, unsigned starts, const float *input,
    const float *target, unsigned long long rows, double *output, const unsigned *indices, unsigned offset) {
  const unsigned state=indices[offset+blockIdx.x],t=threadIdx.x,lane=t&1;
  const SRLMProgram &p=programs[state/starts];
  double sums[23];bool valid=true;
  #pragma unroll
  for(unsigned j=0;j<23;++j)sums[j]=0;
  // Padded, uniform loop: every lane participates in every shuffle, including
  // lanes whose final row is beyond the input. No masked full-warp collectives.
  for(unsigned long long base=0;base<rows;base+=blockDim.x/2){
    const unsigned long long row=base+t/2;
    float prediction=0,g[SR_LM_PARAMETERS];bool row_valid=true;
    #pragma unroll
    for(unsigned j=0;j<SR_LM_PARAMETERS;++j){
      g[j]=0;
      if(row<rows && (j&1)==lane && (j<p.parameters || j==0))
        if(!directional(p,states[state].trial,input,rows,row,j,prediction,g[j]))row_valid=false;
    }
    const unsigned other=__shfl_sync(0xffffffff,(unsigned)row_valid,lane^1,2);
    row_valid=row_valid && other;
    prediction=__shfl_sync(0xffffffff,prediction,0,2);
    #pragma unroll
    for(unsigned j=0;j<SR_LM_PARAMETERS;++j)g[j]=__shfl_sync(0xffffffff,g[j],j&1,2);
    if(!row_valid){valid=false;continue;}
    if(row>=rows)continue;
    const double residual=(double)prediction-(double)target[row];
    #pragma unroll
    for(unsigned at=0;at<23;++at){
      const unsigned index=2*at+lane;
      double product=index==0?residual*residual:0;
      #pragma unroll
      for(unsigned j=0;j<SR_LM_PARAMETERS;++j){
        if(index==1+j)product=(double)g[j]*residual;
        #pragma unroll
        for(unsigned k=0;k<=j;++k)
          if(index==1+SR_LM_PARAMETERS+j*(j+1)/2+k)product=(double)g[j]*g[k];
      }
      sums[at]+=product;
    }
  }
  extern __shared__ double shared[];
  if(!valid && !lane)sums[0]=__longlong_as_double(0x7ff0000000000000ull);
  #pragma unroll
  for(unsigned j=0;j<23;++j){
    double v=sums[j];
    for(unsigned offset=16;offset>=2;offset>>=1)v+=__shfl_down_sync(0xffffffff,v,offset);
    if((t&31)<2 && 2*j+lane<SR_LM_STATISTICS)shared[(t>>5)*SR_LM_STATISTICS+2*j+lane]=v;
  }
  __syncthreads();
  if(!t){
    #pragma unroll
    for(unsigned j=0;j<SR_LM_STATISTICS;++j){
      double v=0;for(unsigned w=0;w<blockDim.x/32;++w)v+=shared[w*SR_LM_STATISTICS+j];
      output[(unsigned long long)state*SR_LM_STATISTICS+j]=v;
    }
  }
}
extern "C" __global__ void lm_statistics_1(const SRLMProgram *programs,
    const SRLMState *states, unsigned starts, const float *input,
    const float *target, unsigned long long rows, double *output,
    const unsigned *indices, unsigned offset) {
  statistics_single<1>(programs,states,starts,input,target,rows,output,indices,offset);
}
extern "C" __global__ void lm_statistics_2(const SRLMProgram *programs,
    const SRLMState *states, unsigned starts, const float *input,
    const float *target, unsigned long long rows, double *output,
    const unsigned *indices, unsigned offset) {
  statistics_single<2>(programs,states,starts,input,target,rows,output,indices,offset);
}
extern "C" __global__ void lm_statistics_4(const SRLMProgram *programs,
    const SRLMState *states, unsigned starts, const float *input,
    const float *target, unsigned long long rows, double *output,
    const unsigned *indices, unsigned offset) {
  statistics_single<4>(programs,states,starts,input,target,rows,output,indices,offset);
}
extern "C" __global__ void lm_statistics_8(const SRLMProgram *programs,
    const SRLMState *states, unsigned starts, const float *input,
    const float *target, unsigned long long rows, double *output,
    const unsigned *indices, unsigned offset) {
  statistics_pair(programs,states,starts,input,target,rows,output,indices,offset);
}
extern "C" __global__ void lm_step(const SRLMProgram *programs, SRLMState *states,
    unsigned starts, unsigned count, const double *trial_stats, double *best_stats,
    unsigned iteration, unsigned propose) {
  const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
  SRLMState &s=states[i];const unsigned n=programs[i/starts].parameters;
  const double *trial=trial_stats+(unsigned long long)i*SR_LM_STATISTICS;
  double *best=best_stats+(unsigned long long)i*SR_LM_STATISTICS;
  bool valid=true;
  #pragma unroll
  for(unsigned j=0;j<SR_LM_STATISTICS;++j)if(!isfinite(trial[j]))valid=false;
  if(valid&&trial[0]<s.loss){
    #pragma unroll
    for(unsigned j=0;j<SR_LM_STATISTICS;++j)best[j]=trial[j];
    #pragma unroll
    for(unsigned j=0;j<SR_LM_PARAMETERS;++j)if(j<n)s.best[j]=s.trial[j];
    s.loss=trial[0];
    if(iteration){s.accepted++;s.damping=fmax(1e-12,s.damping*.3);}
  }else if(iteration)s.damping=fmin(1e12,s.damping*10);
  if(!valid)s.invalid++;
  if(!propose||!isfinite(s.loss)||!n)return;
  // Fixed triangular layout and fully unrolled solve. Inactive dimensions are
  // independent unit diagonals, never fitted or merged with active parameters.
  double matrix[36],delta[SR_LM_PARAMETERS],max_diag=0;
  #pragma unroll
  for(unsigned j=0;j<SR_LM_PARAMETERS;++j){
    #pragma unroll
    for(unsigned k=0;k<=j;++k){
      matrix[j*(j+1)/2+k]=j<n?best[1+SR_LM_PARAMETERS+j*(j+1)/2+k]:(j==k?1.:0.);
    }
    if(j<n)max_diag=fmax(max_diag,matrix[j*(j+1)/2+j]);
  }
  #pragma unroll
  for(unsigned j=0;j<SR_LM_PARAMETERS;++j){
    if(j<n)matrix[j*(j+1)/2+j]+=s.damping*fmax(matrix[j*(j+1)/2+j],fmax(max_diag*1e-12,1e-30));
    delta[j]=j<n?-best[1+j]:0;
  }
  valid=true;
  #pragma unroll
  for(unsigned j=0;j<SR_LM_PARAMETERS;++j){
    #pragma unroll
    for(unsigned k=0;k<=j;++k){
      double v=matrix[j*(j+1)/2+k];
      #pragma unroll
      for(unsigned l=0;l<k;++l)v-=matrix[j*(j+1)/2+l]*matrix[k*(k+1)/2+l];
      if(j==k){if(!(v>0)||!isfinite(v))valid=false;matrix[j*(j+1)/2+k]=sqrt(v);}
      else matrix[j*(j+1)/2+k]=v/matrix[k*(k+1)/2+k];
    }
  }
  #pragma unroll
  for(unsigned j=0;j<SR_LM_PARAMETERS;++j){
    #pragma unroll
    for(unsigned k=0;k<j;++k)delta[j]-=matrix[j*(j+1)/2+k]*delta[k];
    delta[j]/=matrix[j*(j+1)/2+j];
  }
  #pragma unroll
  for(int j=SR_LM_PARAMETERS-1;j>=0;--j){
    #pragma unroll
    for(unsigned k=j+1;k<SR_LM_PARAMETERS;++k)delta[j]-=matrix[k*(k+1)/2+j]*delta[k];
    delta[j]/=matrix[j*(j+1)/2+j];
  }
  #pragma unroll
  for(unsigned j=0;j<SR_LM_PARAMETERS;++j)if(j<n&&!isfinite((float)(s.best[j]+delta[j])))valid=false;
  #pragma unroll
  for(unsigned j=0;j<SR_LM_PARAMETERS;++j)if(j<n)s.trial[j]=valid?(float)(s.best[j]+delta[j]):s.best[j];
  if(!valid)s.damping=fmin(1e12,s.damping*10);
}
