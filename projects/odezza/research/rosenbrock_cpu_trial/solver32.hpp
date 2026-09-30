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
// CPU FP32 state/RHS/Jacobian/stages/LU. FP64 clock, controller and MSE.
// Same Rosenbrock23 stages as solver.cuh; explicit float constants prevent
// accidentally promoting matrix or stage arithmetic to double precision.
namespace fp32 {
template<int N> bool factor(float* a,int* piv) {
    for(int k=0;k<N;++k) {
        int p=k;
        for(int i=k+1;i<N;++i) if(fabsf(a[i*N+k])>fabsf(a[p*N+k]))p=i;
        piv[k]=p;
        if(!isfinite(a[p*N+k]) || fabsf(a[p*N+k])<FLT_MIN)return false;
        if(p!=k)for(int j=0;j<N;++j){float v=a[k*N+j];a[k*N+j]=a[p*N+j];a[p*N+j]=v;}
        for(int i=k+1;i<N;++i){a[i*N+k]/=a[k*N+k];for(int j=k+1;j<N;++j)a[i*N+j]-=a[i*N+k]*a[k*N+j];}
    }
    for(int i=0;i<N*N;++i)if(!isfinite(a[i]))return false;
    return true;
}
template<int N> void solve(const float* a,const int* piv,float* b) {
    for(int k=0;k<N;++k)if(piv[k]!=k){float v=b[k];b[k]=b[piv[k]];b[piv[k]]=v;}
    for(int i=0;i<N;++i)for(int j=0;j<i;++j)b[i]-=a[i*N+j]*b[j];
    for(int i=N-1;i>=0;--i){for(int j=i+1;j<N;++j)b[i]-=a[i*N+j]*b[j];b[i]/=a[i*N+i];}
}
template<class M> void integrate(int begin,int end,const double* times,const double* initial,
        const double* observed,double rtol,double atol,double h_initial,int max_attempts,
        double* prediction,Stats* result) {
    constexpr int N=M::N;
    const float d=0.2928932188134524756f,c32=7.414213562373095049f;
    Stats s={};s.mse=INFINITY;s.min_h=INFINITY;
    float y[N],f[N],j[N*N],a[N*N],k1[N],k2[N],k3[N],mid[N],fm[N],next[N],fe[N];
    int piv[N];double ss=0,correction=0;
    for(int i=0;i<N;++i){y[i]=(float)initial[i];if(prediction)prediction[begin*N+i]=y[i];}
    double t=times[begin],h=h_initial>0?h_initial:(times[begin+1]-t)*.01;
    bool fresh=true;
    for(int row=begin+1;row<end;++row){
        const double target=times[row];
        while(t<target){
            if(s.attempts>=max_attempts){s.status=1;*result=s;return;}
            h=fmin(h,target-t);const float hs=(float)h;
            if(!(h>0)||!(hs>0)||!isfinite(hs)||t+h==t){s.status=2;*result=s;return;}
            ++s.attempts;
            if(fresh){
                M::rhs(y,f);M::jac(y,j);++s.rhs;++s.jac;
                for(int i=0;i<N;++i)if(!isfinite(f[i])){s.status=3;*result=s;return;}
                for(int i=0;i<N*N;++i)if(!isfinite(j[i])){s.status=3;*result=s;return;}
                fresh=false;
            }
            for(int i=0;i<N;++i)for(int q=0;q<N;++q)a[i*N+q]=(i==q?1.f:0.f)-hs*d*j[i*N+q];
            ++s.lu;
            if(!factor<N>(a,piv)){++s.pivot_failures;++s.rejected;h*=.1;continue;}
            for(int i=0;i<N;++i)k1[i]=f[i];solve<N>(a,piv,k1);
            for(int i=0;i<N;++i)mid[i]=y[i]+.5f*hs*k1[i];M::rhs(mid,fm);++s.rhs;
            for(int i=0;i<N;++i)k2[i]=fm[i]-k1[i];solve<N>(a,piv,k2);
            for(int i=0;i<N;++i){k2[i]+=k1[i];next[i]=y[i]+hs*k2[i];}
            M::rhs(next,fe);++s.rhs;
            for(int i=0;i<N;++i)k3[i]=fe[i]-c32*(k2[i]-fm[i])-2.f*(k1[i]-f[i]);
            solve<N>(a,piv,k3);
            double err2=0;bool finite=true;
            for(int i=0;i<N;++i){
                const float defect=(hs/6.f)*(k1[i]-2.f*k2[i]+k3[i]);
                double e=(double)defect/(atol+rtol*fmax(fabs((double)y[i]),fabs((double)next[i])));
                finite=finite&&isfinite(e)&&isfinite(next[i])&&isfinite(fe[i]);err2+=e*e;
            }
            double error=finite?sqrt(err2/N):INFINITY;
            if(error<=1.){
                for(int i=0;i<N;++i)y[i]=next[i];
                t=(h==target-t)?target:t+h;++s.accepted;s.min_h=fmin(s.min_h,h);s.max_h=fmax(s.max_h,h);fresh=true;
            }else ++s.rejected;
            h*=error==0?5.:fmax(.1,fmin(5.,.9*pow(error,-1./3.)));
        }
        for(int i=0;i<N;++i){
            if(prediction)prediction[row*N+i]=y[i];double value=observed[row*N+i];
            if(isfinite(value)){double e=(double)y[i]-value,term=e*e-correction,sum=ss+term;correction=(sum-ss)-term;ss=sum;++s.count;}
        }
    }
    s.mse=s.count?ss/s.count:INFINITY;if(s.count&&!isfinite(s.mse))s.status=4;*result=s;
}
}
