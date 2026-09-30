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
// CPU-only entry points. No CUDA headers, library, context or device required.
#include "solver.cuh"
#include "model.cuh"
#ifdef ODEZZA_CPU_FLOAT
#include "solver32.hpp"
using Real=float;
#else
using Real=double;
#endif

extern "C" void cpu_ros23(int rows, const int* offsets, const double* times,
        const double* ic, const double* obs, double rtol, double atol, double h0,
        int limit, double* prediction, Stats* stats) {
    for (int row=0; row<rows; ++row)
#ifdef ODEZZA_CPU_FLOAT
        fp32::integrate<Model>(offsets[row],offsets[row+1],times,ic+row*Model::N,obs,
#else
        integrate<Model>(offsets[row],offsets[row+1],times,ic+row*Model::N,obs,
#endif
            rtol,atol,h0,limit,prediction,stats+row);
}

extern "C" void cpu_model(const double* y, double* rhs, double* jac) {
    Real yy[Model::N],ff[Model::N],jj[Model::N*Model::N];
    for(int i=0;i<Model::N;++i)yy[i]=(Real)y[i];
    Model::rhs(yy,ff);Model::jac(yy,jj);
    for(int i=0;i<Model::N;++i)rhs[i]=ff[i];
    for(int i=0;i<Model::N*Model::N;++i)jac[i]=jj[i];
}

extern "C" void cpu_rk4(int rows, const int* offsets, const double* times,
        const double* ic, const double* obs, int substeps, double* prediction, Stats* stats) {
    constexpr int N=Model::N;
    for (int row=0; row<rows; ++row) {
        Stats s={}; s.mse=INFINITY; s.min_h=INFINITY;
        Real y[N],a[N],b[N],c[N],d[N],tmp[N];double ss=0,correction=0;
        const int begin=offsets[row],end=offsets[row+1];
        for (int j=0; j<N; ++j) {
            y[j]=ic[row*N+j]; if(prediction) prediction[begin*N+j]=y[j];
        }
        for (int t=begin+1; t<end && s.status==0; ++t) {
            const Real h=(Real)((times[t]-times[t-1])/substeps);
            if (!(h>0) || !isfinite(h)) { s.status=2; break; }
            s.min_h=fmin(s.min_h,h); s.max_h=fmax(s.max_h,h);
            for (int k=0; k<substeps; ++k) {
                Model::rhs(y,a);
                for (int j=0; j<N; ++j) tmp[j]=y[j]+h*Real(0.5)*a[j];
                Model::rhs(tmp,b);
                for (int j=0; j<N; ++j) tmp[j]=y[j]+h*Real(0.5)*b[j];
                Model::rhs(tmp,c);
                for (int j=0; j<N; ++j) tmp[j]=y[j]+h*c[j];
                Model::rhs(tmp,d);
                ++s.attempts; s.rhs+=4;
                for (int j=0; j<N; ++j) {
                    y[j]+=h*(a[j]+2*b[j]+2*c[j]+d[j])/6;
                    if(!isfinite(y[j])) s.status=3;
                }
                if(s.status) break;
                ++s.accepted;
            }
            if(s.status) break;
            for (int j=0; j<N; ++j) {
                if(prediction) prediction[t*N+j]=y[j];
                const double value=obs[t*N+j];
                if(isfinite(value)) {
                    double e=y[j]-value,term=e*e-correction,sum=ss+term;
                    correction=(sum-ss)-term; ss=sum; ++s.count;
                }
            }
        }
        if(s.status==0) {
            s.mse=s.count ? ss/s.count : INFINITY;
            if(s.count && !isfinite(s.mse)) s.status=4;
        }
        stats[row]=s;
    }
}
