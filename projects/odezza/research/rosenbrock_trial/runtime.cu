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
#include <cuda_runtime.h>
#include <stdio.h>
#include "solver.cuh"
#include "model.cuh"

__global__ void rollouts(int rows, int lanes, const int* offsets, const double* times,
        const double* ic, const double* obs, double rtol, double atol, double h0,
        int limit, double* prediction, Stats* stats) {
    int lane=blockIdx.x*blockDim.x+threadIdx.x;
    if (lane>=lanes) return;
    int row=lane%rows;
    // Repetitions are a throughput control, not distinct ASTs/configurations.
    integrate<Model>(offsets[row],offsets[row+1],times,ic+row*Model::N,obs,
        rtol,atol,h0,limit,lane<rows ? prediction : nullptr,stats+lane);
}

extern "C" void cpu_run(int rows, const int* offsets, const double* times,
        const double* ic, const double* obs, double rtol, double atol, double h0,
        int limit, double* prediction, Stats* stats) {
    for (int row=0; row<rows; ++row)
        integrate<Model>(offsets[row],offsets[row+1],times,ic+row*Model::N,obs,
            rtol,atol,h0,limit,prediction,stats+row);
}

extern "C" void model_eval(const double* y, double* rhs, double* jac) {
    Model::rhs(y,rhs); Model::jac(y,jac);
}

extern "C" int gpu_run(int device, int rows, int lanes, int points, const int* offsets,
        const double* times, const double* ic, const double* obs, double rtol,
        double atol, double h0, int limit, double* prediction, Stats* stats,
        float* kernel_ms, float* transfers_ms, char* error, int error_size) {
    cudaError_t code=cudaSuccess;
    cudaStream_t stream=nullptr;
    cudaEvent_t start=nullptr, finish=nullptr, done=nullptr, begin=nullptr;
    int* off=nullptr; double *tt=nullptr,*ii=nullptr,*oo=nullptr,*pp=nullptr; Stats* ss=nullptr;
    size_t values=(size_t)points*Model::N*sizeof(double);
    float all_ms=0;
#define TRY(x) do { code=(x); if (code!=cudaSuccess) { snprintf(error,error_size,"%s: %s",#x,cudaGetErrorString(code)); goto cleanup; } } while(0)
    TRY(cudaSetDevice(device));
    TRY(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    TRY(cudaEventCreate(&begin)); TRY(cudaEventCreate(&start));
    TRY(cudaEventCreate(&finish)); TRY(cudaEventCreate(&done));
    TRY(cudaMalloc(&off,(rows+1)*sizeof(int))); TRY(cudaMalloc(&tt,points*sizeof(double)));
    TRY(cudaMalloc(&ii,rows*Model::N*sizeof(double))); TRY(cudaMalloc(&oo,values));
    if (prediction) TRY(cudaMalloc(&pp,values));
    TRY(cudaMalloc(&ss,lanes*sizeof(Stats)));
    TRY(cudaEventRecord(begin,stream));
    TRY(cudaMemcpyAsync(off,offsets,(rows+1)*sizeof(int),cudaMemcpyHostToDevice,stream));
    TRY(cudaMemcpyAsync(tt,times,points*sizeof(double),cudaMemcpyHostToDevice,stream));
    TRY(cudaMemcpyAsync(ii,ic,rows*Model::N*sizeof(double),cudaMemcpyHostToDevice,stream));
    TRY(cudaMemcpyAsync(oo,obs,values,cudaMemcpyHostToDevice,stream));
    if (pp) TRY(cudaMemsetAsync(pp,0xff,values,stream)); // Unwritten tail stays NaN on failures.
    TRY(cudaEventRecord(start,stream));
    rollouts<<<(lanes+127)/128,128,0,stream>>>(rows,lanes,off,tt,ii,oo,rtol,atol,h0,limit,pp,ss);
    TRY(cudaGetLastError()); TRY(cudaEventRecord(finish,stream));
    if (pp) TRY(cudaMemcpyAsync(prediction,pp,values,cudaMemcpyDeviceToHost,stream));
    TRY(cudaMemcpyAsync(stats,ss,lanes*sizeof(Stats),cudaMemcpyDeviceToHost,stream));
    TRY(cudaEventRecord(done,stream)); TRY(cudaEventSynchronize(done));
    TRY(cudaEventElapsedTime(kernel_ms,start,finish));
    TRY(cudaEventElapsedTime(&all_ms,begin,done)); *transfers_ms=all_ms-*kernel_ms;
cleanup:
    // The successful path has already waited for the completion event.
    // On errors, release this call's resources; never synchronize all devices.
    if(ss) cudaFree(ss); if(pp) cudaFree(pp); if(oo) cudaFree(oo);
    if(ii) cudaFree(ii); if(tt) cudaFree(tt); if(off) cudaFree(off);
    if(done) cudaEventDestroy(done); if(finish) cudaEventDestroy(finish);
    if(start) cudaEventDestroy(start); if(begin) cudaEventDestroy(begin);
    if(stream) cudaStreamDestroy(stream);
    return (int)code;
#undef TRY
}
