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
// Isolated comparison against the unchanged public C API. Never linked into the
// service. Prepared input, allocation, compilation and CPU checks are not timed.
#include "odezza.h"
#include <cuda_runtime.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using U64 = unsigned long long;
using Winner = OdezzaScoreWinner;
using Counts = OdezzaScoreCounts;
#define CU(x) do { cudaError_t checked_cuda_status=(x);if(checked_cuda_status!=cudaSuccess)throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(checked_cuda_status)); } while(0)
#define OD(x) do { OdezzaResult checked_odezza_status=(x);if(checked_odezza_status)throw std::runtime_error(std::string(#x)+": Odezza error "+std::to_string(checked_odezza_status)); } while(0)
#define REQUIRE(x) do { if(!(x))throw std::runtime_error(std::string("check failed: ")+#x); } while(0)

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Buffer {
    void *p=nullptr;size_t bytes=0;
    void allocate(size_t n) { REQUIRE(!p);bytes=n;CU(cudaMalloc(&p,std::max(n,size_t(1)))); }
    template<class T> T *as() const {return static_cast<T*>(p);}
    CUdeviceptr device() const {return reinterpret_cast<CUdeviceptr>(p);}
    ~Buffer() {if(p)(void)cudaFree(p);}
};
struct Events {
    cudaEvent_t e[4];cudaStream_t stream;
    Events() {CU(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));for(auto &x:e)CU(cudaEventCreate(&x));}
    ~Events() {for(auto x:e)(void)cudaEventDestroy(x);(void)cudaStreamDestroy(stream);}
    void mark(int i) {CU(cudaEventRecord(e[i],stream));}
    double elapsed(int a,int b) {float ms=0;CU(cudaEventElapsedTime(&ms,e[a],e[b]));return ms*1000.;}
};
struct Layout {
    U64 systems,banks,permutations,groups,elements,scores;
    unsigned mode;
};
static Layout layout(unsigned systems,unsigned banks,unsigned bits,unsigned mode) {
    Layout l={systems,banks,U64(1)<<bits,0,0,0,mode};
    l.scores=l.systems*l.banks*l.permutations;
    l.groups=mode==2?1:mode==1?l.systems*l.permutations:l.systems;
    l.elements=l.scores/l.groups;return l;
}
__host__ __device__ static U64 score_index(Layout l,U64 g,U64 e) {
    return l.mode==2?e:l.mode==0?g*l.banks*l.permutations+e:
        (g/l.permutations)*l.banks*l.permutations+e*l.permutations+g%l.permutations;
}
__host__ __device__ static Winner empty() {return {FLT_MAX,0,UINT64_MAX};}
__host__ __device__ static bool valid(float f) {return f>=0&&f<FLT_MAX;}
struct AddCounts {
    __device__ Counts operator()(Counts a,Counts b) const {
        return {a.valid+b.valid,a.invalid+b.invalid,a.negative+b.negative};
    }
};
// The temporary Item flags preserve invalid/negative counts without rereading
// all raw scores. Sort keys canonicalize signed zero; final output rereads only
// the k winning raw values to preserve their exact original bits.
struct Item {float key;unsigned flags;U64 index;};
struct ReadItem {
    const float *scores;Layout l;U64 group;
    __device__ Item operator()(U64 e) const {
        U64 index=score_index(l,group,e);float f=scores[index];
        bool ok=valid(f);
        unsigned flags=ok?1:2|(isfinite(f)&&f<0?4:0);
        return {ok?(f==0?0.f:f):FLT_MAX,flags,index};
    }
};
template<int Items> __global__ void block_sort(const float *scores,Layout l,
    unsigned k,Winner *out,Counts *counts) {
    using Load=cub::BlockLoad<Item,256,Items,cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using Sort=cub::BlockRadixSort<float,256,Items,U64>;
    using Reduce=cub::BlockReduce<Counts,256>;
    union Storage {typename Load::TempStorage load;typename Sort::TempStorage sort;typename Reduce::TempStorage reduce;};
    extern __shared__ __align__(16) unsigned char scratch[];
    auto &s=*reinterpret_cast<Storage*>(scratch);
    Item items[Items];float keys[Items];U64 indices[Items];Counts c={0,0,0};
    auto iterator=thrust::make_transform_iterator(thrust::counting_iterator<U64>(0),ReadItem{scores,l,blockIdx.x});
    Load(s.load).Load(iterator,items,static_cast<int>(l.elements),Item{FLT_MAX,0,~U64(0)});
    __syncthreads();
    #pragma unroll
    for(int i=0;i<Items;i++) {
        keys[i]=items[i].key;indices[i]=items[i].index;
        c.valid+=(items[i].flags&1)!=0;c.invalid+=(items[i].flags&2)!=0;c.negative+=(items[i].flags&4)!=0;
    }
    Counts total=Reduce(s.reduce).Reduce(c,AddCounts{});
    if(!threadIdx.x)counts[blockIdx.x]=total;
    __syncthreads();
    Sort(s.sort).Sort(keys,indices);
    #pragma unroll
    for(int i=0;i<Items;i++) {
        unsigned rank=threadIdx.x*Items+i;
        if(rank<k)out[U64(blockIdx.x)*k+rank]=keys[i]<FLT_MAX?Winner{scores[indices[i]],0,indices[i]}:empty();
    }
}

template<int Items> static size_t block_storage() {
    using Load=cub::BlockLoad<Item,256,Items,cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using Sort=cub::BlockRadixSort<float,256,Items,U64>;
    using Reduce=cub::BlockReduce<Counts,256>;
    return std::max({sizeof(typename Load::TempStorage),sizeof(typename Sort::TempStorage),sizeof(typename Reduce::TempStorage)});
}
template<int Items> static bool configure_block(int max_shared) {
    if(block_storage<Items>()>static_cast<size_t>(max_shared))return false;
    CU(cudaFuncSetAttribute(block_sort<Items>,cudaFuncAttributeMaxDynamicSharedMemorySize,int(block_storage<Items>())));return true;
}
template<int Items> static void launch_block(Layout l,const Buffer &scores,unsigned k,Buffer &w,Buffer &c,Events &ev) {
    block_sort<Items><<<unsigned(l.groups),256,block_storage<Items>(),ev.stream>>>(scores.as<float>(),l,k,w.as<Winner>(),c.as<Counts>());
}
#define BLOCK_DISPATCH(action) switch(items) {case 1:action(1);break;case 2:action(2);break;case 4:action(4);break;case 8:action(8);break;case 16:action(16);break;case 32:action(32);break;default:throw std::runtime_error("unsupported block size");}

__global__ void pack_segments(const float *scores,Layout l,float *keys,U64 *indices,Counts *counts) {
    using Reduce=cub::BlockReduce<Counts,256>;
    __shared__ typename Reduce::TempStorage temp;
    U64 g=blockIdx.x;Counts c={0,0,0};ReadItem read{scores,l,g};
    for(U64 e=threadIdx.x;e<l.elements;e+=256) {
        Item x=read(e);keys[g*l.elements+e]=x.key;indices[g*l.elements+e]=x.index;
        c.valid+=(x.flags&1)!=0;c.invalid+=(x.flags&2)!=0;c.negative+=(x.flags&4)!=0;
    }
    Counts total=Reduce(temp).Reduce(c,AddCounts{});
    if(!threadIdx.x)counts[g]=total;
}
__global__ void take_winners(const float *raw,const float *keys,const U64 *indices,
    Layout l,unsigned k,Winner *out) {
    U64 g=blockIdx.x;
    for(unsigned r=threadIdx.x;r<k;r+=256) {
        U64 at=g*l.elements+r;
        out[g*k+r]=r<l.elements&&keys[at]<FLT_MAX?Winner{raw[indices[at]],0,indices[at]}:empty();
    }
}
struct Segments {
    Buffer keys,sorted_keys,indices,sorted_indices,offsets,temp;
    size_t stable_bytes=0,radix_bytes=0;
    void setup(Layout l,Events &ev) {
        REQUIRE(l.scores<=INT32_MAX);
        keys.allocate(l.scores*4);sorted_keys.allocate(l.scores*4);
        indices.allocate(l.scores*8);sorted_indices.allocate(l.scores*8);offsets.allocate((l.groups+1)*sizeof(int));
        std::vector<int> v(l.groups+1);for(U64 i=0;i<=l.groups;i++)v[i]=int(i*l.elements);
        CU(cudaMemcpy(offsets.p,v.data(),offsets.bytes,cudaMemcpyHostToDevice));
        CU(cub::DeviceSegmentedSort::StableSortPairs(nullptr,stable_bytes,keys.as<float>(),sorted_keys.as<float>(),indices.as<U64>(),sorted_indices.as<U64>(),l.scores,l.groups,offsets.as<int>(),offsets.as<int>()+1,ev.stream));
        CU(cub::DeviceSegmentedRadixSort::SortPairs(nullptr,radix_bytes,keys.as<float>(),sorted_keys.as<float>(),indices.as<U64>(),sorted_indices.as<U64>(),l.scores,l.groups,offsets.as<int>(),offsets.as<int>()+1,0,32,ev.stream));
        temp.allocate(std::max(stable_bytes,radix_bytes));
    }
    void sort(Layout l,Events &ev,bool radix) {
        size_t n=radix?radix_bytes:stable_bytes;
        if(radix)CU(cub::DeviceSegmentedRadixSort::SortPairs(temp.p,n,keys.as<float>(),sorted_keys.as<float>(),indices.as<U64>(),sorted_indices.as<U64>(),l.scores,l.groups,offsets.as<int>(),offsets.as<int>()+1,0,32,ev.stream));
        else CU(cub::DeviceSegmentedSort::StableSortPairs(temp.p,n,keys.as<float>(),sorted_keys.as<float>(),indices.as<U64>(),sorted_indices.as<U64>(),l.scores,l.groups,offsets.as<int>(),offsets.as<int>()+1,ev.stream));
    }
    size_t required(bool radix) const {return keys.bytes+sorted_keys.bytes+indices.bytes+sorted_indices.bytes+offsets.bytes+(radix?radix_bytes:stable_bytes);}
};
static bool better(Winner a,Winner b) {return a.mse<b.mse||(a.mse==b.mse&&a.score_index<b.score_index);}
struct Reference {std::vector<Winner> winners;std::vector<Counts> counts;};
static Reference reference(Layout l,const std::vector<float> &raw,unsigned k) {
    Reference ref;ref.winners.assign(l.groups*k,empty());ref.counts.resize(l.groups);
    std::vector<Winner> rows;rows.reserve(l.elements);
    for(U64 g=0;g<l.groups;g++) {
        rows.clear();Counts c={0,0,0};
        for(U64 e=0;e<l.elements;e++) {
            U64 ix=score_index(l,g,e);float f=raw[ix];
            if(valid(f)) {rows.push_back({f,0,ix});++c.valid;}
            else {++c.invalid;if(std::isfinite(f)&&f<0)++c.negative;}
        }
        size_t n=std::min<size_t>(k,rows.size());std::partial_sort(rows.begin(),rows.begin()+n,rows.end(),better);
        std::copy_n(rows.begin(),n,ref.winners.begin()+g*k);ref.counts[g]=c;
    }
    return ref;
}
static void verify(Layout l,unsigned k,const Reference &ref,const std::vector<Winner> &w,
    const std::vector<Counts> &c,const std::vector<float> &values,const std::vector<float> &banks) {
    REQUIRE(!std::memcmp(w.data(),ref.winners.data(),w.size()*sizeof(Winner)));
    REQUIRE(!std::memcmp(c.data(),ref.counts.data(),c.size()*sizeof(Counts)));
    for(size_t i=0;i<w.size();i++) {
        U64 index=w[i].score_index;
        if(index==UINT64_MAX) {REQUIRE(std::isnan(values[2*i])&&std::isnan(values[2*i+1]));continue;}
        REQUIRE(index<l.scores);
        U64 system=index/(l.banks*l.permutations),bank=index/l.permutations%l.banks;
        REQUIRE(!std::memcmp(&values[2*i],&banks[(system*l.banks+bank)*2],8));
        REQUIRE(i/k<l.groups);
    }
}
static std::vector<float> make_scores(Layout l,unsigned pattern) {
    std::vector<float> v(l.scores);
    for(U64 i=0;i<l.scores;i++) {
        uint32_t x=uint32_t(i)+0x9e3779b9u;x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;
        float f=pattern==1?1.f:pattern==3?float(l.scores-i):float(x&0xffffff)/16384.f;
        if(pattern==2) {
            if(i%19==0)f=NAN;if(i%23==0)f=INFINITY;if(i%29==0)f=FLT_MAX;
            if(i%31==0)f=-1;if(i%37==0)f=-INFINITY;if(i%41==0)f=-0.f;if(i%43==0)f=0.f;
            if(i<l.banks*l.permutations)f=NAN;
        }
        v[i]=f;
    }
    return v;
}
struct Sample {double gpu=0,pack=0,sort=0,take=0,host_reduce=0,gather=0,copy=0,total=0;};
static double median(const std::vector<Sample> &s,double Sample::*field) {
    std::vector<double> x;for(auto &v:s)x.push_back(v.*field);std::sort(x.begin(),x.end());return x[x.size()/2];
}
static void run_case(Layout l,unsigned bits,unsigned pattern,unsigned sm,int max_shared,unsigned repeats) {
    auto raw=make_scores(l,pattern);std::vector<float> bank(l.systems*l.banks*2);
    for(size_t i=0;i<bank.size();i++)bank[i]=float(i%1048576)/1024.f;
    Buffer scores,banks;scores.allocate(raw.size()*4);banks.allocate(bank.size()*4);
    CU(cudaMemcpy(scores.p,raw.data(),scores.bytes,cudaMemcpyHostToDevice));CU(cudaMemcpy(banks.p,bank.data(),banks.bytes,cudaMemcpyHostToDevice));
    Events ev;Segments seg;seg.setup(l,ev);
    unsigned items=1;while(items*256<l.elements&&items<=32)items*=2;
    bool block_ok=false;size_t shared=0;
    if(items<=32) {
        #define CONFIG(n) {shared=block_storage<n>();block_ok=configure_block<n>(max_shared);}
        BLOCK_DISPATCH(CONFIG)
        #undef CONFIG
    }
    for(unsigned k: {1u,4u,16u}) {
        auto ref=reference(l,raw,k);
        OdezzaScoreReducer *reducer=nullptr;OD(odezza_score_reducer_create(sm,k,&reducer));
        OdezzaScoringLaunch launch={};launch.constant_bank_count=unsigned(l.banks);launch.active_toggle_count=bits;
        launch.mse_output_device=scores.device();launch.constant_banks_device=banks.device();
        OdezzaScoreReductionSize size;OD(odezza_score_reduction_requirements(&launch,l.systems,OdezzaScoreGrouping(l.mode),k,&size));
        Buffer workspace,winners,counts,gathered;workspace.allocate(size.workspace_bytes);winners.allocate(size.winner_bytes);counts.allocate(size.count_bytes);gathered.allocate(l.groups*k*8);
        std::vector<Winner> host_w(l.groups*k);std::vector<Counts> host_c(l.groups);std::vector<float> host_v(l.groups*k*2);
        std::array<std::vector<Sample>,4> samples;
        for(unsigned rep=0;rep<repeats+3;rep++)for(unsigned turn=0;turn<4;turn++) {
            unsigned algo=(turn+rep)%4;if(algo==3&&!block_ok)continue;
            Sample s;double start=now();
            if(algo==0) {
                OdezzaScoreReductionReport rr;
                OD(odezza_score_reducer_run(reducer,&launch,l.systems,OdezzaScoreGrouping(l.mode),workspace.device(),workspace.bytes,winners.device(),winners.bytes,counts.device(),counts.bytes,&rr));
                s.gpu=rr.kernel_seconds*1e6;s.sort=s.gpu;
            } else {
                ev.mark(0);
                if(algo==3) {
                    #define LAUNCH(n) launch_block<n>(l,scores,k,winners,counts,ev)
                    BLOCK_DISPATCH(LAUNCH)
                    #undef LAUNCH
                } else {
                    pack_segments<<<unsigned(l.groups),256,0,ev.stream>>>(scores.as<float>(),l,seg.keys.as<float>(),seg.indices.as<U64>(),counts.as<Counts>());
                    ev.mark(1);seg.sort(l,ev,algo==2);ev.mark(2);
                    take_winners<<<unsigned(l.groups),32,0,ev.stream>>>(scores.as<float>(),seg.sorted_keys.as<float>(),seg.sorted_indices.as<U64>(),l,k,winners.as<Winner>());
                }
                CU(cudaGetLastError());ev.mark(3);CU(cudaEventSynchronize(ev.e[3]));
                s.gpu=ev.elapsed(0,3);
                if(algo==3)s.sort=s.gpu;
                else {s.pack=ev.elapsed(0,1);s.sort=ev.elapsed(1,2);s.take=ev.elapsed(2,3);}
            }
            double reduced=now();s.host_reduce=(reduced-start)*1e6;
            OD(odezza_score_reducer_gather(reducer,&launch,l.systems,OdezzaScoreGrouping(l.mode),2,winners.device(),winners.bytes,gathered.device(),gathered.bytes));
            double g=now();s.gather=(g-reduced)*1e6;
            CU(cudaMemcpy(host_w.data(),winners.p,winners.bytes,cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(host_c.data(),counts.p,counts.bytes,cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(host_v.data(),gathered.p,gathered.bytes,cudaMemcpyDeviceToHost));
            double end=now();s.copy=(end-g)*1e6;s.total=(end-start)*1e6;
            if(rep==0||rep==repeats+2)verify(l,k,ref,host_w,host_c,host_v,bank);
            if(rep>=3)samples[algo].push_back(s);
        }
        const char *names[]={"odezza_current","cub_segmented_stable","cub_segmented_radix","cub_block_radix"};
        for(unsigned a=0;a<4;a++) {
            printf("{\"algorithm\":\"%s\",\"systems\":%llu,\"banks\":%llu,\"bits\":%u,\"grouping\":%u,\"groups\":%llu,\"elements_per_group\":%llu,\"scores\":%llu,\"pattern\":%u,\"k\":%u,",names[a],l.systems,l.banks,bits,l.mode,l.groups,l.elements,l.scores,pattern,k);
            if(a==3&&!block_ok) {printf("\"skipped\":\"group exceeds trial block capacity or shared memory\"}\n");continue;}
            auto &v=samples[a];size_t extra=a==0?workspace.bytes:a==3?0:seg.required(a==2);
            printf("\"checked\":true,\"repeats\":%u,\"extra_device_bytes\":%zu,\"dynamic_shared_bytes\":%zu,\"output_bytes\":%zu,\"gpu_us\":%.6f,\"pack_us\":%.6f,\"selection_us\":%.6f,\"take_us\":%.6f,\"host_reduce_us\":%.6f,\"coefficient_gather_us\":%.6f,\"download_us\":%.6f,\"complete_us\":%.6f}\n",repeats,extra,a==3?shared:0,winners.bytes+counts.bytes+gathered.bytes,median(v,&Sample::gpu),median(v,&Sample::pack),median(v,&Sample::sort),median(v,&Sample::take),median(v,&Sample::host_reduce),median(v,&Sample::gather),median(v,&Sample::copy),median(v,&Sample::total));
        }
        fflush(stdout);OD(odezza_score_reducer_destroy(reducer));
    }
}
int main(int argc,char **argv) {
    try {
        int device=argc>1?std::stoi(argv[1]):0;bool quick=argc>2&&std::string(argv[2])=="quick";
        CU(cudaSetDevice(device));CU(cudaFree(nullptr));
        cudaDeviceProp props;CU(cudaGetDeviceProperties(&props,device));
        int max_shared=0;CU(cudaDeviceGetAttribute(&max_shared,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
        fprintf(stderr,"device=%s sm=%d max_shared=%d CUB=%d\n",props.name,props.major*10+props.minor,max_shared,CUB_VERSION);
        auto run=[&](unsigned systems,unsigned banks,unsigned bits,unsigned mode,unsigned pattern) {
            run_case(layout(systems,banks,bits,mode),bits,pattern,props.major*10+props.minor,max_shared,quick?11:21);
        };
        run(1024,2048,0,0,0);run(128,2048,4,1,2);
        if(!quick) {
            run(1024,2048,0,0,1);run(1024,2048,0,0,3);
            run(1024,192,0,0,2);run(128,8192,0,0,0);run(128,8193,0,0,2);
            run(128,16384,0,0,0);run(8192,2048,0,0,0);
            run(128,16384,0,2,0);run(3,3,2,1,2);
        }
        fprintf(stderr,"all requested comparisons passed\n");return 0;
    } catch(const std::exception &e) {fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
