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
// Embedded by tests/embed_score_reducer.py. NVRTC + installed CUB; no SASS sites.
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
typedef unsigned long long U64;
struct Winner { float mse; unsigned int reserved; U64 index; };
struct Counts { U64 valid, invalid, negative; };
static_assert(sizeof(Winner)==16 && sizeof(Counts)==24, "reducer ABI");
__device__ Winner empty_winner() { return {__int_as_float(0x7f7fffff),0,~0ULL}; }
__device__ bool valid_score(float f) { return f>=0 && f<__int_as_float(0x7f7fffff); }
__device__ bool better(Winner a, Winner b) {
    return a.mse < b.mse || (a.mse == b.mse && a.index < b.index);
}
struct AddCounts {
    __device__ Counts operator()(Counts a, Counts b) const {
        return {a.valid+b.valid,a.invalid+b.invalid,a.negative+b.negative};
    }
};
struct Aggregate { Winner winner; Counts counts; };
struct Combine {
    __device__ Aggregate operator()(Aggregate a, Aggregate b) const {
        return {better(a.winner,b.winner)?a.winner:b.winner,AddCounts{}(a.counts,b.counts)};
    }
};
struct Item { float key; unsigned flags; U64 index; };
struct RawInput {
    using value_type=Item;
    const float *scores; U64 group, configurations, permutations, base; unsigned mode;
    __device__ Item operator[](int offset) const {
        U64 e=base+offset;
        U64 index=mode==2?e:mode==0?group*configurations+e:
            (group/permutations)*configurations+e*permutations+group%permutations;
        float f=scores[index]; bool ok=valid_score(f);
        return {ok?(f==0?0.f:f):__int_as_float(0x7f7fffff),
            ok?1u:2u|(isfinite(f)&&f<0?4u:0u),index};
    }
};
struct PartialInput {
    using value_type=Item;
    const Winner *partial; U64 base;
    __device__ Item operator[](int offset) const {
        Winner w=partial[base+offset];
        return {w.mse==0?0.f:w.mse,0,w.index};
    }
};
// Each node covers a contiguous original-index range. Stable CUB sorting then
// preserves (MSE, original index) ties at every level, including signed zero.
// Only retained winners reread raw scores, preserving their exact FP32 bits.
template<int Items, class Input> __device__ void sort_node(Input input,int count,
    const float *scores,unsigned k,Winner *out,Counts local,Counts *totals,bool raw) {
    using Load=cub::BlockLoad<Item,256,Items,cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using Sort=cub::BlockRadixSort<float,256,Items,U64>;
    using Reduce=cub::BlockReduce<Counts,256>;
    __shared__ union {typename Load::TempStorage load;typename Sort::TempStorage sort;typename Reduce::TempStorage reduce;} temp;
    Item items[Items];float keys[Items];U64 indices[Items];
    Load(temp.load).Load(input,items,count,Item{__int_as_float(0x7f7fffff),0,~0ULL});
    __syncthreads();
    #pragma unroll
    for(int i=0;i<Items;++i) {
        keys[i]=items[i].key;indices[i]=items[i].index;
        if(raw) {
            local.valid+=(items[i].flags&1)!=0;
            local.invalid+=(items[i].flags&2)!=0;
            local.negative+=(items[i].flags&4)!=0;
        }
    }
    Counts sum=Reduce(temp.reduce).Reduce(local,AddCounts{});
    if(!threadIdx.x)*totals=sum;
    __syncthreads();
    Sort(temp.sort).Sort(keys,indices);
    #pragma unroll
    for(int i=0;i<Items;++i) {
        unsigned rank=threadIdx.x*Items+i;
        if(rank<k)out[rank]=valid_score(keys[i])?Winner{scores[indices[i]],0,indices[i]}:empty_winner();
    }
}
// Small top-k uses repeated CUB argmin over a bounded register-resident tile.
// This avoids sorting all 2048 values to return only two to four rows.
template<class Input> __device__ void small_node(Input input,int n,const float *scores,
    unsigned k,Winner *out,Counts local,Counts *totals,bool raw) {
    using Reduce=cub::BlockReduce<Aggregate,256>;
    __shared__ typename Reduce::TempStorage temp;
    __shared__ Winner selected;
    Item items[8];
    #pragma unroll
    for(int i=0;i<8;i++) {
        int at=threadIdx.x+i*256;
        items[i]=at<n?input[at]:Item{__int_as_float(0x7f7fffff),0,~0ULL};
        if(raw) {
            local.valid+=(items[i].flags&1)!=0;
            local.invalid+=(items[i].flags&2)!=0;
            local.negative+=(items[i].flags&4)!=0;
        }
    }
    for(unsigned rank=0;rank<k;rank++) {
        Winner best=empty_winner();
        #pragma unroll
        for(int i=0;i<8;i++) {
            Winner w=valid_score(items[i].key)?Winner{items[i].key,0,items[i].index}:empty_winner();
            if(better(w,best))best=w;
        }
        Aggregate sum=Reduce(temp).Reduce(Aggregate{best,local},Combine{});
        if(!threadIdx.x) {
            selected=sum.winner;
            out[rank]=selected.index==~0ULL?empty_winner():Winner{scores[selected.index],0,selected.index};
            if(!rank)*totals=sum.counts;
        }
        __syncthreads();
        #pragma unroll
        for(int i=0;i<8;i++)if(items[i].index==selected.index)items[i].key=__int_as_float(0x7f7fffff);
        __syncthreads();
    }
}
template<int Items> __device__ void raw_node(const float *scores,U64 configurations,
    U64 permutations,U64 elements,unsigned tiles,unsigned mode,unsigned k,Winner *out,Counts *counts) {
    U64 group=blockIdx.x/tiles,base=(U64)(blockIdx.x%tiles)*2048;
    int n=(int)(elements-base<2048?elements-base:2048);
    RawInput input{scores,group,configurations,permutations,base,mode};
    if constexpr(Items==0) {
        using Reduce=cub::BlockReduce<Aggregate,256>;
        __shared__ typename Reduce::TempStorage temp;
        Aggregate a={empty_winner(),{0,0,0}};
        for(int i=threadIdx.x;i<n;i+=256) {
            Item x=input[i];
            Aggregate b={valid_score(x.key)?Winner{scores[x.index],0,x.index}:empty_winner(),
                {(x.flags&1)!=0,(x.flags&2)!=0,(x.flags&4)!=0}};
            a=Combine{}(a,b);
        }
        Aggregate sum=Reduce(temp).Reduce(a,Combine{});
        if(!threadIdx.x){out[blockIdx.x]=sum.winner;counts[blockIdx.x]=sum.counts;}
    } else sort_node<Items>(input,n,scores,k,out+(U64)blockIdx.x*k,Counts{0,0,0},counts+blockIdx.x,true);
}
template<int Items> __device__ void merge_node(const float *scores,const Winner *in,
    const Counts *counts,unsigned previous,unsigned tiles,unsigned fanout,unsigned k,Winner *out,Counts *totals) {
    U64 group=blockIdx.x/tiles;unsigned begin=(blockIdx.x%tiles)*fanout;
    unsigned n=previous-begin<fanout?previous-begin:fanout;
    U64 base=group*previous+begin;
    if constexpr(Items==0) {
        using Reduce=cub::BlockReduce<Aggregate,256>;
        __shared__ typename Reduce::TempStorage temp;
        Aggregate a={empty_winner(),{0,0,0}};
        for(unsigned i=threadIdx.x;i<n;i+=256)a=Combine{}(a,Aggregate{in[base+i],counts[base+i]});
        Aggregate sum=Reduce(temp).Reduce(a,Combine{});
        if(!threadIdx.x){out[blockIdx.x]=sum.winner;totals[blockIdx.x]=sum.counts;}
    } else {
        Counts sum={0,0,0};
        for(unsigned i=threadIdx.x;i<n;i+=256)sum=AddCounts{}(sum,counts[base+i]);
        sort_node<Items>(PartialInput{in,base*k},n*k,scores,k,out+(U64)blockIdx.x*k,sum,totals+blockIdx.x,false);
    }
}
#define RAW(I) extern "C" __global__ void reduce_scores_##I(const float *scores,U64 configs,U64 perms,U64 elements,unsigned tiles,unsigned mode,unsigned k,Winner *out,Counts *counts) { raw_node<I>(scores,configs,perms,elements,tiles,mode,k,out,counts); }
#define MERGE(I) extern "C" __global__ void merge_scores_##I(const float *scores,const Winner *in,const Counts *counts,unsigned previous,unsigned tiles,unsigned fanout,unsigned k,Winner *out,Counts *totals) { merge_node<I>(scores,in,counts,previous,tiles,fanout,k,out,totals); }
RAW(0) RAW(1) RAW(2) RAW(4) RAW(8)
MERGE(0) MERGE(1) MERGE(2) MERGE(4) MERGE(8)
extern "C" __global__ void reduce_scores_small(const float *scores,U64 configs,U64 perms,
    U64 elements,unsigned tiles,unsigned mode,unsigned k,Winner *out,Counts *counts) {
    U64 group=blockIdx.x/tiles,base=(U64)(blockIdx.x%tiles)*2048;
    int n=(int)(elements-base<2048?elements-base:2048);
    small_node(RawInput{scores,group,configs,perms,base,mode},n,scores,k,out+(U64)blockIdx.x*k,Counts{0,0,0},counts+blockIdx.x,true);
}
extern "C" __global__ void merge_scores_small(const float *scores,const Winner *in,const Counts *counts,
    unsigned previous,unsigned tiles,unsigned fanout,unsigned k,Winner *out,Counts *totals) {
    U64 group=blockIdx.x/tiles;unsigned begin=(blockIdx.x%tiles)*fanout;
    unsigned n=previous-begin<fanout?previous-begin:fanout;
    U64 base=group*previous+begin;Counts sum={0,0,0};
    for(unsigned i=threadIdx.x;i<n;i+=256)sum=AddCounts{}(sum,counts[base+i]);
    small_node(PartialInput{in,base*k},n*k,scores,k,out+(U64)blockIdx.x*k,sum,totals+blockIdx.x,false);
}
extern "C" __global__ void gather_constants(const Winner *winners, const float *banks,
    U64 entries, U64 configurations, U64 bank_count, unsigned bits, unsigned constants,
    U64 score_count, float *out, const OParameter *parameters, const float *uniform_pool, const float *normal_pool, U64 pool_size) {
    for (U64 i=(U64)blockIdx.x*blockDim.x+threadIdx.x;i<entries*constants;
         i+=(U64)gridDim.x*blockDim.x) {
        U64 index=winners[i/constants].index;
        if (index>=score_count) out[i]=__int_as_float(0x7fc00000);
        else {
            U64 system=index/configurations,bank=(index%configurations)>>bits;
            out[i]=parameters?o_sample_constant(parameters,uniform_pool,normal_pool,pool_size,system*constants+i%constants,(unsigned)bank):banks[(system*bank_count+bank)*constants+i%constants];
        }
    }
}
