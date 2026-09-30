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
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include <cutlass/tfloat32.h>

#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>

namespace kermac_cute_gather_syrk {

inline bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

inline unsigned int get_cuda_device_arch() {
    int device = 0;
    cudaGetDevice(&device);
    int major = 0;
    int minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    return static_cast<unsigned int>(major) * 100u + static_cast<unsigned int>(minor) * 10u;
}

inline float round_to_tf32(float x) {
    return static_cast<float>(cutlass::tfloat32_t(x));
}

struct FloatToTf32 {
    CUTLASS_HOST_DEVICE
    cutlass::tfloat32_t operator()(float x) const {
        return cutlass::tfloat32_t(x);
    }
};

template <int TileM, int TileN>
struct CuteMmaConfig;

template <>
struct CuteMmaConfig<32, 32> {
    using TiledMma = cute::TiledMMA<
        cute::MMA_Atom<cute::SM80_16x8x8_F32TF32TF32F32_TN>,
        cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>, cute::Stride<cute::_2, cute::_1, cute::_1>>,
        cute::Tile<cute::_32, cute::_32, cute::_8>
    >;

    static constexpr int kThreads = decltype(cute::size(TiledMma{}))::value;
};

template <>
struct CuteMmaConfig<16, 16> {
    using TiledMma = cute::TiledMMA<
        cute::MMA_Atom<cute::SM80_16x8x8_F32TF32TF32F32_TN>,
        cute::Layout<cute::Shape<cute::_1, cute::_2, cute::_1>>
    >;

    static constexpr int kThreads = decltype(cute::size(TiledMma{}))::value;
};

template <int TileM, int TileN>
struct CuteSimtMmaConfig;

template <>
struct CuteSimtMmaConfig<32, 32> {
    using TiledMma = decltype(cute::make_tiled_mma(
        cute::UniversalFMA<float, float, float>{},
        cute::Layout<cute::Shape<cute::_16, cute::_16, cute::_1>>{}
    ));

    static constexpr int kThreads = decltype(cute::size(TiledMma{}))::value;
};

template <int Rows, int KTile, int Stages>
using PanelLayout = decltype(cute::tile_to_shape(
    cute::composition(
        cute::Swizzle<3, 2, 3>{},
        cute::Layout<cute::Shape<cute::_8, cute::_32>, cute::Stride<cute::_32, cute::_1>>{}
    ),
    cute::make_shape(cute::Int<Rows>{}, cute::Int<KTile>{}, cute::Int<Stages>{})
));

template <int TileM, int TileN, int KTile, int Stages>
struct SharedStorage {
    cute::ArrayEngine<float, cute::cosize_v<PanelLayout<TileM, KTile, Stages>>> panel;
    int32_t idx[TileM];
};

template <int Stages>
CUTE_DEVICE void cp_async_stage_wait(int tiles_in_flight) {
    constexpr int kMaxGroupsToKeep = Stages <= 1 ? 0 : (Stages - 2);
    int groups_to_keep = tiles_in_flight - 1;
    if (groups_to_keep < 0) {
        groups_to_keep = 0;
    } else if (groups_to_keep > kMaxGroupsToKeep) {
        groups_to_keep = kMaxGroupsToKeep;
    }

    if constexpr (kMaxGroupsToKeep == 0) {
        cute::cp_async_wait<0>();
    } else if constexpr (kMaxGroupsToKeep == 1) {
        if (groups_to_keep == 0) {
            cute::cp_async_wait<0>();
        } else {
            cute::cp_async_wait<1>();
        }
    } else {
        if (groups_to_keep == 0) {
            cute::cp_async_wait<0>();
        } else if (groups_to_keep == 1) {
            cute::cp_async_wait<1>();
        } else {
            cute::cp_async_wait<2>();
        }
    }
}

template <typename T>
CUTE_DEVICE bool is_aligned_16(const T* ptr) {
    return (reinterpret_cast<std::uintptr_t>(ptr) & 0xFu) == 0u;
}

template <typename T>
CUTE_DEVICE uint32_t to_smem_addr(T* ptr) {
#if defined(__CUDA_ARCH__)
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
#else
    (void)ptr;
    return 0u;
#endif
}

template <bool AsyncCopy, int TileRows, int KTile, int Stages, class Tensor>
CUTE_DEVICE void issue_panel_stage_loads(
    const float* __restrict__ x,
    int64_t x_ld,
    const int32_t* __restrict__ indices,
    Tensor& panel,
    int pipe,
    int row0,
    int num_rows
) {
    static_assert(KTile % 4 == 0, "KTile must be a multiple of 4 for 16-byte cp.async vectors");

    using namespace cute;
    using CopyVec = uint128_t;
    using CopyAtom = SM80_CP_ASYNC_CACHEALWAYS_ZFILL<CopyVec>;

    constexpr int kVec = 4;
    constexpr int kVecsPerRow = KTile / kVec;
    constexpr int kTotalVecs = TileRows * kVecsPerRow;

    const int tid = static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x);

    for (int linear = tid; linear < kTotalVecs; linear += threads) {
        const int row = linear / kVecsPerRow;
        const int kv = linear % kVecsPerRow;
        const int k = kv * kVec;
        float* dst_ptr = &panel(row, k, pipe);
        const int32_t feature = indices[row];
        const int src_row = row0 + k;
        const int remaining_rows = num_rows - src_row;
        const int valid_elems = feature >= 0
            ? (remaining_rows <= 0 ? 0 : (remaining_rows < kVec ? remaining_rows : kVec))
            : 0;

        const float* src_ptr = x + static_cast<int64_t>(feature >= 0 ? feature : 0) * x_ld + static_cast<int64_t>(src_row);

        if constexpr (AsyncCopy) {
            if (valid_elems == kVec && is_aligned_16(src_ptr) && is_aligned_16(dst_ptr)) {
                CopyAtom::copy(
                    *reinterpret_cast<CopyVec const*>(src_ptr),
                    *reinterpret_cast<CopyVec*>(dst_ptr),
                    true
                );
                continue;
            }
        }

        #pragma unroll
        for (int i = 0; i < kVec; ++i) {
            const bool valid = i < valid_elems;
            dst_ptr[i] = valid ? src_ptr[i] : 0.0f;
        }
    }
}

template <bool AsyncCopy, int TileM, int TileN, int KTile, int Stages>
__launch_bounds__(CuteMmaConfig<TileM, TileN>::kThreads)
__global__ void cute_gather_gemm_kernel(
    const float* __restrict__ x,
    int64_t x_ld,
    const int32_t* __restrict__ indices,
    int64_t indices_batch_stride,
    float* __restrict__ out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count
) {
    static_assert(KTile % 8 == 0, "KTile must be a multiple of the TF32 MMA K dimension");
    static_assert(TileM > 0 && TileN > 0 && KTile > 0 && Stages > 0, "invalid tile configuration");

    using namespace cute;
    using Config = CuteMmaConfig<TileM, TileN>;
    using TiledMma = typename Config::TiledMma;
    using SmemLayout = PanelLayout<TileM, KTile, Stages>;

    const int batch = static_cast<int>(blockIdx.x);
    if (batch >= batch_count) {
        return;
    }

    const int tile_m = static_cast<int>(blockIdx.y);
    const int tile_n = static_cast<int>(blockIdx.z);
    const int col0_m = tile_m * TileM;
    const int col0_n = tile_n * TileN;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& storage = *reinterpret_cast<SharedStorage<TileM, TileN, KTile, Stages>*>(smem_raw);

    Tensor sPanel = make_tensor(make_smem_ptr(storage.panel.begin()), SmemLayout{});

    const int tid = static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x);

    const int32_t* batch_indices = indices + static_cast<int64_t>(batch) * indices_batch_stride;

    for (int i = tid; i < TileM; i += threads) {
        storage.idx[i] = batch_indices[col0_m + i];
    }
    __syncthreads();

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    float* out_batch = out + static_cast<int64_t>(batch) * out_batch_stride;
    Tensor gC = make_tensor(
        make_gmem_ptr(out_batch + static_cast<int64_t>(col0_m) + static_cast<int64_t>(col0_n) * out_ld),
        make_layout(
            make_shape(Int<TileM>{}, Int<TileN>{}),
            make_stride(Int<1>{}, out_ld)
        )
    );
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    const int total_tiles = (num_rows + KTile - 1) / KTile;
    int next_tile = 0;
    int tiles_in_flight = 0;
    int smem_pipe_read = 0;
    int smem_pipe_write = 0;

    const int prefetch = Stages > 1 ? (Stages - 1) : 0;
    for (int i = 0; i < prefetch && next_tile < total_tiles; ++i) {
        issue_panel_stage_loads<AsyncCopy, TileM, KTile, Stages>(x, x_ld, storage.idx, sPanel, i, next_tile * KTile, num_rows);
        cp_async_fence();
        ++next_tile;
        ++tiles_in_flight;
    }
    smem_pipe_write = prefetch % Stages;

    while (next_tile < total_tiles || tiles_in_flight > 0) {
        if (next_tile < total_tiles) {
            issue_panel_stage_loads<AsyncCopy, TileM, KTile, Stages>(x, x_ld, storage.idx, sPanel, smem_pipe_write, next_tile * KTile, num_rows);
            cp_async_fence();
            ++next_tile;
            ++tiles_in_flight;
            smem_pipe_write = (smem_pipe_write + 1) % Stages;
        }

        cp_async_stage_wait<Stages>(tiles_in_flight);
        __syncthreads();

        cooperative_gemm(
            threadIdx.x,
            tiled_mma,
            sPanel(_, _, smem_pipe_read),
            sPanel(_, _, smem_pipe_read),
            tCrC,
            FloatToTf32{},
            FloatToTf32{},
            SM75_U32x4_LDSM_N{},
            SM75_U32x4_LDSM_N{}
        );

        __syncthreads();
        smem_pipe_read = (smem_pipe_read + 1) % Stages;
        --tiles_in_flight;
    }

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        tCgC(i) = tCrC(i);
    }
}

template <bool AsyncCopy, int TileM, int TileN, int KTile, int Stages>
cudaError_t launch_gather_syrk_batched_kernel_impl(
    const float* x,
    int64_t x_ld,
    const int32_t* indices,
    int64_t indices_batch_stride,
    float* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    static_assert(TileM == 32, "this CuTe kernel currently supports TileM=32 only");
    static_assert(TileN == 32, "this CuTe kernel currently supports TileN=32 only");
    static_assert(KTile == 32, "this CuTe kernel currently supports KTile=32 only");
    static_assert(Stages >= 1 && Stages <= 4, "supported stage counts are 1..4");

    if (get_cuda_device_arch() < 800u) {
        return cudaErrorNotSupported;
    }

    const dim3 grid(
        static_cast<unsigned>(batch_count),
        static_cast<unsigned>(32 / TileM),
        static_cast<unsigned>(32 / TileN)
    );
    const dim3 block(CuteMmaConfig<TileM, TileN>::kThreads);
    const int shared_bytes = static_cast<int>(sizeof(SharedStorage<TileM, TileN, KTile, Stages>));

    auto kernel = cute_gather_gemm_kernel<AsyncCopy, TileM, TileN, KTile, Stages>;
    cudaError_t status = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes
    );
    if (status != cudaSuccess) {
        return status;
    }

    kernel<<<grid, block, shared_bytes, stream>>>(
        x,
        x_ld,
        indices,
        indices_batch_stride,
        out,
        out_ld,
        out_batch_stride,
        num_rows,
        batch_count
    );
    return cudaPeekAtLastError();
}

template <int TileM, int TileN, int KTile, int Stages>
cudaError_t launch_gather_syrk_batched_kernel(
    const float* x,
    int64_t x_ld,
    const int32_t* indices,
    int64_t indices_batch_stride,
    float* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    return launch_gather_syrk_batched_kernel_impl<false, TileM, TileN, KTile, Stages>(
        x,
        x_ld,
        indices,
        indices_batch_stride,
        out,
        out_ld,
        out_batch_stride,
        num_rows,
        batch_count,
        stream
    );
}

template <int TileM, int TileN, int KTile, int Stages>
cudaError_t launch_gather_syrk_batched_kernel_cpasync(
    const float* x,
    int64_t x_ld,
    const int32_t* indices,
    int64_t indices_batch_stride,
    float* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    return launch_gather_syrk_batched_kernel_impl<true, TileM, TileN, KTile, Stages>(
        x,
        x_ld,
        indices,
        indices_batch_stride,
        out,
        out_ld,
        out_batch_stride,
        num_rows,
        batch_count,
        stream
    );
}

template <bool AsyncCopy, int KTile, int Stages>
__launch_bounds__(CuteSimtMmaConfig<32, 32>::kThreads)
__global__ void gather_syrk_fp32_cute_kernel(
    const float* __restrict__ x,
    int64_t x_ld,
    const int32_t* __restrict__ indices,
    int64_t indices_batch_stride,
    float* __restrict__ out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count
) {
    static_assert(KTile > 0 && Stages > 0, "invalid FP32 CuTe kernel configuration");

    using namespace cute;

    constexpr int kCols = 32;
    using Config = CuteSimtMmaConfig<kCols, kCols>;
    using TiledMma = typename Config::TiledMma;
    using SmemLayout = PanelLayout<kCols, KTile, Stages>;

    const int batch = static_cast<int>(blockIdx.x);
    if (batch >= batch_count) {
        return;
    }

    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& storage = *reinterpret_cast<SharedStorage<kCols, kCols, KTile, Stages>*>(smem_raw);
    Tensor sPanel = make_tensor(make_smem_ptr(storage.panel.begin()), SmemLayout{});

    const int tid = static_cast<int>(threadIdx.x);
    const int32_t* batch_indices = indices + static_cast<int64_t>(batch) * indices_batch_stride;
    for (int i = tid; i < kCols; i += static_cast<int>(blockDim.x)) {
        storage.idx[i] = batch_indices[i];
    }
    __syncthreads();

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    float* out_batch = out + static_cast<int64_t>(batch) * out_batch_stride;
    Tensor gC = make_tensor(
        make_gmem_ptr(out_batch),
        make_layout(
            make_shape(Int<kCols>{}, Int<kCols>{}),
            make_stride(Int<1>{}, out_ld)
        )
    );
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    const int total_tiles = (num_rows + KTile - 1) / KTile;
    int next_tile = 0;
    int tiles_in_flight = 0;
    int smem_pipe_read = 0;
    int smem_pipe_write = 0;

    const int prefetch = Stages > 1 ? (Stages - 1) : 0;
    for (int i = 0; i < prefetch && next_tile < total_tiles; ++i) {
        issue_panel_stage_loads<AsyncCopy, kCols, KTile, Stages>(x, x_ld, storage.idx, sPanel, i, next_tile * KTile, num_rows);
        cp_async_fence();
        ++next_tile;
        ++tiles_in_flight;
    }
    smem_pipe_write = prefetch % Stages;

    while (next_tile < total_tiles || tiles_in_flight > 0) {
        if (next_tile < total_tiles) {
            issue_panel_stage_loads<AsyncCopy, kCols, KTile, Stages>(x, x_ld, storage.idx, sPanel, smem_pipe_write, next_tile * KTile, num_rows);
            cp_async_fence();
            ++next_tile;
            ++tiles_in_flight;
            smem_pipe_write = (smem_pipe_write + 1) % Stages;
        }

        cp_async_stage_wait<Stages>(tiles_in_flight);
        __syncthreads();

        cooperative_gemm(
            threadIdx.x,
            tiled_mma,
            sPanel(_, _, smem_pipe_read),
            sPanel(_, _, smem_pipe_read),
            tCrC
        );

        __syncthreads();
        smem_pipe_read = (smem_pipe_read + 1) % Stages;
        --tiles_in_flight;
    }

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        tCgC(i) = tCrC(i);
    }
}

template <bool AsyncCopy, int KTile, int Stages>
cudaError_t launch_gather_syrk_batched_kernel_fp32_cute_impl(
    const float* x,
    int64_t x_ld,
    const int32_t* indices,
    int64_t indices_batch_stride,
    float* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    static_assert(KTile >= 32 && KTile <= 128, "supported FP32 CuTe KTile values are 32..128");
    static_assert(Stages >= 1 && Stages <= 4, "supported FP32 CuTe stage counts are 1..4");

    if (get_cuda_device_arch() < 800u) {
        return cudaErrorNotSupported;
    }

    constexpr int kCols = 32;
    const dim3 grid(static_cast<unsigned>(batch_count), 1u, 1u);
    const dim3 block(CuteSimtMmaConfig<kCols, kCols>::kThreads);
    const int shared_bytes = static_cast<int>(sizeof(SharedStorage<kCols, kCols, KTile, Stages>));

    auto kernel = gather_syrk_fp32_cute_kernel<AsyncCopy, KTile, Stages>;
    cudaError_t status = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes
    );
    if (status != cudaSuccess) {
        return status;
    }

    kernel<<<grid, block, shared_bytes, stream>>>(
        x,
        x_ld,
        indices,
        indices_batch_stride,
        out,
        out_ld,
        out_batch_stride,
        num_rows,
        batch_count
    );
    return cudaPeekAtLastError();
}

template <int KTile, int Stages>
cudaError_t launch_gather_syrk_batched_kernel_fp32_cute(
    const float* x,
    int64_t x_ld,
    const int32_t* indices,
    int64_t indices_batch_stride,
    float* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    return launch_gather_syrk_batched_kernel_fp32_cute_impl<false, KTile, Stages>(
        x,
        x_ld,
        indices,
        indices_batch_stride,
        out,
        out_ld,
        out_batch_stride,
        num_rows,
        batch_count,
        stream
    );
}

template <int KTile, int Stages>
cudaError_t launch_gather_syrk_batched_kernel_fp32_cute_cpasync(
    const float* x,
    int64_t x_ld,
    const int32_t* indices,
    int64_t indices_batch_stride,
    float* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    return launch_gather_syrk_batched_kernel_fp32_cute_impl<true, KTile, Stages>(
        x,
        x_ld,
        indices,
        indices_batch_stride,
        out,
        out_ld,
        out_batch_stride,
        num_rows,
        batch_count,
        stream
    );
}

template <bool AsyncCopy, bool Triangular, int KTile, int Stages, int Threads>
__launch_bounds__(Threads)
__global__ void gather_syrk_fp32_simt_kernel(
    const float* __restrict__ x,
    int64_t x_ld,
    const int32_t* __restrict__ indices,
    int64_t indices_batch_stride,
    float* __restrict__ out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count
) {
    static_assert(KTile % 4 == 0, "KTile must be a multiple of 4");
    static_assert(Stages > 0, "Stages must be positive");
    static_assert(Threads % 32 == 0, "Threads must be warp-aligned");

    using namespace cute;

    constexpr int kCols = 32;
    constexpr int kWarps = Threads / 32;
    constexpr int kColsPerThread = kCols / kWarps;
    static_assert(kCols % kWarps == 0, "Thread count must evenly partition the 32 output columns");

    const int batch = static_cast<int>(blockIdx.x);
    if (batch >= batch_count) {
        return;
    }

    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& storage = *reinterpret_cast<SharedStorage<kCols, kCols, KTile, Stages>*>(smem_raw);
    Tensor sPanel = make_tensor(make_smem_ptr(storage.panel.begin()), PanelLayout<kCols, KTile, Stages>{});

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = lane;
    const int col0 = warp * kColsPerThread;

    const int32_t* batch_indices = indices + static_cast<int64_t>(batch) * indices_batch_stride;
    for (int i = tid; i < kCols; i += Threads) {
        storage.idx[i] = batch_indices[i];
    }
    __syncthreads();

    float acc[kColsPerThread];
    #pragma unroll
    for (int i = 0; i < kColsPerThread; ++i) {
        acc[i] = 0.0f;
    }

    const int total_tiles = (num_rows + KTile - 1) / KTile;
    int next_tile = 0;
    int tiles_in_flight = 0;
    int smem_pipe_read = 0;
    int smem_pipe_write = 0;

    const int prefetch = Stages > 1 ? (Stages - 1) : 0;
    for (int i = 0; i < prefetch && next_tile < total_tiles; ++i) {
        issue_panel_stage_loads<AsyncCopy, kCols, KTile, Stages>(x, x_ld, storage.idx, sPanel, i, next_tile * KTile, num_rows);
        cp_async_fence();
        ++next_tile;
        ++tiles_in_flight;
    }
    smem_pipe_write = prefetch % Stages;

    while (next_tile < total_tiles || tiles_in_flight > 0) {
        if (next_tile < total_tiles) {
            issue_panel_stage_loads<AsyncCopy, kCols, KTile, Stages>(x, x_ld, storage.idx, sPanel, smem_pipe_write, next_tile * KTile, num_rows);
            cp_async_fence();
            ++next_tile;
            ++tiles_in_flight;
            smem_pipe_write = (smem_pipe_write + 1) % Stages;
        }

        cp_async_stage_wait<Stages>(tiles_in_flight);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < KTile; ++k) {
            const float a = sPanel(row, k, smem_pipe_read);
            #pragma unroll
            for (int i = 0; i < kColsPerThread; ++i) {
                const int col = col0 + i;
                if constexpr (Triangular) {
                    if (col <= row) {
                        acc[i] += a * sPanel(col, k, smem_pipe_read);
                    }
                } else {
                    acc[i] += a * sPanel(col, k, smem_pipe_read);
                }
            }
        }

        __syncthreads();
        smem_pipe_read = (smem_pipe_read + 1) % Stages;
        --tiles_in_flight;
    }

    float* out_batch = out + static_cast<int64_t>(batch) * out_batch_stride;
    #pragma unroll
    for (int i = 0; i < kColsPerThread; ++i) {
        const int col = col0 + i;
        if constexpr (Triangular) {
            if (col <= row) {
                out_batch[static_cast<int64_t>(row) + static_cast<int64_t>(col) * out_ld] = acc[i];
                if (col != row) {
                    out_batch[static_cast<int64_t>(col) + static_cast<int64_t>(row) * out_ld] = acc[i];
                }
            }
        } else {
            out_batch[static_cast<int64_t>(row) + static_cast<int64_t>(col) * out_ld] = acc[i];
        }
    }
}

template <bool AsyncCopy, bool Triangular, int KTile, int Stages, int Threads>
cudaError_t launch_gather_syrk_batched_kernel_fp32_simt_impl(
    const float* x,
    int64_t x_ld,
    const int32_t* indices,
    int64_t indices_batch_stride,
    float* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    static_assert(KTile >= 32 && KTile <= 128, "supported FP32 SIMT KTile values are 32..128");
    static_assert(Stages >= 1 && Stages <= 4, "supported FP32 SIMT stage counts are 1..4");
    static_assert(Threads == 128 || Threads == 256, "supported FP32 SIMT thread counts are 128 or 256");

    if (get_cuda_device_arch() < 800u) {
        return cudaErrorNotSupported;
    }

    constexpr int kCols = 32;
    const dim3 grid(static_cast<unsigned>(batch_count), 1u, 1u);
    const dim3 block(static_cast<unsigned>(Threads));
    const int shared_bytes = static_cast<int>(sizeof(SharedStorage<kCols, kCols, KTile, Stages>));

    auto kernel = gather_syrk_fp32_simt_kernel<AsyncCopy, Triangular, KTile, Stages, Threads>;
    cudaError_t status = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes
    );
    if (status != cudaSuccess) {
        return status;
    }

    kernel<<<grid, block, shared_bytes, stream>>>(
        x,
        x_ld,
        indices,
        indices_batch_stride,
        out,
        out_ld,
        out_batch_stride,
        num_rows,
        batch_count
    );
    return cudaPeekAtLastError();
}

template <int KTile, int Stages, int Threads>
cudaError_t launch_gather_syrk_batched_kernel_fp32_simt(
    const float* x,
    int64_t x_ld,
    const int32_t* indices,
    int64_t indices_batch_stride,
    float* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    return launch_gather_syrk_batched_kernel_fp32_simt_impl<false, false, KTile, Stages, Threads>(
        x,
        x_ld,
        indices,
        indices_batch_stride,
        out,
        out_ld,
        out_batch_stride,
        num_rows,
        batch_count,
        stream
    );
}

template <int KTile, int Stages, int Threads>
cudaError_t launch_gather_syrk_batched_kernel_fp32_simt_cpasync(
    const float* x,
    int64_t x_ld,
    const int32_t* indices,
    int64_t indices_batch_stride,
    float* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    return launch_gather_syrk_batched_kernel_fp32_simt_impl<true, false, KTile, Stages, Threads>(
        x,
        x_ld,
        indices,
        indices_batch_stride,
        out,
        out_ld,
        out_batch_stride,
        num_rows,
        batch_count,
        stream
    );
}

template <int KTile, int Stages, int Threads>
cudaError_t launch_gather_syrk_batched_kernel_fp32_syrk(
    const float* x,
    int64_t x_ld,
    const int32_t* indices,
    int64_t indices_batch_stride,
    float* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    return launch_gather_syrk_batched_kernel_fp32_simt_impl<false, true, KTile, Stages, Threads>(
        x,
        x_ld,
        indices,
        indices_batch_stride,
        out,
        out_ld,
        out_batch_stride,
        num_rows,
        batch_count,
        stream
    );
}

template <int KTile, int Stages, int Threads>
cudaError_t launch_gather_syrk_batched_kernel_fp32_syrk_cpasync(
    const float* x,
    int64_t x_ld,
    const int32_t* indices,
    int64_t indices_batch_stride,
    float* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    cudaStream_t stream
) {
    return launch_gather_syrk_batched_kernel_fp32_simt_impl<true, true, KTile, Stages, Threads>(
        x,
        x_ld,
        indices,
        indices_batch_stride,
        out,
        out_ld,
        out_batch_stride,
        num_rows,
        batch_count,
        stream
    );
}

inline void reference_gather_syrk_tf32(
    const std::vector<float>& x,
    int64_t x_ld,
    const std::vector<int32_t>& indices,
    int64_t indices_batch_stride,
    std::vector<float>* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    int cols
) {
    out->assign(static_cast<size_t>(out_batch_stride) * static_cast<size_t>(batch_count), 0.0f);
    for (int batch = 0; batch < batch_count; ++batch) {
        const int32_t* batch_indices = indices.data() + static_cast<int64_t>(batch) * indices_batch_stride;
        float* out_batch = out->data() + static_cast<int64_t>(batch) * out_batch_stride;
        for (int col_j = 0; col_j < cols; ++col_j) {
            const int32_t feature_j = batch_indices[col_j];
            for (int col_i = 0; col_i < cols; ++col_i) {
                const int32_t feature_i = batch_indices[col_i];
                double acc = 0.0;
                for (int row = 0; row < num_rows; ++row) {
                    const float a = feature_i >= 0
                        ? round_to_tf32(x[static_cast<int64_t>(row) + static_cast<int64_t>(feature_i) * x_ld])
                        : 0.0f;
                    const float b = feature_j >= 0
                        ? round_to_tf32(x[static_cast<int64_t>(row) + static_cast<int64_t>(feature_j) * x_ld])
                        : 0.0f;
                    acc += static_cast<double>(a) * static_cast<double>(b);
                }
                out_batch[static_cast<int64_t>(col_i) + static_cast<int64_t>(col_j) * out_ld] =
                    static_cast<float>(acc);
            }
        }
    }
}

inline void reference_gather_syrk_fp32(
    const std::vector<float>& x,
    int64_t x_ld,
    const std::vector<int32_t>& indices,
    int64_t indices_batch_stride,
    std::vector<float>* out,
    int64_t out_ld,
    int64_t out_batch_stride,
    int32_t num_rows,
    int32_t batch_count,
    int cols
) {
    out->assign(static_cast<size_t>(out_batch_stride) * static_cast<size_t>(batch_count), 0.0f);
    for (int batch = 0; batch < batch_count; ++batch) {
        const int32_t* batch_indices = indices.data() + static_cast<int64_t>(batch) * indices_batch_stride;
        float* out_batch = out->data() + static_cast<int64_t>(batch) * out_batch_stride;
        for (int col_j = 0; col_j < cols; ++col_j) {
            const int32_t feature_j = batch_indices[col_j];
            for (int col_i = 0; col_i < cols; ++col_i) {
                const int32_t feature_i = batch_indices[col_i];
                double acc = 0.0;
                for (int row = 0; row < num_rows; ++row) {
                    const float a = feature_i >= 0
                        ? x[static_cast<int64_t>(row) + static_cast<int64_t>(feature_i) * x_ld]
                        : 0.0f;
                    const float b = feature_j >= 0
                        ? x[static_cast<int64_t>(row) + static_cast<int64_t>(feature_j) * x_ld]
                        : 0.0f;
                    acc += static_cast<double>(a) * static_cast<double>(b);
                }
                out_batch[static_cast<int64_t>(col_i) + static_cast<int64_t>(col_j) * out_ld] =
                    static_cast<float>(acc);
            }
        }
    }
}

inline float max_abs_diff(
    const std::vector<float>& lhs,
    const std::vector<float>& rhs
) {
    const size_t n = std::min(lhs.size(), rhs.size());
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(lhs[i] - rhs[i]));
    }
    return max_diff;
}

}  // namespace kermac_cute_gather_syrk
