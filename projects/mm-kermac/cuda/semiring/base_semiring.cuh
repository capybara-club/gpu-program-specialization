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

#include <ptx_inject.h>
#include <cute/tensor.hpp>

enum class Majorness {
    COL_MAJOR,
    ROW_MAJOR
};

enum class Alignment {
    ALIGN_1,
    ALIGN_4
};

enum class KernelMatrixPackedType {
    FULL,
	UPPER_TRIANGLE,
	LOWER_TRIANGLE,
	STAGGERED
};

// Disabled shared-storage staging layout; kept for reference during layout experiments.
#if 0
template <
class SmemLayoutA,
class SmemLayoutB,
class T
>
struct SharedStorageNorm
{
    alignas(16) cute::ArrayEngine<T, cute::cosize_v<SmemLayoutA>> A;
    alignas(16) cute::ArrayEngine<T, cute::cosize_v<SmemLayoutB>> B;
};
#endif

__host__
__device__
__forceinline__
double 
fast_sqrt(
    double theta
) {
    #if defined(__CUDA_ARCH__)
        return ::sqrt(theta);      // device side
    #else
        return std::sqrt(theta);   // host side
    #endif
}

__device__ 
__forceinline__
void 
syrk_lower_index_from_linear(
    int32_t k, 
    int32_t &row, 
    int32_t &col
) {
  // Solve T(row) <= k < T(row+1), where T(r) = r*(r+1)/2
  // row = floor( (sqrt(8*k + 1) - 1) / 2 )
  double x = 8.0 * static_cast<double>(k) + 1.0;
  double r = floor((fast_sqrt(x) - 1.0) * 0.5);

  int32_t i = static_cast<int32_t>(r);
  int64_t ii = static_cast<int64_t>(i);
  int64_t tri = ii * (ii + 1) / 2;  // T(i)

  row = i;
  col = k - static_cast<int32_t>(tri);
}

template <
    bool predicate_reads,
    bool predicate_writes,
    KernelMatrixPackedType kernel_matrix_packed_type,
    class ProblemShape, class CtaTiler, class ThreadTiler,
    class AStride, class ASmemLayout, class TiledCopyA,
    class BStride, class BSmemLayout, class TiledCopyB,
    class CStride, class CSmemLayout,
    class HYPER0Stride,
    class HYPER1Stride,
    class HYPER2Stride,
    class HYPER3Stride,
    class T
>
__device__
__forceinline__
void
kernel_cute_semiring(
    ProblemShape shape_MNKL, CtaTiler cta_tiler, ThreadTiler thread_tiler,
    T const *A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    T const *B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    T       *C, CStride dC, CSmemLayout,
    T *HYPER0, HYPER0Stride dHYPER0,
    T *HYPER1, HYPER1Stride dHYPER1,
    T *HYPER2, HYPER2Stride dHYPER2,
    T *HYPER3, HYPER3Stride dHYPER3
) {
    using namespace cute;

    // Preconditions
    CUTE_STATIC_ASSERT_V(rank(shape_MNKL) == Int<4>{}); // (M, N, K, L)
    CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{}); // (BLK_M, BLK_N, BLK_K)
    CUTE_STATIC_ASSERT_V(rank(thread_tiler) == Int<2>{}); // (THR_M, THR_N)

    CUTE_STATIC_ASSERT_V(size(copy_a) == size(thread_tiler)); // NumThreads
    CUTE_STATIC_ASSERT_V(size(copy_b) == size(thread_tiler)); // NumThreads

    static_assert(is_static<ASmemLayout>::value);
    static_assert(is_static<BSmemLayout>::value);
    static_assert(is_static<CSmemLayout>::value);

    CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
    CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M

    CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
    CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N

    CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
    CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

    CUTE_STATIC_ASSERT_V(congruent(select<0,2,3>(shape_MNKL), dA));       // dA strides for shape MK
    CUTE_STATIC_ASSERT_V(congruent(select<1,2,3>(shape_MNKL), dB));       // dB strides for shape NK
    CUTE_STATIC_ASSERT_V(congruent(select<0,1,3>(shape_MNKL), dC));       // dC strides for shape MNO

    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2,3>(shape_MNKL), dA); // (M,K,L)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2,3>(shape_MNKL), dB); // (N,K,L)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1,3>(shape_MNKL), dC); // (M,N,L)

    auto bidx = blockIdx.x;
    auto bidy = blockIdx.y;
    auto bidz = blockIdx.z;

    Tensor mHYPER0 = make_tensor(make_gmem_ptr(HYPER0), select<3>(shape_MNKL), dHYPER0);
    Tensor mHYPER1 = make_tensor(make_gmem_ptr(HYPER1), select<3>(shape_MNKL), dHYPER1);
    Tensor mHYPER2 = make_tensor(make_gmem_ptr(HYPER2), select<3>(shape_MNKL), dHYPER2);
    Tensor mHYPER3 = make_tensor(make_gmem_ptr(HYPER3), select<3>(shape_MNKL), dHYPER3);

    T hyper0 = HYPER0 ? mHYPER0(bidz) : T(0);
    T hyper1 = HYPER1 ? mHYPER1(bidz) : T(0);
    T hyper2 = HYPER2 ? mHYPER2(bidz) : T(0);
    T hyper3 = HYPER3 ? mHYPER3(bidz) : T(0);

    if constexpr (kernel_matrix_packed_type != KernelMatrixPackedType::FULL) {
        __shared__ int s_bidx, s_bidy;
        if (threadIdx.x == 0) {
            syrk_lower_index_from_linear(bidx, s_bidx, s_bidy);  // uses double sqrt
        }
        __syncthreads();

        if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::LOWER_TRIANGLE) {
            bidx = s_bidx;
            bidy = s_bidy;
        }
        if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::UPPER_TRIANGLE) {
            bidx = s_bidy;
            bidy = s_bidx;
        }
    }

    auto cta_coord = make_coord(bidx, bidy, _); // (m,n,k)
    Tensor gA = local_tile(mA(_,_,bidz), cta_tiler, cta_coord, Step<_1,  X, _1>{}); // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB(_,_,bidz), cta_tiler, cta_coord, Step< X, _1, _1>{}); // (BLK_N,BLK_K,k)
    Tensor gC = local_tile(mC(_,_,bidz), cta_tiler, cta_coord, Step<_1, _1,  X>{}); // (BLK_M,BLK_N)

    auto m_max_coord = size<0>(shape_MNKL) - size<0>(gA) * bidx;  // M - BLK_M * m_coord
    auto n_max_coord = size<1>(shape_MNKL) - size<0>(gB) * bidy;  // N - BLK_N * n_coord
    auto k_residue   = size<2>(shape_MNKL) - size<1>(gA) * size<2>(gA); // K - BLK_K * k_coord_max

    // Need to get the tile count before the offsetting in gA, gB of the k_residue
    int k_tile_count = 0;
    {
        ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
        Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
        // Total count of tiles
        k_tile_count = size<3>(tAgA);
    }

    if constexpr (predicate_reads) {
        // Shift tensor so residue_k is at origin (Can't read any k_coord < residue_k)
        // This aligns the tensor with BLK_K for all but the 0th k_tile
        gA = cute::domain_offset(make_coord(0, k_residue, 0), gA);
        gB = cute::domain_offset(make_coord(0, k_residue, 0), gB);
    }

#if 1
    alignas(16) __shared__ T smem_a[cosize_v<ASmemLayout>];
    alignas(16) __shared__ T smem_b[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smem_a), sA_layout);   // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem_b), sB_layout);   // (BLK_N,BLK_K,PIPE)

#else
    // Shared memory buffers
    extern __shared__ char shared_memory[];
    using SharedStorage = SharedStorageNorm<ASmemLayout, BSmemLayout, T>;
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
    Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K,PIPE)
#endif

    // Tiled copy setups
    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thr_copy_a.partition_D(sA); // (CPY,CPY_M,CPY_K,PIPE)

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); // (CPY,CPY_N,CPY_K,k)
    Tensor tBsB = thr_copy_b.partition_D(sB); // (CPY,CPY_N,CPY_K,PIPE)

    // Pipe size
    auto K_PIPE_MAX = size<3>(tAsA);
    // Current tile index in gmem to read from
    int k_tile_next = 0;
    
    CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA)); // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA)); // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB)); // CPY_N
    CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB)); // CPY_K
    CUTE_STATIC_ASSERT_V(K_PIPE_MAX == size<3>(tAsA)); // PIPE A
    CUTE_STATIC_ASSERT_V(K_PIPE_MAX == size<3>(tBsB)); // PIPE B

    // Partition the tensors
    Tensor tCsA = local_partition(sA, thread_tiler, threadIdx.x, Step<_1,  X>{}); // (THR_M,THR_K,PIPE)
    Tensor tCsB = local_partition(sB, thread_tiler, threadIdx.x, Step< X, _1>{}); // (THR_N,THR_K,PIPE)
    Tensor tCgC = local_partition(gC, thread_tiler, threadIdx.x, Step<_1, _1>{}); // (THR_M,THR_N)

    Tensor tCrA = make_fragment_like(tCsA(_,_,0)); // (THR_M,THR_K)
    Tensor tCrB = make_fragment_like(tCsB(_,_,0)); // (THR_N,THR_K)
    Tensor tCrC = make_fragment_like(tCgC);        // (THR_M,THR_N)

    // Create coordinate tensors for the problem for predication
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA))); // (M,K) -> (m,k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB))); // (N,K) -> (n,k)
    Tensor cC = make_identity_tensor(make_shape(size<0>(gC), size<1>(gC))); // (M,N) -> (m,n)

    // Partition coordinate tensors for predication
    Tensor tAcA = thr_copy_a.partition_S(cA);
    Tensor tBcB = thr_copy_b.partition_S(cB);
    Tensor tCcC = local_partition(cC, thread_tiler, threadIdx.x, Step<_1, _1>{});

    // Current pipe index in smem to read from
    int smem_pipe_read  = 0;
    // Current pipe index in smem to write to
    int smem_pipe_write = K_PIPE_MAX-1;

    // Pipe slice
    Tensor tCsA_p = tCsA(_,_,smem_pipe_read);
    Tensor tCsB_p = tCsB(_,_,smem_pipe_read);

    // Size of the register pipeline
    auto K_BLOCK_MAX = size<1>(tCrA);

    Tensor tApA = make_tensor<bool>(
        make_shape(size<1>(tAsA), size<2>(tAsA)),
        make_stride(Int<1>{}, Int<0>{})
    );
    Tensor tBpB = make_tensor<bool>(
        make_shape(size<1>(tBsB), size<2>(tBsB)),
        make_stride(Int<1>{}, Int<0>{})
    );

    if constexpr (predicate_reads) {
        // Generate the in-bounds/out-of-bounds coordinates for each tensor as a bool predicate
        CUTE_UNROLL
        for (int m = 0; m < size<0>(tApA); m++) {
            tApA(m,0) = get<0>(tAcA(0,m,0)) < m_max_coord;
        }
        CUTE_UNROLL
        for (int n = 0; n < size<0>(tBpB); n++) {
            tBpB(n,0) = get<0>(tBcB(0,n,0)) < n_max_coord;
        }
    }

    // Print all tensor shapes/data here before anything functionally happens such as copies

    if constexpr (predicate_reads) {
        // Clear the smem tiles to account for predicated off loads 
        clear(tAsA);
        clear(tBsB);

        // Start async loads for 0th k-tile, where we take care of the k-residue
        // We already shifted the global memory coordinate over to account for the k-residue
        {
            constexpr int k_pipe = 0;
    
            Tensor tAgAk = tAgA(_,_,_,k_tile_next);
            CUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < size<2>(tAsA); ++k) {
                if (get<1>(tAcA(0,0,k)) >= -k_residue) { // blk_k coord < residue_k (gA shifted)
                    copy_if(copy_a, tApA(_,k), tAgAk(_,_,k), tAsA(_,_,k,k_pipe));
                }
            }
            Tensor tBgBk = tBgB(_,_,_,k_tile_next);
            for (int k = 0; k < size<2>(tBsB); ++k) {
                if (get<1>(tBcB(0,0,k)) >= -k_residue) { // blk_k coord < residue_k (gB shifted)
                    copy_if(copy_b, tBpB(_,k), tBgBk(_,_,k), tBsB(_,_,k,k_pipe));
                }
            }
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_next; }
        }

        // Start async loads for 1st k-tile onwards, no k-residue handling needed
        // Do this for all but the last pipe. Each mainloop iter will schedule a pipeline copy.
        CUTE_UNROLL
        for (int k_pipe = 1; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
            if (k_tile_count <= 0) {
                clear(tApA);
                clear(tBpB);
            }
            copy_if(copy_a, tApA, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
            copy_if(copy_b, tBpB, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_next; }
        }
    } else { // No predicated reads
        // Start async loads for 0th k-tile onwards, no k-residue handling needed
        // Do this for all but the last pipe. Each mainloop iter will schedule a copy.
        CUTE_UNROLL
        for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
            copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
            copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
            cp_async_fence();
            --k_tile_count;
            if (k_tile_count > 0) { ++k_tile_next; }
        }
    }

    // Clear accumulators
    clear(tCrC);

    // PREFETCH register pipeline
    if (K_BLOCK_MAX > 1) {
        // Wait until our first prefetched tile is loaded in
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
    
        // Prefetch the first rmem from the first k-tile
        copy(tCsA_p(_,Int<0>{}), tCrA(_,Int<0>{}));
        copy(tCsB_p(_,Int<0>{}), tCrB(_,Int<0>{}));
    }

    // Main LOOP!
    CUTE_NO_UNROLL
    while (k_tile_count > -(K_PIPE_MAX-1)) {
        CUTE_UNROLL
        for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
            if (k_block == K_BLOCK_MAX - 1) {
                // Slice the smem_pipe_read smem
                tCsA_p = tCsA(_,_,smem_pipe_read);
                tCsB_p = tCsB(_,_,smem_pipe_read);

                // Commit the smem for smem_pipe_read
                cp_async_wait<K_PIPE_MAX-2>();
                __syncthreads();
            }

            // Load A, B shmem->regs for k_block+1
            auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;
            copy(tCsA_p(_,k_block_next), tCrA(_,k_block_next));
            copy(tCsB_p(_,k_block_next), tCrB(_,k_block_next));
            // Copy gmem to smem before computing gemm on each k-pipe
            if (k_block == 0) {
                if constexpr (predicate_reads) {
                    // Set all predicates to false if we are going to overshoot bounds
                    if (k_tile_count <= 0) {
                        clear(tApA);
                        clear(tBpB);
                    }
                    copy_if(copy_a, tApA, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
                    copy_if(copy_b, tBpB, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
                } else { // No predicated reads
                    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
                    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
                }
                cp_async_fence();

                // Advance the gmem tile
                --k_tile_count;
                if (k_tile_count > 0) { ++k_tile_next; }

                // Advance the smem pipe
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
            }

            // Apply inner
            CUTE_UNROLL
            for (int m = 0; m < size<0>(tCrC); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(tCrC); n++) {
                    PTX_INJECT("mma",
                        PTX_IN(F32, a, tCrA(m,k_block)),
                        PTX_IN(F32, b, tCrB(n,k_block)),
                        PTX_IN(F32, h_0, hyper0),
                        PTX_IN(F32, h_1, hyper1),
                        PTX_IN(F32, h_2, hyper2),
                        PTX_IN(F32, h_3, hyper3),
                        PTX_MOD(F32, c, tCrC(m,n))
                    );
                }
            }
        }
    }
   
    auto block_coord_x = size<0>(gA) * bidx;
    auto block_coord_y = size<0>(gB) * bidy;
    // Apply outer
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); i++) {
        auto thread_coord_x = get<0>(tCcC(i));
        auto thread_coord_y = get<1>(tCcC(i));
        auto full_coord_x = block_coord_x + thread_coord_x;
        auto full_coord_y = block_coord_y + thread_coord_y;
        uint32_t is_diag = full_coord_x == full_coord_y;
        PTX_INJECT("epilogue",
            PTX_IN(U32, is_diag, is_diag),
            PTX_IN(F32, h_0, hyper0),
            PTX_IN(F32, h_1, hyper1),
            PTX_IN(F32, h_2, hyper2),
            PTX_IN(F32, h_3, hyper3),
            PTX_MOD(F32, e, tCrC(i))
        );
    }

    // Write accumulators
    if constexpr (predicate_writes) {
        auto bounds = make_coord(m_max_coord,n_max_coord);
        
        if constexpr (kernel_matrix_packed_type != KernelMatrixPackedType::FULL) {
            const bool is_diag_block = (bidx == bidy);
            if (is_diag_block) {
                CUTE_UNROLL
                for (int i = 0; i < size(tCrC); i++) {
                    auto coord = tCcC(i);
                    auto in_bounds = elem_less(coord, bounds);
                    auto m = get<0>(coord);
                    auto n = get<1>(coord);
                    if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::LOWER_TRIANGLE) {
                        in_bounds = in_bounds && (m >= n);
                    }
                    if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::UPPER_TRIANGLE) {
                        in_bounds = in_bounds && (n >= m);
                    }
                    if (in_bounds) {
                        tCgC(i) = tCrC(i);
                    }
                }
                return;
            }
        }
        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); i++) {
            if (elem_less(tCcC(i), bounds)) {
                tCgC(i) = tCrC(i);
            }
        }
    } else {
        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); i++) {
            tCgC(i) = tCrC(i);
        }
    }
}

template <
    Majorness majorness,
    typename LdType, 
    typename BatchStrideType
>
__device__
__forceinline__
auto 
make_general_stride(
    LdType ld, 
    BatchStrideType batch_stride
) {
    using namespace cute;
    if constexpr (majorness == Majorness::COL_MAJOR) {
        return make_stride(Int<1>{}, ld, batch_stride);
    } else {
        return make_stride(ld, Int<1>{}, batch_stride);
    }
};

template <
    Majorness majorness, 
    typename Dim1Type, 
    typename Dim2Type, 
    typename Dim3Type
>
__device__
__forceinline__
auto 
make_general_layout(
    Dim1Type dim1, 
    Dim2Type dim2, 
    Dim3Type dim3
) {
    using namespace cute;
    if constexpr (majorness == Majorness::COL_MAJOR) {
        return make_layout(make_shape(dim1, dim2, dim3));
    } else {
        auto atom = make_layout(
            make_shape(dim1, dim2),
            make_stride(Int<1>{}, dim1 + Int<4>{})
        );
        return tile_to_shape(atom, make_shape(dim1, dim2, dim3));
    }
};

template <
    typename T, 
    Majorness majorness, 
    Alignment align
>
__device__
__forceinline__
auto 
make_general_tiled_copy() {
    using namespace cute;
    if constexpr (majorness == Majorness::COL_MAJOR) {
        if constexpr (align == Alignment::ALIGN_4) {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, T>{},
                Layout<Shape<_32, _8>>{},
                Layout<Shape<_4, _1>>{}
            );
        } else {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<T>, T>{},
                Layout<Shape<_32, _8>>{},
                Layout<Shape<_1, _1>>{}
            );
        }
    } else {
        return make_tiled_copy(
            Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<T>, T>{},
            Layout<Shape<_32,_8>, Stride<_8,_1>>{},
            Layout<Shape<_1,_1>>{}
        );
    }
};

template<
    Majorness majorness_A,
    Majorness majorness_B,
    Alignment align_A,
    Alignment align_B,
    KernelMatrixPackedType kernel_matrix_packed_type
>
__device__
__forceinline__
void
cute_semiring(
    int64_t m, int64_t n, int64_t k, int64_t l,
    float const *A, int64_t ldA,            int64_t batch_stride_A,
    float const *B, int64_t ldB,            int64_t batch_stride_B,
    float       *C, int64_t ldC,            int64_t batch_stride_C,
    float       *HYPER_0,                   int64_t batch_stride_hyper_0,
    float       *HYPER_1,                   int64_t batch_stride_hyper_1,
    float       *HYPER_2,                   int64_t batch_stride_hyper_2,
    float       *HYPER_3,                   int64_t batch_stride_hyper_3
) {
    // Don't support ALIGN_4 specialization for ROW_MAJOR tensors
    static_assert(majorness_A != Majorness::ROW_MAJOR || align_A != Alignment::ALIGN_4);
    static_assert(majorness_B != Majorness::ROW_MAJOR || align_B != Alignment::ALIGN_4);

    using namespace cute;
    using T = float;

    auto M = uint64_t(m);
    auto N = uint64_t(n);
    auto K = uint64_t(k);
    auto L = uint64_t(l);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK);
    auto bP = Int<3>{};
    
    auto prob_shape = make_shape(M,N,K,L);
    auto dA = make_general_stride<majorness_A>(ldA, batch_stride_A);
    auto dB = make_general_stride<majorness_B>(ldB, batch_stride_B);
    auto dC = make_general_stride<Majorness::COL_MAJOR>(ldC, batch_stride_C);

    auto d_hyper_0 = make_stride(batch_stride_hyper_0);
    auto d_hyper_1 = make_stride(batch_stride_hyper_1);
    auto d_hyper_2 = make_stride(batch_stride_hyper_2);
    auto d_hyper_3 = make_stride(batch_stride_hyper_3);

    auto sA = make_general_layout<majorness_A>(bM, bK, bP);
    auto sB = make_general_layout<majorness_B>(bN, bK, bP);
    auto sC = make_layout(make_shape(bM, bN));

    auto copyA = make_general_tiled_copy<T, majorness_A, align_A>();
    auto copyB = make_general_tiled_copy<T, majorness_B, align_B>();

    auto thread_tiler = Layout<Shape<_16,_16>>{}; // M, N

    kernel_cute_semiring<
        true, true,
        kernel_matrix_packed_type
    > (
        prob_shape, cta_tiler, thread_tiler, 
        A, dA, sA, copyA,
        B, dB, sB, copyB,
        C, dC, sC,
        HYPER_0, d_hyper_0,
        HYPER_1, d_hyper_1,
        HYPER_2, d_hyper_2,
        HYPER_3, d_hyper_3
    );
}
