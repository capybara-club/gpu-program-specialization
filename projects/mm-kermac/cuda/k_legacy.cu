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
#include <stdio.h>
#include <stdint.h>

#include <curand.h>
#include <curand_kernel.h>

template <typename T> constexpr T c_one = T(1.0);
template <typename T> constexpr T c_zero = T(0.0);
template <typename T> constexpr T c_two = T(2.0);
template <typename T> constexpr T c_negative_one = T(-1.0);
template <typename T> constexpr T c_negative_two = T(-2.0);

#define KERNEL_TILE_DIM 16
#define KERNEL_BLOCK_ROWS 16

// Staggered means to alternate between upper and lower triangle
enum class KernelMatrixPackedType {
    FULL,
	UPPER_TRIANGLE,
	LOWER_TRIANGLE,
	STAGGERED
};

enum class NormType {
    L1,
    L2,
    P
};

// Set regularization, set diagonal to 1.0 and 0.0 for 
// value and grad respectively
enum class Symmetric {
	YES,
	NO
};

// Copy to the other triangle the indicated values
enum class SymmetricCopy {
	NO,
	VALUE,
	GRAD_VALUE
};

// Compute the value or the gradient value
enum class KernelValue {
	VALUE,
	GRAD_VALUE
};

// divide by the number of samples
enum class NormalizeByNumSamples {
	YES,
	NO
};

template <class T>
static
__forceinline__
__device__
T
_abs(
    T v
) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    if constexpr (std::is_same_v<T, float>) {
        return fabsf(v);
    } else if constexpr (std::is_same_v<T, double>) {
        return fabs(v);
    }
}

// Forces sqrt.approx for float. Skips having to use --use_fast_math
template <class T>
static
__forceinline__
__device__
T
_sqrt(
    T v
) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    if constexpr (std::is_same_v<T, float>) {
        float result;
        asm(
            "sqrt.approx.ftz.f32 %0, %1;\n\t" 
            : "=f"(result) : "f"(v)
        );
        return result;
    } else if constexpr (std::is_same_v<T, double>) {
        return sqrt(v);
    }
}

// Forces mul and ex2.approx for float. Skips having to use --use_fast_math
template <class T>
static
__forceinline__
__device__
T 
_exp(
	T v
) {
	static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
	if constexpr (std::is_same_v<T,float>) {
        float result;
        asm(
            "mul.ftz.f32 %0, %1, 0f3FB8AA3B;\n\t"
            "ex2.approx.ftz.f32 %0, %0;\n\t" 
            : "=f"(result) : "f"(v)
        );
        return result;
    } else if constexpr (std::is_same_v<T,double>) {
        return exp(v);
    }
}


// Forces lg2.approx, mul and ex2.approx for float. Skips having to use --use_fast_math
template <class T>
static
__forceinline__
__device__
T 
_pow(
	T v,
    T p
) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    if constexpr (std::is_same_v<T,float>) {
        float result;
        asm(
            "lg2.approx.ftz.f32 %0, %1;\n\t"
            "mul.ftz.f32 %0, %0, %2;\n\t"
            "ex2.approx.ftz.f32 %0, %0;\n\t"
            : "=f"(result) : "f"(v), "f"(p)
        );
        return result;
    } else if constexpr (std::is_same_v<T,double>) {
        return pow(v,p);
    }
}

template <class T>
static
__forceinline__
__device__
T
_nan() {
	static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

	if constexpr (std::is_same_v<T,float>) {
		return nanf("");
	} else if constexpr (std::is_same_v<T,double>) {
		return nan("");
	}
}

template <class T>
static
__forceinline__
__device__
T 
_signum(
    T v
) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    if constexpr (std::is_same_v<T, float>) {
		return copysign(1.0f, v);
	} else if constexpr (std::is_same_v<T, double>) {
		return copysign(1.0, v);
	}
}

template <
	KernelMatrixPackedType kernel_matrix_packed_type
>
static
__forceinline__
__device__
bool
_kernel_matrix_pass(
	uint64_t M,
	uint64_t N,
	uint64_t row,
	uint64_t column,
	uint64_t batch_num
) {
	if ( row >= M || column >= N) return false;

	bool is_even = batch_num % 2 == 0;

	if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::FULL) {
		return true;
	} else if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::UPPER_TRIANGLE) { 
		return column >= row;
	} else if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::LOWER_TRIANGLE) {
		return row >= column;
	} else if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::STAGGERED) {
		if (is_even && row >= column) return true;
		if (!is_even && column >= row) return true;
		return false;
	}
}

template <
	KernelMatrixPackedType kernel_matrix_packed_type
>
static
__forceinline__
__device__
uint64_t
_kernel_matrix_coords(
	uint64_t M,
	uint64_t N, 
	uint64_t ldM,
	uint64_t ldMN,
	uint64_t row,
	uint64_t column,
	uint64_t batch_num
) {
	if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::FULL) {
		uint64_t idx = batch_num * ldMN + column * ldM + row;
		return idx;
	} else if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::UPPER_TRIANGLE) {
		uint64_t idx = batch_num * ldMN + column * ldM + row;
		return idx;
	} else if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::LOWER_TRIANGLE) {
		uint64_t idx = batch_num * ldMN + column * ldM + row;
		return idx;
	} else if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::STAGGERED) {
		bool is_even = batch_num % 2 == 0;
		uint64_t this_batch_slice = batch_num / 2;
		uint64_t triangle_shift = is_even ? 0 : 1;
		uint64_t idx = this_batch_slice * ldMN + (column + triangle_shift)*ldM + row;
		return idx;
	}
}

template <
	KernelMatrixPackedType kernel_matrix_packed_type,
	class T
>
static
__device__
__forceinline__
void
kernel_copy_diagonals(
	uint64_t N,
	const T* const __restrict__ A,     uint64_t ldA, uint64_t batch_stride_A,
	T* __restrict__ V,                 uint64_t ldV
) {
	static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

	const uint64_t b = blockIdx.z;
	const uint64_t row = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t column = row;

	if (row < N) {
		uint64_t idx = _kernel_matrix_coords<kernel_matrix_packed_type>(N, N, ldA, batch_stride_A, row, column, b);
		V[b * ldV + row] = A[idx];
	}
}

extern "C"
__global__
void
kernel_copy_diagonals_full_f32(
    uint64_t N,
	const float* const __restrict__ A,   uint64_t ldA, uint64_t batch_stride_A,
	float* __restrict__ V,               uint64_t ldV
) {
    kernel_copy_diagonals<KernelMatrixPackedType::FULL, float>(N, A, ldA, batch_stride_A, V, ldV);
}

template <
	KernelMatrixPackedType kernel_matrix_packed_type,
	class T
>
static
__device__
__forceinline__
void
kernel_copy_triangles(
	uint64_t N,
	T diagonal_value,
	T* __restrict__ A, uint64_t ldA, uint64_t batch_stride_A
) {
	static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
	static_assert( 
		kernel_matrix_packed_type == KernelMatrixPackedType::LOWER_TRIANGLE || 
		kernel_matrix_packed_type == KernelMatrixPackedType::UPPER_TRIANGLE
	);

	const uint64_t b = blockIdx.z;

	__shared__ T tile[KERNEL_TILE_DIM][KERNEL_TILE_DIM + 1];

	uint64_t row_M = blockIdx.x * KERNEL_TILE_DIM + threadIdx.x;
	uint64_t column = blockIdx.y * KERNEL_TILE_DIM + threadIdx.y;

	uint64_t row_M_symmetric = blockIdx.y * KERNEL_TILE_DIM + threadIdx.x;
	uint64_t column_symmetric = blockIdx.x * KERNEL_TILE_DIM + threadIdx.y;
	uint64_t symmetric_idx = b * batch_stride_A + column_symmetric * ldA + row_M_symmetric;

	if ( _kernel_matrix_pass<kernel_matrix_packed_type>(N, N, row_M, column, b) ) {
		uint64_t idx = _kernel_matrix_coords<kernel_matrix_packed_type>(N, N, ldA, batch_stride_A, row_M, column, b);
		T v = A[idx];
		tile[threadIdx.y][threadIdx.x] = v;
	}

	__syncthreads();

	if (row_M_symmetric < N && column_symmetric < N) {
		if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::LOWER_TRIANGLE) {
			if (column_symmetric >= row_M_symmetric) {
				A[symmetric_idx] = column_symmetric == row_M_symmetric ? diagonal_value : tile[threadIdx.x][threadIdx.y];
			}	
		} else if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::UPPER_TRIANGLE) {
			if (column_symmetric <= row_M_symmetric) {
				A[symmetric_idx] = column_symmetric == row_M_symmetric ? diagonal_value : tile[threadIdx.x][threadIdx.y];
			}	
		}
	}
}

extern "C"
__global__
void
kernel_copy_triangles_lower_f32(
	uint64_t N,
	float diagonal_value,
	float* __restrict__ A, uint64_t ldA, uint64_t batch_stride_A
) {
	kernel_copy_triangles<KernelMatrixPackedType::LOWER_TRIANGLE, float>(
		N, diagonal_value, A, ldA, batch_stride_A
	);
}

extern "C"
__global__
void
kernel_copy_triangles_upper_f32(
	uint64_t N,
	float diagonal_value,
	float* __restrict__ A, uint64_t ldA, uint64_t batch_stride_A
) {
	kernel_copy_triangles<KernelMatrixPackedType::UPPER_TRIANGLE, float>(
		N, diagonal_value, A, ldA, batch_stride_A
	);
}

template <
	Symmetric symmetric,
	class T
>
__forceinline__
__device__
void
template_laplace_kernel(
	uint64_t num_samples,
    uint64_t row, uint64_t column,
    T bandwidth,
    T regularizer,
    T epsilon,
    T d,
	T* v_ref,
	T* v_grad_ref
) {
	static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

	T constant_negative_gamma = c_negative_one<T> / bandwidth;

    d = max(c_zero<T>, d);

    T sqrt_d = sqrt(d);
    T v = _exp(sqrt_d * constant_negative_gamma);

    v = d < epsilon ? c_one<T> : v;

	if constexpr (symmetric == Symmetric::YES) {
		if (row == column) {
			*v_ref = c_one<T> + regularizer;
			*v_grad_ref = c_zero<T>;
			return;
		}
    }

	if (d < epsilon) {
		*v_ref = c_one<T>;
		*v_grad_ref = c_zero<T>;
		return;
	}

	*v_ref = v;
	*v_grad_ref = v / sqrt_d;
}

template <
	Symmetric symmetric,
	class T
>
__forceinline__
__device__
void
template_kernel(
	uint64_t num_samples,
    uint64_t row, uint64_t column,
	T bandwidth,
	T regularization,
	T epsilon,
    T d,
	T* v_ref,
	T* v_grad_ref
) {
	static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

	template_laplace_kernel<
		symmetric, T
	> (
		num_samples, 
		row, 
		column, 
		bandwidth,
		regularization,
		epsilon,
		d,
		v_ref,
		v_grad_ref
	);

	// } else {
	// 	*v_ref = _nan<T>();
	// 	*v_grad_ref = _nan<T>();
	// }

}

template <
	KernelMatrixPackedType kernel_matrix_packed_type,
	KernelValue kernel_value,
	Symmetric symmetric,
	SymmetricCopy symmetric_copy,
	NormalizeByNumSamples normalize_by_num_samples,
	class T
>
static
__device__
__forceinline__
void
_kernel_euclidean_kernel(
	uint64_t m, uint64_t n,
	T* __restrict__ A, 					uint64_t ldA, uint64_t batch_stride_A,
	const T* const __restrict__ norm_m, 	uint64_t ld_norm_m,
	const T* const __restrict__ norm_n, 	uint64_t ld_norm_n,
	T bandwidth,
	T regularization,
	T epsilon,
	uint64_t offset_m,
	uint64_t offset_n
) {
	const uint64_t l = blockIdx.z;

	__shared__ T outer_x[KERNEL_TILE_DIM];
	__shared__ T outer_y[KERNEL_TILE_DIM];
	__shared__ T tile[KERNEL_TILE_DIM][KERNEL_TILE_DIM + 1];

	uint64_t row_m = blockIdx.x * KERNEL_TILE_DIM + threadIdx.x;
	uint64_t row_n = blockIdx.y * KERNEL_TILE_DIM + threadIdx.x;
	uint64_t column = blockIdx.y * KERNEL_TILE_DIM + threadIdx.y;
	uint64_t thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

	uint64_t row_m_symmetric = blockIdx.y * KERNEL_TILE_DIM + threadIdx.x;
	uint64_t column_symmetric = blockIdx.x * KERNEL_TILE_DIM + threadIdx.y;
	uint64_t symmetric_idx = l * batch_stride_A + column_symmetric * ldA + row_m_symmetric;

	if (thread_idx < KERNEL_TILE_DIM && row_m < m) {
		outer_x[threadIdx.x] = norm_m[l * ld_norm_m + row_m];
	} else if (thread_idx < 2*KERNEL_TILE_DIM && row_n < n) {
		outer_y[threadIdx.x] = norm_n[l * ld_norm_n + row_n];
	}

	__syncthreads();

	if ( _kernel_matrix_pass<kernel_matrix_packed_type>(m, n, row_m, column, l) ) {
		uint64_t idx = _kernel_matrix_coords<kernel_matrix_packed_type>(m, n, ldA, batch_stride_A, row_m, column, l);
		T d = A[idx];
		T n_x = outer_x[threadIdx.x];
		T n_y = outer_y[threadIdx.y];
		d = n_x + n_y - (d + d);
		// A[idx] = d;
#if 1
		T laplace_data;
		T laplace_data_grad;
		template_kernel<
			symmetric,
			T
		> (
			m, 
			row_m + offset_m, 
			column + offset_n,
			bandwidth,
			regularization,
			epsilon,
			d,
			&laplace_data,
			&laplace_data_grad
		);

		if constexpr (normalize_by_num_samples == NormalizeByNumSamples::YES) {
			laplace_data = laplace_data / static_cast<T>(m);
		}

		if constexpr (kernel_value == KernelValue::VALUE) {
			A[idx] = laplace_data;
		} else {
			A[idx] = laplace_data_grad;
		}
		
		if constexpr (symmetric_copy == SymmetricCopy::VALUE) {
			tile[threadIdx.y][threadIdx.x] = laplace_data;
		} else if constexpr (symmetric_copy == SymmetricCopy::GRAD_VALUE) {
			tile[threadIdx.y][threadIdx.x] = laplace_data_grad;
		}
		#endif
	}

	if constexpr(symmetric_copy == SymmetricCopy::NO) {
		return;
	}

	__syncthreads();

	if (row_m_symmetric < m && column_symmetric < n) {
		if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::LOWER_TRIANGLE) {
			if (column_symmetric > row_m_symmetric) {
				A[symmetric_idx] = tile[threadIdx.x][threadIdx.y];
			}	
		} else if constexpr (kernel_matrix_packed_type == KernelMatrixPackedType::UPPER_TRIANGLE) {
			if (column_symmetric < row_m_symmetric) {
				A[symmetric_idx] = tile[threadIdx.x][threadIdx.y];
			}
		}
	}
}

extern "C"
__global__
void
kernel_laplace_full_symm_f32(
	uint64_t m, uint64_t n,
	float* __restrict__ A, 						uint64_t ldA, uint64_t batch_stride_A,
	const float* const __restrict__ norm_m, 	uint64_t ld_norm_m,
	const float* const __restrict__ norm_n, 	uint64_t ld_norm_n,
	float bandwidth,
	float regularization,
	float epsilon,
	uint64_t offset_m,
	uint64_t offset_n
) {
	_kernel_euclidean_kernel<
		KernelMatrixPackedType::FULL,
		KernelValue::VALUE,
		Symmetric::YES,
		SymmetricCopy::NO,
		NormalizeByNumSamples::NO
	>(
		m, n,
		A, ldA, batch_stride_A,
		norm_m, ld_norm_m,
		norm_n, ld_norm_n,
		bandwidth,
		regularization,
		epsilon,
		offset_m,
		offset_n
	);
}

extern "C"
__global__
void
kernel_laplace_full_f32(
	uint64_t m, uint64_t n,
	float* __restrict__ A, 						uint64_t ldA, uint64_t batch_stride_A,
	const float* const __restrict__ norm_m, 	uint64_t ld_norm_m,
	const float* const __restrict__ norm_n, 	uint64_t ld_norm_n,
	float bandwidth,
	float regularization,
	float epsilon,
	uint64_t offset_m,
	uint64_t offset_n
) {
	_kernel_euclidean_kernel<
		KernelMatrixPackedType::FULL,
		KernelValue::VALUE,
		Symmetric::NO,
		SymmetricCopy::NO,
		NormalizeByNumSamples::NO
	>(
		m, n,
		A, ldA, batch_stride_A,
		norm_m, ld_norm_m,
		norm_n, ld_norm_n,
		bandwidth,
		regularization,
		epsilon,
		offset_m,
		offset_n
	);
}

extern "C"
__global__
void
kernel_laplace_lower_symm_copy_grad_f32(
	uint64_t m, uint64_t n,
	float* __restrict__ A, 						uint64_t ldA, uint64_t batch_stride_A,
	const float* const __restrict__ norm_m, 	uint64_t ld_norm_m,
	const float* const __restrict__ norm_n, 	uint64_t ld_norm_n,
	float bandwidth,
	float regularization,
	float epsilon,
	uint64_t offset_m,
	uint64_t offset_n
) {
	_kernel_euclidean_kernel<
		KernelMatrixPackedType::LOWER_TRIANGLE,
		KernelValue::VALUE,
		Symmetric::YES,
		SymmetricCopy::GRAD_VALUE,
		NormalizeByNumSamples::NO
	>(
		m, n,
		A, ldA, batch_stride_A,
		norm_m, ld_norm_m,
		norm_n, ld_norm_n,
		bandwidth,
		regularization,
		epsilon,
		offset_m,
		offset_n
	);
}

extern "C"
__global__
void
kernel_laplace_upper_symm_copy_grad_f32(
	uint64_t m, uint64_t n,
	float* __restrict__ A, 						uint64_t ldA, uint64_t batch_stride_A,
	const float* const __restrict__ norm_m, 	uint64_t ld_norm_m,
	const float* const __restrict__ norm_n, 	uint64_t ld_norm_n,
	float bandwidth,
	float regularization,
	float epsilon,
	uint64_t offset_m,
	uint64_t offset_n
) {
	_kernel_euclidean_kernel<
		KernelMatrixPackedType::UPPER_TRIANGLE,
		KernelValue::VALUE,
		Symmetric::YES,
		SymmetricCopy::GRAD_VALUE,
		NormalizeByNumSamples::NO
	>(
		m, n,
		A, ldA, batch_stride_A,
		norm_m, ld_norm_m,
		norm_n, ld_norm_n,
		bandwidth,
		regularization,
		epsilon,
		offset_m,
		offset_n
	);
}

extern "C"
__global__
void
kernel_pointer_array_f32(
	float* base,
	float** out,
	int64_t batch_stride,
	int64_t num_batches
) {
	int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_batches) {
        out[i] = base + i * batch_stride;
    }
}
