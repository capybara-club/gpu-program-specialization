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
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <kermac.hpp>
#include <check_result_helper.h>

using namespace kermac;

static const size_t NUM_BYTES = 1ull << 26;
static const float kTol = 1e-4f;

static bool has_cuda_device() {
    if (cuInit(0) != CUDA_SUCCESS) {
        return false;
    }
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
        return false;
    }
    return count > 0;
}

static int64_t effective_cols(const KermacTensor& tensor) {
    int64_t cols = 1;
    for (uint32_t i = 1; i < tensor.num_modes; ++i) {
        cols *= tensor.extent[i];
    }
    return cols;
}

static int64_t batch_extent(const KermacTensor& tensor) {
    return tensor.num_modes < 3 ? 1 : tensor.extent[tensor.num_modes - 1];
}

static void fill_tensors(HostTensor<float>& h_a, HostTensor<float>& h_b) {
    const KermacTensor a = h_a.raw();
    const KermacTensor b = h_b.raw();

    const int64_t M = a.extent[0];
    const int64_t a_batch = batch_extent(a);
    const int64_t b_batch = batch_extent(b);
    const int64_t N = effective_cols(a) / a_batch;

    const int64_t a_ld = a.stride[1];
    const int64_t b_ld = b.stride[1];
    const int64_t batch_stride_a = a_batch == 1 ? 0 : a.stride[a.num_modes - 1];
    const int64_t batch_stride_b = b_batch == 1 ? 0 : b.stride[b.num_modes - 1];

    float* a_ptr = h_a.ptr();
    float* b_ptr = h_b.ptr();

    if (N == 1) {
        for (int64_t batch = 0; batch < b_batch; ++batch) {
            const int64_t b_batch_idx = b_batch == 1 ? 0 : batch;
            for (int64_t row = 0; row < M; ++row) {
                const bool b_positive = ((row + b_batch_idx) % 2) == 0;
                const float b_val = b_positive ? 0.8f : 0.2f;
                const int64_t idx = b_batch_idx * batch_stride_b + row;
                b_ptr[idx] = b_val;
            }
        }

        for (int64_t batch = 0; batch < a_batch; ++batch) {
            for (int64_t row = 0; row < M; ++row) {
                const bool a_positive = ((row + batch) % 3) != 0;
                const float a_val = a_positive ? 0.9f : 0.1f;
                const int64_t idx = batch * batch_stride_a + row;
                a_ptr[idx] = a_val;
            }
        }
        return;
    }

    for (int64_t batch = 0; batch < b_batch; ++batch) {
        const int64_t b_batch_idx = b_batch == 1 ? 0 : batch;
        for (int64_t row = 0; row < M; ++row) {
            const int64_t gt_label = (row + b_batch_idx) % N;
            const float base = 0.02f * (float)(row + 1) + 0.002f * (float)(b_batch_idx + 1);
            for (int64_t col = 0; col < N; ++col) {
                const float b_val = (col == gt_label)
                    ? (1.5f + base)
                    : (0.2f + 0.05f * (float)col + base);
                const int64_t idx = b_batch_idx * batch_stride_b + col * b_ld + row;
                b_ptr[idx] = b_val;
            }
        }
    }

    for (int64_t batch = 0; batch < a_batch; ++batch) {
        const int64_t b_batch_idx = b_batch == 1 ? 0 : batch;
        for (int64_t row = 0; row < M; ++row) {
            const int64_t gt_label = (row + b_batch_idx) % N;
            const int64_t pred_label =
                ((row + batch) % 3 == 0) ? gt_label : (gt_label + 1) % N;
            const float base = 0.01f * (float)(row + 1) + 0.001f * (float)(batch + 1);
            for (int64_t col = 0; col < N; ++col) {
                const float a_val = (col == pred_label)
                    ? (2.0f + base)
                    : (0.5f + 0.1f * (float)col + base);
                const int64_t idx = batch * batch_stride_a + col * a_ld + row;
                a_ptr[idx] = a_val;
            }
        }
    }
}

static void compute_expected(
    const HostTensor<float>& h_a,
    const HostTensor<float>& h_b,
    std::vector<float>& mse_out,
    std::vector<float>& acc_out
) {
    const KermacTensor a = h_a.raw();
    const KermacTensor b = h_b.raw();

    const int64_t M = a.extent[0];
    const int64_t a_batch = batch_extent(a);
    const int64_t b_batch = batch_extent(b);
    const int64_t N = effective_cols(a) / a_batch;

    const int64_t a_ld = a.stride[1];
    const int64_t b_ld = b.stride[1];
    const int64_t batch_stride_a = a_batch == 1 ? 0 : a.stride[a.num_modes - 1];
    const int64_t batch_stride_b = b_batch == 1 ? 0 : b.stride[b.num_modes - 1];

    const float* a_ptr = h_a.ptr();
    const float* b_ptr = h_b.ptr();

    mse_out.assign(a_batch, 0.0f);
    acc_out.assign(a_batch, 0.0f);

    for (int64_t batch = 0; batch < a_batch; ++batch) {
        double mse_sum = 0.0;
        int64_t acc_count = 0;
        for (int64_t row = 0; row < M; ++row) {
            float max_a = -std::numeric_limits<float>::infinity();
            float max_b = -std::numeric_limits<float>::infinity();
            int64_t max_idx_a = 0;
            int64_t max_idx_b = 0;
            for (int64_t col = 0; col < N; ++col) {
                const int64_t a_idx = batch * batch_stride_a + col * a_ld + row;
                const int64_t b_idx = batch * batch_stride_b + col * b_ld + row;
                const float a_val = a_ptr[a_idx];
                const float b_val = b_ptr[b_idx];
                const float diff = a_val - b_val;
                mse_sum += (double)diff * (double)diff;
                if (a_val > max_a) {
                    max_a = a_val;
                    max_idx_a = col;
                }
                if (b_val > max_b) {
                    max_b = b_val;
                    max_idx_b = col;
                }
            }
            if (N == 1) {
                const float a_val = a_ptr[batch * batch_stride_a + row];
                const float b_val = b_ptr[batch * batch_stride_b + row];
                if ((a_val >= 0.5f) == (b_val >= 0.5f)) {
                    acc_count += 1;
                }
            } else {
                if (max_idx_a == max_idx_b) {
                    acc_count += 1;
                }
            }
        }
        mse_out[batch] = (float)(mse_sum / (double)(M * N));
        acc_out[batch] = (float)acc_count / (float)M;
    }
}

template <typename TensorType, typename AllocatorType>
static std::unique_ptr<TensorType> make_tensor(
    AllocatorType& alloc,
    int64_t rows,
    int64_t cols,
    bool has_batch,
    int64_t batch
) {
    if (has_batch) {
        return std::unique_ptr<TensorType>(new TensorType(alloc, rows, cols, batch));
    }
    return std::unique_ptr<TensorType>(new TensorType(alloc, rows, cols));
}

static int run_case(
    const char* label,
    Kermac& kermac,
    HostStackAllocator& hsa,
    DeviceStackAllocator& dsa,
    CUstream stream,
    int64_t rows,
    int64_t cols,
    bool a_has_batch,
    int64_t a_batch,
    bool b_has_batch,
    int64_t b_batch
) {
    auto h_a = make_tensor<HostTensor<float>>(hsa, rows, cols, a_has_batch, a_batch);
    auto h_b = make_tensor<HostTensor<float>>(hsa, rows, cols, b_has_batch, b_batch);
    auto d_a = make_tensor<DeviceTensor<float>>(dsa, rows, cols, a_has_batch, a_batch);
    auto d_b = make_tensor<DeviceTensor<float>>(dsa, rows, cols, b_has_batch, b_batch);

    const int64_t batch = a_has_batch ? a_batch : 1;

    DeviceTensor<float> d_mse(dsa, batch);
    DeviceTensor<float> d_acc(dsa, batch);
    HostTensor<float> h_mse(hsa, d_mse.extent());
    HostTensor<float> h_acc(hsa, d_acc.extent());

    fill_tensors(*h_a, *h_b);

    d_a->copy_from(*h_a, stream);
    d_b->copy_from(*h_b, stream);

    mse_accuracy(kermac, dsa, *d_a, *d_b, d_mse, d_acc, stream);

    if (!dsa.get().is_dry) {
        h_mse.copy_from(d_mse, stream);
        h_acc.copy_from(d_acc, stream);
        cuCheck(cuStreamSynchronize(stream));

        std::vector<float> expected_mse;
        std::vector<float> expected_acc;
        compute_expected(*h_a, *h_b, expected_mse, expected_acc);

        float* mse_ptr = h_mse.ptr();
        float* acc_ptr = h_acc.ptr();
        for (int64_t i = 0; i < batch; ++i) {
            const float mse_diff = std::fabs(mse_ptr[i] - expected_mse[i]);
            const float acc_diff = std::fabs(acc_ptr[i] - expected_acc[i]);
            if (mse_diff > kTol || acc_diff > kTol) {
                std::cerr << "Mismatch in " << label
                          << " (batch " << i
                          << ", mse diff=" << mse_diff
                          << ", acc diff=" << acc_diff << ")\n";
                return -1;
            }
        }
    }

    return 0;
}

int main() {
    if (!has_cuda_device()) {
        std::cerr << "SKIP: no CUDA device available\n";
        return 77;
    }
    try {
        Kermac kermac(2);
        CUstream stream;
        cuCheck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

        void* host_mem = host_alloc(NUM_BYTES);
        void* device_mem = device_alloc(kermac, NUM_BYTES);
        {
            HostStackAllocator hsa(host_mem, NUM_BYTES);
            DeviceStackAllocator dsa(device_mem, NUM_BYTES);

            struct TestCase {
                const char* label;
                int64_t rows;
                int64_t cols;
                bool a_has_batch;
                int64_t a_batch;
                bool b_has_batch;
                int64_t b_batch;
            };

            const TestCase cases[] = {
                {"A[M,N,B] B[M,N,B]", 8, 3, true, 4, true, 4},
                {"A[M,N,B] B[M,N]", 8, 3, true, 4, false, 1},
                {"A[M,N,B] B[M,N,1]", 8, 3, true, 4, true, 1},
                {"A[M,N] B[M,N]", 7, 3, false, 1, false, 1},
                {"A[M,1,B] B[M,1]", 9, 1, true, 5, false, 1},
            };

            for (const auto& test_case : cases) {
                if (run_case(
                    test_case.label,
                    kermac,
                    hsa,
                    dsa,
                    stream,
                    test_case.rows,
                    test_case.cols,
                    test_case.a_has_batch,
                    test_case.a_batch,
                    test_case.b_has_batch,
                    test_case.b_batch
                ) != 0) {
                    return -1;
                }
            }
        }

        cuCheck(cuStreamSynchronize(stream));
        host_free(host_mem);
        device_free(kermac, device_mem);
        cuCheck(cuStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
