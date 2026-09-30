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
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <exception>
#include <cstdio>

#include <cuda.h>

#include <kermac.hpp>
#include <check_result_helper.h>

using namespace kermac;

constexpr std::size_t NUM_SECONDARY_STREAMS = 2;
static const MatrixPackedType packed_type = KERMAC_MATRIX_PACKED_TYPE_LOWER_TRIANGLE;
static const TensorCoreMode tensor_core_mode = KERMAC_TENSOR_CORE_MODE_F32;
static const std::size_t HOST_BYTES = 1ull << 20;

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

struct RegressionCase {
    const char* name;
    int64_t num_train;
    int64_t num_test;
    int64_t num_features;
    int64_t num_outputs;
    int64_t batch;
    const float* x_train;
    const float* x_test;
    const float* weights;
    double weight_tol;
    double pred_tol;
};

static int64_t tensor_index_3d(
    const KermacTensor& tensor,
    int64_t row,
    int64_t col,
    int64_t batch
) {
    return row * tensor.stride[0]
        + col * tensor.stride[1]
        + batch * tensor.stride[2];
}

static int64_t case_index_3d(
    int64_t dim1,
    int64_t dim2,
    int64_t dim0_idx,
    int64_t dim1_idx,
    int64_t dim2_idx
) {
    return (dim0_idx * dim1 + dim1_idx) * dim2 + dim2_idx;
}

static void run_linear_regression_case(
    Kermac& handle,
    LegacyHandle& legacy,
    HostStackAllocator& hsa,
    DeviceStackAllocator& dsa,
    const RegressionCase& cfg,
    CUstream stream
) {
    const bool is_dry = dsa.get().is_dry;
    const bool use_legacy_solver = cfg.num_outputs == 1;

    DeviceTensor<float> d_x_train(dsa, cfg.num_train, cfg.num_features, cfg.batch);
    DeviceTensor<float> d_y_train(dsa, cfg.num_train, cfg.num_outputs, cfg.batch);
    DeviceTensor<float> d_x_test(dsa, cfg.num_test, cfg.num_features, cfg.batch);
    DeviceTensor<float> d_xtx(dsa, cfg.num_features, cfg.num_features, cfg.batch);
    DeviceTensor<float> d_weights(dsa, cfg.num_features, cfg.num_outputs, cfg.batch);
    DeviceTensor<float> d_preds(dsa, cfg.num_test, cfg.num_outputs, cfg.batch);

    HostTensor<float> h_x_train(hsa, d_x_train.extent());
    HostTensor<float> h_y_train(hsa, d_y_train.extent());
    HostTensor<float> h_x_test(hsa, d_x_test.extent());
    HostTensor<float> h_weights_expected(hsa, d_weights.extent());
    HostTensor<float> h_preds_expected(hsa, d_preds.extent());

    float* h_x_train_ptr = h_x_train.ptr();
    float* h_y_train_ptr = h_y_train.ptr();
    float* h_x_test_ptr = h_x_test.ptr();
    float* h_weights_expected_ptr = h_weights_expected.ptr();
    float* h_preds_expected_ptr = h_preds_expected.ptr();

    const KermacTensor h_x_train_raw = h_x_train.raw();
    const KermacTensor h_y_train_raw = h_y_train.raw();
    const KermacTensor h_x_test_raw = h_x_test.raw();
    const KermacTensor h_weights_expected_raw = h_weights_expected.raw();
    const KermacTensor h_preds_expected_raw = h_preds_expected.raw();

    for (int64_t batch = 0; batch < cfg.batch; ++batch) {
        for (int64_t row = 0; row < cfg.num_train; ++row) {
            for (int64_t feature = 0; feature < cfg.num_features; ++feature) {
                int64_t src_idx = case_index_3d(
                    cfg.num_train,
                    cfg.num_features,
                    batch,
                    row,
                    feature
                );
                h_x_train_ptr[tensor_index_3d(h_x_train_raw, row, feature, batch)] =
                    cfg.x_train[src_idx];
            }
        }

        for (int64_t row = 0; row < cfg.num_test; ++row) {
            for (int64_t feature = 0; feature < cfg.num_features; ++feature) {
                int64_t src_idx = case_index_3d(
                    cfg.num_test,
                    cfg.num_features,
                    batch,
                    row,
                    feature
                );
                h_x_test_ptr[tensor_index_3d(h_x_test_raw, row, feature, batch)] =
                    cfg.x_test[src_idx];
            }
        }

        for (int64_t feature = 0; feature < cfg.num_features; ++feature) {
            for (int64_t output = 0; output < cfg.num_outputs; ++output) {
                int64_t src_idx = case_index_3d(
                    cfg.num_features,
                    cfg.num_outputs,
                    batch,
                    feature,
                    output
                );
                h_weights_expected_ptr[tensor_index_3d(
                    h_weights_expected_raw,
                    feature,
                    output,
                    batch
                )] = cfg.weights[src_idx];
            }
        }

        for (int64_t row = 0; row < cfg.num_train; ++row) {
            for (int64_t output = 0; output < cfg.num_outputs; ++output) {
                float sum = 0.0f;
                for (int64_t feature = 0; feature < cfg.num_features; ++feature) {
                    float x_value = cfg.x_train[case_index_3d(
                        cfg.num_train,
                        cfg.num_features,
                        batch,
                        row,
                        feature
                    )];
                    float w_value = cfg.weights[case_index_3d(
                        cfg.num_features,
                        cfg.num_outputs,
                        batch,
                        feature,
                        output
                    )];
                    sum += x_value * w_value;
                }
                h_y_train_ptr[tensor_index_3d(h_y_train_raw, row, output, batch)] = sum;
            }
        }

        for (int64_t row = 0; row < cfg.num_test; ++row) {
            for (int64_t output = 0; output < cfg.num_outputs; ++output) {
                float sum = 0.0f;
                for (int64_t feature = 0; feature < cfg.num_features; ++feature) {
                    float x_value = cfg.x_test[case_index_3d(
                        cfg.num_test,
                        cfg.num_features,
                        batch,
                        row,
                        feature
                    )];
                    float w_value = cfg.weights[case_index_3d(
                        cfg.num_features,
                        cfg.num_outputs,
                        batch,
                        feature,
                        output
                    )];
                    sum += x_value * w_value;
                }
                h_preds_expected_ptr[tensor_index_3d(h_preds_expected_raw, row, output, batch)] = sum;
            }
        }
    }

    d_x_train.copy_from(h_x_train, stream);
    d_y_train.copy_from(h_y_train, stream);
    d_x_test.copy_from(h_x_test, stream);

    contraction(
        handle,
        dsa,
        tensor_core_mode,
        1.0f,
        d_x_train, "kml",
        d_x_train, "knl",
        0.0f,
        d_xtx, "mnl",
        d_xtx, "mnl",
        stream
    );

    contraction(
        handle,
        dsa,
        tensor_core_mode,
        1.0f,
        d_x_train, "kml",
        d_y_train, "kcl",
        0.0f,
        d_weights, "mcl",
        d_weights, "mcl",
        stream
    );

    if (use_legacy_solver) {
        DeviceTensor<float*> d_xtx_array(dsa, cfg.batch);
        DeviceTensor<float*> d_weights_array(dsa, cfg.batch);
        DeviceTensor<int> d_factor_info(dsa, cfg.batch);
        DeviceTensor<int> d_solve_info(dsa, 1);

        solve_compute_array(legacy, d_xtx, d_xtx_array, stream);
        solve_compute_array(legacy, d_weights, d_weights_array, stream);

        solve(
            handle,
            packed_type,
            d_xtx,
            d_weights,
            d_xtx_array,
            d_weights_array,
            d_factor_info,
            d_solve_info,
            stream
        );

        if (!is_dry) {
            HostTensor<int> h_factor_info(hsa, d_factor_info.extent());
            HostTensor<int> h_solve_info(hsa, d_solve_info.extent());

            h_factor_info.copy_from(d_factor_info, stream);
            h_solve_info.copy_from(d_solve_info, stream);
            cuCheck(cuStreamSynchronize(stream));

            const int* factor_info_ptr = h_factor_info.ptr();
            const int* solve_info_ptr = h_solve_info.ptr();
            for (int64_t batch = 0; batch < cfg.batch; ++batch) {
                ASSERT(factor_info_ptr[batch] == 0);
            }
            ASSERT(solve_info_ptr[0] == 0);
        }
    } else {
        DeviceTensor<int32_t> d_factor_info(dsa, cfg.batch);
        DeviceTensor<int32_t> d_solve_info(dsa, cfg.batch);

        solve(
            handle,
            packed_type,
            dsa,
            d_xtx,
            d_weights,
            d_factor_info,
            d_solve_info,
            stream
        );

        if (!is_dry) {
            HostTensor<int32_t> h_factor_info(hsa, d_factor_info.extent());
            HostTensor<int32_t> h_solve_info(hsa, d_solve_info.extent());

            h_factor_info.copy_from(d_factor_info, stream);
            h_solve_info.copy_from(d_solve_info, stream);
            cuCheck(cuStreamSynchronize(stream));

            const int32_t* factor_info_ptr = h_factor_info.ptr();
            const int32_t* solve_info_ptr = h_solve_info.ptr();
            for (int64_t batch = 0; batch < cfg.batch; ++batch) {
                ASSERT(factor_info_ptr[batch] == 0);
                ASSERT(solve_info_ptr[batch] == 0);
            }
        }
    }

    contraction(
        handle,
        dsa,
        tensor_core_mode,
        1.0f,
        d_x_test, "nml",
        d_weights, "mcl",
        0.0f,
        d_preds, "ncl",
        d_preds, "ncl",
        stream
    );

    if (is_dry) {
        return;
    }

    HostTensor<float> h_weights_actual(hsa, d_weights.extent());
    HostTensor<float> h_preds_actual(hsa, d_preds.extent());

    h_weights_actual.copy_from(d_weights, stream);
    h_preds_actual.copy_from(d_preds, stream);
    cuCheck(cuStreamSynchronize(stream));

    const float* h_weights_actual_ptr = h_weights_actual.ptr();
    const float* h_preds_actual_ptr = h_preds_actual.ptr();

    double max_weight_diff = 0.0;
    double max_pred_diff = 0.0;
    const KermacTensor h_weights_actual_raw = h_weights_actual.raw();
    const KermacTensor h_preds_actual_raw = h_preds_actual.raw();

    for (int64_t batch = 0; batch < cfg.batch; ++batch) {
        for (int64_t output = 0; output < cfg.num_outputs; ++output) {
            for (int64_t feature = 0; feature < cfg.num_features; ++feature) {
                double actual = static_cast<double>(h_weights_actual_ptr[tensor_index_3d(
                    h_weights_actual_raw,
                    feature,
                    output,
                    batch
                )]);
                double expected = static_cast<double>(h_weights_expected_ptr[tensor_index_3d(
                    h_weights_expected_raw,
                    feature,
                    output,
                    batch
                )]);
                double diff = std::fabs(actual - expected);
                if (diff > max_weight_diff) {
                    max_weight_diff = diff;
                }
            }
        }
    }

    for (int64_t batch = 0; batch < cfg.batch; ++batch) {
        for (int64_t output = 0; output < cfg.num_outputs; ++output) {
            for (int64_t row = 0; row < cfg.num_test; ++row) {
                double actual = static_cast<double>(h_preds_actual_ptr[tensor_index_3d(
                    h_preds_actual_raw,
                    row,
                    output,
                    batch
                )]);
                double expected = static_cast<double>(h_preds_expected_ptr[tensor_index_3d(
                    h_preds_expected_raw,
                    row,
                    output,
                    batch
                )]);
                double diff = std::fabs(actual - expected);
                if (diff > max_pred_diff) {
                    max_pred_diff = diff;
                }
            }
        }
    }

    ASSERT(max_weight_diff < cfg.weight_tol);
    ASSERT(max_pred_diff < cfg.pred_tol);
}

static const float kMultiOutputTrainX[2][5][3] = {
    {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 0.0f},
        {2.0f, -1.0f, 1.0f}
    },
    {
        {2.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 1.0f},
        {3.0f, -1.0f, 2.0f},
        {-1.0f, 2.0f, 1.0f}
    }
};

static const float kMultiOutputTestX[2][3][3] = {
    {
        {1.0f, 2.0f, 3.0f},
        {-1.0f, 1.0f, 0.0f},
        {0.5f, -0.5f, 2.0f}
    },
    {
        {1.0f, 1.0f, 1.0f},
        {2.0f, 0.0f, -1.0f},
        {-2.0f, 1.0f, 2.0f}
    }
};

static const float kMultiOutputWeights[2][3][2] = {
    {
        {2.0f, -1.0f},
        {-3.0f, 4.0f},
        {1.0f, 0.5f}
    },
    {
        {1.5f, 2.0f},
        {-2.0f, 1.0f},
        {0.5f, -1.0f}
    }
};

static const RegressionCase kMultiOutputCase = {
    "multi_output",
    5,
    3,
    3,
    2,
    2,
    &kMultiOutputTrainX[0][0][0],
    &kMultiOutputTestX[0][0][0],
    &kMultiOutputWeights[0][0][0],
    1e-4,
    1e-4
};

static const float kSingleOutputTrainX[3][4][2] = {
    {
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f},
        {2.0f, -1.0f}
    },
    {
        {2.0f, 1.0f},
        {-1.0f, 1.0f},
        {0.0f, 1.0f},
        {1.0f, 2.0f}
    },
    {
        {1.0f, 2.0f},
        {2.0f, 1.0f},
        {3.0f, 0.0f},
        {-1.0f, 1.0f}
    }
};

static const float kSingleOutputTestX[3][2][2] = {
    {
        {3.0f, 1.0f},
        {-1.0f, 2.0f}
    },
    {
        {2.0f, -2.0f},
        {0.5f, 1.5f}
    },
    {
        {1.0f, 1.0f},
        {4.0f, -1.0f}
    }
};

static const float kSingleOutputWeights[3][2][1] = {
    {
        {3.0f},
        {-2.0f}
    },
    {
        {1.5f},
        {0.5f}
    },
    {
        {-1.0f},
        {4.0f}
    }
};

static const RegressionCase kSingleOutputCase = {
    "single_output",
    4,
    2,
    2,
    1,
    3,
    &kSingleOutputTrainX[0][0][0],
    &kSingleOutputTestX[0][0][0],
    &kSingleOutputWeights[0][0][0],
    1e-4,
    1e-4
};

int main() {
    if (!has_cuda_device()) {
        std::fprintf(stderr, "SKIP: no CUDA device available\n");
        return 77;
    }

    try {
        Kermac handle(NUM_SECONDARY_STREAMS);
        LegacyHandle legacy(handle);
        CUstream stream;
        cuCheck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

        void* host_memory_ptr = host_alloc(HOST_BYTES);
        HostStackAllocator hsa(host_memory_ptr, HOST_BYTES);

        std::size_t device_bytes = 0;
        {
            DeviceStackAllocator dsa_dry(nullptr, 0);
            run_linear_regression_case(handle, legacy, hsa, dsa_dry, kMultiOutputCase, stream);
            run_linear_regression_case(handle, legacy, hsa, dsa_dry, kSingleOutputCase, stream);
            device_bytes = static_cast<std::size_t>(dsa_dry.get().largest_total_offset);
        }

        void* device_memory_ptr = device_alloc(handle, device_bytes);
        {
            DeviceStackAllocator dsa(device_memory_ptr, device_bytes);
            run_linear_regression_case(handle, legacy, hsa, dsa, kMultiOutputCase, stream);
            run_linear_regression_case(handle, legacy, hsa, dsa, kSingleOutputCase, stream);
        }

        device_free(handle, device_memory_ptr);
        host_free(host_memory_ptr);
        cuCheck(cuStreamDestroy(stream));
        return 0;
    }
    catch (const kermac::Error& e) {
        std::fprintf(stderr, "Kermac error: %s\n", e.what());
        return 1;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Exception: %s\n", e.what());
        return 1;
    }
}
