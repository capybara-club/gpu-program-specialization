# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 Charles Durham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
CC ?= cc
CFLAGS ?= -O3 -std=c99 -Wall -Wextra -Wpedantic
CUDA_ARCH ?= sm_89
CUDA_ROOT ?= /usr/local/cuda

BUILD_DIR := build
PROGRAM := $(BUILD_DIR)/astaxanthin_reproduction
BENCHMARK := $(BUILD_DIR)/benchmark_candidate
CUDA_COMPILER := $(BUILD_DIR)/resident_lm_compile
CUDA_CUBIN := $(BUILD_DIR)/score_only_rk4_$(CUDA_ARCH).cubin
DYNAMIC_CUDA_CUBIN := $(BUILD_DIR)/score_dynamic_leaves_rk4_$(CUDA_ARCH).cubin
CUDA_BENCHMARK := $(BUILD_DIR)/score_only_rk4_runner
CPU_NATIVE_BENCHMARK := $(BUILD_DIR)/score_only_rk4_cpu_native
CPU_SCALAR_BENCHMARK := $(BUILD_DIR)/score_only_rk4_cpu_scalar
CPU_FMA_PEAK := $(BUILD_DIR)/avx512_fma_peak

.PHONY: all bench clean cpu-bench cpu-peak-avx512 cuda-bench cuda-dynamic-leaf-bench data test

all: $(PROGRAM)

$(PROGRAM): astaxanthin_ground_truth.c astaxanthin_ground_truth.h main.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) astaxanthin_ground_truth.c main.c -o $(PROGRAM)

$(BENCHMARK): astaxanthin_ground_truth.c astaxanthin_ground_truth.h benchmark_candidate.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) astaxanthin_ground_truth.c benchmark_candidate.c -o $(BENCHMARK)

bench: $(BENCHMARK)
	$(BENCHMARK) 0.5 100000
	$(BENCHMARK) 0.1 50000
	$(BENCHMARK) 0.01 5000

$(CUDA_COMPILER): resident_lm_compile.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -isystem $(CUDA_ROOT)/include resident_lm_compile.c \
		-L$(CUDA_ROOT)/lib64 -Wl,-rpath,$(CUDA_ROOT)/lib64 -lnvrtc -o $(CUDA_COMPILER)

$(CUDA_CUBIN): score_only_rk4_kernel.cu $(CUDA_COMPILER)
	$(CUDA_COMPILER) $(CUDA_ARCH) score_only_rk4_kernel.cu $(CUDA_CUBIN)

$(DYNAMIC_CUDA_CUBIN): score_dynamic_leaves_rk4_kernel.cu $(CUDA_COMPILER)
	$(CUDA_COMPILER) $(CUDA_ARCH) score_dynamic_leaves_rk4_kernel.cu $(DYNAMIC_CUDA_CUBIN)

$(CUDA_BENCHMARK): score_only_rk4_runner.c astaxanthin_ground_truth.c astaxanthin_ground_truth.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -isystem $(CUDA_ROOT)/include score_only_rk4_runner.c \
		astaxanthin_ground_truth.c -L$(CUDA_ROOT)/lib64 \
		-Wl,-rpath,$(CUDA_ROOT)/lib64 -lcuda -lm -o $(CUDA_BENCHMARK)

cuda-bench: $(CUDA_CUBIN) $(CUDA_BENCHMARK)
	$(CUDA_BENCHMARK) $(CUDA_CUBIN)

cuda-dynamic-leaf-bench: $(DYNAMIC_CUDA_CUBIN) $(CUDA_BENCHMARK)
	$(CUDA_BENCHMARK) $(DYNAMIC_CUDA_CUBIN)

$(CPU_NATIVE_BENCHMARK): score_only_rk4_cpu_benchmark.c astaxanthin_ground_truth.c astaxanthin_ground_truth.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -march=native -ffast-math -fopenmp \
		score_only_rk4_cpu_benchmark.c astaxanthin_ground_truth.c -lm -o $(CPU_NATIVE_BENCHMARK)

$(CPU_SCALAR_BENCHMARK): score_only_rk4_cpu_benchmark.c astaxanthin_ground_truth.c astaxanthin_ground_truth.h
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -march=native -ffast-math -fopenmp -fno-tree-vectorize \
		-DASTAXANTHIN_SCALAR=1 score_only_rk4_cpu_benchmark.c \
		astaxanthin_ground_truth.c -lm -o $(CPU_SCALAR_BENCHMARK)

$(CPU_FMA_PEAK): avx512_fma_peak.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -march=native -ffast-math -fopenmp \
		avx512_fma_peak.c -o $(CPU_FMA_PEAK)

cpu-bench: $(CPU_NATIVE_BENCHMARK) $(CPU_SCALAR_BENCHMARK)
	OMP_PROC_BIND=close OMP_PLACES=cores $(CPU_NATIVE_BENCHMARK)

cpu-peak-avx512: $(CPU_FMA_PEAK)
	OMP_PROC_BIND=close OMP_PLACES=cores $(CPU_FMA_PEAK)

data: $(PROGRAM)
	@mkdir -p generated
	$(PROGRAM) 0.01 > generated/equation_6_ground_truth.csv
	@echo "wrote generated/equation_6_ground_truth.csv"

test: $(PROGRAM)
	python3 verify_reproduction.py $(PROGRAM)

clean:
	rm -rf $(BUILD_DIR) generated
