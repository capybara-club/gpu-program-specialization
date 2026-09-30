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
CUDA_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_PATH)/bin/nvcc
SM ?= 120
ODEZZA_BUILD ?= ../../build
BUILD ?= ../../build/score_topk
GLIBC_COMPAT ?= 0
ifeq ($(GLIBC_COMPAT),1)
COMPAT_FLAGS = -U_GNU_SOURCE -D_DEFAULT_SOURCE -include glibc_compat.h
endif

.PHONY: all
all: $(BUILD)/benchmark
$(BUILD)/benchmark: benchmark.cu glibc_compat.h ../../core/odezza.h $(ODEZZA_BUILD)/libodezza.so
	mkdir -p $(BUILD)
	$(NVCC) -O3 -std=c++17 -arch=sm_$(SM) --ptxas-options=-v $(COMPAT_FLAGS) -I../../core -I$(CUDA_PATH)/include/cccl $< -L$(ODEZZA_BUILD) -Xlinker -rpath -Xlinker $(abspath $(ODEZZA_BUILD)) -lodezza -lcuda -o $@
