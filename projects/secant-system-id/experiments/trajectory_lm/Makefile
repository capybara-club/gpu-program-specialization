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
ROOT := ../..
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CC ?= cc
CUDA_ARCH ?= sm_89
BUILD_DIR ?= $(ROOT)/generated/trajectory_lm
SETTINGS ?= 8192
STARTS ?= 4
THREADS ?= 256
STEPS ?= 16
LM_ITERATIONS ?= 20
DAMPING_ATTEMPTS ?= 8
REPEATS ?= 3

CUBIN := $(BUILD_DIR)/fedbatch_thread_lm.cubin
PRIMAL_SITE := $(BUILD_DIR)/fedbatch_primal_site.inc
PARTIAL_SITE := $(BUILD_DIR)/fedbatch_partial_site.inc
SPLIT_SITE_INSPECTION := $(BUILD_DIR)/fedbatch_split_sites.json
SPLIT_SITE_STAMP := $(BUILD_DIR)/fedbatch_split_sites.stamp
DERIVATIVE_TEMPLATE_CUBIN := $(BUILD_DIR)/fedbatch_thread_lm_derivative_template.cubin
DERIVATIVE_CUBIN := $(BUILD_DIR)/fedbatch_thread_lm_derivative.cubin
DERIVATIVE_INSPECTION := $(BUILD_DIR)/fedbatch_thread_lm_derivative.json
RUNNER := $(BUILD_DIR)/fedbatch_thread_lm_runner
SPECIALIZATION_SOURCES := \
	$(ROOT)/src/secant_system_id/autodiff.py \
	$(ROOT)/src/secant_system_id/cubin.py \
	$(ROOT)/src/secant_system_id/derivative_sass.py \
	$(ROOT)/src/secant_system_id/lm_specialization.py

.PHONY: all specialized run run-specialized

all: $(CUBIN) $(RUNNER)

specialized: $(DERIVATIVE_CUBIN) $(RUNNER)

$(BUILD_DIR):
	mkdir -p $@

$(CUBIN): fedbatch_thread_lm.cu | $(BUILD_DIR)
	$(NVCC) -O3 --use_fast_math -arch=$(CUDA_ARCH) -cubin -Xptxas=-v \
		-Xcompiler=-U_GNU_SOURCE $< -o $@

$(SPLIT_SITE_STAMP): $(SPECIALIZATION_SOURCES) | $(BUILD_DIR)
	PYTHONPATH=$(ROOT)/src python3 -m secant_system_id.lm_specialization \
		generate-sites --primal-output $(PRIMAL_SITE) \
		--partial-output $(PARTIAL_SITE) \
		--inspection-output $(SPLIT_SITE_INSPECTION) --quiet
	touch $@

$(DERIVATIVE_TEMPLATE_CUBIN): fedbatch_thread_lm.cu $(SPLIT_SITE_STAMP) | $(BUILD_DIR)
	$(NVCC) -O3 --use_fast_math -arch=$(CUDA_ARCH) -cubin -Xptxas=-v \
		-Xcompiler=-U_GNU_SOURCE -DSSID_SPECIALIZED_GRADIENT \
		-I$(BUILD_DIR) $< -o $@

$(DERIVATIVE_CUBIN): $(DERIVATIVE_TEMPLATE_CUBIN) $(SPECIALIZATION_SOURCES)
	PYTHONPATH=$(ROOT)/src python3 -m secant_system_id.lm_specialization \
		specialize $< --output $@ --inspection-output $(DERIVATIVE_INSPECTION) --quiet

$(RUNNER): fedbatch_thread_lm_runner.c astaxanthin_ground_truth.c astaxanthin_ground_truth.h | $(BUILD_DIR)
	$(CC) -O3 -march=native -std=c99 -I$(CUDA_HOME)/include \
		fedbatch_thread_lm_runner.c astaxanthin_ground_truth.c -lcuda -lm -o $@

run: all
	$(RUNNER) $(CUBIN) $(SETTINGS) $(STARTS) $(THREADS) $(STEPS) $(LM_ITERATIONS) $(DAMPING_ATTEMPTS) $(REPEATS)

run-specialized: specialized
	$(RUNNER) $(DERIVATIVE_CUBIN) $(SETTINGS) $(STARTS) $(THREADS) $(STEPS) $(LM_ITERATIONS) $(DAMPING_ATTEMPTS) $(REPEATS)
