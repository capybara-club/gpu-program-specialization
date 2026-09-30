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
.DEFAULT_GOAL := all
AR ?= ar
CFLAGS ?= -O2 -fPIC -std=c99 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS ?= -Icore
BUILD_DIR ?= build
CUDA_PATH ?= /usr/local/cuda
CPPFLAGS += -isystem $(CUDA_PATH)/include
LDLIBS += -L$(CUDA_PATH)/lib64 -lnvrtc -lcuda -lm -pthread

LIBRARY_OBJECTS := \
	$(BUILD_DIR)/o_elf.o \
	$(BUILD_DIR)/o_sass_assembler.o \
	$(BUILD_DIR)/o_generate_lm_cuda.o \
	$(BUILD_DIR)/o_lm_source.o \
	$(BUILD_DIR)/o_lm_ast.o \
	$(BUILD_DIR)/o_inspect_lm_cubin.o \
	$(BUILD_DIR)/o_specialize_lm_cubin.o \
	$(BUILD_DIR)/o_lm_pipeline.o \
	$(BUILD_DIR)/o_philox.o \
	$(BUILD_DIR)/o_score_reduce.o \
	$(BUILD_DIR)/o_ast.o \
	$(BUILD_DIR)/o_generate_scoring_cuda.o \
	$(BUILD_DIR)/o_inspect_scoring_cubin.o \
	$(BUILD_DIR)/o_nvrtc_compilation.o \
	$(BUILD_DIR)/o_scoring_pipeline.o \
	$(BUILD_DIR)/o_scoring_template.o \
	$(BUILD_DIR)/o_specialize_scoring_cubin.o

.PHONY: all shared test test-c test-cli test-cuda test-python test-style test-ast-tools parity test-lm-core test-lm-failures test-public-api

.PHONY: test-grammar
test-grammar:
	PYTHONPATH="$(CURDIR)/python" python3 -m unittest discover -s tests/grammar_compiler/tests -q
	PYTHONPATH="$(CURDIR)/python" python3 -m unittest discover -s tests/grammar -p 'test_*.py' -q

-include $(LIBRARY_OBJECTS:.o=.d)

all: $(BUILD_DIR)/libodezza.a $(BUILD_DIR)/odezza-internal-generate-scoring-cuda $(BUILD_DIR)/odezza-internal-pipeline-benchmark

$(LIBRARY_OBJECTS): core/o_sass.h core/odezza.h core/o_odezza_internal.h

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: core/%.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/o_generate_scoring_cuda.o: core/o_generate_scoring_cuda.c core/odezza.h core/o_odezza_internal.h core/o_sha256.h core/o_writer.h

$(BUILD_DIR)/o_ast.o: core/o_ast.c core/odezza.h core/o_odezza_internal.h

$(BUILD_DIR)/o_inspect_scoring_cubin.o: core/o_inspect_scoring_cubin.c core/odezza.h core/o_odezza_internal.h

$(BUILD_DIR)/o_nvrtc_compilation.o: core/o_nvrtc_compilation.c core/odezza.h core/o_odezza_internal.h

$(BUILD_DIR)/o_scoring_pipeline.o: core/o_scoring_pipeline.c core/odezza.h core/o_odezza_internal.h core/o_sha256.h

$(BUILD_DIR)/o_specialize_scoring_cubin.o: core/o_specialize_scoring_cubin.c core/odezza.h core/o_odezza_internal.h core/o_sha256.h

$(BUILD_DIR)/libodezza.a: $(LIBRARY_OBJECTS)
	$(RM) $@
	$(AR) rcs $@ $^

$(BUILD_DIR)/odezza-internal-generate-scoring-cuda: app/o_generate_scoring_cuda_main.c $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD_DIR)/libodezza.a $(LDLIBS) -o $@

$(BUILD_DIR)/odezza-internal-pipeline-benchmark: app/o_scoring_pipeline_benchmark_main.c $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD_DIR)/libodezza.a $(LDLIBS) -o $@

$(BUILD_DIR)/o_generate_scoring_cuda_test: tests/o_generate_scoring_cuda_test.c $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD_DIR)/libodezza.a $(LDLIBS) -o $@

$(BUILD_DIR)/o_inspect_scoring_cubin_test: tests/o_inspect_scoring_cubin_test.c $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD_DIR)/libodezza.a $(LDLIBS) -o $@

$(BUILD_DIR)/o_nvrtc_compilation_test: tests/o_nvrtc_compilation_test.c $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD_DIR)/libodezza.a $(LDLIBS) -o $@

$(BUILD_DIR)/o_specialize_scoring_cubin_test: tests/o_specialize_scoring_cubin_test.c tests/o_scoring_fixture.h $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD_DIR)/libodezza.a $(LDLIBS) -o $@

$(BUILD_DIR)/o_sass_test: tests/o_sass_test.c core/o_sass.h $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD_DIR)/libodezza.a $(LDLIBS) -o $@

$(BUILD_DIR)/o_sass_corpus_test: tests/o_sass_corpus_test.c tests/o_scoring_fixture.h core/o_sha256.h $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD_DIR)/libodezza.a $(LDLIBS) -o $@

$(BUILD_DIR)/o_scoring_pipeline_test: tests/o_scoring_pipeline_test.c $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD_DIR)/libodezza.a $(LDLIBS) -o $@

$(BUILD_DIR)/o_scoring_cuda_test: tests/o_scoring_cuda_test.c $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD_DIR)/libodezza.a $(LDLIBS) -o $@

test-cuda: $(BUILD_DIR)/o_scoring_cuda_test $(BUILD_DIR)/o_score_reduce_test $(BUILD_DIR)/o_philox_test
	$(BUILD_DIR)/o_philox_test
	$(BUILD_DIR)/o_score_reduce_test
	$(BUILD_DIR)/o_scoring_cuda_test 1 0 1
	$(BUILD_DIR)/o_scoring_cuda_test 8 8 1
	$(BUILD_DIR)/o_scoring_cuda_test 8 8 8

$(BUILD_DIR)/o_scoring_template_test: tests/o_scoring_template_test.c $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD_DIR)/libodezza.a $(LDLIBS) -o $@

.PHONY: test-scoring-template
test-scoring-template: $(BUILD_DIR)/o_scoring_template_test
	CUDA_MODULE_LOADING=EAGER $(BUILD_DIR)/o_scoring_template_test

test-c: $(BUILD_DIR)/o_generate_scoring_cuda_test $(BUILD_DIR)/o_inspect_scoring_cubin_test $(BUILD_DIR)/o_nvrtc_compilation_test $(BUILD_DIR)/o_specialize_scoring_cubin_test $(BUILD_DIR)/o_scoring_pipeline_test $(BUILD_DIR)/o_sass_test $(BUILD_DIR)/o_sass_corpus_test
	$(BUILD_DIR)/o_generate_scoring_cuda_test
	$(BUILD_DIR)/o_inspect_scoring_cubin_test
	$(BUILD_DIR)/o_nvrtc_compilation_test
	$(BUILD_DIR)/o_specialize_scoring_cubin_test
	$(BUILD_DIR)/o_sass_test
	$(BUILD_DIR)/o_sass_corpus_test
	$(BUILD_DIR)/o_scoring_pipeline_test

test-python:
	PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_*.py' -v
	PYTHONPATH=python python3 -m unittest discover -s scratch/c99_cubin_runner -p 'test_*.py' -v

test-style:
	cmake -DROOT=$(CURDIR) -P tests/o_verify_no_goto.cmake

test-ast-tools:
	$(MAKE) -C scratch/ast_tools test

# Compatibility target: validate the runtime C99 artifact, whose ABI differs
# from the historical fixed-trajectory Python generator.
parity: $(BUILD_DIR)/o_generate_scoring_cuda_test $(BUILD_DIR)/odezza-internal-generate-scoring-cuda
	$(BUILD_DIR)/o_generate_scoring_cuda_test $(BUILD_DIR)/o_generate_scoring_cuda.cu
	python3 tests/o_verify_c_generator.py $(BUILD_DIR)/o_generate_scoring_cuda.cu --generator $(BUILD_DIR)/odezza-internal-generate-scoring-cuda

test-cli: parity

test: test-style test-c test-cli test-python test-ast-tools

$(BUILD_DIR)/o_score_reduce.o: core/o_score_reduce.h core/o_score_reduce_source.h
$(BUILD_DIR)/o_score_reduce.o: CPPFLAGS += -DODEZZA_CUDA_INCLUDE_DIR='"$(CUDA_PATH)/include"'

$(BUILD_DIR)/o_score_reduce_test: tests/o_score_reduce_test.c $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD_DIR)/libodezza.a $(LDLIBS) -o $@

$(BUILD_DIR)/o_philox.o: core/o_philox.h core/o_philox_source.h
$(BUILD_DIR)/o_generate_scoring_cuda.o $(BUILD_DIR)/o_score_reduce.o: core/o_sampled_constants_source.h

$(BUILD_DIR)/o_philox_test: tests/o_philox_test.c $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD_DIR)/libodezza.a $(LDLIBS) -o $@

$(BUILD_DIR)/o_lm_core_test: tests/o_lm_core_test.c $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD_DIR)/libodezza.a $(LDLIBS) -o $@

test-lm-core: $(BUILD_DIR)/o_lm_core_test
	CUDA_MODULE_LOADING=EAGER $(BUILD_DIR)/o_lm_core_test
	CUDA_MODULE_LOADING=EAGER $(BUILD_DIR)/o_lm_core_test 2 coupled
	CUDA_MODULE_LOADING=EAGER $(BUILD_DIR)/o_lm_core_test 4 coupled
	CUDA_MODULE_LOADING=EAGER $(BUILD_DIR)/o_lm_core_test 8 coupled

$(BUILD_DIR)/libodezza.so: $(LIBRARY_OBJECTS) core/odezza.exports
	$(CC) -shared $(LIBRARY_OBJECTS) $(LDLIBS) -Wl,--version-script=core/odezza.exports -o $@

shared: $(BUILD_DIR)/libodezza.so

$(BUILD_DIR)/o_lm_failure_test: tests/o_lm_core_test.c tests/o_lm_cuda_failures.c $(BUILD_DIR)/libodezza.a
	$(CC) $(CPPFLAGS) $(CFLAGS) -DODEZZA_LM_FAULT_TEST tests/o_lm_core_test.c tests/o_lm_cuda_failures.c $(BUILD_DIR)/libodezza.a $(LDLIBS) -Wl,--wrap=cuModuleLoadData,--wrap=cuModuleUnload,--wrap=cuStreamCreate,--wrap=cuStreamDestroy_v2,--wrap=cuEventRecord,--wrap=cuLaunchKernel,--wrap=o_lm_specialize -o $@

test-lm-failures: $(BUILD_DIR)/o_lm_failure_test
	CUDA_MODULE_LOADING=EAGER $(BUILD_DIR)/o_lm_failure_test
	CUDA_MODULE_LOADING=EAGER $(BUILD_DIR)/o_lm_failure_test 1 coupled fallback

test-public-api: shared
	python3 tests/o_verify_public_exports.py --library $(BUILD_DIR)/libodezza.so

# Independent CPU request frontend; CUDA is needed only for its scoring bridge.
.PHONY: frontend test-frontend benchmark-frontend
frontend:
	$(MAKE) -C frontend all
test-frontend:
	$(MAKE) -C frontend test
benchmark-frontend:
	$(MAKE) -C frontend benchmark

# Optional JSON service runtime; C-owned template cache, no policy in the core.
.PHONY: request-runtime test-request-runtime
request-runtime: $(BUILD_DIR)/libodezza.so
	$(MAKE) -C runtime

test-request-runtime: request-runtime
	CUDA_MODULE_LOADING=EAGER PYTHONPATH="$(CURDIR)/python" python3 tests/runtime/gpu_conformance.py --root /tmp/odezza-runtime-test-cache --output /tmp/odezza-runtime-test-report.json

.PHONY: test-request-robustness
ROBUSTNESS_DEVICE ?= 0
ROBUSTNESS_SEED ?= 20260911
ROBUSTNESS_OUTPUT ?= /tmp/odezza-runtime-robustness.json
test-request-robustness: request-runtime
	CUDA_MODULE_LOADING=EAGER PYTHONPATH="$(CURDIR)/python" python3 tests/runtime/robustness.py --device $(ROBUSTNESS_DEVICE) --seed $(ROBUSTNESS_SEED) --output $(ROBUSTNESS_OUTPUT)

.PHONY: test-request-service
SERVICE_TEST_OUTPUT ?= /tmp/odezza-runtime-service.json
test-request-service: request-runtime
	CUDA_MODULE_LOADING=EAGER PYTHONPATH="$(CURDIR)/python" python3 tests/runtime/service_upgrade.py --output $(SERVICE_TEST_OUTPUT)
	CUDA_MODULE_LOADING=EAGER PYTHONPATH="$(CURDIR)/python" python3 tests/runtime/scheduling_checks.py --output /tmp/odezza-runtime-scheduling.json
