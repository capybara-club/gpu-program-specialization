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
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS += -I.
BUILD ?= ../build/frontend
SOURCES = odr_json.c odr_allocations.c odr_expr.c odr_trajectories.c odr_static.c odr_grammar.c odr_producer.c odr_sampling.c
OBJECTS = $(SOURCES:%.c=$(BUILD)/%.o)
CUDA_PATH ?= /usr/local/cuda
include ../third_party/sqlite/sqlite.mk
.PHONY: all scoring benchmark test test-gpu clean
all: $(BUILD)/libodezza_request.a $(BUILD)/inspect
$(BUILD):
	mkdir -p $@
$(BUILD)/%.o: %.c odezza_request.h odr_internal.h odr_grammar_internal.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
$(BUILD)/libodezza_request.a: $(OBJECTS)
	$(AR) rcs $@ $^
$(BUILD)/inspect: ../tests/frontend/inspect.c $(BUILD)/libodezza_request.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD)/libodezza_request.a -lm -o $@
$(BUILD)/parse_test: ../tests/frontend/parse_test.c $(BUILD)/libodezza_request.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD)/libodezza_request.a -lm -o $@
$(BUILD)/grammar_test: ../tests/frontend/grammar_test.c $(BUILD)/libodezza_request.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD)/libodezza_request.a -lm -o $@
$(BUILD)/contracts_test: ../tests/frontend/contracts_test.c $(BUILD)/libodezza_request.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD)/libodezza_request.a -lm -o $@
$(BUILD)/mutation_test: ../tests/frontend/mutation_test.c $(BUILD)/libodezza_request.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD)/libodezza_request.a -lm -o $@
$(BUILD)/work_test: ../tests/frontend/work_test.c $(BUILD)/libodezza_request.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD)/libodezza_request.a -lm -o $@
test: all $(BUILD)/work_test $(BUILD)/parse_test $(BUILD)/grammar_test $(BUILD)/contracts_test $(BUILD)/mutation_test
	$(BUILD)/work_test
	$(BUILD)/parse_test
	$(BUILD)/grammar_test
	$(BUILD)/contracts_test
	$(BUILD)/mutation_test
	python3 ../tests/frontend/allocation_audit.py $(BUILD)/libodezza_request.a
	PYTHONPATH=../python python3 ../tests/frontend/compare.py $(abspath $(BUILD)/inspect)
$(BUILD)/gpu_test: ../tests/frontend/gpu_test.c odr_scoring.c odr_scoring_payload.c odr_cache.c odezza_scoring_request.h $(BUILD)/libodezza_request.a ../build/libodezza.so $(SQLITE_OBJECT)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I../core -isystem $(CUDA_PATH)/include ../tests/frontend/gpu_test.c odr_scoring.c odr_scoring_payload.c odr_cache.c $(BUILD)/libodezza_request.a -L../build -Wl,-rpath,$(abspath ../build) -lodezza -L$(CUDA_PATH)/lib64 -lcuda -lnvrtc $(SQLITE_OBJECT) -lm -pthread -o $@
test-gpu: $(BUILD)/gpu_test
	CUDA_MODULE_LOADING=EAGER $(BUILD)/gpu_test ../examples/grammar/submit.json $(BUILD)/template-cache
clean:
	rm -f $(BUILD)/mutation_test
	rm -f $(OBJECTS) $(BUILD)/libodezza_request.a $(BUILD)/inspect $(BUILD)/parse_test $(BUILD)/grammar_test $(BUILD)/contracts_test $(BUILD)/gpu_test $(BUILD)/benchmark $(BUILD)/odr_scoring.o $(BUILD)/odr_scoring_payload.o $(BUILD)/libodezza_scoring_request.a

$(BUILD)/odr_scoring.o: odr_scoring.c odezza_scoring_request.h ../core/odezza.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I../core -isystem $(CUDA_PATH)/include -c $< -o $@
$(BUILD)/odr_scoring_payload.o: odr_scoring_payload.c odezza_scoring_request.h odezza_request.h odr_internal.h ../core/odezza.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I../core -isystem $(CUDA_PATH)/include -c $< -o $@
$(BUILD)/odr_cache.o: odr_cache.c odr_cache.h ../core/o_sha256.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I../core -isystem $(CUDA_PATH)/include -c $< -o $@
$(BUILD)/libodezza_scoring_request.a: $(SQLITE_OBJECT) $(BUILD)/odr_cache.o $(BUILD)/odr_scoring.o $(BUILD)/odr_scoring_payload.o
	$(AR) rcs $@ $^
scoring: $(BUILD)/libodezza_request.a $(BUILD)/libodezza_scoring_request.a
$(BUILD)/benchmark: ../tests/frontend/benchmark.c $(BUILD)/libodezza_request.a
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(BUILD)/libodezza_request.a -lm -o $@
benchmark: $(BUILD)/benchmark
	$(BUILD)/benchmark ../examples/grammar/million.json

# Pure CPU tests use the repository's existing opaque CUDA test declarations;
# no CUDA driver functions are mocked. The cache test uses vendored SQLite.
.PHONY: test-cache
test-cache: $(BUILD)/cache_test
	$(BUILD)/cache_test
$(BUILD)/cache_test: ../tests/frontend/cache_test.c odr_cache.c odr_cache.h $(SQLITE_OBJECT)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I../tests/cpu_headers -I../core $< odr_cache.c $(SQLITE_OBJECT) -lm -pthread -o $@
