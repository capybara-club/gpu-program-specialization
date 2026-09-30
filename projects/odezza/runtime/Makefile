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
.DEFAULT_GOAL := all
.DEFAULT_GOAL := all
CC ?= cc
CUDA_PATH ?= /usr/local/cuda
BUILD ?= ../build/runtime
include ../third_party/sqlite/sqlite.mk
CFLAGS ?= -O2 -g -std=c99 -fPIC -Wall -Wextra -Wpedantic -Werror -MMD -MP
CPPFLAGS += -I. -I../frontend -I../core -isystem $(CUDA_PATH)/include
SOURCES = odz_job.c odz_workers.c odz_buffers.c odz_memory.c odz_status.c odz_gpu.c odz_execute.c odz_sizing.c odz_retention.c odz_report.c odz_report_bound.c
FRONTEND = odr_json odr_allocations odr_expr odr_trajectories odr_static odr_grammar odr_producer odr_sampling odr_scoring odr_scoring_payload odr_cache
OBJECTS = $(SQLITE_OBJECT) $(SOURCES:%.c=$(BUILD)/%.o) $(FRONTEND:%=$(BUILD)/%.o)
-include $(OBJECTS:.o=.d)
.PHONY: all
all: $(BUILD)/libodezza_runtime.so
$(BUILD):
	mkdir -p $@
$(BUILD)/%.o: %.c odz_internal.h odezza_runtime.h numeric_source.h ../frontend/odezza_request.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
$(BUILD)/%.o: ../frontend/%.c ../frontend/odezza_request.h ../frontend/odr_grammar_internal.h | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
$(BUILD)/libodezza_runtime.so: $(OBJECTS) ../build/libodezza.so odezza_runtime.exports
	$(CC) $(LDFLAGS) -Wl,--version-script=odezza_runtime.exports -shared -o $@ $(OBJECTS) -L../build -Wl,-rpath,$(abspath ../build) -lodezza -L$(CUDA_PATH)/lib64 -lcuda -lnvrtc -lm -pthread

.PHONY: embed
embed:
	python3 ../tests/runtime/embed.py

.PHONY: test-sizing-cpu
test-sizing-cpu: $(BUILD)/sizing_cpu
	$(BUILD)/sizing_cpu
$(BUILD)/sizing_cpu: ../tests/runtime/sizing_cpu.c odz_sizing.c odz_sizing.h odezza_runtime.h | $(BUILD)
	$(CC) -O1 -g -std=c99 -Wall -Wextra -Wpedantic -Werror -fsanitize=address,undefined -I. $< odz_sizing.c -lm -o $@

.PHONY: test-memory
test-memory: $(BUILD)/memory_test
	$(BUILD)/memory_test
$(BUILD)/memory_test: ../tests/runtime/memory_test.c odz_memory.c odz_internal.h odz_sizing.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsanitize=address,undefined -fno-omit-frame-pointer $< odz_memory.c -L$(CUDA_PATH)/lib64 -lcuda -lm -pthread -o $@

.PHONY: test-retention-cpu
test-retention-cpu: $(BUILD)/retention_cpu $(BUILD)/report_bound_cpu
	$(BUILD)/retention_cpu
	$(BUILD)/report_bound_cpu ../examples/grammar/submit.json
$(BUILD)/retention_cpu: ../tests/runtime/retention_cpu.c ../tests/runtime/retention_support.h odz_retention.c odz_internal.h odz_sizing.h ../build/frontend/libodezza_request.a | $(BUILD)
	$(CC) -O1 -g -std=c99 -Wall -Wextra -Wpedantic -Werror -fsanitize=address,undefined -I../tests/cpu_headers -I../frontend -I../core -I. $< odz_retention.c ../build/frontend/libodezza_request.a -lm -pthread -o $@
$(BUILD)/report_bound_cpu: ../tests/runtime/report_bound_cpu.c ../tests/runtime/retention_support.h odz_retention.c odz_report_bound.c odz_internal.h odz_sizing.h ../build/frontend/libodezza_request.a | $(BUILD)
	$(CC) -O1 -g -std=c99 -Wall -Wextra -Wpedantic -Werror -fsanitize=address,undefined -I../tests/cpu_headers -I../frontend -I../core -I. $< odz_retention.c odz_report_bound.c ../build/frontend/libodezza_request.a -lm -pthread -o $@
