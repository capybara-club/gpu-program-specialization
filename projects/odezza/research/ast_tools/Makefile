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
CFLAGS ?= -O2 -std=c99 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off
BUILD_DIR ?= build
LDLIBS += -lm
CUDA_PATH ?= /usr/local/cuda
ODEZZA_LIBRARY ?= ../../build/libodezza.a

COMMON_SOURCES := o_ast_space.c o_ast_cpu.c o_ast_file.c

.PHONY: all engine-benchmark test clean

all: $(BUILD_DIR)/odezza-ast-tools

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/odezza-ast-tools: o_ast_tools_main.c $(COMMON_SOURCES) o_ast_tools.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) o_ast_tools_main.c $(COMMON_SOURCES) $(LDLIBS) -o $@

$(BUILD_DIR)/o_ast_tools_test: o_ast_tools_test.c $(COMMON_SOURCES) o_ast_tools.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) o_ast_tools_test.c $(COMMON_SOURCES) $(LDLIBS) -o $@

$(BUILD_DIR)/odezza-ast-engine-benchmark: o_ast_engine_benchmark.c $(COMMON_SOURCES) o_ast_tools.h $(ODEZZA_LIBRARY) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I../../src -isystem $(CUDA_PATH)/include o_ast_engine_benchmark.c $(COMMON_SOURCES) $(ODEZZA_LIBRARY) -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 -lnvrtc -lcuda $(LDLIBS) -pthread -o $@

engine-benchmark: $(BUILD_DIR)/odezza-ast-engine-benchmark

test: $(BUILD_DIR)/odezza-ast-tools $(BUILD_DIR)/o_ast_tools_test
	$(BUILD_DIR)/o_ast_tools_test
	PYTHONPATH=../../python ODEZZA_AST_TOOL=$(CURDIR)/$(BUILD_DIR)/odezza-ast-tools python3 -m unittest -v test_ast_tools.py

clean:
	$(RM) -r $(BUILD_DIR)
