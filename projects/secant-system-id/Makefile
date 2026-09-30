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
CFLAGS ?= -O3 -std=c99 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Werror -pedantic -fPIC
CPPFLAGS ?= -I csrc

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
LIBRARY := build/libsecant_system_id_c99.dylib
SHARED_FLAGS := -dynamiclib -pthread -lm
MODULE_LIFECYCLE_LIBS := -pthread
else
LIBRARY := build/libsecant_system_id_c99.so
SHARED_FLAGS := -shared -pthread -ldl -lm
MODULE_LIFECYCLE_LIBS := -pthread -ldl
endif

SOURCES := csrc/ssid_common.c csrc/ssid_autodiff.c csrc/ssid_specializer.c csrc/ssid_benchmark.c csrc/ssid_pipeline.c csrc/ssid_gp.c
OBJECTS := $(SOURCES:csrc/%.c=build/c99/%.o)

.PHONY: c99 clean-c99 module-lifecycle-benchmark

c99: $(LIBRARY)

$(LIBRARY): $(OBJECTS)
	$(CC) $(OBJECTS) $(SHARED_FLAGS) -o $@

build/c99/%.o: csrc/%.c csrc/secant_system_id.h csrc/ssid_internal.h
	mkdir -p build/c99
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

clean-c99:
	rm -rf build/c99 $(LIBRARY)

module-lifecycle-benchmark: build/module_lifecycle_benchmark

build/module_lifecycle_benchmark: experiments/module_lifecycle_benchmark.c
	mkdir -p build
	$(CC) -O3 -std=c99 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Werror -pedantic $< $(MODULE_LIFECYCLE_LIBS) -o $@
