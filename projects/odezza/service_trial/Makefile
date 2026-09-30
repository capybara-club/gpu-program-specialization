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
BUILD ?= ../build/service_trial
NATS_PREFIX ?= /opt/homebrew
NATS_CFLAGS ?= -I$(NATS_PREFIX)/include
NATS_LIBS ?= -L$(NATS_PREFIX)/lib -Wl,-rpath,$(NATS_PREFIX)/lib -lnats
.PHONY: service worker
service: $(BUILD)/orchestrator
$(BUILD)/orchestrator: orchestrator.c admission.c admission.h trial.c trial.h ../runtime/odezza_runtime.h ../runtime/odz_sizing.h ../build/frontend/libodezza_request.a
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(NATS_CFLAGS) -I../frontend -I../runtime orchestrator.c admission.c trial.c ../build/frontend/libodezza_request.a $(NATS_LIBS) -lm -pthread -o $@
worker: $(BUILD)/worker
$(BUILD)/worker: worker.c trial.c trial.h ../build/runtime/libodezza_runtime.so ../build/frontend/libodezza_request.a
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(NATS_CFLAGS) -I../runtime -I../frontend worker.c trial.c ../build/frontend/libodezza_request.a -L../build/runtime -Wl,-rpath,$(abspath ../build/runtime) -lodezza_runtime $(NATS_LIBS) -lm -pthread -o $@
.PHONY: prepare-benchmark
prepare-benchmark: $(BUILD)/prepare_benchmark
../build/frontend/libodezza_request.a:
	$(MAKE) -C ../frontend
$(BUILD)/prepare_benchmark: prepare_benchmark.c ../build/frontend/libodezza_request.a
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -I../frontend $< ../build/frontend/libodezza_request.a -lm -pthread -o $@

.PHONY: test-control
test-control: $(BUILD)/tests_control
	$(BUILD)/tests_control
$(BUILD)/tests_control: tests_control.c worker.c trial.c trial.h ../build/runtime/libodezza_runtime.so
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) $(NATS_CFLAGS) -I../runtime -I../frontend -I../core -isystem /usr/local/cuda/include tests_control.c trial.c ../build/frontend/libodezza_request.a -L../build/runtime -Wl,-rpath,$(abspath ../build/runtime) -lodezza_runtime $(NATS_LIBS) -lm -pthread -o $@

.PHONY: test-admission
test-admission: $(BUILD)/tests_admission
	$(BUILD)/tests_admission
$(BUILD)/tests_admission: tests_admission.c admission.c admission.h ../build/frontend/libodezza_request.a
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -I../frontend tests_admission.c admission.c ../build/frontend/libodezza_request.a -lm -o $@
