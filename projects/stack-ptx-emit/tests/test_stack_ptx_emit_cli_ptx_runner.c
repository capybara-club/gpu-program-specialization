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
/*
 * SPDX-FileCopyrightText: 2026 Charles Durham
 *
 * SPDX-License-Identifier: MIT
 */

#define PTX_INJECT_IMPLEMENTATION
#include <ptx_inject.h>

#include <check_result_helper.h>
#include <ptx_inject_helper.h>
#include <nvptx_helper.h>
#include <cuda_helper.h>

#include <cuda.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static
bool
has_cuda_device(void) {
	if (cuInit(0) != CUDA_SUCCESS) {
		return false;
	}
	int count = 0;
	if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
		return false;
	}
	return count > 0;
}

static
char*
read_file(
	const char* path,
	size_t* num_bytes_out
) {
	FILE* file = fopen(path, "rb");
	if (file == NULL) {
		return NULL;
	}

	ASSERT(fseek(file, 0, SEEK_END) == 0);
	long file_size = ftell(file);
	ASSERT(file_size >= 0);
	ASSERT(fseek(file, 0, SEEK_SET) == 0);

	char* buffer = (char*)malloc((size_t)file_size + 1);
	ASSERT(buffer != NULL);

	size_t bytes_read = fread(buffer, 1, (size_t)file_size, file);
	ASSERT(bytes_read == (size_t)file_size);
	ASSERT(fclose(file) == 0);

	buffer[bytes_read] = '\0';
	if (num_bytes_out != NULL) {
		*num_bytes_out = bytes_read;
	}
	return buffer;
}

int
main(
	int argc,
	char** argv
) {
	if (argc != 2) {
		fprintf(stderr, "Usage: %s STUB_PATH\n", argv[0]);
		return 1;
	}

	if (!has_cuda_device()) {
		fprintf(stderr, "No CUDA device available; skipping PTX runner.\n");
		return 77;
	}

	char* kernel_ptx = read_file(PTX_KERNEL, NULL);
	ASSERT(kernel_ptx != NULL);

	char* stub = read_file(argv[1], NULL);
	ASSERT(stub != NULL);

	CUdevice cu_device;
	cuCheck( cuInit(0) );
	cuCheck( cuDeviceGet(&cu_device, 0) );

	CUcontext cu_context;
	cuCheck( cuContextCreate(&cu_context, cu_device) );

	int device_compute_capability_major = 0;
	int device_compute_capability_minor = 0;
	get_device_capability(cu_device, &device_compute_capability_major, &device_compute_capability_minor);

	PtxInjectHandle ptx_inject;
	ptxInjectCheck( ptx_inject_create(&ptx_inject, kernel_ptx) );
	free(kernel_ptx);

	size_t inject_func_idx = 0;
	ptxInjectCheck( ptx_inject_inject_info_by_name(ptx_inject, "func", &inject_func_idx, NULL, NULL) );

	const char* ptx_stubs[1];
	ptx_stubs[inject_func_idx] = stub;

	size_t rendered_ptx_num_bytes = 0;
	char* rendered_ptx = render_injected_ptx(ptx_inject, ptx_stubs, 1, &rendered_ptx_num_bytes);
	ptxInjectCheck( ptx_inject_destroy(ptx_inject) );
	free(stub);

	char* sass = nvptx_compile(
		device_compute_capability_major,
		device_compute_capability_minor,
		rendered_ptx,
		rendered_ptx_num_bytes,
		NULL,
		false
	);
	free(rendered_ptx);

	CUmodule cu_module;
	cuCheck( cuModuleLoadDataEx(&cu_module, sass, 0, NULL, NULL) );
	free(sass);

	CUfunction cu_function;
	cuCheck( cuModuleGetFunction(&cu_function, cu_module, "kernel") );

	float h_out = 0.0f;
	CUdeviceptr d_out;
	cuCheck( cuMemAlloc(&d_out, sizeof(float)) );

	void* args[] = {
		(void*)&d_out
	};

	cuCheck(
		cuLaunchKernel(
			cu_function,
			1, 1, 1,
			1, 1, 1,
			0,
			0,
			args,
			NULL
		)
	);
	cuCheck( cuCtxSynchronize() );
	cuCheck( cuMemcpyDtoH(&h_out, d_out, sizeof(float)) );
	cuCheck( cuMemFree(d_out) );
	cuCheck( cuModuleUnload(cu_module) );
	cuCheck( cuCtxDestroy(cu_context) );

	union {
		float f;
		uint32_t u;
	} bits = {
		.f = h_out
	};
	printf("0x%08X\n", (unsigned)bits.u);
	return 0;
}
