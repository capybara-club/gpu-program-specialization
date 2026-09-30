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
# SPDX-FileCopyrightText: Copyright (c) 2026 Charles Durham
# SPDX-License-Identifier: MIT

add_executable(cubin_function_patch_runtime_compile_test
  runtime_compile_test.c
  runtime_compile_sources.c)

target_compile_features(cubin_function_patch_runtime_compile_test PRIVATE c_std_99)
target_include_directories(cubin_function_patch_runtime_compile_test PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/thirdparty/incbin
  ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(cubin_function_patch_runtime_compile_test PRIVATE
  CFP_TEST_PATCH_CALLER_CU_PATH="${CUBIN_FUNCTION_PATCH_TEST_COMMON_CUDA_DIR}/patch_caller.cu"
  CFP_TEST_PATCH_RESERVED_CU_PATH="${CUBIN_FUNCTION_PATCH_TEST_COMMON_CUDA_DIR}/patch_reserved.cu"
  CFP_TEST_PATCH_SIMPLE_CU_PATH="${CUBIN_FUNCTION_PATCH_TEST_COMMON_CUDA_DIR}/patch_simple.cu"
  CFP_TEST_PATCH_POLY_CU_PATH="${CUBIN_FUNCTION_PATCH_TEST_COMMON_CUDA_DIR}/patch_poly.cu"
  CFP_TEST_PATCH_SWITCH_RESERVED_CU_PATH="${CUBIN_FUNCTION_PATCH_TEST_COMMON_CUDA_DIR}/patch_switch_reserved.cu"
  CFP_TEST_PATCH_SWITCH_REPLACEMENT_CU_PATH="${CUBIN_FUNCTION_PATCH_TEST_COMMON_CUDA_DIR}/patch_switch_replacement.cu")
target_link_libraries(cubin_function_patch_runtime_compile_test PRIVATE
  cubin_function_patch
  CUDA::cuda_driver
  CUDA::nvrtc
  CUDA::nvJitLink
  ${CUBIN_FUNCTION_PATCH_NVPTXCOMPILER_TARGET}
  m)

add_test(
  NAME cubin_function_patch_runtime_compile_test
  COMMAND cubin_function_patch_runtime_compile_test)
