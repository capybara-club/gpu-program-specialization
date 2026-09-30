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

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER_TARGET "")
if(TARGET CUDA::nvptxcompiler_static)
  set(CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER_TARGET CUDA::nvptxcompiler_static)
elseif(TARGET CUDA::nvptxcompiler)
  set(CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER_TARGET CUDA::nvptxcompiler)
elseif(TARGET cfp_nvptxcompiler_static)
  set(CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER_TARGET cfp_nvptxcompiler_static)
elseif(TARGET cfp_nvptxcompiler)
  set(CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER_TARGET cfp_nvptxcompiler)
else()
  set(_cfp_bench_nvptx_paths "")
  if(DEFINED CUDAToolkit_LIBRARY_DIR)
    list(APPEND _cfp_bench_nvptx_paths "${CUDAToolkit_LIBRARY_DIR}")
  endif()
  if(DEFINED CUDAToolkit_LIBRARY_DIRS)
    list(APPEND _cfp_bench_nvptx_paths ${CUDAToolkit_LIBRARY_DIRS})
  endif()
  list(APPEND _cfp_bench_nvptx_paths
    "$ENV{CUDA_HOME}/lib64"
    "$ENV{CUDA_PATH}/lib64"
    "/usr/local/cuda/lib64")
  list(REMOVE_DUPLICATES _cfp_bench_nvptx_paths)
  find_library(CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER_STATIC
    NAMES nvptxcompiler_static
    PATHS ${_cfp_bench_nvptx_paths})
  find_library(CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER
    NAMES nvptxcompiler
    PATHS ${_cfp_bench_nvptx_paths})
  if(CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER_STATIC)
    add_library(cfp_bench_nvptxcompiler_static UNKNOWN IMPORTED)
    set_target_properties(cfp_bench_nvptxcompiler_static PROPERTIES
      IMPORTED_LOCATION "${CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER_STATIC}")
    set(CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER_TARGET cfp_bench_nvptxcompiler_static)
  elseif(CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER)
    add_library(cfp_bench_nvptxcompiler UNKNOWN IMPORTED)
    set_target_properties(cfp_bench_nvptxcompiler PROPERTIES
      IMPORTED_LOCATION "${CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER}")
    set(CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER_TARGET cfp_bench_nvptxcompiler)
  endif()
endif()

if(CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER_TARGET STREQUAL "")
  message(FATAL_ERROR "Could not find nvPTXCompiler")
endif()

add_executable(cubin_function_patch_compile_scaling_bench
  cubin_function_patch_compile_scaling_bench.c)

target_compile_features(cubin_function_patch_compile_scaling_bench PRIVATE c_std_99)
target_include_directories(cubin_function_patch_compile_scaling_bench PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(cubin_function_patch_compile_scaling_bench PRIVATE
  cubin_function_patch
  CUDA::cuda_driver
  CUDA::nvrtc
  CUDA::nvJitLink
  ${CUBIN_FUNCTION_PATCH_BENCH_NVPTXCOMPILER_TARGET}
  Threads::Threads
  m)

if(BUILD_TESTING)
  add_test(
    NAME cubin_function_patch_compile_scaling_bench_sanity
    COMMAND cubin_function_patch_compile_scaling_bench
      --modules 2
      --workers 1
      --repeats 1
      --warmup 0)
endif()
