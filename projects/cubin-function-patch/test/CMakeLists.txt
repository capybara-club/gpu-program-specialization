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

set(CUBIN_FUNCTION_PATCH_TEST_COMMON_CUDA_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/common/cuda")

set(CUBIN_FUNCTION_PATCH_NVPTXCOMPILER_TARGET "")
if(TARGET CUDA::nvptxcompiler_static)
  set(CUBIN_FUNCTION_PATCH_NVPTXCOMPILER_TARGET CUDA::nvptxcompiler_static)
elseif(TARGET CUDA::nvptxcompiler)
  set(CUBIN_FUNCTION_PATCH_NVPTXCOMPILER_TARGET CUDA::nvptxcompiler)
else()
  set(_cfp_nvptx_paths "")
  if(DEFINED CUDAToolkit_LIBRARY_DIR)
    list(APPEND _cfp_nvptx_paths "${CUDAToolkit_LIBRARY_DIR}")
  endif()
  if(DEFINED CUDAToolkit_LIBRARY_DIRS)
    list(APPEND _cfp_nvptx_paths ${CUDAToolkit_LIBRARY_DIRS})
  endif()
  list(APPEND _cfp_nvptx_paths
    "$ENV{CUDA_HOME}/lib64"
    "$ENV{CUDA_PATH}/lib64"
    "/usr/local/cuda/lib64")
  list(REMOVE_DUPLICATES _cfp_nvptx_paths)
  find_library(CUBIN_FUNCTION_PATCH_NVPTXCOMPILER_STATIC
    NAMES nvptxcompiler_static
    PATHS ${_cfp_nvptx_paths})
  find_library(CUBIN_FUNCTION_PATCH_NVPTXCOMPILER
    NAMES nvptxcompiler
    PATHS ${_cfp_nvptx_paths})
  if(CUBIN_FUNCTION_PATCH_NVPTXCOMPILER_STATIC)
    add_library(cfp_nvptxcompiler_static UNKNOWN IMPORTED)
    set_target_properties(cfp_nvptxcompiler_static PROPERTIES
      IMPORTED_LOCATION "${CUBIN_FUNCTION_PATCH_NVPTXCOMPILER_STATIC}")
    set(CUBIN_FUNCTION_PATCH_NVPTXCOMPILER_TARGET cfp_nvptxcompiler_static)
  elseif(CUBIN_FUNCTION_PATCH_NVPTXCOMPILER)
    add_library(cfp_nvptxcompiler UNKNOWN IMPORTED)
    set_target_properties(cfp_nvptxcompiler PROPERTIES
      IMPORTED_LOCATION "${CUBIN_FUNCTION_PATCH_NVPTXCOMPILER}")
    set(CUBIN_FUNCTION_PATCH_NVPTXCOMPILER_TARGET cfp_nvptxcompiler)
  endif()
endif()

if(CUBIN_FUNCTION_PATCH_NVPTXCOMPILER_TARGET STREQUAL "")
  message(FATAL_ERROR "Could not find nvPTXCompiler")
endif()

find_program(CUBIN_FUNCTION_PATCH_NVCC
  NAMES nvcc
  HINTS "${CUDAToolkit_BIN_DIR}" "$ENV{CUDA_HOME}/bin" "$ENV{CUDA_PATH}/bin")
find_program(CUBIN_FUNCTION_PATCH_NVLINK
  NAMES nvlink
  HINTS "${CUDAToolkit_BIN_DIR}" "$ENV{CUDA_HOME}/bin" "$ENV{CUDA_PATH}/bin")

if(NOT CUBIN_FUNCTION_PATCH_NVCC)
  message(FATAL_ERROR "Could not find nvcc")
endif()
if(NOT CUBIN_FUNCTION_PATCH_NVLINK)
  message(FATAL_ERROR "Could not find nvlink")
endif()

set(CUBIN_FUNCTION_PATCH_TEST_SM "" CACHE STRING
  "CUDA SM target for build-time cubin tests, such as 120 or 90.")
if(CUBIN_FUNCTION_PATCH_TEST_SM STREQUAL "")
  find_program(CUBIN_FUNCTION_PATCH_NVIDIA_SMI nvidia-smi)
  if(CUBIN_FUNCTION_PATCH_NVIDIA_SMI)
    execute_process(
      COMMAND "${CUBIN_FUNCTION_PATCH_NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader,nounits
      OUTPUT_VARIABLE _cfp_compute_cap
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" _cfp_first_compute_cap "${_cfp_compute_cap}")
    if(_cfp_first_compute_cap)
      string(REPLACE "." "" CUBIN_FUNCTION_PATCH_TEST_SM "${_cfp_first_compute_cap}")
    endif()
  endif()
endif()
if(CUBIN_FUNCTION_PATCH_TEST_SM STREQUAL "")
  set(CUBIN_FUNCTION_PATCH_TEST_SM "120")
endif()
message(STATUS "cubin-function-patch build-time test SM: sm_${CUBIN_FUNCTION_PATCH_TEST_SM}")

add_subdirectory(runtime_compile)
add_subdirectory(cmake_rdc)
