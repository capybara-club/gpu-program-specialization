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
cmake_minimum_required(VERSION 3.22)

project(stack-ptx-emit LANGUAGES C CXX)

set(STACK_PTX_EMIT_MM_PTX_DIR_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/../mm-ptx")
set(STACK_PTX_EMIT_MM_PTX_DIR "${STACK_PTX_EMIT_MM_PTX_DIR_DEFAULT}" CACHE PATH "Path to the mm-ptx repository")

if(NOT TARGET mm_ptx_headers)
    if(NOT EXISTS "${STACK_PTX_EMIT_MM_PTX_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "mm-ptx repository not found at ${STACK_PTX_EMIT_MM_PTX_DIR}")
    endif()

    set(MMPTX_BUILD_BENCH OFF CACHE BOOL "Build mm-ptx benchmarks" FORCE)
    set(MMPTX_BUILD_EXAMPLES OFF CACHE BOOL "Build mm-ptx examples" FORCE)
    add_subdirectory("${STACK_PTX_EMIT_MM_PTX_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/mm-ptx")
endif()

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_CUDA_COMPILER_LOADED)
    enable_language(CUDA)
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

if(NOT COMMAND mm_add_stack_ptx_header)
    list(APPEND CMAKE_MODULE_PATH "${STACK_PTX_EMIT_MM_PTX_DIR}/cmake")
    include(mm_ptx_codegen)
endif()

add_library(stack_ptx_emit_headers INTERFACE)
target_include_directories(stack_ptx_emit_headers INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(stack_ptx_emit_headers INTERFACE
    mm_ptx_headers
    stack_ptx_emit_generated
)

add_library(stack_ptx_emit_mm_ptx_common INTERFACE)
target_include_directories(stack_ptx_emit_mm_ptx_common INTERFACE
    "${STACK_PTX_EMIT_MM_PTX_DIR}/common"
    "${STACK_PTX_EMIT_MM_PTX_DIR}/routine_libraries"
)

add_subdirectory(tools)

option(STACK_PTX_EMIT_BUILD_EXAMPLES "Build stack-ptx-emit examples" ON)
if(STACK_PTX_EMIT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
