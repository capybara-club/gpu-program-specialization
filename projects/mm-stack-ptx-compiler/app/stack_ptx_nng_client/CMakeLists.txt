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
cmake_minimum_required(VERSION 3.24)
project(app_stack_ptx_nng_client C)

if(NOT TARGET stack_ptx_compiler_nng)
  message(FATAL_ERROR "stack_ptx_compiler_nng target not found. Ensure nng is available.")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(STACK_PTX_NNG_CLIENT_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(STACK_PTX_NNG_CLIENT_DESCRIPTIONS_JSON "${CMAKE_CURRENT_SOURCE_DIR}/stack_ptx_nng_client_descriptions.json")
set(STACK_PTX_NNG_CLIENT_GENERATED_HDR "${STACK_PTX_NNG_CLIENT_GENERATED_DIR}/stack_ptx_nng_client_descriptions.h")

mm_add_stack_ptx_header(
  OUT_HDR "${STACK_PTX_NNG_CLIENT_GENERATED_HDR}"
  JSON    "${STACK_PTX_NNG_CLIENT_DESCRIPTIONS_JSON}"
  LANG    c
)

add_custom_target(stack_ptx_nng_client_generated_headers
  DEPENDS
    "${STACK_PTX_NNG_CLIENT_GENERATED_HDR}"
)

add_executable(stack_ptx_nng_client
  stack_ptx_nng_client.c
)
target_link_libraries(stack_ptx_nng_client PRIVATE stack_ptx_compiler_nng)
target_include_directories(stack_ptx_nng_client PRIVATE "${STACK_PTX_NNG_CLIENT_GENERATED_DIR}")
target_compile_features(stack_ptx_nng_client PRIVATE c_std_11)
add_dependencies(stack_ptx_nng_client stack_ptx_nng_client_generated_headers)
