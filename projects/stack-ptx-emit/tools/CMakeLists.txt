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
set(STACK_PTX_EMIT_TOOLS_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(STACK_PTX_EMIT_DESCRIPTIONS_JSON "${CMAKE_CURRENT_SOURCE_DIR}/../descriptions/stack_ptx_descriptions.json")
set(STACK_PTX_EMIT_GENERATED_C_HDR "${STACK_PTX_EMIT_TOOLS_GENERATED_DIR}/stack_ptx_tools_descriptions.h")

mm_add_stack_ptx_header(
    OUT_HDR "${STACK_PTX_EMIT_GENERATED_C_HDR}"
    JSON    "${STACK_PTX_EMIT_DESCRIPTIONS_JSON}"
    LANG    c
)

add_custom_target(stack_ptx_emit_generated_headers
    DEPENDS
        "${STACK_PTX_EMIT_GENERATED_C_HDR}"
)

add_library(stack_ptx_emit_generated INTERFACE)
target_include_directories(stack_ptx_emit_generated INTERFACE "${STACK_PTX_EMIT_TOOLS_GENERATED_DIR}")
add_dependencies(stack_ptx_emit_generated stack_ptx_emit_generated_headers)

add_executable(stack_ptx_emit_cli stack_ptx_emit_cli.c)
target_link_libraries(
    stack_ptx_emit_cli
    PRIVATE
    m
    stack_ptx_emit_headers
    stack_ptx_emit_generated
    stack_ptx_emit_mm_ptx_common
)

add_executable(
    stack_ptx_emit_linear_regression_cli
    stack_ptx_emit_linear_regression_cli.c
)
target_link_libraries(
    stack_ptx_emit_linear_regression_cli
    PRIVATE
    m
    stack_ptx_emit_headers
    stack_ptx_emit_generated
    stack_ptx_emit_mm_ptx_common
)
