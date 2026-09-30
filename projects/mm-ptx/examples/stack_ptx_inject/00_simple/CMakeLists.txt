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
project(stack_ptx_inject_00_simple)

set(KERNEL_TARGET "${PROJECT_NAME}_kernel_ptx")

add_library(${KERNEL_TARGET} OBJECT kernel.cu)
target_link_libraries(${KERNEL_TARGET} PRIVATE mm_ptx_headers)
set_target_properties(${KERNEL_TARGET} PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES native
)

add_executable(${PROJECT_NAME} main.c)
add_dependencies(${PROJECT_NAME} ${KERNEL_TARGET})
target_compile_definitions(${PROJECT_NAME} PRIVATE PTX_KERNEL="$<TARGET_OBJECTS:${KERNEL_TARGET}>")
set(KERNEL_PTX "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${KERNEL_TARGET}.dir/${CMAKE_CFG_INTDIR}/kernel.ptx")
set_source_files_properties(main.c PROPERTIES OBJECT_DEPENDS "${KERNEL_PTX};${CMAKE_CURRENT_SOURCE_DIR}/kernel.cu")

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    CUDA::cuda_driver
    CUDA::nvrtc
    CUDA::nvptxcompiler_static
    m
    mm_ptx_headers
    stack_ptx_example_generated
)
