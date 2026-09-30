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

project(cusr_amd_probe C CXX)

set(CUSR_AMD_GPU_TARGET "gfx1201" CACHE STRING "AMDGPU target used for HIP code objects")

find_program(CUSR_AMD_HIPCC hipcc REQUIRED)
find_program(CUSR_AMD_LLVM_OBJDUMP llvm-objdump HINTS /usr/lib/llvm-21/bin REQUIRED)
find_program(CUSR_AMD_CLANG_OFFLOAD_BUNDLER clang-offload-bundler HINTS /usr/lib/llvm-21/bin REQUIRED)
find_library(CUSR_AMD_HIP_RUNTIME amdhip64 REQUIRED)

set(CUSR_AMD_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(CUSR_AMD_PROBE_BUNDLE "${CUSR_AMD_GENERATED_DIR}/cusr_amd_island_probe.bundle")
set(CUSR_AMD_PROBE_HSACO "${CUSR_AMD_GENERATED_DIR}/cusr_amd_island_probe.hsaco")
set(CUSR_AMD_PROBE_ISA "${CUSR_AMD_GENERATED_DIR}/cusr_amd_island_probe.isa")

add_custom_command(
    OUTPUT "${CUSR_AMD_PROBE_BUNDLE}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CUSR_AMD_GENERATED_DIR}"
    COMMAND
        "${CUSR_AMD_HIPCC}"
        -O1
        --offload-arch=${CUSR_AMD_GPU_TARGET}
        --genco
        "${CMAKE_CURRENT_SOURCE_DIR}/probe/cusr_amd_island_probe.hip"
        -o "${CUSR_AMD_PROBE_BUNDLE}"
    DEPENDS probe/cusr_amd_island_probe.hip
    VERBATIM
)

add_custom_command(
    OUTPUT "${CUSR_AMD_PROBE_HSACO}"
    COMMAND
        "${CUSR_AMD_CLANG_OFFLOAD_BUNDLER}"
        -unbundle
        -type=o
        -input=${CUSR_AMD_PROBE_BUNDLE}
        -output=${CUSR_AMD_PROBE_HSACO}
        -targets=hipv4-amdgcn-amd-amdhsa--${CUSR_AMD_GPU_TARGET}
    DEPENDS "${CUSR_AMD_PROBE_BUNDLE}"
    VERBATIM
)

add_custom_command(
    OUTPUT "${CUSR_AMD_PROBE_ISA}"
    COMMAND
        "${CUSR_AMD_LLVM_OBJDUMP}"
        --disassemble
        --demangle
        --mcpu=${CUSR_AMD_GPU_TARGET}
        "${CUSR_AMD_PROBE_HSACO}"
        > "${CUSR_AMD_PROBE_ISA}"
    DEPENDS "${CUSR_AMD_PROBE_HSACO}"
    VERBATIM
)

add_custom_target(cusr_amd_probe ALL DEPENDS
    "${CUSR_AMD_PROBE_HSACO}"
    "${CUSR_AMD_PROBE_ISA}"
)

add_executable(cusr_amd_patch_probe
    probe/cusr_amd_patch_probe.c
)

target_compile_features(cusr_amd_patch_probe PRIVATE c_std_99)
target_compile_definitions(cusr_amd_patch_probe PRIVATE __HIP_PLATFORM_AMD__=1)
target_compile_options(cusr_amd_patch_probe PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(cusr_amd_patch_probe PRIVATE "${CUSR_AMD_HIP_RUNTIME}" m)
add_dependencies(cusr_amd_patch_probe cusr_amd_probe)

enable_testing()
add_test(
    NAME cusr_amd_patch_probe
    COMMAND cusr_amd_patch_probe "${CUSR_AMD_PROBE_HSACO}"
)
