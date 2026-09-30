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

set(_cfp_cmake_rdc_dir "${CMAKE_CURRENT_BINARY_DIR}/cuda")
set(_cfp_arch "sm_${CUBIN_FUNCTION_PATCH_TEST_SM}")

function(cfp_add_rdc_cubin output_var source_name)
  set(source "${CUBIN_FUNCTION_PATCH_TEST_COMMON_CUDA_DIR}/${source_name}.cu")
  set(output "${_cfp_cmake_rdc_dir}/${source_name}.rdc.cubin")
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_cfp_cmake_rdc_dir}"
    COMMAND "${CUBIN_FUNCTION_PATCH_NVCC}"
      -arch=${_cfp_arch}
      -rdc=true
      --cubin
      "${source}"
      -o "${output}"
    DEPENDS "${source}"
    VERBATIM)
  set(${output_var} "${output}" PARENT_SCOPE)
endfunction()

cfp_add_rdc_cubin(CFP_CALLER_RDC_CUBIN patch_caller)
cfp_add_rdc_cubin(CFP_RESERVED_RDC_CUBIN patch_reserved)
cfp_add_rdc_cubin(CFP_SIMPLE_RDC_CUBIN patch_simple)
cfp_add_rdc_cubin(CFP_POLY_RDC_CUBIN patch_poly)
cfp_add_rdc_cubin(CFP_SWITCH_RESERVED_RDC_CUBIN patch_switch_reserved)
cfp_add_rdc_cubin(CFP_SWITCH_REPLACEMENT_RDC_CUBIN patch_switch_replacement)
cfp_add_rdc_cubin(CFP_PANEL_CALLER_RDC_CUBIN patch_panel_caller)
cfp_add_rdc_cubin(CFP_PANEL_RESERVED_RDC_CUBIN patch_panel_reserved)
cfp_add_rdc_cubin(CFP_PANEL_REPLACEMENT_RDC_CUBIN patch_panel_replacement)
cfp_add_rdc_cubin(CFP_MULTI_PANEL_CALLER_RDC_CUBIN patch_multi_panel_caller)
cfp_add_rdc_cubin(CFP_MULTI_PANEL_RESERVED_RDC_CUBIN patch_multi_panel_reserved)
cfp_add_rdc_cubin(CFP_MULTI_PANEL_REPLACEMENT_RDC_CUBIN patch_multi_panel_replacement)
cfp_add_rdc_cubin(CFP_MULTI_PANEL_SPARSE_REPLACEMENT_RDC_CUBIN patch_multi_panel_sparse_replacement)

set(CFP_RESERVED_LINKED_CUBIN "${_cfp_cmake_rdc_dir}/reserved.linked.cubin")
set(CFP_SIMPLE_LINKED_CUBIN "${_cfp_cmake_rdc_dir}/simple.linked.cubin")
set(CFP_POLY_LINKED_CUBIN "${_cfp_cmake_rdc_dir}/poly.linked.cubin")
set(CFP_SWITCH_RESERVED_LINKED_CUBIN "${_cfp_cmake_rdc_dir}/switch_reserved.linked.cubin")
set(CFP_SWITCH_REPLACEMENT_LINKED_CUBIN "${_cfp_cmake_rdc_dir}/switch_replacement.linked.cubin")
set(CFP_PANEL_RESERVED_LINKED_CUBIN "${_cfp_cmake_rdc_dir}/panel_reserved.linked.cubin")
set(CFP_PANEL_REPLACEMENT_LINKED_CUBIN "${_cfp_cmake_rdc_dir}/panel_replacement.linked.cubin")
set(CFP_MULTI_PANEL_RESERVED_LINKED_CUBIN "${_cfp_cmake_rdc_dir}/multi_panel_reserved.linked.cubin")
set(CFP_MULTI_PANEL_REPLACEMENT_LINKED_CUBIN "${_cfp_cmake_rdc_dir}/multi_panel_replacement.linked.cubin")
set(CFP_MULTI_PANEL_SPARSE_REPLACEMENT_LINKED_CUBIN "${_cfp_cmake_rdc_dir}/multi_panel_sparse_replacement.linked.cubin")

function(cfp_add_linked_cubin output caller site)
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_cfp_cmake_rdc_dir}"
    COMMAND "${CUBIN_FUNCTION_PATCH_NVLINK}"
      -arch=${_cfp_arch}
      "${caller}"
      "${site}"
      -o "${output}"
    DEPENDS "${caller}" "${site}"
    VERBATIM)
endfunction()

cfp_add_linked_cubin("${CFP_RESERVED_LINKED_CUBIN}" "${CFP_CALLER_RDC_CUBIN}" "${CFP_RESERVED_RDC_CUBIN}")
cfp_add_linked_cubin("${CFP_SIMPLE_LINKED_CUBIN}" "${CFP_CALLER_RDC_CUBIN}" "${CFP_SIMPLE_RDC_CUBIN}")
cfp_add_linked_cubin("${CFP_POLY_LINKED_CUBIN}" "${CFP_CALLER_RDC_CUBIN}" "${CFP_POLY_RDC_CUBIN}")
cfp_add_linked_cubin("${CFP_SWITCH_RESERVED_LINKED_CUBIN}" "${CFP_CALLER_RDC_CUBIN}" "${CFP_SWITCH_RESERVED_RDC_CUBIN}")
cfp_add_linked_cubin("${CFP_SWITCH_REPLACEMENT_LINKED_CUBIN}" "${CFP_CALLER_RDC_CUBIN}" "${CFP_SWITCH_REPLACEMENT_RDC_CUBIN}")
cfp_add_linked_cubin("${CFP_PANEL_RESERVED_LINKED_CUBIN}" "${CFP_PANEL_CALLER_RDC_CUBIN}" "${CFP_PANEL_RESERVED_RDC_CUBIN}")
cfp_add_linked_cubin("${CFP_PANEL_REPLACEMENT_LINKED_CUBIN}" "${CFP_PANEL_CALLER_RDC_CUBIN}" "${CFP_PANEL_REPLACEMENT_RDC_CUBIN}")
cfp_add_linked_cubin("${CFP_MULTI_PANEL_RESERVED_LINKED_CUBIN}" "${CFP_MULTI_PANEL_CALLER_RDC_CUBIN}" "${CFP_MULTI_PANEL_RESERVED_RDC_CUBIN}")
cfp_add_linked_cubin("${CFP_MULTI_PANEL_REPLACEMENT_LINKED_CUBIN}" "${CFP_MULTI_PANEL_CALLER_RDC_CUBIN}" "${CFP_MULTI_PANEL_REPLACEMENT_RDC_CUBIN}")
cfp_add_linked_cubin("${CFP_MULTI_PANEL_SPARSE_REPLACEMENT_LINKED_CUBIN}" "${CFP_MULTI_PANEL_CALLER_RDC_CUBIN}" "${CFP_MULTI_PANEL_SPARSE_REPLACEMENT_RDC_CUBIN}")

set(_cfp_cmake_rdc_incbin_artifacts
  "${CFP_RESERVED_LINKED_CUBIN}"
  "${CFP_SIMPLE_LINKED_CUBIN}"
  "${CFP_POLY_LINKED_CUBIN}"
  "${CFP_SIMPLE_RDC_CUBIN}"
  "${CFP_POLY_RDC_CUBIN}"
  "${CFP_SWITCH_RESERVED_LINKED_CUBIN}"
  "${CFP_SWITCH_REPLACEMENT_LINKED_CUBIN}"
  "${CFP_SWITCH_REPLACEMENT_RDC_CUBIN}"
  "${CFP_PANEL_RESERVED_LINKED_CUBIN}"
  "${CFP_PANEL_REPLACEMENT_LINKED_CUBIN}"
  "${CFP_PANEL_REPLACEMENT_RDC_CUBIN}"
  "${CFP_MULTI_PANEL_RESERVED_LINKED_CUBIN}"
  "${CFP_MULTI_PANEL_REPLACEMENT_LINKED_CUBIN}"
  "${CFP_MULTI_PANEL_SPARSE_REPLACEMENT_LINKED_CUBIN}"
  "${CFP_MULTI_PANEL_REPLACEMENT_RDC_CUBIN}"
  "${CFP_MULTI_PANEL_SPARSE_REPLACEMENT_RDC_CUBIN}")

add_custom_target(cubin_function_patch_cmake_rdc_artifacts
  DEPENDS ${_cfp_cmake_rdc_incbin_artifacts})

add_executable(cubin_function_patch_cmake_rdc_test
  cmake_rdc_test.c
  cmake_rdc_artifacts.c)
add_dependencies(cubin_function_patch_cmake_rdc_test
  cubin_function_patch_cmake_rdc_artifacts)

set_source_files_properties(cmake_rdc_artifacts.c PROPERTIES
  OBJECT_DEPENDS "${_cfp_cmake_rdc_incbin_artifacts}")

target_compile_features(cubin_function_patch_cmake_rdc_test PRIVATE c_std_99)
target_include_directories(cubin_function_patch_cmake_rdc_test PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/thirdparty/incbin
  ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(cubin_function_patch_cmake_rdc_test PRIVATE
  CFP_CMAKE_RDC_TEST_SM=${CUBIN_FUNCTION_PATCH_TEST_SM}
  CFP_RESERVED_LINKED_CUBIN_PATH="${CFP_RESERVED_LINKED_CUBIN}"
  CFP_SIMPLE_LINKED_CUBIN_PATH="${CFP_SIMPLE_LINKED_CUBIN}"
  CFP_POLY_LINKED_CUBIN_PATH="${CFP_POLY_LINKED_CUBIN}"
  CFP_SIMPLE_RDC_CUBIN_PATH="${CFP_SIMPLE_RDC_CUBIN}"
  CFP_POLY_RDC_CUBIN_PATH="${CFP_POLY_RDC_CUBIN}"
  CFP_SWITCH_RESERVED_LINKED_CUBIN_PATH="${CFP_SWITCH_RESERVED_LINKED_CUBIN}"
  CFP_SWITCH_REPLACEMENT_LINKED_CUBIN_PATH="${CFP_SWITCH_REPLACEMENT_LINKED_CUBIN}"
  CFP_SWITCH_REPLACEMENT_RDC_CUBIN_PATH="${CFP_SWITCH_REPLACEMENT_RDC_CUBIN}"
  CFP_PANEL_RESERVED_LINKED_CUBIN_PATH="${CFP_PANEL_RESERVED_LINKED_CUBIN}"
  CFP_PANEL_REPLACEMENT_LINKED_CUBIN_PATH="${CFP_PANEL_REPLACEMENT_LINKED_CUBIN}"
  CFP_PANEL_REPLACEMENT_RDC_CUBIN_PATH="${CFP_PANEL_REPLACEMENT_RDC_CUBIN}"
  CFP_MULTI_PANEL_RESERVED_LINKED_CUBIN_PATH="${CFP_MULTI_PANEL_RESERVED_LINKED_CUBIN}"
  CFP_MULTI_PANEL_REPLACEMENT_LINKED_CUBIN_PATH="${CFP_MULTI_PANEL_REPLACEMENT_LINKED_CUBIN}"
  CFP_MULTI_PANEL_SPARSE_REPLACEMENT_LINKED_CUBIN_PATH="${CFP_MULTI_PANEL_SPARSE_REPLACEMENT_LINKED_CUBIN}"
  CFP_MULTI_PANEL_REPLACEMENT_RDC_CUBIN_PATH="${CFP_MULTI_PANEL_REPLACEMENT_RDC_CUBIN}"
  CFP_MULTI_PANEL_SPARSE_REPLACEMENT_RDC_CUBIN_PATH="${CFP_MULTI_PANEL_SPARSE_REPLACEMENT_RDC_CUBIN}")
target_link_libraries(cubin_function_patch_cmake_rdc_test PRIVATE
  cubin_function_patch
  CUDA::cuda_driver
  m)

add_test(
  NAME cubin_function_patch_cmake_rdc_test
  COMMAND cubin_function_patch_cmake_rdc_test)
