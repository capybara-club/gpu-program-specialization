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
function(implicit_sindy_add_cuda_root_candidate out_var root)
  if(root)
    get_filename_component(_implicit_sindy_cuda_root "${root}" ABSOLUTE)
    if(EXISTS "${_implicit_sindy_cuda_root}/include" AND EXISTS "${_implicit_sindy_cuda_root}/lib")
      list(APPEND ${out_var} "${_implicit_sindy_cuda_root}")
      set(${out_var} "${${out_var}}" PARENT_SCOPE)
    endif()
  endif()
endfunction()

function(implicit_sindy_add_cuda_roots_from_base out_var base)
  if(NOT base)
    return()
  endif()
  get_filename_component(_implicit_sindy_cuda_base "${base}" ABSOLUTE)
  foreach(_implicit_sindy_cuda_name cu13 cu12)
    implicit_sindy_add_cuda_root_candidate(
      ${out_var}
      "${_implicit_sindy_cuda_base}/nvidia/${_implicit_sindy_cuda_name}")
    file(GLOB _implicit_sindy_cuda_site_roots
      "${_implicit_sindy_cuda_base}/lib/python*/site-packages/nvidia/${_implicit_sindy_cuda_name}"
      "${_implicit_sindy_cuda_base}/lib/*/python*/site-packages/nvidia/${_implicit_sindy_cuda_name}"
      "${_implicit_sindy_cuda_base}/Lib/site-packages/nvidia/${_implicit_sindy_cuda_name}")
    foreach(_implicit_sindy_cuda_site_root IN LISTS _implicit_sindy_cuda_site_roots)
      implicit_sindy_add_cuda_root_candidate(${out_var} "${_implicit_sindy_cuda_site_root}")
    endforeach()
  endforeach()
  set(${out_var} "${${out_var}}" PARENT_SCOPE)
endfunction()

function(implicit_sindy_cuda_root_has_jit_stack root out_var)
  find_library(_implicit_sindy_probe_nvrtc
    NAMES nvrtc libnvrtc.so.13 libnvrtc.so.12
    PATHS "${root}/lib"
    NO_DEFAULT_PATH)
  find_library(_implicit_sindy_probe_nvjitlink
    NAMES nvJitLink libnvJitLink.so.13 libnvJitLink.so.12
    PATHS "${root}/lib"
    NO_DEFAULT_PATH)
  find_library(_implicit_sindy_probe_nvptxcompiler_static
    NAMES nvptxcompiler_static libnvptxcompiler_static.a
    PATHS "${root}/lib"
    NO_DEFAULT_PATH)
  find_library(_implicit_sindy_probe_nvptxcompiler_shared
    NAMES nvptxcompiler libnvptxcompiler.so libnvptxcompiler.so.13 libnvptxcompiler.so.12
    PATHS "${root}/lib"
    NO_DEFAULT_PATH)
  if(_implicit_sindy_probe_nvrtc AND _implicit_sindy_probe_nvjitlink AND
      (_implicit_sindy_probe_nvptxcompiler_static OR _implicit_sindy_probe_nvptxcompiler_shared))
    set(${out_var} ON PARENT_SCOPE)
  else()
    set(${out_var} OFF PARENT_SCOPE)
  endif()
  unset(_implicit_sindy_probe_nvrtc CACHE)
  unset(_implicit_sindy_probe_nvjitlink CACHE)
  unset(_implicit_sindy_probe_nvptxcompiler_static CACHE)
  unset(_implicit_sindy_probe_nvptxcompiler_shared CACHE)
endfunction()

function(implicit_sindy_detect_python_cuda_root)
  set(IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR "" PARENT_SCOPE)
  set(IMPLICIT_SINDY_PYTHON_CUDA_LIBRARY_DIR "" PARENT_SCOPE)
  if(NOT BUILD_PYTHON_BINDINGS OR NOT IMPLICIT_SINDY_PREFER_PYTHON_CUDA)
    return()
  endif()

  set(_implicit_sindy_python_cuda_root_candidates)
  if(IMPLICIT_SINDY_PYTHON_CUDA_ROOT)
    implicit_sindy_add_cuda_root_candidate(
      _implicit_sindy_python_cuda_root_candidates
      "${IMPLICIT_SINDY_PYTHON_CUDA_ROOT}")
  endif()
  if(NOT IMPLICIT_SINDY_PYTHON_CUDA_ROOT)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
        "import pathlib, site, sys, sysconfig\npaths=[]\nfor key in ('purelib','platlib'):\n    value=sysconfig.get_paths().get(key)\n    if value: paths.append(value)\ntry:\n    paths.extend(site.getsitepackages())\nexcept Exception:\n    pass\ntry:\n    paths.append(site.getusersitepackages())\nexcept Exception:\n    pass\npaths.extend(sys.path)\nseen=set()\nfor base in paths:\n    if not base or base in seen: continue\n    seen.add(base)\n    for name in ('cu13','cu12'):\n        root=pathlib.Path(base)/'nvidia'/name\n        if (root/'include').is_dir() and (root/'lib').is_dir():\n            print(root)\n            sys.exit(0)\nsys.exit(0)\n"
      RESULT_VARIABLE IMPLICIT_SINDY_PYTHON_CUDA_STATUS
      OUTPUT_VARIABLE IMPLICIT_SINDY_PYTHON_CUDA_ROOT_DETECTED
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(IMPLICIT_SINDY_PYTHON_CUDA_STATUS EQUAL 0 AND NOT IMPLICIT_SINDY_PYTHON_CUDA_ROOT_DETECTED STREQUAL "")
      implicit_sindy_add_cuda_root_candidate(
        _implicit_sindy_python_cuda_root_candidates
        "${IMPLICIT_SINDY_PYTHON_CUDA_ROOT_DETECTED}")
    endif()
  endif()
  foreach(_implicit_sindy_prefix IN LISTS CMAKE_PREFIX_PATH)
    implicit_sindy_add_cuda_roots_from_base(
      _implicit_sindy_python_cuda_root_candidates
      "${_implicit_sindy_prefix}")
  endforeach()
  if(DEFINED ENV{CMAKE_PREFIX_PATH})
    cmake_path(CONVERT "$ENV{CMAKE_PREFIX_PATH}" TO_CMAKE_PATH_LIST _implicit_sindy_env_prefixes)
    foreach(_implicit_sindy_prefix IN LISTS _implicit_sindy_env_prefixes)
      implicit_sindy_add_cuda_roots_from_base(
        _implicit_sindy_python_cuda_root_candidates
        "${_implicit_sindy_prefix}")
    endforeach()
  endif()

  list(REMOVE_DUPLICATES _implicit_sindy_python_cuda_root_candidates)
  set(_implicit_sindy_python_cuda_partial_root "")
  set(_implicit_sindy_python_cuda_complete_root "")
  foreach(_implicit_sindy_root IN LISTS _implicit_sindy_python_cuda_root_candidates)
    find_library(_implicit_sindy_candidate_nvrtc
      NAMES nvrtc libnvrtc.so.13 libnvrtc.so.12
      PATHS "${_implicit_sindy_root}/lib"
      NO_DEFAULT_PATH)
    find_library(_implicit_sindy_candidate_nvjitlink
      NAMES nvJitLink libnvJitLink.so.13 libnvJitLink.so.12
      PATHS "${_implicit_sindy_root}/lib"
      NO_DEFAULT_PATH)
    if(_implicit_sindy_candidate_nvrtc AND _implicit_sindy_candidate_nvjitlink AND NOT _implicit_sindy_python_cuda_partial_root)
      set(_implicit_sindy_python_cuda_partial_root "${_implicit_sindy_root}")
    endif()
    unset(_implicit_sindy_candidate_nvrtc CACHE)
    unset(_implicit_sindy_candidate_nvjitlink CACHE)
    implicit_sindy_cuda_root_has_jit_stack("${_implicit_sindy_root}" _implicit_sindy_has_jit_stack)
    if(_implicit_sindy_has_jit_stack)
      set(_implicit_sindy_python_cuda_complete_root "${_implicit_sindy_root}")
      break()
    endif()
  endforeach()

  if(_implicit_sindy_python_cuda_complete_root)
    set(IMPLICIT_SINDY_PYTHON_CUDA_ROOT "${_implicit_sindy_python_cuda_complete_root}")
    set(IMPLICIT_SINDY_PYTHON_CUDA_ROOT "${_implicit_sindy_python_cuda_complete_root}" CACHE PATH
      "Optional nvidia/cuXX root from a Python CUDA package. Empty auto-detects from Python site-packages when building Python bindings." FORCE)
    set(IMPLICIT_SINDY_PYTHON_CUDA_ROOT "${_implicit_sindy_python_cuda_complete_root}" PARENT_SCOPE)
  elseif(_implicit_sindy_python_cuda_partial_root)
    set(IMPLICIT_SINDY_PYTHON_CUDA_ROOT "${_implicit_sindy_python_cuda_partial_root}")
    set(IMPLICIT_SINDY_PYTHON_CUDA_ROOT "${_implicit_sindy_python_cuda_partial_root}" CACHE PATH
      "Optional nvidia/cuXX root from a Python CUDA package. Empty auto-detects from Python site-packages when building Python bindings." FORCE)
    set(IMPLICIT_SINDY_PYTHON_CUDA_ROOT "${_implicit_sindy_python_cuda_partial_root}" PARENT_SCOPE)
  endif()

  if(IMPLICIT_SINDY_PYTHON_CUDA_ROOT)
    set(_implicit_sindy_python_cuda_include_dir "${IMPLICIT_SINDY_PYTHON_CUDA_ROOT}/include")
    set(_implicit_sindy_python_cuda_library_dir "${IMPLICIT_SINDY_PYTHON_CUDA_ROOT}/lib")
    if(EXISTS "${_implicit_sindy_python_cuda_include_dir}" AND EXISTS "${_implicit_sindy_python_cuda_library_dir}")
      message(STATUS "implicit-sindy: preferring Python CUDA root ${IMPLICIT_SINDY_PYTHON_CUDA_ROOT}")
      set(IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR "${_implicit_sindy_python_cuda_include_dir}" PARENT_SCOPE)
      set(IMPLICIT_SINDY_PYTHON_CUDA_LIBRARY_DIR "${_implicit_sindy_python_cuda_library_dir}" PARENT_SCOPE)
    else()
      message(WARNING "IMPLICIT_SINDY_PYTHON_CUDA_ROOT does not contain include/lib: ${IMPLICIT_SINDY_PYTHON_CUDA_ROOT}")
      set(IMPLICIT_SINDY_PYTHON_CUDA_INCLUDE_DIR "" PARENT_SCOPE)
      set(IMPLICIT_SINDY_PYTHON_CUDA_LIBRARY_DIR "" PARENT_SCOPE)
    endif()
  endif()
endfunction()
