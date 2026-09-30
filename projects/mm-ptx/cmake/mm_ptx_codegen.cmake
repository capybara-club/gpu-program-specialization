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
# cmake/mm_ptx_codegen.cmake

function(mm_add_stack_ptx_header)
  set(options)
  set(oneValueArgs OUT_HDR JSON SCRIPT LANG)
  set(multiValueArgs DEPENDS)
  cmake_parse_arguments(MM "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT MM_OUT_HDR)
    message(FATAL_ERROR "mm_add_stack_ptx_header: OUT_HDR is required")
  endif()
  if(NOT MM_JSON)
    message(FATAL_ERROR "mm_add_stack_ptx_header: JSON is required")
  endif()

  if(NOT MM_SCRIPT)
    set(MM_SCRIPT "${MM_PTX_TOOLS_DIR}/stack_ptx_generate_infos.py")
  endif()

  if(NOT MM_LANG)
    set(MM_LANG c)
  endif()

  get_filename_component(_out_dir "${MM_OUT_HDR}" DIRECTORY)

  add_custom_command(
    OUTPUT  "${MM_OUT_HDR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${_out_dir}"
    COMMAND ${Python3_EXECUTABLE} "${MM_SCRIPT}"
            --input  "${MM_JSON}"
            --output "${MM_OUT_HDR}"
            --lang   "${MM_LANG}"
    DEPENDS "${MM_JSON}" "${MM_SCRIPT}" ${MM_DEPENDS}
    COMMENT "Generating Stack-PTX header from ${MM_JSON}"
    VERBATIM
  )

  set_source_files_properties("${MM_OUT_HDR}"
    PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE
  )

  set(${MM_OUT_HDR} "${MM_OUT_HDR}" PARENT_SCOPE)
endfunction()
