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
project(kermac)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

enable_language(CUDA)

string(REGEX MATCH "^[0-9]+" CUDA_MAJOR_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")

# cuContextCreate changed in cuda 13
if(CUDA_MAJOR_VERSION EQUAL 13)
    add_definitions(-DCUDA_VERSION_13)
endif()

add_subdirectory(cutensor)

set(KERMAC_INC         "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(GENERATED_INC_DIR  "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${GENERATED_INC_DIR}")

get_target_property(MM_PTX_HEADER_DIRS mm_ptx_headers INTERFACE_INCLUDE_DIRECTORIES)
set(MM_PTX_COMMON_DIR "")
foreach(dir IN LISTS MM_PTX_HEADER_DIRS)
    if (EXISTS "${dir}/common/philox.cuh")
        set(MM_PTX_COMMON_DIR "${dir}/common")
        break()
    endif()
endforeach()
if (NOT MM_PTX_COMMON_DIR)
    message(FATAL_ERROR "mm-ptx common directory not found; required for philox.cuh")
endif()

option(KERMAC_DEBUG "Enable Kermac debug macros" OFF)

set(STACK_PTX_DESCRIPTIONS_JSON "${CMAKE_CURRENT_SOURCE_DIR}/type_descriptions/stack_ptx_descriptions.json")
set(STACK_PTX_GENERATED_HDR     "${GENERATED_INC_DIR}/kermac_stack_ptx_descriptions.h")
mm_add_stack_ptx_header(OUT_HDR "${STACK_PTX_GENERATED_HDR}" JSON "${STACK_PTX_DESCRIPTIONS_JSON}" LANG c)

function(kermac_add_ptx_obj tgt src)
    add_library(${tgt} OBJECT "${src}")
    target_link_libraries(${tgt} PRIVATE mm_ptx_headers)
    set_target_properties(${tgt} PROPERTIES CUDA_PTX_COMPILATION ON CUDA_ARCHITECTURES native)
endfunction()

function(kermac_add_cubin tgt src out_cubin)
    add_custom_command(
        OUTPUT  "${out_cubin}"
        COMMAND ${CMAKE_CUDA_COMPILER} -O3 -Xptxas -v -arch=native --use_fast_math
                -I "${MM_PTX_COMMON_DIR}"
                -cubin "${src}" -o "${out_cubin}"
        DEPENDS "${src}"
        COMMENT "nvcc ${src} -> ${out_cubin}"
        VERBATIM
    )
    add_custom_target(${tgt} DEPENDS "${out_cubin}")
endfunction()

set(KERNELS_SRC   "${CMAKE_CURRENT_SOURCE_DIR}/cuda/k_kernels.cu")
set(KERNELS_CUBIN "${CMAKE_CURRENT_BINARY_DIR}/k_kernels.cubin")
kermac_add_cubin(kermac_kernels_cubin "${KERNELS_SRC}" "${KERNELS_CUBIN}")

set(KLEGACY_SRC   "${CMAKE_CURRENT_SOURCE_DIR}/cuda/k_legacy.cu")
set(KLEGACY_CUBIN "${CMAKE_CURRENT_BINARY_DIR}/k_legacy.cubin")
kermac_add_cubin(kermac_legacy_cubin "${KLEGACY_SRC}" "${KLEGACY_CUBIN}")

kermac_add_ptx_obj(elementwise_target "${CMAKE_CURRENT_SOURCE_DIR}/cuda/k_elementwise.cu")

kermac_add_ptx_obj(semiring_nt_target       "${CMAKE_CURRENT_SOURCE_DIR}/cuda/semiring/semiring_nt.cu")
kermac_add_ptx_obj(semiring_nt_lower_target "${CMAKE_CURRENT_SOURCE_DIR}/cuda/semiring/semiring_nt_lower.cu")
kermac_add_ptx_obj(semiring_nt_upper_target "${CMAKE_CURRENT_SOURCE_DIR}/cuda/semiring/semiring_nt_upper.cu")
kermac_add_ptx_obj(semiring_nt_lower_copy_grad_target "${CMAKE_CURRENT_SOURCE_DIR}/cuda/semiring/semiring_nt_lower_copy_grad.cu")
kermac_add_ptx_obj(semiring_nt_upper_copy_grad_target "${CMAKE_CURRENT_SOURCE_DIR}/cuda/semiring/semiring_nt_upper_copy_grad.cu")
kermac_add_ptx_obj(semiring_grad_target     "${CMAKE_CURRENT_SOURCE_DIR}/cuda/semiring/semiring_grad.cu")

set(SEMIRING_INCS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/semiring" "${KERMAC_INC}")
if(DEFINED MM_PTX_CUTLASS_INCLUDE_DIR)
    list(APPEND SEMIRING_INCS "${MM_PTX_CUTLASS_INCLUDE_DIR}")
endif()

foreach(t semiring_nt_target semiring_nt_lower_target semiring_nt_upper_target semiring_nt_lower_copy_grad_target semiring_nt_upper_copy_grad_target semiring_grad_target)
    target_include_directories(${t} PRIVATE ${SEMIRING_INCS})
    target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endforeach()

add_library(${PROJECT_NAME} STATIC
    k_core.c 
    k_tensor.c 
    k_cutensor.c 
    k_agop.c 
    k_solver.c 
    k_compare.c
    k_elementwise.c 
    k_semiring.c 
    k_semiring_copy_grad.c
    k_semiring_grad.c 
    k_legacy.c 
    k_elites.c 
    k_compiler.c
    k_nvrtc_compiler.c
    ${PTX_INJECT_GENERATED_HDR} 
    ${STACK_PTX_GENERATED_HDR}
)

set_source_files_properties(
    k_core.c
    PROPERTIES
    OBJECT_DEPENDS "${KERNELS_CUBIN}"
)
set_source_files_properties(
    k_legacy.c
    PROPERTIES
    OBJECT_DEPENDS "${KLEGACY_CUBIN}"
)

target_include_directories(${PROJECT_NAME} PUBLIC "${GENERATED_INC_DIR}" "${KERMAC_INC}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${PROJECT_NAME}
    cutensor 
    CUDA::cuda_driver 
    CUDA::cublas 
    CUDA::cusolver 
    CUDA::nvrtc
    CUDA::nvptxcompiler_static 
    m 
    mm_ptx_headers
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    KERMAC_KERNELS_CUBIN="${KERNELS_CUBIN}"
    KERMAC_LEGACY_KERNELS_CUBIN="${KLEGACY_CUBIN}"
    PTX_ELEMENTWISE="$<TARGET_OBJECTS:elementwise_target>"
    KERMAC_SEMIRING_NT_PTX="$<TARGET_OBJECTS:semiring_nt_target>"
    KERMAC_SEMIRING_NT_LOWER_PTX="$<TARGET_OBJECTS:semiring_nt_lower_target>"
    KERMAC_SEMIRING_NT_UPPER_PTX="$<TARGET_OBJECTS:semiring_nt_upper_target>"
    KERMAC_SEMIRING_NT_LOWER_COPY_GRAD_PTX="$<TARGET_OBJECTS:semiring_nt_lower_copy_grad_target>"
    KERMAC_SEMIRING_NT_UPPER_COPY_GRAD_PTX="$<TARGET_OBJECTS:semiring_nt_upper_copy_grad_target>"
    KERMAC_SEMIRING_GRAD_PTX="$<TARGET_OBJECTS:semiring_grad_target>"
)

add_dependencies(${PROJECT_NAME}
    elementwise_target
    semiring_nt_target 
    semiring_nt_lower_target 
    semiring_nt_upper_target
    semiring_nt_lower_copy_grad_target
    semiring_nt_upper_copy_grad_target
    semiring_grad_target
    kermac_kernels_cubin
    kermac_legacy_cubin
)

if(KERMAC_DEBUG)
    target_compile_definitions(${PROJECT_NAME} PUBLIC KERMAC_DEBUG)
    message(WARNING "KERMAC_DEBUG macros ENABLED for 'kermac' library.")
endif()

add_subdirectory(cpu)

add_subdirectory(bench)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)

set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/imgui")
set(IMGUI_BACKENDS_DIR "${IMGUI_DIR}/backends")

add_executable(
    kermac_agop_viewer
    apps/kermac_agop_viewer.cpp
    "${IMGUI_BACKENDS_DIR}/imgui_impl_glfw.cpp"
    "${IMGUI_BACKENDS_DIR}/imgui_impl_vulkan.cpp"
)

if(TARGET glfw)
    set(GLFW_TARGET glfw)
elseif(TARGET glfw3)
    set(GLFW_TARGET glfw3)
else()
    message(FATAL_ERROR "glfw target not found after find_package(glfw3)")
endif()

target_include_directories(kermac_agop_viewer PRIVATE
    "${IMGUI_DIR}"
    "${IMGUI_BACKENDS_DIR}"
)

target_link_libraries(kermac_agop_viewer PRIVATE
    kermac
    imgui
    ${GLFW_TARGET}
    Vulkan::Vulkan
    ${CMAKE_DL_LIBS}
)

target_compile_definitions(kermac_agop_viewer PRIVATE
    IMGUI_VULKAN_DEMO_FONT_PATH="/usr/share/fonts/noto/NotoSans-Regular.ttf"
    IMGUI_VULKAN_DEMO_FALLBACK_FONT_PATH="${IMGUI_DIR}/misc/fonts/Roboto-Medium.ttf"
)

target_compile_features(kermac_agop_viewer PRIVATE cxx_std_11)

add_executable(
    kermac_data_viewer
    apps/kermac_data_viewer.cpp
    "${IMGUI_BACKENDS_DIR}/imgui_impl_glfw.cpp"
    "${IMGUI_BACKENDS_DIR}/imgui_impl_vulkan.cpp"
)

target_include_directories(kermac_data_viewer PRIVATE
    "${IMGUI_DIR}"
    "${IMGUI_BACKENDS_DIR}"
)

target_link_libraries(kermac_data_viewer PRIVATE
    kermac
    kermac_cpu
    imgui
    ${GLFW_TARGET}
    Vulkan::Vulkan
    ${CMAKE_DL_LIBS}
)

target_compile_definitions(kermac_data_viewer PRIVATE
    IMGUI_VULKAN_DEMO_FONT_PATH="/usr/share/fonts/noto/NotoSans-Regular.ttf"
    IMGUI_VULKAN_DEMO_FALLBACK_FONT_PATH="${IMGUI_DIR}/misc/fonts/Roboto-Medium.ttf"
)

target_compile_features(kermac_data_viewer PRIVATE cxx_std_11)

add_executable(
    kermac_random_tensor
    apps/kermac_random_tensor.cpp
)

target_link_libraries(kermac_random_tensor PRIVATE
    kermac
)

target_compile_features(kermac_random_tensor PRIVATE cxx_std_11)
