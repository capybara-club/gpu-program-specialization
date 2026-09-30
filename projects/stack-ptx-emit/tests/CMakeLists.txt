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
set(STACK_PTX_EMIT_TESTS_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

set(STACK_PTX_EMIT_SIMPLE_KERNEL "${CMAKE_CURRENT_SOURCE_DIR}/ptx_inject_simple_kernel.cu")
set(STACK_PTX_EMIT_SIMPLE_KERNEL_TARGET "stack_ptx_emit_simple_kernel_ptx")

add_library(${STACK_PTX_EMIT_SIMPLE_KERNEL_TARGET} OBJECT "${STACK_PTX_EMIT_SIMPLE_KERNEL}")
target_link_libraries(${STACK_PTX_EMIT_SIMPLE_KERNEL_TARGET} PRIVATE mm_ptx_headers)
set_target_properties(${STACK_PTX_EMIT_SIMPLE_KERNEL_TARGET} PROPERTIES
    CUDA_PTX_COMPILATION ON
    CUDA_ARCHITECTURES native
)

set(STACK_PTX_EMIT_SIMPLE_KERNEL_PTX "${STACK_PTX_EMIT_TESTS_GENERATED_DIR}/stack_ptx_emit_simple_kernel.ptx")
add_custom_command(
    OUTPUT "${STACK_PTX_EMIT_SIMPLE_KERNEL_PTX}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${STACK_PTX_EMIT_TESTS_GENERATED_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_OBJECTS:${STACK_PTX_EMIT_SIMPLE_KERNEL_TARGET}>
            "${STACK_PTX_EMIT_SIMPLE_KERNEL_PTX}"
    DEPENDS ${STACK_PTX_EMIT_SIMPLE_KERNEL_TARGET}
    COMMENT "Preparing PTX for stack_ptx_emit_cli_e2e"
    VERBATIM
)
add_custom_target(stack_ptx_emit_simple_kernel_ptx_file DEPENDS "${STACK_PTX_EMIT_SIMPLE_KERNEL_PTX}")

add_executable(stack_ptx_emit_cli_ptx_runner test_stack_ptx_emit_cli_ptx_runner.c)
add_dependencies(stack_ptx_emit_cli_ptx_runner stack_ptx_emit_simple_kernel_ptx_file)
add_dependencies(stack_ptx_emit_cli_ptx_runner stack_ptx_emit_cli)
set_source_files_properties(
    test_stack_ptx_emit_cli_ptx_runner.c
    PROPERTIES
    OBJECT_DEPENDS "${STACK_PTX_EMIT_SIMPLE_KERNEL_PTX}"
)
target_compile_definitions(stack_ptx_emit_cli_ptx_runner PRIVATE PTX_KERNEL="${STACK_PTX_EMIT_SIMPLE_KERNEL_PTX}")
target_link_libraries(
    stack_ptx_emit_cli_ptx_runner
    PRIVATE
    CUDA::cuda_driver
    CUDA::nvptxcompiler_static
    m
    stack_ptx_emit_headers
    stack_ptx_emit_mm_ptx_common
)

add_executable(
    stack_ptx_emit_linear_regression_e2e_generator
    test_stack_ptx_emit_linear_regression_e2e_generator.c
)
target_link_libraries(
    stack_ptx_emit_linear_regression_e2e_generator
    PRIVATE
    m
    stack_ptx_emit_headers
    stack_ptx_emit_generated
    stack_ptx_emit_mm_ptx_common
)

add_executable(
    stack_ptx_emit_linear_regression_latex_e2e_generator
    test_stack_ptx_emit_linear_regression_latex_e2e_generator.c
)
target_link_libraries(
    stack_ptx_emit_linear_regression_latex_e2e_generator
    PRIVATE
    m
    stack_ptx_emit_headers
    stack_ptx_emit_generated
    stack_ptx_emit_mm_ptx_common
)

add_executable(
    stack_ptx_emit_linear_regression_markdown_factor
    test_stack_ptx_emit_linear_regression_markdown_factor.c
)
target_link_libraries(
    stack_ptx_emit_linear_regression_markdown_factor
    PRIVATE
    m
    stack_ptx_emit_headers
    stack_ptx_emit_generated
    stack_ptx_emit_mm_ptx_common
)

find_program(STACK_PTX_EMIT_RUSTC_EXECUTABLE NAMES rustc PATHS "$ENV{HOME}/.cargo/bin" NO_DEFAULT_PATH)
if(NOT STACK_PTX_EMIT_RUSTC_EXECUTABLE)
    find_program(STACK_PTX_EMIT_RUSTC_EXECUTABLE NAMES rustc)
endif()
find_program(STACK_PTX_EMIT_PDFLATEX_EXECUTABLE NAMES pdflatex)

set(STACK_PTX_EMIT_CLI_PATH "${CMAKE_CURRENT_BINARY_DIR}/../tools/stack_ptx_emit_cli")
set(STACK_PTX_EMIT_LINEAR_REGRESSION_CLI_PATH "${CMAKE_CURRENT_BINARY_DIR}/../tools/stack_ptx_emit_linear_regression_cli")
set(STACK_PTX_EMIT_PTX_RUNNER_PATH "${CMAKE_CURRENT_BINARY_DIR}/stack_ptx_emit_cli_ptx_runner")
set(STACK_PTX_EMIT_LINEAR_REGRESSION_E2E_GENERATOR_PATH "${CMAKE_CURRENT_BINARY_DIR}/stack_ptx_emit_linear_regression_e2e_generator")
set(STACK_PTX_EMIT_LINEAR_REGRESSION_LATEX_E2E_GENERATOR_PATH "${CMAKE_CURRENT_BINARY_DIR}/stack_ptx_emit_linear_regression_latex_e2e_generator")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/stack_ptx_emit_cli_e2e.sh.in"
    "${CMAKE_CURRENT_BINARY_DIR}/stack_ptx_emit_cli_e2e.sh"
    @ONLY
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/stack_ptx_emit_linear_regression_e2e.sh.in"
    "${CMAKE_CURRENT_BINARY_DIR}/stack_ptx_emit_linear_regression_e2e.sh"
    @ONLY
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/stack_ptx_emit_linear_regression_latex_e2e.sh.in"
    "${CMAKE_CURRENT_BINARY_DIR}/stack_ptx_emit_linear_regression_latex_e2e.sh"
    @ONLY
)

add_test(
    NAME stack_ptx_emit.stack_ptx_emit_cli_e2e
    COMMAND bash "${CMAKE_CURRENT_BINARY_DIR}/stack_ptx_emit_cli_e2e.sh"
)
set_tests_properties(stack_ptx_emit.stack_ptx_emit_cli_e2e PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME stack_ptx_emit.stack_ptx_emit_linear_regression_cli_json
    COMMAND
    ${Python3_EXECUTABLE}
    "${CMAKE_CURRENT_SOURCE_DIR}/test_stack_ptx_emit_linear_regression_cli_json.py"
    "${STACK_PTX_EMIT_LINEAR_REGRESSION_CLI_PATH}"
)

add_test(
    NAME stack_ptx_emit.stack_ptx_emit_linear_regression_e2e
    COMMAND bash "${CMAKE_CURRENT_BINARY_DIR}/stack_ptx_emit_linear_regression_e2e.sh"
)
set_tests_properties(stack_ptx_emit.stack_ptx_emit_linear_regression_e2e PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME stack_ptx_emit.stack_ptx_emit_linear_regression_latex_e2e
    COMMAND bash "${CMAKE_CURRENT_BINARY_DIR}/stack_ptx_emit_linear_regression_latex_e2e.sh"
)
set_tests_properties(stack_ptx_emit.stack_ptx_emit_linear_regression_latex_e2e PROPERTIES SKIP_RETURN_CODE 77)

if(TARGET stack_ptx_emit_linear_regression_codegen_example)
    add_test(
        NAME stack_ptx_emit.linear_regression_codegen_example_c
        COMMAND bash -lc
        "output=\"$($<TARGET_FILE:stack_ptx_emit_linear_regression_codegen_example> --backend c --show-spec --input-names age,income,score,balance)\" && grep -F 'Evaluate the hard-coded linear regression model for one observation.' <<<\"$output\" >/dev/null && grep -F '@param x Input vector with layout:' <<<\"$output\" >/dev/null && grep -F 'x[0] = age' <<<\"$output\" >/dev/null && grep -F 'beta_standardized' <<<\"$output\" >/dev/null && grep -F 'intercept' <<<\"$output\" >/dev/null && grep -F '0.0' <<<\"$output\" >/dev/null && grep -F '2.5' <<<\"$output\" >/dev/null && grep -F 'const float age = x[0];' <<<\"$output\" >/dev/null && grep -F 'const float b0 = 2.5f;' <<<\"$output\" >/dev/null && grep -F 'const float intercept = -5.8125f;' <<<\"$output\" >/dev/null && grep -F 'float accum = intercept;' <<<\"$output\" >/dev/null"
    )

    add_test(
        NAME stack_ptx_emit.linear_regression_codegen_example_c_default_names
        COMMAND bash -lc
        "output=\"$($<TARGET_FILE:stack_ptx_emit_linear_regression_codegen_example> --backend c)\" && grep -F 'const float x0 = x[0];' <<<\"$output\" >/dev/null && grep -F 'const float b0 = 2.5f;' <<<\"$output\" >/dev/null && grep -F 'const float b1 = -0.375f;' <<<\"$output\" >/dev/null && grep -F 'const float intercept = -5.8125f;' <<<\"$output\" >/dev/null && grep -F 'float accum = intercept;' <<<\"$output\" >/dev/null"
    )

    add_test(
        NAME stack_ptx_emit.linear_regression_codegen_example_rust
        COMMAND bash -lc
        "output=\"$($<TARGET_FILE:stack_ptx_emit_linear_regression_codegen_example> --backend rust --show-spec --input-names age,income,score,balance)\" && grep -F '/// Evaluate the hard-coded linear regression model for one observation.' <<<\"$output\" >/dev/null && grep -F '/// # Arguments' <<<\"$output\" >/dev/null && grep -F '///   * `x[0] = age`' <<<\"$output\" >/dev/null && grep -F 'beta_standardized' <<<\"$output\" >/dev/null && grep -F 'intercept' <<<\"$output\" >/dev/null && grep -F '0.0' <<<\"$output\" >/dev/null && grep -F '2.5' <<<\"$output\" >/dev/null && grep -F 'let age: f32 = x[0];' <<<\"$output\" >/dev/null && grep -F 'let b0: f32 = 2.5f32;' <<<\"$output\" >/dev/null && grep -F 'let intercept: f32 = -5.8125f32;' <<<\"$output\" >/dev/null && grep -F 'let mut accum: f32 = intercept;' <<<\"$output\" >/dev/null"
    )

    add_test(
        NAME stack_ptx_emit.linear_regression_codegen_example_numpy
        COMMAND bash -lc
        "output=\"$($<TARGET_FILE:stack_ptx_emit_linear_regression_codegen_example> --backend numpy --show-spec --input-names age,income,score,balance)\" && grep -F 'def mock_linear_regression_infer(x):' <<<\"$output\" >/dev/null && grep -F 'Evaluate the hard-coded linear regression model for a batch of observations.' <<<\"$output\" >/dev/null && grep -F 'Parameters' <<<\"$output\" >/dev/null && grep -F 'x[:, 0] = age' <<<\"$output\" >/dev/null && grep -F 'beta_standardized' <<<\"$output\" >/dev/null && grep -F 'intercept' <<<\"$output\" >/dev/null && grep -F '0.0' <<<\"$output\" >/dev/null && grep -F '2.5' <<<\"$output\" >/dev/null && grep -F 'import numpy as np' <<<\"$output\" >/dev/null && grep -F 'age = x[:, 0]' <<<\"$output\" >/dev/null && grep -F 'b0 = np.float32(2.5)' <<<\"$output\" >/dev/null && grep -F 'accum = intercept' <<<\"$output\" >/dev/null && GENERATED_NUMPY=\"$output\" python3 -c 'import os; import numpy as np; ns = {}; src = os.environ[\"GENERATED_NUMPY\"]; compile(src, \"<generated_numpy>\", \"exec\"); exec(src, ns); x = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, -2.0, 0.0]], dtype=np.float32); y = ns[\"mock_linear_regression_infer\"](x); expected = np.array([6.0625, -12.1875], dtype=np.float32); assert y.shape == (2,); assert np.allclose(y, expected)'"
    )

    add_test(
        NAME stack_ptx_emit.linear_regression_codegen_example_latex
        COMMAND bash -lc
        "output=\"$($<TARGET_FILE:stack_ptx_emit_linear_regression_codegen_example> --backend latex --input-names age,income,score,balance)\" && grep -F '\\documentclass[11pt,landscape]{article}' <<<\"$output\" >/dev/null && grep -F '\\pagestyle{plain}' <<<\"$output\" >/dev/null && grep -F '\\begin{longtable}' <<<\"$output\" >/dev/null && grep -F '$j$ & $B_j$ & $\\phi_j(x)$' <<<\"$output\" >/dev/null"
    )

    add_test(
        NAME stack_ptx_emit.linear_regression_codegen_example_markdown
        COMMAND bash -lc
        "output=\"$($<TARGET_FILE:stack_ptx_emit_linear_regression_codegen_example> --backend markdown --input-names age,income,score,balance)\" && grep -F '# Linear Regression Model Specification' <<<\"$output\" >/dev/null && grep -F '## Inputs' <<<\"$output\" >/dev/null && grep -F '## Model Form' <<<\"$output\" >/dev/null && grep -F '## Feature Table' <<<\"$output\" >/dev/null && grep -F '<div align=\"center\">' <<<\"$output\" >/dev/null && grep -F '<table>' <<<\"$output\" >/dev/null && grep -F '<thead>' <<<\"$output\" >/dev/null && grep -F '<tbody>' <<<\"$output\" >/dev/null"
    )

    add_test(
        NAME stack_ptx_emit.linear_regression_markdown_factor
        COMMAND $<TARGET_FILE:stack_ptx_emit_linear_regression_markdown_factor>
    )
endif()
