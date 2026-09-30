#!/usr/bin/env python3
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

import argparse
from pathlib import Path
import sys

from corpus import DEFAULT_CORPUS, load_corpus, postorder


OP_ENCODERS = {
    "add": "secant_ast_encode_add_f32",
    "sub": "secant_ast_encode_sub_f32",
    "mul": "secant_ast_encode_mul_f32",
    "div": "secant_ast_encode_div_f32",
    "neg": "secant_ast_encode_neg_f32",
    "sqrt": "secant_ast_encode_sqrt_f32",
    "rsqrt": "secant_ast_encode_rsqrt_f32",
    "abs": "secant_ast_encode_abs_f32",
    "min": "secant_ast_encode_min_f32",
    "max": "secant_ast_encode_max_f32",
    "sin": "secant_ast_encode_sin_f32",
    "cos": "secant_ast_encode_cos_f32",
    "ex2": "secant_ast_encode_ex2_f32",
    "lg2": "secant_ast_encode_lg2_f32",
    "tanh": "secant_ast_encode_tanh_f32",
}

SIMD_OPERATORS = {
    "add": "_mm256_add_ps",
    "sub": "_mm256_sub_ps",
    "mul": "_mm256_mul_ps",
    "div": "_mm256_div_ps",
    "min": "_mm256_min_ps",
    "max": "_mm256_max_ps",
}

SCALAR_OPERATORS = {
    "min": "fminf",
    "max": "fmaxf",
}

ROUTINE_NAMES = ("safe_div", "safe_sqrt", "safe_rsqrt")


def c_float(value):
    return f"{float(value).hex()}f"


def render_native(expression, simd):
    if isinstance(expression, int):
        return f"x{expression}"
    operation = expression[0]
    if operation == "constant":
        value = c_float(expression[1])
        return f"_mm256_set1_ps({value})" if simd else value
    arguments = [
        render_native(argument, simd)
        for argument in expression[1:]
    ]
    if simd:
        if operation in SIMD_OPERATORS:
            return (
                f"{SIMD_OPERATORS[operation]}"
                f"({arguments[0]}, {arguments[1]})")
        if operation == "neg":
            return (
                "_mm256_xor_ps("
                f"{arguments[0]}, _mm256_set1_ps(-0.0f))")
        if operation == "abs":
            return (
                "_mm256_andnot_ps("
                f"_mm256_set1_ps(-0.0f), {arguments[0]})")
        if operation == "sqrt":
            return f"_mm256_sqrt_ps({arguments[0]})"
        if operation == "rsqrt":
            return (
                "_mm256_div_ps(_mm256_set1_ps(1.0f), "
                f"_mm256_sqrt_ps({arguments[0]}))")
        if operation in ("sin", "cos", "ex2", "lg2", "tanh"):
            return f"secant_native_{operation}_ps({arguments[0]})"
        if operation == "safe_div":
            return (
                "secant_native_safe_div_ps"
                f"({arguments[0]}, {arguments[1]})")
        if operation in ("safe_sqrt", "safe_rsqrt"):
            radicand = (
                "_mm256_add_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0f), "
                f"{arguments[0]}), _mm256_set1_ps(0.01f))")
            if operation == "safe_sqrt":
                return f"_mm256_sqrt_ps({radicand})"
            return (
                "_mm256_div_ps(_mm256_set1_ps(1.0f), "
                f"_mm256_sqrt_ps({radicand}))")
        raise ValueError(f"unsupported native SIMD operation: {operation}")
    if operation == "add":
        return f"(({arguments[0]}) + ({arguments[1]}))"
    if operation == "sub":
        return f"(({arguments[0]}) - ({arguments[1]}))"
    if operation == "mul":
        return f"(({arguments[0]}) * ({arguments[1]}))"
    if operation == "div":
        return f"(({arguments[0]}) / ({arguments[1]}))"
    if operation in SCALAR_OPERATORS:
        return (
            f"{SCALAR_OPERATORS[operation]}"
            f"({arguments[0]}, {arguments[1]})")
    if operation == "safe_div":
        return (
            "secant_native_safe_div_f32"
            f"({arguments[0]}, {arguments[1]})")
    if operation == "neg":
        return f"(-({arguments[0]}))"
    if operation == "abs":
        return f"fabsf({arguments[0]})"
    if operation in ("sin", "cos", "sqrt", "tanh"):
        return f"{operation}f({arguments[0]})"
    if operation == "ex2":
        return f"exp2f({arguments[0]})"
    if operation == "lg2":
        return f"log2f({arguments[0]})"
    if operation == "rsqrt":
        return f"(1.0f / sqrtf({arguments[0]}))"
    if operation == "safe_sqrt":
        return f"sqrtf(fabsf({arguments[0]}) + 0.01f)"
    if operation == "safe_rsqrt":
        return f"(1.0f / sqrtf(fabsf({arguments[0]}) + 0.01f))"
    raise ValueError(f"unsupported native scalar operation: {operation}")


def render_routine(lines, name):
    if name == "safe_div":
        instructions = (
            "secant_ast_encode_routine_arg_f32(0u)",
            "secant_ast_encode_routine_arg_f32(1u)",
            "secant_ast_encode_mul_f32",
            "secant_ast_encode_routine_arg_f32(1u)",
            "secant_ast_encode_routine_arg_f32(1u)",
            "secant_ast_encode_mul_f32",
            "secant_ast_encode_constant_f32(0.01f)",
            "secant_ast_encode_add_f32",
            "secant_ast_encode_rcp_f32",
            "secant_ast_encode_mul_f32",
        )
    else:
        instructions = (
            "secant_ast_encode_routine_arg_f32(0u)",
            "secant_ast_encode_abs_f32",
            "secant_ast_encode_constant_f32(0.01f)",
            "secant_ast_encode_add_f32",
            (
                "secant_ast_encode_sqrt_f32"
                if name == "safe_sqrt"
                else "secant_ast_encode_rsqrt_f32"
            ),
        )
    lines.append(
        "static const SecantAstInstruction "
        f"secant_portable_alu_routine_{name}[] = {{")
    lines.extend(f"    {instruction}," for instruction in instructions)
    lines.append("    secant_ast_encode_return_f32")
    lines.append("};")
    lines.append("")


def render_header(corpus):
    expressions = corpus["expressions"]
    routines = tuple(
        name for name in ROUTINE_NAMES
        if name in corpus["operations"])
    routine_indices = {
        name: routine_idx
        for routine_idx, name in enumerate(routines)
    }
    program_instructions = max(
        len(postorder(expression)) + 1
        for expression in expressions)
    lines = [
        "#ifndef SECANT_PORTABLE_ALU_V1_H_INCLUDED",
        "#define SECANT_PORTABLE_ALU_V1_H_INCLUDED",
        "",
        f"#define SECANT_PORTABLE_ALU_CORPUS_NAME \"{corpus['name']}\"",
        f"#define SECANT_PORTABLE_ALU_CORPUS_HASH \"{corpus['hash']}\"",
        f"#define SECANT_PORTABLE_ALU_NUM_INPUTS {corpus['num_inputs']}u",
        f"#define SECANT_PORTABLE_ALU_NUM_CASES {len(expressions)}u",
        f"#define SECANT_PORTABLE_ALU_TARGET_CASE {corpus['target_expression']}u",
        f"#define SECANT_PORTABLE_ALU_PROGRAM_INSTRUCTIONS {program_instructions}u",
        f"#define SECANT_PORTABLE_ALU_NUM_ROUTINES {len(routines)}u",
        "",
        "#define SECANT_PORTABLE_ALU_FOR_EACH_CASE(X) \\",
    ]
    for expression_idx in range(len(expressions)):
        suffix = " \\" if expression_idx + 1 < len(expressions) else ""
        lines.append(f"    X({expression_idx}){suffix}")
    lines.extend([
        "",
        "#ifdef SECANT_PORTABLE_ALU_INCLUDE_AST",
        "#include \"secant.h\"",
        "",
    ])
    for routine in routines:
        render_routine(lines, routine)
    if routines:
        lines.append(
            "static const SecantAstInstruction* const "
            "secant_portable_alu_routines"
            "[SECANT_PORTABLE_ALU_NUM_ROUTINES] = {")
        lines.extend(
            f"    secant_portable_alu_routine_{routine},"
            for routine in routines)
        lines.extend([
            "};",
            "static const char* const "
            "secant_portable_alu_routine_names"
            "[SECANT_PORTABLE_ALU_NUM_ROUTINES] = {",
        ])
        lines.extend(f'    "{routine}",' for routine in routines)
        lines.extend([
            "};",
            "",
        ])
    for expression_idx, expression in enumerate(expressions):
        lines.append(
            "static const SecantAstInstruction "
            f"secant_portable_alu_ast_{expression_idx}"
            "[SECANT_PORTABLE_ALU_PROGRAM_INSTRUCTIONS] = {")
        for operation, argument in postorder(expression):
            if operation == "input":
                instruction = f"secant_ast_encode_input_f32({argument}u)"
            elif operation == "constant":
                instruction = f"secant_ast_encode_constant_f32({c_float(argument)})"
            elif operation in routine_indices:
                num_args = 2 if operation == "safe_div" else 1
                instruction = (
                    f"secant_ast_encode_routine_f32("
                    f"{routine_indices[operation]}u, {num_args}u)")
            else:
                instruction = OP_ENCODERS[operation]
            lines.append(f"    {instruction},")
        lines.append("    secant_ast_encode_return_f32")
        lines.append("};")
        lines.append("")
    lines.append(
        "static const SecantAstInstruction* const "
        "secant_portable_alu_asts[SECANT_PORTABLE_ALU_NUM_CASES] = {")
    for expression_idx in range(len(expressions)):
        lines.append(f"    secant_portable_alu_ast_{expression_idx},")
    lines.extend([
        "};",
        "#endif",
        "",
        "#ifdef SECANT_PORTABLE_ALU_INCLUDE_NATIVE",
    ])
    arguments = ", ".join(f"x{idx}" for idx in range(corpus["num_inputs"]))
    for expression_idx, expression in enumerate(expressions):
        lines.append(
            f"#define SECANT_PORTABLE_ALU_SIMD_EXPR_{expression_idx}({arguments}) "
            f"{render_native(expression, True)}")
        lines.append(
            f"#define SECANT_PORTABLE_ALU_SCALAR_EXPR_{expression_idx}({arguments}) "
            f"{render_native(expression, False)}")
    lines.append(
        f"#define SECANT_PORTABLE_ALU_SCALAR_TARGET({arguments}) "
        f"SECANT_PORTABLE_ALU_SCALAR_EXPR_{corpus['target_expression']}"
        f"({arguments})")
    lines.extend([
        "#endif",
        "",
        "#endif /* SECANT_PORTABLE_ALU_V1_H_INCLUDED */",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Generate C representations of a portable expression corpus.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rendered = render_header(load_corpus(args.corpus))
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"generated corpus header is stale: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
