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
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Sequence

from .autodiff import SystemDerivativeBundle
from .cubin import inspect_named_marker_site
from .derivative_sass import specialize_program_groups
from .fed_batch import FED_BATCH_SHAPE, planted_programs
from .genome import SystemGenome
from .shape import KernelShape
from .template import FIRST_MARKER_BITS


DEFAULT_PRIMAL_PATCH_CAPACITY = 256
DEFAULT_PARTIAL_PATCH_CAPACITY = 1024
PRIMAL_MARKER_BITS = FIRST_MARKER_BITS
PARTIAL_MARKER_BITS = FIRST_MARKER_BITS + 64


@dataclass(frozen=True)
class SplitLMSpecializationResult:
    cubin: bytes
    primal_instruction_counts: tuple[int, ...]
    partial_instruction_counts: tuple[int, ...]
    register_count: int
    primal_pressure_fallback_groups: tuple[int, ...]
    partial_pressure_fallback_groups: tuple[int, ...]


def primal_output_count(shape: KernelShape) -> int:
    return shape.ast_count


def partial_output_count(shape: KernelShape) -> int:
    return shape.input_count


def _render_marker(
    shape: KernelShape,
    output_count: int,
    patch_capacity: int,
    marker_bits: int,
    prefix: str,
) -> tuple[list[str], str]:
    for name, value in (
        ("output_count", output_count),
        ("patch_capacity", patch_capacity),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if shape.input_count + output_count + 1 > 255:
        raise ValueError("LM marker ABI cannot use more than 255 registers")

    marked = [f"{prefix}_marked_{index}" for index in range(shape.input_count)]
    outputs = [f"{prefix}_output_{index}" for index in range(output_count)]
    keepalive = f"{prefix}_keepalive"
    address = f"{prefix}_keepalive_address"
    asm_output_count = shape.input_count + output_count + 1
    keepalive_operand = asm_output_count - 1
    input_operand_start = asm_output_count
    lines = ["{", f".reg .u32 {address};", "brkpt;"]
    for index in range(shape.input_count):
        marker = marker_bits + index
        lines.append(
            f"add.rn.ftz.f32 %{index}, %{input_operand_start + index}, "
            f"0f{marker:08x};"
        )
    lines.append("brkpt;")
    for index in range(1, shape.input_count):
        lines.append(f"add.rn.ftz.f32 %0, %0, %{index};")
    for index in range(output_count):
        marker = marker_bits + shape.input_count + index
        lines.append(
            f"add.rn.ftz.f32 %{shape.input_count + index}, %0, "
            f"0f{marker:08x};"
        )
    lines.append("brkpt;")

    output_operands = [shape.input_count + index for index in range(output_count)]
    if output_count == 1:
        lines.append(
            f"add.rn.ftz.f32 %{keepalive_operand}, %{output_operands[0]}, "
            f"%{input_operand_start};"
        )
        first_input = 1
    else:
        lines.append(
            f"add.rn.ftz.f32 %{keepalive_operand}, %{output_operands[0]}, "
            f"%{output_operands[1]};"
        )
        for operand in output_operands[2:]:
            lines.append(
                f"add.rn.ftz.f32 %{keepalive_operand}, "
                f"%{keepalive_operand}, %{operand};"
            )
        first_input = 0
    for index in range(first_input, shape.input_count):
        lines.append(
            f"add.rn.ftz.f32 %{keepalive_operand}, %{keepalive_operand}, "
            f"%{input_operand_start + index};"
        )
    lines.extend(
        (
            f"mov.u32 {address}, 0;",
            f"st.volatile.shared.f32 [{address}], %{keepalive_operand};",
        )
    )
    lines.extend("brkpt;" for _ in range(patch_capacity))
    lines.append("}")

    declarations = [
        *(f"float {name} __attribute__((unused));" for name in marked),
        *(f"float {name};" for name in outputs),
        f"float {keepalive} __attribute__((unused));",
    ]
    template = "\n".join(f'    "{line}\\n\\t"' for line in lines)
    constraints = ", ".join(
        f'"=&f"({name})' for name in marked + outputs + [keepalive]
    )
    inputs = ", ".join(
        f'"f"(leaves[{index}])' for index in range(shape.input_count)
    )
    assembly = "\n".join(
        [
            *declarations,
            "asm volatile(",
            template,
            f"    : {constraints}",
            f"    : {inputs}",
            '    : "memory"',
            ");",
        ]
    )
    return outputs, assembly


def render_primal_site(
    shape: KernelShape,
    patch_capacity: int = DEFAULT_PRIMAL_PATCH_CAPACITY,
) -> str:
    outputs, assembly = _render_marker(
        shape,
        primal_output_count(shape),
        patch_capacity,
        PRIMAL_MARKER_BITS,
        "ssid_primal",
    )
    materialization = [
        f"const float specific_growth_{site + 1} = {outputs[site]};"
        for site in range(shape.ast_count)
    ]
    return "\n".join(
        [
            "/* Generated primal LM site; source of truth is postorder AD tape v1. */",
            assembly,
            *materialization,
            "",
        ]
    )


def render_partial_site(
    shape: KernelShape,
    patch_capacity: int = DEFAULT_PARTIAL_PATCH_CAPACITY,
) -> str:
    outputs, assembly = _render_marker(
        shape,
        partial_output_count(shape),
        patch_capacity,
        PARTIAL_MARKER_BITS,
        "ssid_partial",
    )
    materialization: list[str] = []
    output_index = 0
    for site, leaf_count in enumerate(shape.ast_leaf_counts):
        partials = ", ".join(
            outputs[output_index + leaf] for leaf in range(leaf_count)
        )
        materialization.append(
            f"float leaf_partials_{site + 1}[{leaf_count}] = {{{partials}}};"
        )
        output_index += leaf_count
    return "\n".join(
        [
            "/* Generated statistics-only partial site from postorder AD tape v1. */",
            assembly,
            *materialization,
            "",
        ]
    )


def specialize_split_lm_cubin(
    cubin: bytes,
    genome: SystemGenome,
    shape: KernelShape,
    primal_patch_capacity: int = DEFAULT_PRIMAL_PATCH_CAPACITY,
    partial_patch_capacity: int = DEFAULT_PARTIAL_PATCH_CAPACITY,
    kernel_name: str = "secant_cubin_materialize_000",
) -> tuple[SplitLMSpecializationResult, SystemDerivativeBundle]:
    bundle = SystemDerivativeBundle.from_genome(genome, shape)
    primal_plan = inspect_named_marker_site(
        cubin,
        shape.input_count,
        primal_output_count(shape),
        primal_patch_capacity,
        PRIMAL_MARKER_BITS,
        kernel_name,
    )
    partial_plan = inspect_named_marker_site(
        cubin,
        shape.input_count,
        partial_output_count(shape),
        partial_patch_capacity,
        PARTIAL_MARKER_BITS,
        kernel_name,
    )

    primal_programs = tuple(site.primal for site in bundle.sites)
    primal = specialize_program_groups(
        cubin,
        primal_plan,
        (primal_programs,),
    )
    partial = specialize_program_groups(
        primal.cubin,
        partial_plan,
        tuple(site.partials for site in bundle.sites),
    )
    register_count = max(primal.register_count, partial.register_count)
    output = bytearray(partial.cubin)
    register_offsets = set(primal_plan.register_count_offsets)
    register_offsets.update(partial_plan.register_count_offsets)
    for offset in register_offsets:
        struct.pack_into("<I", output, offset, register_count)
    register_header_offsets = set(primal_plan.register_count_header_offsets)
    register_header_offsets.update(partial_plan.register_count_header_offsets)
    for offset in register_header_offsets:
        struct.pack_into("<B", output, offset, register_count)
    return (
        SplitLMSpecializationResult(
            bytes(output),
            primal.instruction_counts,
            partial.instruction_counts,
            register_count,
            primal.pressure_fallback_groups,
            partial.pressure_fallback_groups,
        ),
        bundle,
    )


def _write(path: str, data: str | bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        target.write_bytes(data)
    else:
        target.write_text(data)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or specialize split trajectory-LM postorder sites"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate-sites")
    generate.add_argument("--primal-output", required=True)
    generate.add_argument("--partial-output", required=True)
    generate.add_argument("--inspection-output")
    generate.add_argument("--quiet", action="store_true")
    generate.add_argument(
        "--primal-patch-capacity",
        type=int,
        default=DEFAULT_PRIMAL_PATCH_CAPACITY,
    )
    generate.add_argument(
        "--partial-patch-capacity",
        type=int,
        default=DEFAULT_PARTIAL_PATCH_CAPACITY,
    )
    specialize = subparsers.add_parser("specialize")
    specialize.add_argument("template_cubin")
    specialize.add_argument("--output", required=True)
    specialize.add_argument("--inspection-output")
    specialize.add_argument("--quiet", action="store_true")
    specialize.add_argument(
        "--primal-patch-capacity",
        type=int,
        default=DEFAULT_PRIMAL_PATCH_CAPACITY,
    )
    specialize.add_argument(
        "--partial-patch-capacity",
        type=int,
        default=DEFAULT_PARTIAL_PATCH_CAPACITY,
    )
    arguments = parser.parse_args(argv)

    genome = SystemGenome(planted_programs(FED_BATCH_SHAPE))
    bundle = SystemDerivativeBundle.from_genome(genome, FED_BATCH_SHAPE)
    if arguments.command == "generate-sites":
        _write(
            arguments.primal_output,
            render_primal_site(
                FED_BATCH_SHAPE,
                arguments.primal_patch_capacity,
            ),
        )
        _write(
            arguments.partial_output,
            render_partial_site(
                FED_BATCH_SHAPE,
                arguments.partial_patch_capacity,
            ),
        )
        document = {
            "schema": "secant-system-id.trajectory-lm-split-sites",
            "schema_version": 1,
            "primal": {
                "marker_bits": f"0x{PRIMAL_MARKER_BITS:08x}",
                "patch_capacity": arguments.primal_patch_capacity,
                "output_count": primal_output_count(FED_BATCH_SHAPE),
            },
            "partial": {
                "marker_bits": f"0x{PARTIAL_MARKER_BITS:08x}",
                "patch_capacity": arguments.partial_patch_capacity,
                "output_count": partial_output_count(FED_BATCH_SHAPE),
            },
            "derivatives": bundle.inspection(),
        }
    else:
        cubin = Path(arguments.template_cubin).read_bytes()
        result, bundle = specialize_split_lm_cubin(
            cubin,
            genome,
            FED_BATCH_SHAPE,
            arguments.primal_patch_capacity,
            arguments.partial_patch_capacity,
        )
        _write(arguments.output, result.cubin)
        document = {
            "schema": "secant-system-id.trajectory-lm-split-specialization",
            "schema_version": 1,
            "register_count": result.register_count,
            "primal_pressure_fallback_groups": list(
                result.primal_pressure_fallback_groups
            ),
            "partial_pressure_fallback_groups": list(
                result.partial_pressure_fallback_groups
            ),
            "primal_sass_instruction_counts": list(
                result.primal_instruction_counts
            ),
            "partial_sass_instruction_counts": list(
                result.partial_instruction_counts
            ),
            "derivatives": bundle.inspection(),
        }
    rendered = json.dumps(document, indent=2) + "\n"
    if arguments.inspection_output:
        _write(arguments.inspection_output, rendered)
    if not arguments.quiet:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
