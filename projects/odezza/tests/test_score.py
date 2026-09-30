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

import struct
import unittest

from odezza.ast import Program, constant_source, source, state_source
from odezza.elf import Function
from odezza.manifest import parse_cuda_manifest, shape_from_manifest
from odezza.ast import OutputPrograms, SystemGroup
from odezza.elf import CubinPlan, SitePlan
from odezza.example import benchmark_bundle, lotka_volterra_benchmark_group, lotka_volterra_group
from odezza.model import KERNEL_NAME, KernelShape
from odezza.sass import specialize_cubin
from odezza.sass import OPCODE_LOP3_IMM, OPCODE_MOV
from odezza.template import generate_cuda


def synthetic_plan(shape: KernelShape) -> tuple[bytes, CubinPlan]:
    shared_start = 16
    dispatch_offset = shared_start + shape.shared_patch_capacity * 16 + 32
    arena_start = dispatch_offset + 64
    arena_size = shape.arena_instruction_count * 16
    continuation = arena_start + arena_size
    scaffold_end = continuation + shape.output_count * 16 + 32
    function_size = scaffold_end + 16
    target_table_start = function_size
    cubin = bytes(function_size + 4 * shape.system_capacity)
    plan = CubinPlan(
        cubin_size=len(cubin),
        architecture=120,
        input_count=shape.input_count,
        output_count=shape.output_count,
        system_capacity=shape.system_capacity,
        register_count=32,
        register_count_offsets=(),
        register_count_header_offsets=(),
        function=Function(1, 0, function_size),
        site=SitePlan(
            scaffold_start_offset=0,
            scaffold_end_offset=scaffold_end,
            shared_start_offset=shared_start,
            shared_instruction_count=shape.shared_patch_capacity,
            dispatch_offsets=(dispatch_offset,),
            dispatch_instructions=((0x1234, 0x5678),),
            arena_start_offset=arena_start,
            arena_instruction_count=shape.arena_instruction_count,
            arena_end_offset=arena_start + arena_size,
            continuation_offset=continuation,
            incoming_wait_mask=0,
            input_registers=tuple(range(shape.input_count)),
            output_registers=tuple(range(shape.input_count, shape.input_count + shape.output_count)),
            final_output_registers=tuple(range(shape.input_count + shape.output_count, shape.input_count + 2 * shape.output_count)),
            output_materialization_offsets=tuple(
                continuation + 16 * index for index in range(shape.output_count)
            ),
            available_registers=tuple(range(20, 32)),
            predicate_register=1,
            permutation_register=19,
            toggle_test_instruction=(
                (1 << 32) | (19 << 24) | (255 << 16) | OPCODE_LOP3_IMM,
                0x000FC0000782C0FF,
            ),
            cleanup_offsets=(),
            target_table_offsets=tuple(target_table_start + 4 * index for index in range(shape.system_capacity)),
            original_target_values=tuple(arena_start + 16 * index for index in range(shape.system_capacity)),
        ),
    )
    return cubin, plan


class ScoreTests(unittest.TestCase):
    def test_manifest_round_trip_has_grid_system_dimension(self) -> None:
        shape = KernelShape(system_capacity=4, shared_patch_capacity=64, system_patch_capacity=80)
        source_text = generate_cuda(shape)
        manifest, payload = parse_cuda_manifest(source_text)
        self.assertEqual(shape_from_manifest(manifest), shape)
        self.assertIn(KERNEL_NAME, payload)
        self.assertIn("const unsigned int system_index = blockIdx.y;", payload)
        self.assertIn("brx.idx.uni", payload)
        self.assertIn("equations: every shared and candidate expression", payload)

    def test_group_requires_shared_and_branch_outputs_to_form_complete_rhs(self) -> None:
        shape = KernelShape(system_capacity=2, shared_patch_capacity=64, system_patch_capacity=64)
        x = Program.from_expression(source(state_source(0)), shape)
        y = Program.from_expression(source(state_source(1)), shape)
        valid = SystemGroup(OutputPrograms((0,), (x,)), (OutputPrograms((1,), (y,)),))
        valid.validate(shape)
        invalid = SystemGroup(OutputPrograms((0,), (x,)), (OutputPrograms((0,), (y,)),))
        with self.assertRaisesRegex(ValueError, "overwrites"):
            invalid.validate(shape)

    def test_specializer_writes_shared_prelude_and_distinct_branch_targets(self) -> None:
        shape = KernelShape(system_capacity=4, shared_patch_capacity=64, system_patch_capacity=64)
        cubin, plan = synthetic_plan(shape)
        group = lotka_volterra_group(shape)
        result = specialize_cubin(cubin, plan, group, shape)
        targets = tuple(
            struct.unpack_from("<I", result.cubin, offset)[0]
            for offset in plan.site.target_table_offsets
        )
        self.assertNotEqual(targets[0], targets[1])
        self.assertEqual(targets[2:], (targets[0], targets[0]))
        self.assertEqual(result.body_offsets[:2], targets[:2])
        self.assertEqual(len(result.shared_instruction_counts), 1)
        self.assertEqual(len(result.system_instruction_counts), 2)
        for offset, live_register, final_register in zip(
            plan.site.output_materialization_offsets,
            plan.site.output_registers,
            plan.site.final_output_registers,
        ):
            word0 = struct.unpack_from("<Q", result.cubin, offset)[0]
            self.assertEqual(word0 & 0xFFFF, OPCODE_MOV)
            self.assertEqual((word0 >> 32) & 0xFF, live_register)
            self.assertEqual((word0 >> 16) & 0xFF, final_register)

    def test_toggle_bounds_apply_to_every_branch(self) -> None:
        shape = KernelShape(system_capacity=2, shared_patch_capacity=64, system_patch_capacity=64)
        shared = Program.from_expression(source(state_source(0)), shape)
        branch = Program.from_expression(source(constant_source(0)), shape)
        group = SystemGroup(
            OutputPrograms((0,), (shared,)),
            (OutputPrograms((1,), (branch,)),),
        )
        document = group.to_document(shape)
        self.assertEqual(document["required_toggle_bits"], 0)

    def test_benchmark_fills_branch_table_and_constant_banks_are_system_major(self) -> None:
        shape = KernelShape()
        group = lotka_volterra_benchmark_group(shape)
        bundle = benchmark_bundle(shape, bank_count=3)
        self.assertEqual(len(group.systems), shape.system_capacity)
        self.assertEqual(group.required_toggle_bits, 5)
        self.assertEqual(bundle["system_count"], 8)
        self.assertEqual(len(bundle["constant_banks"]), 8)
        self.assertTrue(all(len(banks) == 3 for banks in bundle["constant_banks"]))
        self.assertEqual(bundle["constant_banks"][0][0], [1.5, 1.0, 0.75, 1.0])


if __name__ == "__main__":
    unittest.main()
