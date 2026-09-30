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

from odezza.ast import OutputPrograms, Program, SystemGroup, constant_source, source, state_source, toggle2
from odezza.elf import CubinPlan, Function, SitePlan
from odezza.model import KernelShape
from odezza.sass import CONTROL_ALU_STALL, OPCODE_FSEL, OPCODE_LOP3_IMM, specialize_cubin


def synthetic_plan(
    shape: KernelShape,
    predicate_register: int = 0,
) -> tuple[bytes, CubinPlan]:
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
            output_registers=tuple(
                range(shape.input_count, shape.input_count + shape.output_count)
            ),
            final_output_registers=tuple(
                range(
                    shape.input_count + shape.output_count,
                    shape.input_count + 2 * shape.output_count,
                )
            ),
            output_materialization_offsets=tuple(
                continuation + 16 * index
                for index in range(shape.output_count)
            ),
            available_registers=tuple(range(20, 32)),
            predicate_register=predicate_register,
            permutation_register=19,
            toggle_test_instruction=(
                (1 << 32) | (19 << 24) | (255 << 16) | OPCODE_LOP3_IMM,
                0x000FC0000780C0FF | (predicate_register << 17),
            ),
            cleanup_offsets=(),
            target_table_offsets=tuple(
                target_table_start + 4 * index
                for index in range(shape.system_capacity)
            ),
            original_target_values=tuple(
                arena_start + 16 * index
                for index in range(shape.system_capacity)
            ),
        ),
    )
    return cubin, plan


class SassTests(unittest.TestCase):
    def test_specializer_emits_predicate_and_select_for_toggle(self) -> None:
        shape = KernelShape(
            system_capacity=1,
            shared_patch_capacity=64,
            system_patch_capacity=64,
        )
        cubin, plan = synthetic_plan(shape)
        first = Program.from_expression(source(state_source(1)), shape)
        toggle = toggle2(0, state_source(0), constant_source(0))
        second = Program.from_expression(toggle * source(state_source(1)), shape)
        group = SystemGroup(
            OutputPrograms((0,), (first,)),
            (OutputPrograms((1,), (second,)),),
        )
        result = specialize_cubin(cubin, plan, group, shape)
        opcodes = [
            struct.unpack_from(
                "<Q",
                result.cubin,
                plan.site.arena_start_offset + offset * 16,
            )[0]
            & 0xFFFF
            for offset in range(result.system_instruction_counts[0][0])
        ]
        self.assertIn(OPCODE_LOP3_IMM, opcodes)
        self.assertIn(OPCODE_FSEL, opcodes)
        self.assertEqual(result.system_instruction_counts[0][0], 4)

    def test_source_only_programs_do_not_emit_toggles(self) -> None:
        shape = KernelShape(
            system_capacity=1,
            shared_patch_capacity=64,
            system_patch_capacity=64,
        )
        cubin, plan = synthetic_plan(shape)
        first = Program.from_expression(source(state_source(0)), shape)
        second = Program.from_expression(source(constant_source(0)), shape)
        result = specialize_cubin(
            cubin,
            plan,
            SystemGroup(
                OutputPrograms((0,), (first,)),
                (OutputPrograms((1,), (second,)),),
            ),
            shape,
        )
        self.assertEqual(result.shared_instruction_counts, (1,))
        self.assertEqual(result.system_instruction_counts, ((1,),))

    def test_specializer_uses_inspected_predicate_register(self) -> None:
        shape = KernelShape(
            system_capacity=1,
            shared_patch_capacity=64,
            system_patch_capacity=64,
        )
        cubin, plan = synthetic_plan(shape, predicate_register=3)
        toggle = Program.from_expression(
            toggle2(0, state_source(0), state_source(1)),
            shape,
        )
        source_only = Program.from_expression(source(state_source(0)), shape)
        result = specialize_cubin(
            cubin,
            plan,
            SystemGroup(
                OutputPrograms((0,), (source_only,)),
                (OutputPrograms((1,), (toggle,)),),
            ),
            shape,
        )
        first_set_word1 = struct.unpack_from(
            "<Q",
            result.cubin,
            plan.site.arena_start_offset + 8,
        )[0]
        first_select_word1 = struct.unpack_from(
            "<Q",
            result.cubin,
            plan.site.arena_start_offset + 16 + 8,
        )[0]
        self.assertEqual((first_set_word1 >> 17) & 0x7, 3)
        self.assertEqual((first_set_word1 >> 40) & 0xF, CONTROL_ALU_STALL)
        self.assertEqual((first_select_word1 >> 23) & 0x7, 3)


if __name__ == "__main__":
    unittest.main()
