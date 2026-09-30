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

from secant_system_id.ast import Program, input_slot
from secant_system_id.cubin import Function, INSTRUCTION_BYTES
from secant_system_id.fed_batch import FED_BATCH_SHAPE
from secant_system_id.genome import SystemGenome
from secant_system_id.packed_cubin import PackedCubinPlan, PackedSitePlan
from secant_system_id.packed_sass import specialize_packed_cubin


def _plan(incoming_wait_mask: int, dispatch_instruction_count: int = 4) -> PackedCubinPlan:
    entry_offset = 512
    dispatch_offsets = tuple(
        576 + index * INSTRUCTION_BYTES
        for index in range(dispatch_instruction_count)
    )
    arena_start = 576 + dispatch_instruction_count * INSTRUCTION_BYTES
    arena_instruction_count = 64
    arena_end = arena_start + arena_instruction_count * INSTRUCTION_BYTES
    return PackedCubinPlan(
        cubin_size=2048,
        architecture=89,
        input_count=FED_BATCH_SHAPE.input_count,
        output_count=FED_BATCH_SHAPE.ast_count,
        genome_capacity=2,
        register_count=64,
        register_count_offsets=(104,),
        register_count_header_offsets=(111,),
        function=Function(symbol_index=1, file_offset=entry_offset, size=1536),
        site=PackedSitePlan(
            entry_offset=entry_offset,
            dispatch_offsets=dispatch_offsets,
            dispatch_instructions=tuple(
                (0x1000 + index, 0x2000 + index)
                for index in range(len(dispatch_offsets))
            ),
            arena_start_offset=arena_start,
            arena_instruction_count=arena_instruction_count,
            arena_end_offset=arena_end,
            continuation_offset=arena_end + 32,
            incoming_wait_mask=incoming_wait_mask,
            input_registers=tuple(range(8, 24)),
            output_registers=(24, 25),
            available_registers=tuple(range(26, 48)),
            cleanup_offsets=(528, 544, 560),
            target_table_offsets=(64, 68),
            original_target_values=(128, 256),
        ),
    )


class PackedSassDependencyTest(unittest.TestCase):
    def test_incoming_wait_precedes_dispatch_instead_of_first_ast_instruction(self) -> None:
        wait_mask = 0x1F
        plan = _plan(wait_mask)
        genome = SystemGenome(
            (
                Program.from_expression(input_slot(0) + 1.0),
                Program.from_expression(input_slot(8) + 2.0),
            )
        )

        result = specialize_packed_cubin(
            bytes(plan.cubin_size), plan, (genome,), FED_BATCH_SHAPE
        )

        _entry_word0, entry_word1 = struct.unpack_from(
            "<QQ", result.cubin, plan.site.entry_offset
        )
        first_body_offset = plan.function.file_offset + result.body_offsets[0]
        _first_word0, first_word1 = struct.unpack_from(
            "<QQ", result.cubin, first_body_offset
        )
        self.assertEqual((entry_word1 >> 52) & 0x3F, wait_mask)
        self.assertEqual((first_word1 >> 52) & 0x3F, 0)

    def test_five_instruction_warp_dispatch_is_preserved(self) -> None:
        plan = _plan(0x07, dispatch_instruction_count=5)
        genome = SystemGenome(
            (
                Program.from_expression(input_slot(0) + 1.0),
                Program.from_expression(input_slot(8) + 2.0),
            )
        )

        result = specialize_packed_cubin(
            bytes(plan.cubin_size), plan, (genome,), FED_BATCH_SHAPE
        )

        self.assertEqual(
            [struct.unpack_from("<QQ", result.cubin, offset) for offset in plan.site.dispatch_offsets],
            list(plan.site.dispatch_instructions),
        )


if __name__ == "__main__":
    unittest.main()
