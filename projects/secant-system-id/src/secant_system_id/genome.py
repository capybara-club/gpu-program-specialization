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

from dataclasses import dataclass

from .ast import Program
from .shape import KernelShape


@dataclass(frozen=True)
class SystemGenome:
    """One independent post-order program per missing mathematical site."""

    programs: tuple[Program, ...]

    def validate(self, shape: KernelShape) -> None:
        if len(self.programs) != shape.ast_count:
            raise ValueError("system genome must contain one AST per missing site")
        for site, (program, offset, count) in enumerate(
            zip(self.programs, shape.ast_input_offsets, shape.ast_leaf_counts)
        ):
            program.validate(shape.input_count)
            allowed = set(range(offset, offset + count))
            unexpected = program.input_indices() - allowed
            if unexpected:
                raise ValueError(
                    f"AST {site} references leaf slots belonging to another missing site: "
                    f"{sorted(unexpected)}"
                )

    def replace_site(self, site: int, program: Program, shape: KernelShape) -> "SystemGenome":
        if not 0 <= site < len(self.programs):
            raise IndexError("missing-site index is outside the genome")
        programs = list(self.programs)
        programs[site] = program
        result = SystemGenome(tuple(programs))
        result.validate(shape)
        return result
