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
"""Register-only common-subexpression lowering for grouped LM outputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

from .ast import InstructionType, Program, _ARITY
from .lm_ast import lm_sass_shape, lower_lm_program
from .lm_elf import LMCubinPlan, LMScalarSitePlan
from .lm_model import LMKernelShape
from .sass import (
    LN_TWO_BITS,
    LOG2_E_BITS,
    NEG_ONE_BITS,
    REGISTER_RZ,
    REST_FMUL_RZ,
    REST_MUFU_COS,
    REST_MUFU_EX2,
    REST_MUFU_LG2,
    REST_MUFU_RCP,
    REST_MUFU_RSQ,
    REST_MUFU_SIN,
    REST_MUFU_SQRT,
    REST_MUFU_TANH,
    SIN_COS_SCALE_BITS,
    Assembler,
    Operand,
    SassAssemblyError,
    _fadd_imm,
    _fadd_reg,
    _ffma,
    _fmul_imm,
    _fmul_reg,
    _fsel,
    _mov,
    _mufu,
    _nop,
    _toggle_test,
)


@dataclass(frozen=True)
class _Node:
    kind: InstructionType
    operand: int | tuple[int, ...] | None = None
    arguments: tuple["_Node", ...] = ()


def _program_roots(
    programs: Sequence[Program],
    shape: LMKernelShape,
) -> tuple[_Node, ...]:
    interned: dict[_Node, _Node] = {}
    roots: list[_Node] = []

    def intern(node: _Node) -> _Node:
        return interned.setdefault(node, node)

    for original in programs:
        program = lower_lm_program(original, shape)
        stack: list[_Node] = []
        returned = False
        for instruction in program.instructions():
            kind = instruction.kind
            if kind in {
                InstructionType.STATE_F32,
                InstructionType.CONSTANT_F32,
                InstructionType.LITERAL_F32,
            }:
                node = _Node(kind, instruction.operands[0])
            elif kind in {
                InstructionType.TOGGLE2_F32,
                InstructionType.TOGGLE4_F32,
            }:
                choice_count = 1 << len(instruction.operands)
                if len(stack) < choice_count:
                    raise SassAssemblyError("grouped LM toggle underflow")
                arguments = tuple(stack[-choice_count:])
                del stack[-choice_count:]
                node = _Node(kind, tuple(instruction.operands), arguments)
            elif kind == InstructionType.RETURN_F32:
                if returned or len(stack) != 1:
                    raise SassAssemblyError("invalid grouped LM postorder program")
                roots.append(stack[-1])
                returned = True
                continue
            elif kind in _ARITY:
                arity = _ARITY[kind]
                if len(stack) < arity:
                    raise SassAssemblyError("grouped LM program underflow")
                arguments = tuple(stack[-arity:])
                del stack[-arity:]
                if kind == InstructionType.EXP_F32:
                    scale = intern(_Node(InstructionType.LITERAL_F32, LOG2_E_BITS))
                    product = intern(
                        _Node(InstructionType.MUL_F32, arguments=(arguments[0], scale))
                    )
                    node = _Node(InstructionType.EX2_F32, arguments=(product,))
                elif kind == InstructionType.LOG_F32:
                    logarithm = intern(
                        _Node(InstructionType.LG2_F32, arguments=arguments)
                    )
                    scale = intern(_Node(InstructionType.LITERAL_F32, LN_TWO_BITS))
                    node = _Node(
                        InstructionType.MUL_F32,
                        arguments=(logarithm, scale),
                    )
                else:
                    node = _Node(kind, arguments=arguments)
            else:
                raise SassAssemblyError(
                    f"grouped LM CSE does not support {kind.name}"
                )
            stack.append(intern(node))
        if not returned:
            raise SassAssemblyError("grouped LM program has no return")
    return tuple(roots)


def _copy(value: Operand) -> Operand:
    return Operand(
        immediate_bits=value.immediate_bits,
        register=value.register,
        owned=value.owned,
        barrier_slot=value.barrier_slot,
        barrier_token=value.barrier_token,
    )


class GroupedLMCSEAssembler:
    """Emit a group of scalar outputs while retaining each shared node once."""

    def __init__(
        self,
        plan: LMCubinPlan,
        site: LMScalarSitePlan,
        shape: LMKernelShape,
    ) -> None:
        if not site.output_registers:
            raise SassAssemblyError("an LM group needs at least one output")
        available = tuple(
            dict.fromkeys(
                (*reversed(site.output_registers), *site.available_registers)
            )
        )
        site_plan = SimpleNamespace(
            input_registers=site.input_registers,
            output_registers=(),
            predicate_register=site.predicate_register,
            permutation_register=site.permutation_register,
            toggle_test_instruction=site.toggle_test_instruction,
            available_registers=available,
        )
        assembler_plan = SimpleNamespace(
            architecture=plan.architecture,
            register_count=plan.register_count,
            site=site_plan,
        )
        self.assembler = Assembler(
            assembler_plan,
            lm_sass_shape(shape),
            REGISTER_RZ,
            site.incoming_wait_mask,
        )
        self.outputs = site.output_registers
        self.values: dict[_Node, Operand] = {}
        self.remaining: Counter[_Node] = Counter()
        self.pinned: set[int] = set()

    def _count_uses(self, roots: Sequence[_Node]) -> None:
        visited: set[_Node] = set()

        def visit(node: _Node) -> None:
            if node in visited or not node.arguments:
                return
            visited.add(node)
            for argument in node.arguments:
                self.remaining[argument] += 1
                visit(argument)

        for root in roots:
            self.remaining[root] += 1
            visit(root)

    def _allocate(self) -> Operand:
        while True:
            value = self.assembler.allocate()
            if value.register not in self.pinned:
                return value

    def _last_owned(self, node: _Node, value: Operand) -> bool:
        return (
            value.owned
            and not value.immediate
            and value.register not in self.pinned
            and self.remaining[node] == 1
        )

    def _consume(self, node: _Node, keep: Operand | None = None) -> None:
        self.remaining[node] -= 1
        if self.remaining[node] < 0:
            raise SassAssemblyError("grouped LM CSE use-count underflow")
        if self.remaining[node] == 0:
            value = self.values.pop(node, None)
            if value is not None and (
                value.immediate or value.register not in self.pinned
            ):
                self.assembler.release(value, keep)

    def _materialize(self, value: Operand) -> tuple[Operand, bool]:
        if not value.immediate:
            return value, False
        target = self._allocate()
        self.assembler.emit(
            _fadd_imm(target.register, REGISTER_RZ, int(value.immediate_bits), 0)
        )
        return target, True

    def _relocate_output_conflict(self, output: int, root: _Node) -> None:
        conflicts = [
            node
            for node, value in self.values.items()
            if node != root
            and self.remaining[node] > 0
            and not value.immediate
            and value.register == output
        ]
        if not conflicts:
            return
        source = self.values[conflicts[0]]
        target = self._allocate()
        self.assembler.emit(
            _mov(target.register, source.register, self.assembler.wait_mask_for((source,)))
        )
        for node in conflicts:
            self.values[node] = _copy(target)

    def _binary(
        self,
        lhs_node: _Node,
        rhs_node: _Node,
        lhs: Operand,
        rhs: Operand,
        multiply: bool,
    ) -> Operand:
        lhs_temporary = False
        if lhs.immediate and not rhs.immediate:
            lhs_node, rhs_node, lhs, rhs = rhs_node, lhs_node, rhs, lhs
        elif lhs.immediate and rhs.immediate:
            lhs, lhs_temporary = self._materialize(lhs)
            if not lhs_temporary:
                raise AssertionError("literal materialization failed")
        target = (
            _copy(lhs)
            if self._last_owned(lhs_node, lhs)
            else (
                _copy(rhs)
                if self._last_owned(rhs_node, rhs)
                else self._allocate()
            )
        )
        wait = self.assembler.wait_mask_for((lhs, rhs))
        if multiply:
            instruction = (
                _fmul_imm(target.register, lhs.register, int(rhs.immediate_bits), wait)
                if rhs.immediate
                else _fmul_reg(target.register, lhs.register, rhs.register, wait)
            )
        else:
            instruction = (
                _fadd_imm(target.register, lhs.register, int(rhs.immediate_bits), wait)
                if rhs.immediate
                else _fadd_reg(target.register, lhs.register, rhs.register, wait)
            )
        self.assembler.emit(instruction)
        self._consume(lhs_node, target)
        self._consume(rhs_node, target)
        if lhs_temporary:
            self.assembler.release(lhs, target)
        target.barrier_slot = None
        return target

    def _subtract(
        self,
        lhs_node: _Node,
        rhs_node: _Node,
        lhs: Operand,
        rhs: Operand,
    ) -> Operand:
        if rhs.immediate:
            negated = Operand(immediate_bits=int(rhs.immediate_bits) ^ 0x80000000)
            return self._binary(lhs_node, rhs_node, lhs, negated, False)
        negated = _copy(rhs) if self._last_owned(rhs_node, rhs) else self._allocate()
        self.assembler.emit(
            _fmul_imm(
                negated.register,
                rhs.register,
                NEG_ONE_BITS,
                self.assembler.wait_mask_for((rhs,)),
            )
        )
        lhs_register, lhs_temporary = self._materialize(lhs)
        target = (
            _copy(lhs_register)
            if lhs_temporary or self._last_owned(lhs_node, lhs_register)
            else negated
        )
        self.assembler.emit(
            _fadd_reg(
                target.register,
                lhs_register.register,
                negated.register,
                self.assembler.wait_mask_for((lhs_register, negated)),
            )
        )
        self._consume(lhs_node, target)
        self._consume(rhs_node, target)
        if lhs_temporary:
            self.assembler.release(lhs_register, target)
        if target.register != negated.register:
            self.assembler.release(negated)
        target.barrier_slot = None
        return target

    def _divide(
        self,
        lhs_node: _Node,
        rhs_node: _Node,
        lhs: Operand,
        rhs: Operand,
    ) -> Operand:
        rhs_register, rhs_temporary = self._materialize(rhs)
        lhs_register, lhs_temporary = self._materialize(lhs)
        reciprocal = (
            _copy(rhs_register)
            if rhs_temporary or self._last_owned(rhs_node, rhs_register)
            else self._allocate()
        )
        slot, token = self.assembler.allocate_barrier()
        self.assembler.emit(
            _mufu(
                reciprocal.register,
                rhs_register.register,
                self.assembler.wait_mask_for((rhs_register,)),
                slot,
                REST_MUFU_RCP,
            )
        )
        reciprocal.barrier_slot = slot
        reciprocal.barrier_token = token
        target = (
            _copy(lhs_register)
            if lhs_temporary or self._last_owned(lhs_node, lhs_register)
            else reciprocal
        )
        self.assembler.emit(
            _fmul_reg(
                target.register,
                lhs_register.register,
                reciprocal.register,
                self.assembler.wait_mask_for((lhs_register, reciprocal)),
            )
        )
        self._consume(lhs_node, target)
        self._consume(rhs_node, target)
        if lhs_temporary:
            self.assembler.release(lhs_register, target)
        if rhs_temporary and rhs_register.register != reciprocal.register:
            self.assembler.release(rhs_register, target)
        if reciprocal.register != target.register:
            self.assembler.release(reciprocal)
        target.barrier_slot = None
        return target

    def _unary(self, node: _Node, argument_node: _Node, argument: Operand) -> Operand:
        source, source_temporary = self._materialize(argument)
        target = (
            _copy(source)
            if source_temporary or self._last_owned(argument_node, source)
            else self._allocate()
        )
        wait = self.assembler.wait_mask_for((source,))
        if node.kind == InstructionType.NEG_F32:
            self.assembler.emit(
                _fmul_imm(target.register, source.register, NEG_ONE_BITS, wait)
            )
            target.barrier_slot = None
        else:
            rest = {
                InstructionType.SQRT_F32: REST_MUFU_SQRT,
                InstructionType.RCP_F32: REST_MUFU_RCP,
                InstructionType.SIN_F32: REST_MUFU_SIN,
                InstructionType.COS_F32: REST_MUFU_COS,
                InstructionType.EX2_F32: REST_MUFU_EX2,
                InstructionType.LG2_F32: REST_MUFU_LG2,
                InstructionType.RSQRT_F32: REST_MUFU_RSQ,
                InstructionType.TANH_F32: REST_MUFU_TANH,
            }[node.kind]
            source_register = source.register
            if node.kind in {InstructionType.SIN_F32, InstructionType.COS_F32}:
                self.assembler.emit(
                    _fmul_imm(
                        target.register,
                        source.register,
                        SIN_COS_SCALE_BITS,
                        wait,
                        REST_FMUL_RZ,
                    )
                )
                source_register = target.register
                wait = 0
            slot, token = self.assembler.allocate_barrier()
            self.assembler.emit(
                _mufu(target.register, source_register, wait, slot, rest)
            )
            target.barrier_slot = slot
            target.barrier_token = token
        self._consume(argument_node, target)
        if source_temporary:
            self.assembler.release(source, target)
        return target

    def _toggle(self, node: _Node, arguments: Sequence[Operand]) -> Operand:
        values = [(value, False) for value in arguments]
        bit_indices = node.operand
        if not isinstance(bit_indices, tuple):
            raise SassAssemblyError("grouped LM toggle has no bit indices")
        consumed_arguments = False
        for bit_index in bit_indices:
            self.assembler.emit(
                _toggle_test(
                    self.assembler.plan.site.toggle_test_instruction,
                    self.assembler.plan.site.permutation_register,
                    bit_index,
                )
            )
            reduced: list[tuple[Operand, bool]] = []
            for index in range(0, len(values), 2):
                false_value, false_ephemeral = values[index]
                true_value, true_ephemeral = values[index + 1]
                false_register, false_temporary = self._materialize(false_value)
                true_register, true_temporary = self._materialize(true_value)
                target = self._allocate()
                self.assembler.emit(
                    _fsel(
                        target.register,
                        true_register.register,
                        false_register.register,
                        self.assembler.plan.site.predicate_register,
                    )
                )
                if false_temporary or false_ephemeral:
                    self.assembler.release(false_register, target)
                if true_temporary or true_ephemeral:
                    self.assembler.release(true_register, target)
                reduced.append((target, True))
            if not consumed_arguments:
                for argument_node in node.arguments:
                    self._consume(argument_node)
                consumed_arguments = True
            values = reduced
        if len(values) != 1:
            raise SassAssemblyError("grouped LM toggle did not reduce to one value")
        return values[0][0]

    def _emit_node(self, node: _Node) -> Operand:
        cached = self.values.get(node)
        if cached is not None:
            return cached
        if node.kind in {InstructionType.STATE_F32, InstructionType.CONSTANT_F32}:
            value = self.assembler.source_operand(node.kind, int(node.operand))
        elif node.kind == InstructionType.LITERAL_F32:
            value = Operand(immediate_bits=int(node.operand))
        else:
            arguments = [self._emit_node(argument) for argument in node.arguments]
            if node.kind in {InstructionType.ADD_F32, InstructionType.MUL_F32}:
                value = self._binary(
                    node.arguments[0],
                    node.arguments[1],
                    arguments[0],
                    arguments[1],
                    node.kind == InstructionType.MUL_F32,
                )
            elif node.kind == InstructionType.SUB_F32:
                value = self._subtract(
                    node.arguments[0],
                    node.arguments[1],
                    arguments[0],
                    arguments[1],
                )
            elif node.kind == InstructionType.DIV_F32:
                value = self._divide(
                    node.arguments[0],
                    node.arguments[1],
                    arguments[0],
                    arguments[1],
                )
            elif node.kind in {
                InstructionType.NEG_F32,
                InstructionType.SQRT_F32,
                InstructionType.RCP_F32,
                InstructionType.SIN_F32,
                InstructionType.COS_F32,
                InstructionType.EX2_F32,
                InstructionType.LG2_F32,
                InstructionType.RSQRT_F32,
                InstructionType.TANH_F32,
            }:
                value = self._unary(node, node.arguments[0], arguments[0])
            elif node.kind == InstructionType.FMA_F32:
                registers: list[Operand] = []
                temporaries: list[bool] = []
                for argument in arguments:
                    register, temporary = self._materialize(argument)
                    registers.append(register)
                    temporaries.append(temporary)
                target = next(
                    (
                        _copy(register)
                        for argument_node, register, temporary in zip(
                            node.arguments,
                            registers,
                            temporaries,
                        )
                        if temporary or self._last_owned(argument_node, register)
                    ),
                    None,
                )
                if target is None:
                    target = self._allocate()
                self.assembler.emit(
                    _ffma(
                        target.register,
                        registers[0].register,
                        registers[1].register,
                        registers[2].register,
                        self.assembler.wait_mask_for(registers),
                    )
                )
                for argument_node, register, temporary in zip(
                    node.arguments,
                    registers,
                    temporaries,
                ):
                    self._consume(argument_node, target)
                    if temporary:
                        self.assembler.release(register, target)
                target.barrier_slot = None
                value = target
            elif node.kind in {
                InstructionType.TOGGLE2_F32,
                InstructionType.TOGGLE4_F32,
            }:
                value = self._toggle(node, arguments)
            else:
                raise SassAssemblyError(
                    f"grouped LM CSE does not support {node.kind.name}"
                )
        self.values[node] = value
        return value

    def compile(
        self,
        programs: Sequence[Program],
        shape: LMKernelShape,
    ) -> tuple[tuple[int, ...], int]:
        roots = _program_roots(programs, shape)
        if len(roots) != len(self.outputs):
            raise SassAssemblyError("grouped LM output count differs from its site")
        self._count_uses(roots)
        counts: list[int] = []
        previous = 0
        for root, output in zip(roots, self.outputs):
            value = self._emit_node(root)
            self._relocate_output_conflict(output, root)
            if output in self.assembler.free_registers:
                self.assembler.free_registers.remove(output)
            wait = self.assembler.wait_mask_for((value,))
            if value.immediate:
                self.assembler.emit(
                    _fadd_imm(output, REGISTER_RZ, int(value.immediate_bits), wait)
                )
            elif value.register != output:
                self.assembler.emit(_mov(output, value.register, wait))
            self._consume(root, Operand(register=output, owned=True))
            self.pinned.add(output)
            current = len(self.assembler.instructions)
            counts.append(current - previous)
            previous = current
        if self.assembler.active_barriers:
            self.assembler.emit(
                _nop(sum(1 << slot for slot in self.assembler.active_barriers))
            )
            self.assembler.active_barriers.clear()
            counts[-1] += 1
        return tuple(counts), self.assembler.high_water_register

def compile_lm_program_group(
    plan: LMCubinPlan,
    site: LMScalarSitePlan,
    programs: Sequence[Program],
    shape: LMKernelShape,
) -> tuple[list[tuple[int, int]], tuple[int, ...], int]:
    assembler = GroupedLMCSEAssembler(plan, site, shape)
    counts, high_water = assembler.compile(programs, shape)
    return assembler.assembler.instructions, counts, high_water
