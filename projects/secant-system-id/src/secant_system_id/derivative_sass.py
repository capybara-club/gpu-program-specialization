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

from collections import Counter
from dataclasses import dataclass
import struct
from typing import Sequence

from .ast import InstructionType, Program
from .autodiff import SystemDerivativeBundle
from .cubin import CubinPlan, INSTRUCTION_BYTES
from .sass import (
    LN_TWO_BITS,
    LOG2_E_BITS,
    NEG_ONE_BITS,
    REGISTER_PAD,
    REGISTER_RZ,
    REST_FADD_ABS,
    REST_FMNMX_MAX,
    REST_FMNMX_MIN,
    REST_MUFU_COS,
    REST_MUFU_EX2,
    REST_MUFU_LG2,
    REST_MUFU_RCP,
    REST_MUFU_RSQ,
    REST_MUFU_SIN,
    REST_MUFU_SQRT,
    REST_MUFU_TANH,
    REST_FMUL_RZ,
    SIN_COS_SCALE_BITS,
    Assembler,
    Operand,
    SassAssemblyError,
    SpecializationResult,
    _abs,
    _branch,
    _fadd_imm,
    _fadd_reg,
    _ffma,
    _fmnmx,
    _fmul_imm,
    _fmul_reg,
    _mov,
    _mufu,
    _nop,
)


_ARITY = {
    InstructionType.ADD_F32: 2,
    InstructionType.SUB_F32: 2,
    InstructionType.MUL_F32: 2,
    InstructionType.DIV_F32: 2,
    InstructionType.NEG_F32: 1,
    InstructionType.SQRT_F32: 1,
    InstructionType.RCP_F32: 1,
    InstructionType.ABS_F32: 1,
    InstructionType.MIN_F32: 2,
    InstructionType.MAX_F32: 2,
    InstructionType.FMA_F32: 3,
    InstructionType.SIN_F32: 1,
    InstructionType.COS_F32: 1,
    InstructionType.EX2_F32: 1,
    InstructionType.LG2_F32: 1,
    InstructionType.RSQRT_F32: 1,
    InstructionType.TANH_F32: 1,
    InstructionType.EXP_F32: 1,
    InstructionType.LOG_F32: 1,
}


@dataclass(frozen=True)
class _Node:
    kind: InstructionType
    operand: int | None = None
    arguments: tuple["_Node", ...] = ()


def _program_roots(programs: Sequence[Program]) -> tuple[_Node, ...]:
    interned: dict[_Node, _Node] = {}
    roots: list[_Node] = []

    def intern(node: _Node) -> _Node:
        return interned.setdefault(node, node)

    for program in programs:
        stack: list[_Node] = []
        returned = False
        for instruction in program.instructions():
            kind = instruction.kind
            if kind in {
                InstructionType.STATIC_COLUMN_INPUT_F32,
                InstructionType.CONSTANT_BITS_F32,
            }:
                node = _Node(kind, instruction.operand)
            elif kind == InstructionType.RETURN_F32:
                if returned or len(stack) != 1:
                    raise SassAssemblyError("invalid scalar derivative postorder program")
                roots.append(stack[-1])
                returned = True
                continue
            elif kind in _ARITY:
                arity = _ARITY[kind]
                if len(stack) < arity:
                    raise SassAssemblyError("derivative program underflow")
                arguments = tuple(stack[-arity:])
                del stack[-arity:]
                if kind == InstructionType.EXP_F32:
                    scale = intern(
                        _Node(InstructionType.CONSTANT_BITS_F32, LOG2_E_BITS)
                    )
                    product = intern(
                        _Node(
                            InstructionType.MUL_F32,
                            arguments=(arguments[0], scale),
                        )
                    )
                    node = _Node(InstructionType.EX2_F32, arguments=(product,))
                elif kind == InstructionType.LOG_F32:
                    logarithm = intern(
                        _Node(InstructionType.LG2_F32, arguments=arguments)
                    )
                    scale = intern(
                        _Node(InstructionType.CONSTANT_BITS_F32, LN_TWO_BITS)
                    )
                    node = _Node(
                        InstructionType.MUL_F32,
                        arguments=(logarithm, scale),
                    )
                else:
                    node = _Node(kind, arguments=arguments)
            else:
                raise SassAssemblyError(
                    f"CSE derivative writer does not support {kind.name}"
                )
            node = intern(node)
            stack.append(node)
        if not returned:
            raise SassAssemblyError("derivative program has no return")
    return tuple(roots)


def _copy_operand(value: Operand) -> Operand:
    return Operand(
        immediate_bits=value.immediate_bits,
        register=value.register,
        owned=value.owned,
        barrier_slot=value.barrier_slot,
        barrier_token=value.barrier_token,
    )


class _SiteCSEAssembler:
    """Hash-cons scalar derivative programs and emit each shared node once."""

    def __init__(
        self,
        plan: CubinPlan,
        output_registers: Sequence[int],
        incoming_wait_mask: int,
    ):
        if not output_registers:
            raise SassAssemblyError("a derivative site needs at least one output")
        explicit_new = tuple(
            range(plan.register_count, min(REGISTER_RZ - REGISTER_PAD, REGISTER_RZ))
        )
        pool = (
            explicit_new
            + tuple(reversed(output_registers))
            + tuple(plan.site.available_registers)
        )
        self.assembler = Assembler(
            plan.site.input_registers,
            (),
            pool,
            REGISTER_RZ,
            incoming_wait_mask,
            REGISTER_RZ - REGISTER_PAD,
        )
        # Every legal expansion register is already explicit in the pool. Keep
        # the final two registers as padding, and track only registers that are
        # actually allocated instead of forcing every derivative to 255 regs.
        self.assembler.next_register = REGISTER_RZ
        self.assembler.high_water_register = plan.register_count
        self.inputs = plan.site.input_registers
        self.outputs = tuple(output_registers)
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
        return value.owned and not value.immediate and self.remaining[node] == 1

    def _consume(self, node: _Node, keep: Operand | None = None) -> None:
        self.remaining[node] -= 1
        if self.remaining[node] < 0:
            raise SassAssemblyError("CSE derivative use count underflow")
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
            _fadd_imm(target.register, REGISTER_RZ, value.immediate_bits, 0)
        )
        return target, True

    def _relocate_live_output_conflicts(self, output: int, root: _Node) -> None:
        conflicts = [
            (node, operand)
            for node, operand in self.values.items()
            if node != root
            and self.remaining[node] > 0
            and not operand.immediate
            and operand.register == output
        ]
        if not conflicts:
            return
        source = conflicts[0][1]
        target = self._allocate()
        wait = self.assembler.wait_mask_for((source,))
        self.assembler.emit(_mov(target.register, source.register, wait))
        target.barrier_slot = None
        target.barrier_token = 0
        for node, _operand in conflicts:
            self.values[node] = _copy_operand(target)

    def _binary_add_multiply(
        self,
        node: _Node,
        lhs_node: _Node,
        rhs_node: _Node,
        lhs: Operand,
        rhs: Operand,
        multiply: bool,
    ) -> Operand:
        if lhs.immediate and not rhs.immediate:
            lhs_node, rhs_node, lhs, rhs = rhs_node, lhs_node, rhs, lhs
        elif lhs.immediate and rhs.immediate:
            lhs, lhs_temporary = self._materialize(lhs)
            if not lhs_temporary:
                raise AssertionError("constant materialization did not allocate")
        target = (
            _copy_operand(lhs)
            if self._last_owned(lhs_node, lhs)
            else (
                _copy_operand(rhs)
                if self._last_owned(rhs_node, rhs)
                else self._allocate()
            )
        )
        wait = self.assembler.wait_mask_for((lhs, rhs))
        if multiply:
            instruction = (
                _fmul_imm(target.register, lhs.register, rhs.immediate_bits, wait)
                if rhs.immediate
                else _fmul_reg(target.register, lhs.register, rhs.register, wait)
            )
        else:
            instruction = (
                _fadd_imm(target.register, lhs.register, rhs.immediate_bits, wait)
                if rhs.immediate
                else _fadd_reg(target.register, lhs.register, rhs.register, wait)
            )
        self.assembler.emit(instruction)
        self._consume(lhs_node, target)
        self._consume(rhs_node, target)
        target.barrier_slot = None
        return target

    def _emit_subtract(
        self,
        lhs_node: _Node,
        rhs_node: _Node,
        lhs: Operand,
        rhs: Operand,
    ) -> Operand:
        if rhs.immediate:
            negated = Operand(immediate_bits=rhs.immediate_bits ^ 0x80000000)
            return self._binary_add_multiply(
                _Node(InstructionType.ADD_F32),
                lhs_node,
                rhs_node,
                lhs,
                negated,
                False,
            )
        negated = (
            _copy_operand(rhs)
            if self._last_owned(rhs_node, rhs)
            else self._allocate()
        )
        wait = self.assembler.wait_mask_for((rhs,))
        self.assembler.emit(
            _fmul_imm(negated.register, rhs.register, NEG_ONE_BITS, wait)
        )
        negated.barrier_slot = None
        lhs_register, lhs_temporary = self._materialize(lhs)
        target = (
            _copy_operand(lhs_register)
            if lhs_temporary or self._last_owned(lhs_node, lhs_register)
            else negated
        )
        wait = self.assembler.wait_mask_for((lhs_register, negated))
        self.assembler.emit(
            _fadd_reg(target.register, lhs_register.register, negated.register, wait)
        )
        self._consume(lhs_node, target)
        self._consume(rhs_node, target)
        if lhs_temporary:
            self.assembler.release(lhs_register, target)
        if target.register != negated.register:
            self.assembler.release(negated)
        target.barrier_slot = None
        return target

    def _emit_divide(
        self,
        lhs_node: _Node,
        rhs_node: _Node,
        lhs: Operand,
        rhs: Operand,
    ) -> Operand:
        rhs_register, rhs_temporary = self._materialize(rhs)
        lhs_register, lhs_temporary = self._materialize(lhs)
        reciprocal = (
            _copy_operand(rhs_register)
            if rhs_temporary or self._last_owned(rhs_node, rhs_register)
            else self._allocate()
        )
        wait = self.assembler.wait_mask_for((rhs_register,))
        slot, token = self.assembler.allocate_barrier()
        self.assembler.emit(
            _mufu(
                reciprocal.register,
                rhs_register.register,
                wait,
                slot,
                REST_MUFU_RCP,
            )
        )
        reciprocal.barrier_slot = slot
        reciprocal.barrier_token = token
        target = (
            _copy_operand(lhs_register)
            if lhs_temporary or self._last_owned(lhs_node, lhs_register)
            else reciprocal
        )
        wait = self.assembler.wait_mask_for((lhs_register, reciprocal))
        self.assembler.emit(
            _fmul_reg(
                target.register,
                lhs_register.register,
                reciprocal.register,
                wait,
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

    def _emit_unary(
        self,
        node: _Node,
        argument_node: _Node,
        argument: Operand,
    ) -> Operand:
        kind = node.kind
        source, source_temporary = self._materialize(argument)
        target = (
            _copy_operand(source)
            if source_temporary or self._last_owned(argument_node, source)
            else self._allocate()
        )
        wait = self.assembler.wait_mask_for((source,))
        if kind == InstructionType.NEG_F32:
            self.assembler.emit(
                _fmul_imm(target.register, source.register, NEG_ONE_BITS, wait)
            )
            target.barrier_slot = None
        elif kind == InstructionType.ABS_F32:
            self.assembler.emit(_abs(target.register, source.register, wait))
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
            }[kind]
            source_register = source.register
            if kind in {InstructionType.SIN_F32, InstructionType.COS_F32}:
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

    def _emit_node(self, node: _Node) -> Operand:
        cached = self.values.get(node)
        if cached is not None:
            return cached
        if node.kind == InstructionType.STATIC_COLUMN_INPUT_F32:
            value = Operand(register=self.inputs[node.operand])
        elif node.kind == InstructionType.CONSTANT_BITS_F32:
            value = Operand(immediate_bits=node.operand)
        else:
            arguments = [self._emit_node(argument) for argument in node.arguments]
            kind = node.kind
            if kind in {InstructionType.ADD_F32, InstructionType.MUL_F32}:
                value = self._binary_add_multiply(
                    node,
                    node.arguments[0],
                    node.arguments[1],
                    arguments[0],
                    arguments[1],
                    kind == InstructionType.MUL_F32,
                )
            elif kind == InstructionType.SUB_F32:
                value = self._emit_subtract(
                    node.arguments[0],
                    node.arguments[1],
                    arguments[0],
                    arguments[1],
                )
            elif kind == InstructionType.DIV_F32:
                value = self._emit_divide(
                    node.arguments[0],
                    node.arguments[1],
                    arguments[0],
                    arguments[1],
                )
            elif kind in {
                InstructionType.NEG_F32,
                InstructionType.SQRT_F32,
                InstructionType.RCP_F32,
                InstructionType.ABS_F32,
                InstructionType.SIN_F32,
                InstructionType.COS_F32,
                InstructionType.EX2_F32,
                InstructionType.LG2_F32,
                InstructionType.RSQRT_F32,
                InstructionType.TANH_F32,
            }:
                value = self._emit_unary(node, node.arguments[0], arguments[0])
            elif kind in {InstructionType.MIN_F32, InstructionType.MAX_F32}:
                lhs, lhs_temporary = self._materialize(arguments[0])
                rhs, rhs_temporary = self._materialize(arguments[1])
                target = (
                    _copy_operand(lhs)
                    if lhs_temporary or self._last_owned(node.arguments[0], lhs)
                    else (
                        _copy_operand(rhs)
                        if rhs_temporary or self._last_owned(node.arguments[1], rhs)
                        else self._allocate()
                    )
                )
                wait = self.assembler.wait_mask_for((lhs, rhs))
                self.assembler.emit(
                    _fmnmx(
                        target.register,
                        lhs.register,
                        rhs.register,
                        REST_FMNMX_MAX
                        if kind == InstructionType.MAX_F32
                        else REST_FMNMX_MIN,
                        wait,
                    )
                )
                self._consume(node.arguments[0], target)
                self._consume(node.arguments[1], target)
                if lhs_temporary:
                    self.assembler.release(lhs, target)
                if rhs_temporary:
                    self.assembler.release(rhs, target)
                target.barrier_slot = None
                value = target
            elif kind == InstructionType.FMA_F32:
                registers: list[Operand] = []
                temporaries: list[bool] = []
                for argument in arguments:
                    register, temporary = self._materialize(argument)
                    registers.append(register)
                    temporaries.append(temporary)
                target: Operand | None = None
                for argument_node, register, temporary in zip(
                    node.arguments, registers, temporaries
                ):
                    if temporary or self._last_owned(argument_node, register):
                        target = _copy_operand(register)
                        break
                if target is None:
                    target = self._allocate()
                wait = self.assembler.wait_mask_for(registers)
                self.assembler.emit(
                    _ffma(
                        target.register,
                        registers[0].register,
                        registers[1].register,
                        registers[2].register,
                        wait,
                    )
                )
                for argument_node, register, temporary in zip(
                    node.arguments, registers, temporaries
                ):
                    self._consume(argument_node, target)
                    if temporary:
                        self.assembler.release(register, target)
                target.barrier_slot = None
                value = target
            else:
                raise SassAssemblyError(
                    f"CSE derivative writer does not support {kind.name}"
                )
        self.values[node] = value
        return value

    def compile(self, programs: Sequence[Program]) -> tuple[tuple[int, ...], int]:
        roots = _program_roots(programs)
        if len(roots) != len(self.outputs):
            raise SassAssemblyError("derivative output count does not match marker ABI")
        self._count_uses(roots)
        instruction_counts: list[int] = []
        previous_count = 0
        for root, output in zip(roots, self.outputs):
            value = self._emit_node(root)
            self._relocate_live_output_conflicts(output, root)
            if output in self.assembler.free_registers:
                self.assembler.free_registers.remove(output)
            wait = self.assembler.wait_mask_for((value,))
            if value.immediate:
                self.assembler.emit(
                    _fadd_imm(output, REGISTER_RZ, value.immediate_bits, wait)
                )
            elif value.register != output:
                self.assembler.emit(_mov(output, value.register, wait))
            self._consume(root, Operand(register=output, owned=True))
            self.pinned.add(output)
            current_count = len(self.assembler.instructions)
            instruction_counts.append(current_count - previous_count)
            previous_count = current_count
        if self.assembler.active_barriers:
            self.assembler.emit(
                _nop(sum(1 << slot for slot in self.assembler.active_barriers))
            )
            self.assembler.active_barriers.clear()
            instruction_counts[-1] += 1
        return tuple(instruction_counts), self.assembler.high_water_register

    def compile_independent(
        self,
        programs: Sequence[Program],
    ) -> tuple[tuple[int, ...], int]:
        """Compile high-pressure outputs independently, hardest first.

        Cross-output CSE is normally both smaller and faster. It can retain a
        large primal subgraph while completed derivative outputs progressively
        pin the small register pool exposed by a register-heavy LM template.
        This fallback deliberately gives up cross-output reuse. Each partial
        releases all of its temporaries before the next one begins, and the
        largest symbolic derivatives run while the most output registers are
        still available as scratch.
        """

        roots = _program_roots(programs)
        if len(roots) != len(self.outputs):
            raise SassAssemblyError("derivative output count does not match marker ABI")
        order = sorted(
            range(len(programs)),
            key=lambda index: (
                sum(1 for _ in programs[index].instructions()),
                -index,
            ),
            reverse=True,
        )
        instruction_counts = [0] * len(programs)
        for index in order:
            self.values.clear()
            self.remaining.clear()
            root = roots[index]
            output = self.outputs[index]
            self._count_uses((root,))
            previous_count = len(self.assembler.instructions)
            value = self._emit_node(root)
            self._relocate_live_output_conflicts(output, root)
            if output in self.assembler.free_registers:
                self.assembler.free_registers.remove(output)
            wait = self.assembler.wait_mask_for((value,))
            if value.immediate:
                self.assembler.emit(
                    _fadd_imm(output, REGISTER_RZ, value.immediate_bits, wait)
                )
            elif value.register != output:
                self.assembler.emit(_mov(output, value.register, wait))
            self._consume(root, Operand(register=output, owned=True))
            self.pinned.add(output)
            self.assembler.free_registers = [
                register
                for register in self.assembler.free_registers
                if register not in self.pinned
            ]
            if any(count != 0 for count in self.remaining.values()):
                raise SassAssemblyError("independent derivative left live values")
            self.values.clear()
            instruction_counts[index] = (
                len(self.assembler.instructions) - previous_count
            )
        if self.assembler.active_barriers:
            self.assembler.emit(
                _nop(sum(1 << slot for slot in self.assembler.active_barriers))
            )
            self.assembler.active_barriers.clear()
            instruction_counts[order[-1]] += 1
        return tuple(instruction_counts), self.assembler.high_water_register


_REGISTER_PRESSURE_ERRORS = {
    "specialized expression exhausted the register file",
    "CSE allocator attempted to reuse a derivative output",
}


def _compile_program_group(
    plan: CubinPlan,
    output_registers: Sequence[int],
    incoming_wait_mask: int,
    programs: Sequence[Program],
) -> tuple[_SiteCSEAssembler, tuple[int, ...], int, bool]:
    assembler = _SiteCSEAssembler(plan, output_registers, incoming_wait_mask)
    try:
        counts, high_water = assembler.compile(programs)
        return assembler, counts, high_water, False
    except SassAssemblyError as error:
        if str(error) not in _REGISTER_PRESSURE_ERRORS:
            raise
    assembler = _SiteCSEAssembler(plan, output_registers, incoming_wait_mask)
    counts, high_water = assembler.compile_independent(programs)
    return assembler, counts, high_water, True


def specialize_derivative_bundle(
    cubin: bytes,
    plan: CubinPlan,
    bundle: SystemDerivativeBundle,
) -> SpecializationResult:
    """Specialize primal/partials with CSE across each missing-site bundle."""

    return specialize_program_groups(
        cubin,
        plan,
        tuple(site.programs for site in bundle.sites),
    )


def specialize_program_groups(
    cubin: bytes,
    plan: CubinPlan,
    program_groups: Sequence[Sequence[Program]],
) -> SpecializationResult:
    """Specialize ordered output groups with CSE local to each group."""

    if len(cubin) != plan.cubin_size:
        raise SassAssemblyError("CUBIN size changed after inspection")
    output_count = sum(len(group) for group in program_groups)
    if output_count != len(plan.site.output_registers):
        raise SassAssemblyError("program groups do not match marker outputs")
    if not program_groups or any(not group for group in program_groups):
        raise SassAssemblyError("program groups must be nonempty")

    generated: list[tuple[int, int]] = []
    counts: list[int] = []
    pressure_fallback_groups: list[int] = []
    expanded_register_count = plan.register_count
    output_offset = 0
    for group_index, programs in enumerate(program_groups):
        site_output_count = len(programs)
        site_outputs = plan.site.output_registers[
            output_offset : output_offset + site_output_count
        ]
        (
            assembler,
            site_counts,
            high_water,
            _used_pressure_fallback,
        ) = _compile_program_group(
            plan,
            site_outputs,
            plan.site.incoming_wait_mask if group_index == 0 else 0,
            programs,
        )
        generated.extend(assembler.assembler.instructions)
        counts.extend(site_counts)
        if _used_pressure_fallback:
            pressure_fallback_groups.append(group_index)
        expanded_register_count = max(expanded_register_count, high_water)
        output_offset += site_output_count

    if len(generated) > plan.site.instruction_count:
        raise SassAssemblyError(
            f"specialization needs {len(generated)} SASS instructions but "
            f"the site has {plan.site.instruction_count}"
        )
    remaining = plan.site.instruction_count - len(generated)
    if remaining:
        generated.append(_nop() if remaining == 1 else _branch(remaining, plan.architecture))

    output = bytearray(cubin)
    for index, (word0, word1) in enumerate(generated):
        struct.pack_into(
            "<QQ",
            output,
            plan.site.start_offset + index * INSTRUCTION_BYTES,
            word0,
            word1,
        )
    struct.pack_into("<QQ", output, plan.site.load_fence_offset, *_nop())

    final_register_count = plan.register_count
    if expanded_register_count > plan.register_count:
        if expanded_register_count > REGISTER_RZ - REGISTER_PAD:
            raise SassAssemblyError("specialized derivative leaves no register padding")
        final_register_count = expanded_register_count + REGISTER_PAD
        for offset in plan.register_count_offsets:
            struct.pack_into("<I", output, offset, final_register_count)
        for offset in plan.register_count_header_offsets:
            struct.pack_into("<B", output, offset, final_register_count)
    return SpecializationResult(
        bytes(output),
        tuple(counts),
        final_register_count,
        tuple(pressure_fallback_groups),
    )
