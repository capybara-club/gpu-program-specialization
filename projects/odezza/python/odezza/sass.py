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
"""Python SASS specializer reference."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Sequence

from .ast import InstructionType, OutputPrograms, Program, SystemGroup, _ARITY
from .elf import CubinPlan, INSTRUCTION_BYTES, WAIT_BARRIER_MASK
from .model import MAX_TOGGLE_BITS, KernelShape


REGISTER_RZ = 255
REGISTER_PAD = 2
BARRIER_SLOTS = 6

NOP_WORD0 = 0x0000000000007918
NOP_WORD1 = 0x000FC00000000000
OPCODE_MOV = 0x7202
OPCODE_FSEL = 0x7208
OPCODE_FMNMX_REG = 0x7209
OPCODE_FMUL_REG = 0x7220
OPCODE_FADD_REG = 0x7221
OPCODE_FFMA_REG = 0x7223
OPCODE_FADD_IMM = 0x7421
OPCODE_FMUL_IMM = 0x7820
OPCODE_LOP3_IMM = 0x7812
OPCODE_MUFU = 0x7308
OPCODE_BRA = 0x7947

CONTROL_ALU_HI = 0xFC
CONTROL_ALU_STALL = 0xC
CONTROL_BARRIER_BASE = 0xE2
CONTROL_MUFU_STALL = 0x6
REST_FADD_IMM = 0x0000010000
REST_FADD_ABS = 0x0000010200
REST_FMUL = 0x0000410000
REST_FMUL_RZ = 0x000040C000
REST_FMNMX_MIN = 0x0003810000
REST_FMNMX_MAX = 0x0007810000
REST_FFMA = 0x0000010000
REST_MOV = 0x0000000F00
REST_MUFU_COS = 0x0000000000
REST_MUFU_SIN = 0x0000000400
REST_MUFU_EX2 = 0x0000000800
REST_MUFU_LG2 = 0x0000000C00
REST_MUFU_RCP = 0x0000001000
REST_MUFU_RSQ = 0x0000001400
REST_MUFU_SQRT = 0x0000002000
REST_MUFU_TANH = 0x0000002400
REST_BRA = 0x0003800000

NEG_ONE_BITS = 0xBF800000
SIN_COS_SCALE_BITS = 0x3E22F983
LN_TWO_BITS = 0x3F317218
LOG2_E_BITS = 0x3FB8AA3B


class SassAssemblyError(ValueError):
    pass


@dataclass
class Operand:
    immediate_bits: int | None = None
    register: int = 0
    owned: bool = False
    barrier_slot: int | None = None
    barrier_token: int = 0

    @property
    def immediate(self) -> bool:
        return self.immediate_bits is not None


def _control(wait_mask: int, high: int, stall: int, rest: int) -> int:
    return ((wait_mask & WAIT_BARRIER_MASK) << 52) | ((high & 0xFF) << 44) | ((stall & 0xF) << 40) | rest


def _alu_control(wait_mask: int, rest: int = 0) -> int:
    return _control(wait_mask, CONTROL_ALU_HI, CONTROL_ALU_STALL, rest)


def _barrier_control(wait_mask: int, slot: int, rest: int) -> int:
    return _control(wait_mask, CONTROL_BARRIER_BASE + 4 * slot, CONTROL_MUFU_STALL, rest)


def _nop(wait_mask: int = 0) -> tuple[int, int]:
    return NOP_WORD0, ((wait_mask & WAIT_BARRIER_MASK) << 52) | NOP_WORD1


def _mov(dst: int, src: int, wait_mask: int) -> tuple[int, int]:
    return (src << 32) | (dst << 16) | OPCODE_MOV, _alu_control(wait_mask, REST_MOV)


def _fadd_reg(dst: int, lhs: int, rhs: int, wait_mask: int) -> tuple[int, int]:
    return (rhs << 32) | (lhs << 24) | (dst << 16) | OPCODE_FADD_REG, _alu_control(wait_mask)


def _fadd_imm(dst: int, src: int, bits: int, wait_mask: int) -> tuple[int, int]:
    return ((bits & 0xFFFFFFFF) << 32) | (src << 24) | (dst << 16) | OPCODE_FADD_IMM, _alu_control(wait_mask, REST_FADD_IMM)


def _fmul_reg(dst: int, lhs: int, rhs: int, wait_mask: int) -> tuple[int, int]:
    return (rhs << 32) | (lhs << 24) | (dst << 16) | OPCODE_FMUL_REG, _alu_control(wait_mask, REST_FMUL)


def _fmul_imm(dst: int, src: int, bits: int, wait_mask: int, rest: int = REST_FMUL) -> tuple[int, int]:
    return ((bits & 0xFFFFFFFF) << 32) | (src << 24) | (dst << 16) | OPCODE_FMUL_IMM, _alu_control(wait_mask, rest)


def _fsel(dst: int, true_value: int, false_value: int, predicate: int) -> tuple[int, int]:
    return (false_value << 32) | (true_value << 24) | (dst << 16) | OPCODE_FSEL, _alu_control(
        0, predicate << 23
    )


def _toggle_test(template: tuple[int, int], permutation_register: int, bit_index: int) -> tuple[int, int]:
    word0, word1 = template
    word0 &= 0x0000000000FFFFFF
    word0 |= permutation_register << 24
    word0 |= (1 << bit_index) << 32
    word1 &= ~((0x3F << 52) | (0xF << 40))
    word1 |= CONTROL_ALU_STALL << 40
    return word0, word1


def _absolute(dst: int, src: int, wait_mask: int) -> tuple[int, int]:
    return (0x800000FF << 32) | (src << 24) | (dst << 16) | OPCODE_FADD_REG, _alu_control(wait_mask, REST_FADD_ABS)


def _fmnmx(dst: int, lhs: int, rhs: int, maximum: bool, wait_mask: int) -> tuple[int, int]:
    rest = REST_FMNMX_MAX if maximum else REST_FMNMX_MIN
    return (rhs << 32) | (lhs << 24) | (dst << 16) | OPCODE_FMNMX_REG, _alu_control(wait_mask, rest)


def _ffma(dst: int, lhs: int, rhs: int, addend: int, wait_mask: int) -> tuple[int, int]:
    return (rhs << 32) | (lhs << 24) | (dst << 16) | OPCODE_FFMA_REG, _alu_control(wait_mask, REST_FFMA | addend)


def _mufu(dst: int, src: int, wait_mask: int, slot: int, rest: int) -> tuple[int, int]:
    return (src << 32) | (dst << 16) | OPCODE_MUFU, _barrier_control(wait_mask, slot, rest)


def _branch(remaining: int, architecture: int) -> tuple[int, int]:
    if architecture // 10 == 8:
        target = 16 * remaining - 16
        word0 = (target << 32) | OPCODE_BRA
    else:
        target = 4 * remaining - 4
        word0 = ((target & 0xFF) << 16) | ((target & ~0xFF) << 26) | OPCODE_BRA
    return word0, _alu_control(0, REST_BRA)


class Assembler:
    def __init__(self, plan: CubinPlan, shape: KernelShape, result_register: int, incoming_wait_mask: int):
        if not 0 <= plan.site.predicate_register < 7:
            raise SassAssemblyError("the inspected toggle predicate must be P0 through P6")
        self.plan = plan
        self.shape = shape
        self.instructions: list[tuple[int, int]] = []
        self.stack: list[Operand] = []
        reserved = set(plan.site.input_registers) | set(plan.site.output_registers) | {plan.site.permutation_register, REGISTER_RZ}
        self.free_registers = [value for value in plan.site.available_registers if value not in reserved]
        self.next_register = plan.register_count
        self.high_water_register = plan.register_count
        self.result_register = result_register
        self.pending_wait_mask = incoming_wait_mask
        self.active_barriers: dict[int, int] = {}
        self.reserved_barriers: set[int] = set()
        self.next_barrier_token = 1

    def emit(self, instruction: tuple[int, int]) -> None:
        word0, word1 = instruction
        if self.pending_wait_mask:
            word1 |= (self.pending_wait_mask & WAIT_BARRIER_MASK) << 52
            self.pending_wait_mask = 0
        self.instructions.append((word0, word1))

    def allocate(self) -> Operand:
        if self.free_registers:
            register = self.free_registers.pop()
            self.high_water_register = max(self.high_water_register, register + 1)
            return Operand(register=register, owned=True)
        if self.next_register >= REGISTER_RZ:
            raise SassAssemblyError("specialized expression exhausted the register file")
        result = Operand(register=self.next_register, owned=True)
        self.next_register += 1
        self.high_water_register = max(self.high_water_register, self.next_register)
        return result

    def release(self, operand: Operand, keep: Operand | None = None) -> None:
        if not operand.owned or operand.immediate or (keep is not None and operand.register == keep.register):
            return
        self.pending_wait_mask |= self.wait_mask_for((operand,))
        if operand.register not in self.free_registers:
            self.free_registers.append(operand.register)

    def wait_mask_for(self, operands: Sequence[Operand]) -> int:
        mask = 0
        for operand in operands:
            slot = operand.barrier_slot
            if slot is not None and self.active_barriers.get(slot) == operand.barrier_token:
                mask |= 1 << slot
                del self.active_barriers[slot]
        return mask

    def allocate_barrier(self) -> tuple[int, int]:
        for slot in range(BARRIER_SLOTS):
            if slot not in self.active_barriers and slot not in self.reserved_barriers:
                token = self.next_barrier_token
                self.next_barrier_token += 1
                self.active_barriers[slot] = token
                return slot, token
        if not self.active_barriers:
            raise SassAssemblyError("all dependency barriers are reserved")
        self.emit(_nop(sum(1 << slot for slot in self.active_barriers)))
        self.active_barriers.clear()
        return self.allocate_barrier()

    def ensure_register(self, operand: Operand) -> Operand:
        if not operand.immediate:
            return operand
        target = self.allocate()
        self.emit(_fadd_imm(target.register, REGISTER_RZ, int(operand.immediate_bits), 0))
        return target

    def choose_output(self, *operands: Operand) -> Operand:
        for operand in operands:
            if operand.owned and not operand.immediate:
                return Operand(operand.immediate_bits, operand.register, True, operand.barrier_slot, operand.barrier_token)
        return self.allocate()

    def source_operand(self, kind: InstructionType, index: int) -> Operand:
        if kind == InstructionType.STATE_F32:
            if not 0 <= index < self.shape.state_count:
                raise SassAssemblyError("state index is outside the register ABI")
            encoded = index
        elif kind == InstructionType.CONSTANT_F32:
            if not 0 <= index < self.shape.constant_count:
                raise SassAssemblyError("constant index is outside the register ABI")
            encoded = self.shape.state_count + index
        else:
            raise SassAssemblyError("source operand requires a state or constant leaf")
        return Operand(register=self.plan.site.input_registers[encoded])

    def select_level(self, bit_index: int, values: Sequence[Operand]) -> list[Operand]:
        if not 0 <= bit_index < MAX_TOGGLE_BITS or len(values) < 2 or len(values) % 2:
            raise SassAssemblyError("invalid toggle selection level")
        self.emit(_toggle_test(self.plan.site.toggle_test_instruction, self.plan.site.permutation_register, bit_index))
        result: list[Operand] = []
        for index in range(0, len(values), 2):
            false_value = self.ensure_register(values[index])
            true_value = self.ensure_register(values[index + 1])
            target = self.choose_output(false_value, true_value)
            self.emit(_fsel(target.register, true_value.register, false_value.register, self.plan.site.predicate_register))
            self.release(false_value, target)
            self.release(true_value, target)
            result.append(target)
        return result

    def select(self, bit_indices: Sequence[int], values: Sequence[Operand]) -> Operand:
        reduced = list(values)
        for bit_index in bit_indices:
            reduced = self.select_level(bit_index, reduced)
        if len(reduced) != 1:
            raise SassAssemblyError("toggle selection did not reduce to one value")
        return reduced[0]

    def add_or_multiply(self, multiply: bool) -> None:
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        if lhs.immediate and not rhs.immediate:
            lhs, rhs = rhs, lhs
        elif lhs.immediate and rhs.immediate:
            lhs = self.ensure_register(lhs)
        target = self.choose_output(lhs, rhs)
        wait = self.wait_mask_for((lhs, rhs))
        if multiply:
            instruction = _fmul_imm(target.register, lhs.register, int(rhs.immediate_bits), wait) if rhs.immediate else _fmul_reg(target.register, lhs.register, rhs.register, wait)
        else:
            instruction = _fadd_imm(target.register, lhs.register, int(rhs.immediate_bits), wait) if rhs.immediate else _fadd_reg(target.register, lhs.register, rhs.register, wait)
        self.emit(instruction)
        self.release(lhs, target)
        self.release(rhs, target)
        target.barrier_slot = None
        self.stack.append(target)

    def subtract(self) -> None:
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        if rhs.immediate:
            rhs.immediate_bits = int(rhs.immediate_bits) ^ 0x80000000
            self.stack.extend((lhs, rhs))
            self.add_or_multiply(False)
            return
        negated = rhs if rhs.owned else self.allocate()
        self.emit(_fmul_imm(negated.register, rhs.register, NEG_ONE_BITS, self.wait_mask_for((rhs,))))
        lhs = self.ensure_register(lhs)
        target = self.choose_output(lhs, negated)
        self.emit(_fadd_reg(target.register, lhs.register, negated.register, self.wait_mask_for((lhs, negated))))
        self.release(rhs, target if rhs.register == negated.register else None)
        self.release(lhs, target)
        self.release(negated, target)
        self.stack.append(target)

    def divide(self) -> None:
        rhs = self.ensure_register(self.stack.pop())
        lhs = self.ensure_register(self.stack.pop())
        reciprocal = rhs if rhs.owned else self.allocate()
        slot, token = self.allocate_barrier()
        self.emit(_mufu(reciprocal.register, rhs.register, self.wait_mask_for((rhs,)), slot, REST_MUFU_RCP))
        reciprocal.barrier_slot = slot
        reciprocal.barrier_token = token
        target = lhs if lhs.owned else reciprocal
        self.emit(_fmul_reg(target.register, lhs.register, reciprocal.register, self.wait_mask_for((lhs, reciprocal))))
        self.release(lhs, target)
        if rhs.register != reciprocal.register:
            self.release(rhs, target)
        self.release(reciprocal, target)
        target.barrier_slot = None
        self.stack.append(target)

    def unary(self, name: str) -> None:
        value = self.stack.pop()
        if name == "neg" and value.immediate:
            value.immediate_bits = int(value.immediate_bits) ^ 0x80000000
            self.stack.append(value)
            return
        value = self.ensure_register(value)
        target = value if value.owned else self.allocate()
        wait = self.wait_mask_for((value,))
        if name == "neg":
            self.emit(_fmul_imm(target.register, value.register, NEG_ONE_BITS, wait))
        elif name == "abs":
            self.emit(_absolute(target.register, value.register, wait))
        else:
            rest = {
                "sqrt": REST_MUFU_SQRT,
                "rcp": REST_MUFU_RCP,
                "sin": REST_MUFU_SIN,
                "cos": REST_MUFU_COS,
                "exp2": REST_MUFU_EX2,
                "log2": REST_MUFU_LG2,
                "rsqrt": REST_MUFU_RSQ,
                "tanh": REST_MUFU_TANH,
            }[name]
            if name in {"sin", "cos"}:
                self.emit(_fmul_imm(target.register, value.register, SIN_COS_SCALE_BITS, wait, REST_FMUL_RZ))
                wait = 0
            slot, token = self.allocate_barrier()
            self.emit(_mufu(target.register, target.register if name in {"sin", "cos"} else value.register, wait, slot, rest))
            target.barrier_slot = slot
            target.barrier_token = token
        self.release(value, target)
        self.stack.append(target)

    def minmax(self, maximum: bool) -> None:
        rhs = self.ensure_register(self.stack.pop())
        lhs = self.ensure_register(self.stack.pop())
        target = self.choose_output(lhs, rhs)
        self.emit(_fmnmx(target.register, lhs.register, rhs.register, maximum, self.wait_mask_for((lhs, rhs))))
        self.release(lhs, target)
        self.release(rhs, target)
        self.stack.append(target)

    def fused_multiply_add(self) -> None:
        addend = self.ensure_register(self.stack.pop())
        rhs = self.ensure_register(self.stack.pop())
        lhs = self.ensure_register(self.stack.pop())
        target = self.choose_output(lhs, rhs, addend)
        self.emit(_ffma(target.register, lhs.register, rhs.register, addend.register, self.wait_mask_for((lhs, rhs, addend))))
        self.release(lhs, target)
        self.release(rhs, target)
        self.release(addend, target)
        self.stack.append(target)

    def compile(self, program: Program) -> None:
        program.validate(self.shape)
        returned = False
        for instruction in program.instructions():
            kind = instruction.kind
            if kind in {InstructionType.STATE_F32, InstructionType.CONSTANT_F32}:
                self.stack.append(self.source_operand(kind, instruction.operands[0]))
            elif kind in {InstructionType.TOGGLE2_F32, InstructionType.TOGGLE4_F32}:
                choice_count = 1 << len(instruction.operands)
                choices = self.stack[-choice_count:]
                del self.stack[-choice_count:]
                self.stack.append(self.select(instruction.operands, choices))
            elif kind == InstructionType.LITERAL_F32:
                self.stack.append(Operand(immediate_bits=instruction.operands[0]))
            elif kind == InstructionType.ADD_F32:
                self.add_or_multiply(False)
            elif kind == InstructionType.MUL_F32:
                self.add_or_multiply(True)
            elif kind == InstructionType.SUB_F32:
                self.subtract()
            elif kind == InstructionType.DIV_F32:
                self.divide()
            elif kind == InstructionType.NEG_F32:
                self.unary("neg")
            elif kind == InstructionType.ABS_F32:
                self.unary("abs")
            elif kind == InstructionType.MIN_F32:
                self.minmax(False)
            elif kind == InstructionType.MAX_F32:
                self.minmax(True)
            elif kind == InstructionType.FMA_F32:
                self.fused_multiply_add()
            elif kind in {
                InstructionType.SQRT_F32,
                InstructionType.RCP_F32,
                InstructionType.SIN_F32,
                InstructionType.COS_F32,
                InstructionType.EX2_F32,
                InstructionType.LG2_F32,
                InstructionType.RSQRT_F32,
                InstructionType.TANH_F32,
            }:
                names = {
                    InstructionType.SQRT_F32: "sqrt",
                    InstructionType.RCP_F32: "rcp",
                    InstructionType.SIN_F32: "sin",
                    InstructionType.COS_F32: "cos",
                    InstructionType.EX2_F32: "exp2",
                    InstructionType.LG2_F32: "log2",
                    InstructionType.RSQRT_F32: "rsqrt",
                    InstructionType.TANH_F32: "tanh",
                }
                self.unary(names[kind])
            elif kind == InstructionType.EXP_F32:
                self.stack.append(Operand(immediate_bits=LOG2_E_BITS))
                self.add_or_multiply(True)
                self.unary("exp2")
            elif kind == InstructionType.LOG_F32:
                self.unary("log2")
                self.stack.append(Operand(immediate_bits=LN_TWO_BITS))
                self.add_or_multiply(True)
            elif kind == InstructionType.RETURN_F32:
                returned = True
                break
            elif kind in _ARITY:
                raise SassAssemblyError(f"SASS emission is not implemented for {kind.name}")
            else:
                raise SassAssemblyError(f"unsupported instruction {kind.name}")
        if not returned or len(self.stack) != 1:
            raise SassAssemblyError("postorder program did not produce one result")
        root = self.stack.pop()
        if root.immediate:
            self.emit(_fadd_imm(self.result_register, REGISTER_RZ, int(root.immediate_bits), 0))
        else:
            self.emit(_mov(self.result_register, root.register, self.wait_mask_for((root,))))
        if self.active_barriers:
            self.emit(_nop(sum(1 << slot for slot in self.active_barriers)))
            self.active_barriers.clear()


@dataclass(frozen=True)
class SpecializationResult:
    cubin: bytes
    shared_instruction_counts: tuple[int, ...]
    system_instruction_counts: tuple[tuple[int, ...], ...]
    body_instruction_counts: tuple[int, ...]
    body_offsets: tuple[int, ...]
    register_count: int


def _compile_outputs(
    plan: CubinPlan,
    shape: KernelShape,
    outputs: OutputPrograms,
    incoming_wait_mask: int,
) -> tuple[list[tuple[int, int]], tuple[int, ...], int]:
    generated: list[tuple[int, int]] = []
    counts: list[int] = []
    high_water = plan.register_count
    pending_wait = incoming_wait_mask
    for output_index, program in zip(outputs.output_indices, outputs.programs):
        assembler = Assembler(plan, shape, plan.site.output_registers[output_index], pending_wait)
        assembler.compile(program)
        generated.extend(assembler.instructions)
        counts.append(len(assembler.instructions))
        high_water = max(high_water, assembler.high_water_register)
        pending_wait = 0
    return generated, tuple(counts), high_water


def specialize_cubin(
    cubin: bytes,
    plan: CubinPlan,
    group: SystemGroup,
    shape: KernelShape,
) -> SpecializationResult:
    """Specialize one shared RHS prelude and a compact set of system branches."""

    if len(cubin) != plan.cubin_size:
        raise SassAssemblyError("CUBIN size changed after inspection")
    if plan.input_count != shape.input_count or plan.output_count != shape.output_count:
        raise SassAssemblyError("kernel shape does not match the inspected register ABI")
    if plan.system_capacity != shape.system_capacity:
        raise SassAssemblyError("system capacity does not match the inspected branch table")
    group.validate(shape)
    expanded_register_count = plan.register_count

    shared, shared_counts, shared_high_water = _compile_outputs(
        plan,
        shape,
        group.shared,
        plan.site.incoming_wait_mask,
    )
    expanded_register_count = max(expanded_register_count, shared_high_water)
    dispatch_start = plan.site.dispatch_offsets[0]
    shared_branch_offset = plan.site.shared_start_offset + len(shared) * INSTRUCTION_BYTES
    if shared_branch_offset >= dispatch_start:
        raise SassAssemblyError("shared derivative programs exhausted the common patch site")
    shared_remaining = (dispatch_start - shared_branch_offset) // INSTRUCTION_BYTES
    shared_branch = _branch(shared_remaining, plan.architecture)
    if not shared:
        shared_branch = (
            shared_branch[0],
            shared_branch[1] | ((plan.site.incoming_wait_mask & WAIT_BARRIER_MASK) << 52),
        )
    shared.append(shared_branch)
    if len(shared) > plan.site.shared_instruction_count:
        raise SassAssemblyError("shared derivative programs exhausted the common patch site")

    bodies: list[tuple[int, int]] = []
    body_offsets: list[int] = []
    body_counts: list[int] = []
    system_counts: list[tuple[int, ...]] = []
    for system in group.systems:
        body, counts, high_water = _compile_outputs(plan, shape, system, 0)
        expanded_register_count = max(expanded_register_count, high_water)
        if len(body) + 1 > shape.system_patch_capacity:
            raise SassAssemblyError(
                f"candidate body needs {len(body) + 1} instructions; per-system capacity is {shape.system_patch_capacity}"
            )
        body_start = plan.site.arena_start_offset + len(bodies) * INSTRUCTION_BYTES
        branch_offset = body_start + len(body) * INSTRUCTION_BYTES
        if branch_offset >= plan.site.arena_end_offset:
            raise SassAssemblyError("candidate bodies exhausted the system arena")
        remaining = (plan.site.continuation_offset - branch_offset) // INSTRUCTION_BYTES
        body.append(_branch(remaining, plan.architecture))
        if len(bodies) + len(body) > plan.site.arena_instruction_count:
            raise SassAssemblyError("candidate bodies exhausted the system arena")
        body_offsets.append(body_start - plan.function.file_offset)
        body_counts.append(len(body))
        system_counts.append(counts)
        bodies.extend(body)

    output = bytearray(cubin)
    nop = _nop()
    for index in range(plan.site.shared_instruction_count):
        struct.pack_into("<QQ", output, plan.site.shared_start_offset + index * INSTRUCTION_BYTES, *nop)
    for index in range(plan.site.arena_instruction_count):
        struct.pack_into("<QQ", output, plan.site.arena_start_offset + index * INSTRUCTION_BYTES, *nop)
    for offset in plan.site.cleanup_offsets:
        struct.pack_into("<QQ", output, offset, *nop)
    for offset, final_register, live_register in zip(
        plan.site.output_materialization_offsets,
        plan.site.final_output_registers,
        plan.site.output_registers,
    ):
        struct.pack_into("<QQ", output, offset, *_mov(final_register, live_register, 0))
    for offset, instruction in zip(plan.site.dispatch_offsets, plan.site.dispatch_instructions):
        struct.pack_into("<QQ", output, offset, *instruction)
    for index, instruction in enumerate(shared):
        struct.pack_into("<QQ", output, plan.site.shared_start_offset + index * INSTRUCTION_BYTES, *instruction)
    for index, instruction in enumerate(bodies):
        struct.pack_into("<QQ", output, plan.site.arena_start_offset + index * INSTRUCTION_BYTES, *instruction)

    fallback = body_offsets[0]
    for index, table_offset in enumerate(plan.site.target_table_offsets):
        target = body_offsets[index] if index < len(body_offsets) else fallback
        struct.pack_into("<I", output, table_offset, target)

    final_register_count = plan.register_count
    if expanded_register_count > plan.register_count:
        if expanded_register_count > REGISTER_RZ - REGISTER_PAD:
            raise SassAssemblyError("expressions leave no register padding")
        final_register_count = expanded_register_count + REGISTER_PAD
        for offset in plan.register_count_offsets:
            struct.pack_into("<I", output, offset, final_register_count)
        for offset in plan.register_count_header_offsets:
            struct.pack_into("<B", output, offset, final_register_count)

    return SpecializationResult(
        cubin=bytes(output),
        shared_instruction_counts=shared_counts,
        system_instruction_counts=tuple(system_counts),
        body_instruction_counts=tuple(body_counts),
        body_offsets=tuple(body_offsets),
        register_count=final_register_count,
    )
