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
import struct
from typing import Sequence

from .ast import InstructionType, Program
from .cubin import CubinPlan, INSTRUCTION_BYTES, WAIT_BARRIER_MASK


REGISTER_RZ = 255
REGISTER_PAD = 2
BARRIER_SLOTS = 6

NOP_WORD0 = 0x0000000000007918
NOP_WORD1 = 0x000FC00000000000
OPCODE_MOV = 0x7202
OPCODE_FMNMX_REG = 0x7209
OPCODE_FMUL_REG = 0x7220
OPCODE_FADD_REG = 0x7221
OPCODE_FFMA_REG = 0x7223
OPCODE_FADD_IMM = 0x7421
OPCODE_FMUL_IMM = 0x7820
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


@dataclass(frozen=True)
class SpecializationResult:
    cubin: bytes
    instruction_counts: tuple[int, ...]
    register_count: int
    pressure_fallback_groups: tuple[int, ...] = ()


def _control(wait_mask: int, high: int, stall: int, rest: int) -> int:
    return (
        ((wait_mask & WAIT_BARRIER_MASK) << 52)
        | ((high & 0xFF) << 44)
        | ((stall & 0xF) << 40)
        | rest
    )


def _alu_control(wait_mask: int, rest: int = 0) -> int:
    return _control(wait_mask, CONTROL_ALU_HI, CONTROL_ALU_STALL, rest)


def _barrier_control(wait_mask: int, slot: int, rest: int) -> int:
    return _control(wait_mask, CONTROL_BARRIER_BASE + 4 * slot, CONTROL_MUFU_STALL, rest)


def _nop(wait_mask: int = 0) -> tuple[int, int]:
    return NOP_WORD0, ((wait_mask & WAIT_BARRIER_MASK) << 52) | NOP_WORD1


def _mov(dst: int, src: int, wait_mask: int) -> tuple[int, int]:
    return (src << 32) | (dst << 16) | OPCODE_MOV, _alu_control(wait_mask, REST_MOV)


def _fadd_reg(dst: int, lhs: int, rhs: int, wait_mask: int) -> tuple[int, int]:
    return (
        (rhs << 32) | (lhs << 24) | (dst << 16) | OPCODE_FADD_REG,
        _alu_control(wait_mask),
    )


def _fadd_imm(dst: int, src: int, bits: int, wait_mask: int) -> tuple[int, int]:
    return (
        ((bits & 0xFFFFFFFF) << 32) | (src << 24) | (dst << 16) | OPCODE_FADD_IMM,
        _alu_control(wait_mask, REST_FADD_IMM),
    )


def _fmul_reg(dst: int, lhs: int, rhs: int, wait_mask: int) -> tuple[int, int]:
    return (
        (rhs << 32) | (lhs << 24) | (dst << 16) | OPCODE_FMUL_REG,
        _alu_control(wait_mask, REST_FMUL),
    )


def _fmul_imm(dst: int, src: int, bits: int, wait_mask: int, rest: int = REST_FMUL) -> tuple[int, int]:
    return (
        ((bits & 0xFFFFFFFF) << 32) | (src << 24) | (dst << 16) | OPCODE_FMUL_IMM,
        _alu_control(wait_mask, rest),
    )


def _abs(dst: int, src: int, wait_mask: int) -> tuple[int, int]:
    return (
        (0x800000FF << 32) | (src << 24) | (dst << 16) | OPCODE_FADD_REG,
        _alu_control(wait_mask, REST_FADD_ABS),
    )


def _fmnmx(dst: int, lhs: int, rhs: int, rest: int, wait_mask: int) -> tuple[int, int]:
    return (
        (rhs << 32) | (lhs << 24) | (dst << 16) | OPCODE_FMNMX_REG,
        _alu_control(wait_mask, rest),
    )


def _ffma(dst: int, lhs: int, rhs: int, addend: int, wait_mask: int) -> tuple[int, int]:
    return (
        (rhs << 32) | (lhs << 24) | (dst << 16) | OPCODE_FFMA_REG,
        _alu_control(wait_mask, REST_FFMA | addend),
    )


def _mufu(dst: int, src: int, wait_mask: int, slot: int, rest: int) -> tuple[int, int]:
    return (
        (src << 32) | (dst << 16) | OPCODE_MUFU,
        _barrier_control(wait_mask, slot, rest),
    )


def _branch(remaining: int, architecture: int) -> tuple[int, int]:
    major = architecture // 10
    if major == 8:
        target = 16 * remaining - 16
        word0 = (target << 32) | OPCODE_BRA
    else:
        target = 4 * remaining - 4
        word0 = ((target & 0xFF) << 16) | ((target & ~0xFF) << 26) | OPCODE_BRA
    return word0, _alu_control(0, REST_BRA)


class Assembler:
    def __init__(
        self,
        input_registers: Sequence[int],
        output_registers: Sequence[int],
        available_registers: Sequence[int],
        result_register: int,
        incoming_wait_mask: int,
        first_new_register: int,
    ):
        self.instructions: list[tuple[int, int]] = []
        self.stack: list[Operand] = []
        reserved = set(input_registers) | set(output_registers) | {REGISTER_RZ}
        self.free_registers = [reg for reg in available_registers if reg not in reserved]
        self.next_register = first_new_register
        self.high_water_register = first_new_register
        self.result_register = result_register
        self.pending_wait_mask = incoming_wait_mask
        self.active_barriers: dict[int, int] = {}
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
            self.high_water_register = max(
                self.high_water_register, register + 1
            )
            return Operand(register=register, owned=True)
        if self.next_register >= REGISTER_RZ:
            raise SassAssemblyError("specialized expression exhausted the register file")
        operand = Operand(register=self.next_register, owned=True)
        self.next_register += 1
        self.high_water_register = max(self.high_water_register, self.next_register)
        return operand

    def release(self, operand: Operand, keep: Operand | None = None) -> None:
        if not operand.owned or operand.immediate:
            return
        if keep is not None and operand.register == keep.register:
            return
        self.pending_wait_mask |= self.wait_mask_for((operand,))
        if operand.register not in self.free_registers:
            self.free_registers.append(operand.register)

    def ensure_register(self, operand: Operand) -> Operand:
        if not operand.immediate:
            return operand
        target = self.allocate()
        self.emit(_fadd_imm(target.register, REGISTER_RZ, operand.immediate_bits, 0))
        return target

    def choose_output(self, *operands: Operand) -> Operand:
        for operand in operands:
            if operand.owned and not operand.immediate:
                return Operand(
                    register=operand.register,
                    owned=True,
                    barrier_slot=operand.barrier_slot,
                    barrier_token=operand.barrier_token,
                )
        return self.allocate()

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
            if slot not in self.active_barriers:
                token = self.next_barrier_token
                self.next_barrier_token += 1
                self.active_barriers[slot] = token
                return slot, token
        mask = sum(1 << slot for slot in self.active_barriers)
        self.emit(_nop(mask))
        self.active_barriers.clear()
        token = self.next_barrier_token
        self.next_barrier_token += 1
        self.active_barriers[0] = token
        return 0, token

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
        self.emit(instruction)
        self.release(lhs, target)
        self.release(rhs, target)
        target.barrier_slot = None
        self.stack.append(target)

    def subtract(self) -> None:
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        if rhs.immediate:
            rhs.immediate_bits ^= 0x80000000
            self.stack.extend((lhs, rhs))
            self.add_or_multiply(False)
            return
        negated = rhs if rhs.owned else self.allocate()
        wait = self.wait_mask_for((rhs,))
        self.emit(_fmul_imm(negated.register, rhs.register, NEG_ONE_BITS, wait))
        negated.barrier_slot = None
        lhs = self.ensure_register(lhs)
        target = self.choose_output(lhs, negated)
        wait = self.wait_mask_for((lhs, negated))
        self.emit(_fadd_reg(target.register, lhs.register, negated.register, wait))
        self.release(rhs, target if rhs.register == negated.register else None)
        self.release(lhs, target)
        self.release(negated, target)
        target.barrier_slot = None
        self.stack.append(target)

    def divide(self) -> None:
        rhs = self.ensure_register(self.stack.pop())
        lhs = self.ensure_register(self.stack.pop())
        reciprocal = rhs if rhs.owned else self.allocate()
        wait = self.wait_mask_for((rhs,))
        slot, token = self.allocate_barrier()
        self.emit(_mufu(reciprocal.register, rhs.register, wait, slot, REST_MUFU_RCP))
        reciprocal.barrier_slot = slot
        reciprocal.barrier_token = token
        target = lhs if lhs.owned else reciprocal
        wait = self.wait_mask_for((lhs, reciprocal))
        self.emit(_fmul_reg(target.register, lhs.register, reciprocal.register, wait))
        self.release(lhs, target)
        if rhs.register != reciprocal.register:
            self.release(rhs, target)
        self.release(reciprocal, target)
        target.barrier_slot = None
        self.stack.append(target)

    def negate(self) -> None:
        value = self.stack.pop()
        if value.immediate:
            value.immediate_bits ^= 0x80000000
            self.stack.append(value)
            return
        target = value if value.owned else self.allocate()
        wait = self.wait_mask_for((value,))
        self.emit(_fmul_imm(target.register, value.register, NEG_ONE_BITS, wait))
        self.release(value, target)
        target.barrier_slot = None
        self.stack.append(target)

    def unary_alu(self, operation: str) -> None:
        value = self.ensure_register(self.stack.pop())
        target = value if value.owned else self.allocate()
        wait = self.wait_mask_for((value,))
        if operation == "abs":
            self.emit(_abs(target.register, value.register, wait))
            target.barrier_slot = None
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
            }[operation]
            if operation in {"sin", "cos"}:
                self.emit(_fmul_imm(target.register, value.register, SIN_COS_SCALE_BITS, wait, REST_FMUL_RZ))
                wait = 0
            slot, token = self.allocate_barrier()
            self.emit(_mufu(target.register, target.register if operation in {"sin", "cos"} else value.register, wait, slot, rest))
            target.barrier_slot = slot
            target.barrier_token = token
        self.release(value, target)
        self.stack.append(target)

    def minmax(self, maximum: bool) -> None:
        rhs = self.ensure_register(self.stack.pop())
        lhs = self.ensure_register(self.stack.pop())
        target = self.choose_output(lhs, rhs)
        wait = self.wait_mask_for((lhs, rhs))
        self.emit(_fmnmx(target.register, lhs.register, rhs.register, REST_FMNMX_MAX if maximum else REST_FMNMX_MIN, wait))
        self.release(lhs, target)
        self.release(rhs, target)
        target.barrier_slot = None
        self.stack.append(target)

    def fused_multiply_add(self) -> None:
        addend = self.ensure_register(self.stack.pop())
        rhs = self.ensure_register(self.stack.pop())
        lhs = self.ensure_register(self.stack.pop())
        target = self.choose_output(lhs, rhs, addend)
        wait = self.wait_mask_for((lhs, rhs, addend))
        self.emit(_ffma(target.register, lhs.register, rhs.register, addend.register, wait))
        self.release(lhs, target)
        self.release(rhs, target)
        self.release(addend, target)
        target.barrier_slot = None
        self.stack.append(target)

    def compile(self, program: Program, input_registers: Sequence[int]) -> None:
        program.validate(len(input_registers))
        returned = False
        for instruction in program.instructions():
            kind = instruction.kind
            if kind == InstructionType.STATIC_COLUMN_INPUT_F32:
                self.stack.append(Operand(register=input_registers[instruction.operand]))
            elif kind == InstructionType.CONSTANT_BITS_F32:
                self.stack.append(Operand(immediate_bits=instruction.operand))
            elif kind == InstructionType.ADD_F32:
                self.add_or_multiply(False)
            elif kind == InstructionType.MUL_F32:
                self.add_or_multiply(True)
            elif kind == InstructionType.SUB_F32:
                self.subtract()
            elif kind == InstructionType.DIV_F32:
                self.divide()
            elif kind == InstructionType.NEG_F32:
                self.negate()
            elif kind == InstructionType.ABS_F32:
                self.unary_alu("abs")
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
                self.unary_alu(names[kind])
            elif kind == InstructionType.EXP_F32:
                self.stack.append(Operand(immediate_bits=LOG2_E_BITS))
                self.add_or_multiply(True)
                self.unary_alu("exp2")
            elif kind == InstructionType.LOG_F32:
                self.unary_alu("log2")
                self.stack.append(Operand(immediate_bits=LN_TWO_BITS))
                self.add_or_multiply(True)
            elif kind == InstructionType.RETURN_F32:
                returned = True
                break
            else:
                raise SassAssemblyError(f"Python SASS writer does not support {kind.name}")
        if not returned or len(self.stack) != 1:
            raise SassAssemblyError("post-order program did not produce one result")
        root = self.stack.pop()
        if root.immediate:
            self.emit(_fadd_imm(self.result_register, REGISTER_RZ, root.immediate_bits, 0))
        else:
            wait = self.wait_mask_for((root,))
            self.emit(_mov(self.result_register, root.register, wait))
        if self.active_barriers:
            self.emit(_nop(sum(1 << slot for slot in self.active_barriers)))
            self.active_barriers.clear()


def specialize_cubin(cubin: bytes, plan: CubinPlan, programs: Sequence[Program]) -> SpecializationResult:
    if len(cubin) != plan.cubin_size:
        raise SassAssemblyError("CUBIN size changed after inspection")
    if len(programs) != len(plan.site.output_registers):
        raise SassAssemblyError("one post-order AST is required for each missing output")

    generated: list[tuple[int, int]] = []
    instruction_counts: list[int] = []
    expanded_register_count = plan.register_count
    for index, program in enumerate(programs):
        assembler = Assembler(
            plan.site.input_registers,
            plan.site.output_registers,
            plan.site.available_registers,
            plan.site.output_registers[index],
            plan.site.incoming_wait_mask if index == 0 else 0,
            plan.register_count,
        )
        assembler.compile(program, plan.site.input_registers)
        generated.extend(assembler.instructions)
        instruction_counts.append(len(assembler.instructions))
        expanded_register_count = max(expanded_register_count, assembler.high_water_register)

    if len(generated) > plan.site.instruction_count:
        raise SassAssemblyError(
            f"specialization needs {len(generated)} SASS instructions but the site has {plan.site.instruction_count}"
        )
    remaining = plan.site.instruction_count - len(generated)
    if remaining:
        generated.append(_nop() if remaining == 1 else _branch(remaining, plan.architecture))

    output = bytearray(cubin)
    for index, (word0, word1) in enumerate(generated):
        struct.pack_into("<QQ", output, plan.site.start_offset + index * INSTRUCTION_BYTES, word0, word1)
    struct.pack_into("<QQ", output, plan.site.load_fence_offset, *_nop())

    final_register_count = plan.register_count
    if expanded_register_count > plan.register_count:
        if expanded_register_count > REGISTER_RZ - REGISTER_PAD:
            raise SassAssemblyError("specialized expression leaves no register padding")
        final_register_count = expanded_register_count + REGISTER_PAD
        for offset in plan.register_count_offsets:
            struct.pack_into("<I", output, offset, final_register_count)
        for offset in plan.register_count_header_offsets:
            struct.pack_into("<B", output, offset, final_register_count)
    return SpecializationResult(bytes(output), tuple(instruction_counts), final_register_count)
