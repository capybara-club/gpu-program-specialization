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
# SPDX-FileCopyrightText: 2026 Charles Durham
#
# SPDX-License-Identifier: MIT

from ._impl import (
    StackPtxResult,
    StackPtxInstruction,
    StackPtxInstructionType,
    StackPtxPayload,
    StackPtxPTXArgs,
    StackPtxMetaInstruction,
    stack_ptx_compile,
    stack_ptx_compile_workspace_size,
    stack_ptx_result_to_string,
)
from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional

class _AutoEnum(IntEnum):
    """Base to start auto() at 0,1,2,... consistently."""
    def _generate_next_value_(name, start, count, last_values):
        return count

class Instr:
    def __init__(self, instruction_type, stack_idx=0, ret_idx=0, idx=0, name="Instr", payload_params=None):
        self.instruction_type = instruction_type
        self.stack_idx = stack_idx
        self.ret_idx = ret_idx
        self.idx = idx
        self.name = name
        self.payload_params = payload_params or {}

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def realize(self):
        payload_type = self.payload_params.get('type')
        if payload_type == 'u':
            payload = StackPtxPayload.from_u(self.payload_params['v'])
        elif payload_type == 'f':
            payload = StackPtxPayload.from_f(self.payload_params['v'])
        elif payload_type == 's':
            payload = StackPtxPayload.from_s(self.payload_params['v'])
        elif payload_type == 'meta_constant':
            payload = StackPtxPayload.from_meta_constant(self.payload_params['v'])
        else:
            payload = StackPtxPayload()
        return StackPtxInstruction(
            self.instruction_type,
            self.stack_idx,
            self.ret_idx,
            self.idx,
            payload
        )

_META_NAMES = {
    StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_DUP: "dup",
    StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_YANK_DUP: "yank_dup",
    StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_SWAP: "swap",
    StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_SWAP_WITH: "swap_with",
    StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_REPLACE: "replace",
    StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_DROP: "drop",
    StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_ROTATE: "rotate",
    StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_REVERSE: "reverse",
}

class StackTypeEnum(_AutoEnum):
    def __new__(
        cls,
        value: int,
        literal_prefix: str,
    ):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.literal_prefix = literal_prefix
        return obj

    def constant(self, v):
        name = f"{self.literal_prefix}_constant({v})"
        payload_type = self.literal_prefix
        if payload_type not in ["u32", "f32", "s32"]:
            raise ValueError("Can only create a constant from f32, s32, or u32 literal types")
        if payload_type == "u32":
            ptype = 'u'
        elif payload_type == "f32":
            ptype = 'f'
        else:
            ptype = 's'
        return Instr(
            instruction_type=StackPtxInstructionType.STACK_PTX_INSTRUCTION_TYPE_CONSTANT,
            stack_idx=self.value,
            name=name,
            payload_params={'type': ptype, 'v': v}
        )

    @classmethod
    def meta_constant(cls, v):
        return Instr(
            instruction_type=StackPtxInstructionType.STACK_PTX_INSTRUCTION_TYPE_META,
            idx=StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_CONSTANT,
            name=f"meta_constant({v})",
            payload_params={'type': 'meta_constant', 'v': v}
        )

    def _encode_meta(self, meta_instruction):
        meta_name = _META_NAMES.get(meta_instruction, "unknown_meta")
        name = f"{self.literal_prefix}_{meta_name}"
        return Instr(
            instruction_type=StackPtxInstructionType.STACK_PTX_INSTRUCTION_TYPE_META,
            stack_idx=self.value,
            idx=meta_instruction,
            name=name
        )

    @property
    def dup(self):
        return self._encode_meta(StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_DUP)

    @property
    def yank_dup(self):
        return self._encode_meta(StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_YANK_DUP)

    @property
    def swap(self):
        return self._encode_meta(StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_SWAP)

    @property
    def swap_with(self):
        return self._encode_meta(StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_SWAP_WITH)

    @property
    def replace(self):
        return self._encode_meta(StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_REPLACE)

    @property
    def drop(self):
        return self._encode_meta(StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_DROP)

    @property
    def rotate(self):
        return self._encode_meta(StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_ROTATE)

    @property
    def reverse(self):
        return self._encode_meta(StackPtxMetaInstruction.STACK_PTX_META_INSTRUCTION_REVERSE)

    def store(self, idx):
        return Instr(
            instruction_type=StackPtxInstructionType.STACK_PTX_INSTRUCTION_TYPE_STORE,
            stack_idx=self.value,
            idx=idx,
            name=f"store({idx})"
        )

    @classmethod
    def load(cls, idx):
        return Instr(
            instruction_type=StackPtxInstructionType.STACK_PTX_INSTRUCTION_TYPE_LOAD,
            idx=idx,
            name=f"load({idx})"
        )

class ArgTypeEnum(_AutoEnum):
    def __new__(
        cls,
        value: int,
        stack_type: int,
        stack_num_vec_elems: int = 0
    ):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.stack_type = stack_type
        obj.stack_num_vec_elems = stack_num_vec_elems
        return obj

def create_instruction_enum(arg_type_enum):
    class PtxInstructionEnum(_AutoEnum):
        def __new__(
            cls,
            value,
            ptx_str,
            args,
            rets,
            aligned = False
        ):
            obj = int.__new__(cls, value)
            obj._value_ = value
            obj.ptx_str = ptx_str
            if not isinstance(args, list) or len(args) > 4 or not all(isinstance(a, arg_type_enum) for a in args):
                raise ValueError("Invalid args: must be list of at most 4 ArgType instances")
            if not isinstance(rets, list) or len(rets) > 2 or not all(isinstance(r, arg_type_enum) for r in rets):
                raise ValueError("Invalid rets: must be list of at most 2 ArgType instances")
            obj.args = args
            obj.rets = rets
            obj.aligned = aligned
            return obj

        def __str__(self):
            return self.ptx_str

        def __repr__(self):
            return self.ptx_str

        def realize(self):
            sentinel = len(arg_type_enum)
            arg_values = [a.value for a in self.args] + [sentinel] * (4 - len(self.args))
            ret_values = [r.value for r in self.rets] + [sentinel] * (2 - len(self.rets))
            payload = StackPtxPayload.from_ptx_args(
                StackPtxPTXArgs(
                    arg_0=arg_values[0],
                    arg_1=arg_values[1],
                    arg_2=arg_values[2],
                    arg_3=arg_values[3],
                    ret_0=ret_values[0],
                    ret_1=ret_values[1],
                    flag_is_aligned=self.aligned
                )
            )
            return StackPtxInstruction(
                StackPtxInstructionType.STACK_PTX_INSTRUCTION_TYPE_PTX,
                0,
                0,
                self.value,
                payload
            )

    return PtxInstructionEnum

def create_special_register_enum(arg_type_enum):
    class SpecialRegisterEnum(_AutoEnum):
        def __new__(
            cls,
            value,
            ptx_str,
            arg,
        ):
            obj = int.__new__(cls, value)
            obj._value_ = value
            obj.ptx_str = ptx_str
            if not isinstance(arg, arg_type_enum):
                raise ValueError("Invalid args: must be a ArgType instances")
            obj.arg = arg
            return obj

        def __str__(self):
            return self.ptx_str

        def __repr__(self):
            return self.ptx_str

        def realize(self):
            payload = StackPtxPayload.from_special_arg(self.arg)
            return StackPtxInstruction(
                StackPtxInstructionType.STACK_PTX_INSTRUCTION_TYPE_SPECIAL,
                0,
                0,
                self.value,
                payload
            )

    return SpecialRegisterEnum

def _encode_input_instruction(
    value: int
):
    return \
        StackPtxInstruction(
            instruction_type=StackPtxInstructionType.STACK_PTX_INSTRUCTION_TYPE_INPUT,
            stack_idx=0,
            ret_idx=0,
            idx=value,
            payload=StackPtxPayload()
        )

class RoutineEnum(_AutoEnum):
    def __new__(
        cls,
        value: int,
        instructions
    ):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.instructions = instructions
        return obj

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def realize(self):
        return StackPtxInstruction(
            StackPtxInstructionType.STACK_PTX_INSTRUCTION_TYPE_ROUTINE,
            0,
            0,
            self.value,
            StackPtxPayload()
        )
    
class RegisterRegistry:
    @dataclass(frozen=True)
    class RegisterRecord:
        idx: int
        register_name: str  # e.g., "in_0"
        stack_type: int     # e.g., Stack.u32
        name: str           # Optional display name, defaults to register_name

    class _RegSymbol:
        """
        Proxy for attribute access (e.g., Register.in_0).
        """
        __slots__ = ("_reg", "_rec")

        def __init__(
            self,
            registry: "RegisterRegistry",
            record: "RegisterRegistry.RegisterRecord",
        ):
            self._reg = registry
            self._rec = record

        def req(self):
            return self._rec.idx

        def realize(self):
            return _encode_input_instruction(self._rec.idx)

        @property
        def meta(self) -> "RegisterRegistry.RegisterRecord":
            return self._rec

        def __repr__(self) -> str:
            return (
                f"<Register {self._rec.register_name} "
                f"idx={self._rec.idx} stack={self._rec.stack_type}>"
            )

        def __str__(self) -> str:
            return self._rec.register_name

    def __init__(self):
        self._items: List["RegisterRegistry.RegisterRecord"] = []
        self._by_name: Dict[str, "RegisterRegistry.RegisterRecord"] = {}
        self._by_idx: Dict[int, "RegisterRegistry.RegisterRecord"] = {}
        self._next_idx = 0
        self._frozen = False

    def add(
        self,
        register_name: str,
        stack_type: int,
        name: Optional[str] = None,
    ) -> "RegisterRegistry.RegisterRecord":
        """Add a new register with a register_name, stack type, and optional display name."""
        if self._frozen:
            raise RuntimeError("RegisterRegistry is frozen; cannot add more registers.")
        name = register_name if name is None else name
        idx = self._next_idx
        self._next_idx += 1
        rec = self.RegisterRecord(idx, register_name, int(stack_type), name)
        self._items.append(rec)
        self._by_name[name] = rec
        self._by_idx[idx] = rec
        return rec

    def freeze(self) -> None:
        """Prevent further modifications to the registry."""
        self._frozen = True

    def get(self, name: str) -> "RegisterRegistry.RegisterRecord":
        """Retrieve a register by name."""
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"Name '{name}' not found")

    def by_idx(self, idx: int) -> "RegisterRegistry.RegisterRecord":
        """Retrieve a register by index."""
        try:
            return self._by_idx[idx]
        except KeyError:
            raise KeyError(f"Register with index {idx} not found")

    def get_register_names(self) -> List[str]:
        """Return a list of all register names."""
        return [rec.register_name for rec in self._items]

    def get_stack_types(self) -> List[int]:
        """Return a list of all stack types."""
        return [int(rec.stack_type) for rec in self._items]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterable["RegisterRegistry.RegisterRecord"]:
        return iter(self._items)

    def __getattr__(self, name: str) -> "_RegSymbol":
        """Support dynamic attribute access like register.in_0."""
        try:
            rec = self.get(name)
        except KeyError as e:
            raise AttributeError(f"Register '{name}' not found") from None
        return self._RegSymbol(self, rec)


class StackPtxError(RuntimeError):
    """Raised when a Stack PTX C API call fails."""
    @staticmethod
    def _check_result(ret: StackPtxResult) -> None:
        if ret != StackPtxResult.STACK_PTX_SUCCESS:
            raise StackPtxError(stack_ptx_result_to_string(ret))

class StackPtx:
    def __init__(
        self,
        stack_enum,
        arg_enum,
        ptx_instruction_enum,
        special_register_enum,
    ):
        self.stack_enum = stack_enum
        self.arg_enum = arg_enum
        self.ptx_instruction_strings = [instr.ptx_str for instr in ptx_instruction_enum]
        self.special_register_strings = [instr.ptx_str for instr in special_register_enum]
        self.stack_literal_prefixes = [stack.literal_prefix for stack in stack_enum]
        self.arg_stack_indices = [arg.stack_type for arg in arg_enum]
        self.arg_num_stack_elems = [arg.stack_num_vec_elems for arg in arg_enum]

    def compile(
        self,
        registry,
        instructions,
        requests,
        execution_limit,
        routine_enum = None,
        max_ast_size = 100,
        max_ast_to_visit_stack_depth = 10,
        stack_size = 128,
        max_frame_depth = 4,
        store_size = 16,
    ):
        def realize(instr):
            if isinstance(instr, StackPtxInstruction):
                return instr
            elif hasattr(instr, 'realize'):
                return instr.realize()
            else:
                raise ValueError(f"Invalid instruction: {instr}")

        def request_to_idx(request):
            if hasattr(request, "_rec"):
                idx = int(request._rec.idx)
            elif hasattr(request, "idx"):
                idx = int(request.idx)
            else:
                try:
                    idx = int(request)
                except (TypeError, ValueError):
                    raise TypeError(
                        "requests must be RegisterRegistry symbols, RegisterRecords, or integer indices"
                    ) from None
            if idx < 0:
                raise ValueError("requests indices must be >= 0")
            return idx

        instruction_ret = StackPtxInstruction(
            instruction_type=StackPtxInstructionType.STACK_PTX_INSTRUCTION_TYPE_RETURN,
            stack_idx=0,
            ret_idx=0,
            idx=0,
            payload=StackPtxPayload()
        )
        # Realize the instructions
        realized_instructions = [realize(instr) for instr in instructions] + [instruction_ret]
        register_names = registry.get_register_names()
        register_stack_types = registry.get_stack_types()
        requests = [request_to_idx(request) for request in requests]
        routines = None
        if routine_enum is not None:
            routines = [[realize(instr) for instr in routine.instructions] + [instruction_ret] for routine in routine_enum]
        ret, workspace_size = stack_ptx_compile_workspace_size(
            max_ast_size,
            max_ast_to_visit_stack_depth,
            stack_size,
            max_frame_depth,
            store_size,
            len(self.stack_enum),
            len(self.arg_enum)
        )
        StackPtxError._check_result(ret)
        workspace = bytearray(workspace_size)
        ret, buffer_size = stack_ptx_compile(
            max_ast_size,
            max_ast_to_visit_stack_depth,
            stack_size,
            max_frame_depth,
            store_size,
            self.ptx_instruction_strings,
            self.special_register_strings,
            self.stack_literal_prefixes,
            self.arg_stack_indices,
            self.arg_num_stack_elems,
            realized_instructions,
            register_names,
            register_stack_types,
            routines,
            requests,
            execution_limit,
            workspace,
            None
        )
        StackPtxError._check_result(ret)
        buffer = bytearray(buffer_size+1)
        ret, buffer_size = stack_ptx_compile(
            max_ast_size,
            max_ast_to_visit_stack_depth,
            stack_size,
            max_frame_depth,
            store_size,
            self.ptx_instruction_strings,
            self.special_register_strings,
            self.stack_literal_prefixes,
            self.arg_stack_indices,
            self.arg_num_stack_elems,
            realized_instructions,
            register_names,
            register_stack_types,
            routines,
            requests,
            execution_limit,
            workspace,
            buffer
        )
        StackPtxError._check_result(ret)
        return buffer[:buffer_size].decode('utf-8')
