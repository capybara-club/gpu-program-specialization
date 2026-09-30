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

from enum import IntEnum

from .ast import (
    ABS,
    ADD,
    DIV,
    EX2,
    LG2,
    MUL,
    RSQRT,
    SQRT,
    RETURN,
    encode_constant_bits,
    encode_routine,
    encode_routine_arg,
    pack_programs,
)


class Routine(IntEnum):
    EXP = 0
    SAFE_LOG = 1
    SAFE_LOG10 = 2
    SAFE_POW = 3
    SAFE_SQRT = 4
    SAFE_RSQRT = 5
    SAFE_DIV = 6
    SAFE_RCP = 7
    LOG = 8
    LOG10 = 9
    POW = 10


EPSILON_BITS = 0x322BCC77
LOG2E_BITS = 0x3FB8AA3B
LN2_BITS = 0x3F317218
LOG10_2_BITS = 0x3E9A209B

EXP = encode_routine(Routine.EXP, 1)
SAFE_LOG = encode_routine(Routine.SAFE_LOG, 1)
SAFE_LOG10 = encode_routine(Routine.SAFE_LOG10, 1)
SAFE_POW = encode_routine(Routine.SAFE_POW, 2)
SAFE_SQRT = encode_routine(Routine.SAFE_SQRT, 1)
SAFE_RSQRT = encode_routine(Routine.SAFE_RSQRT, 1)
SAFE_DIV = encode_routine(Routine.SAFE_DIV, 2)
SAFE_RCP = encode_routine(Routine.SAFE_RCP, 1)
LOG = encode_routine(Routine.LOG, 1)
LOG10 = encode_routine(Routine.LOG10, 1)
POW = encode_routine(Routine.POW, 2)


DEFAULT_ROUTINES = pack_programs(
    (
        (encode_routine_arg(0), encode_constant_bits(LOG2E_BITS), MUL, EX2, RETURN),
        (
            encode_routine_arg(0), ABS, encode_constant_bits(EPSILON_BITS), ADD, LG2,
            encode_constant_bits(LN2_BITS), MUL, RETURN,
        ),
        (
            encode_routine_arg(0), ABS, encode_constant_bits(EPSILON_BITS), ADD, LG2,
            encode_constant_bits(LOG10_2_BITS), MUL, RETURN,
        ),
        (
            encode_routine_arg(0), ABS, encode_constant_bits(EPSILON_BITS), ADD, LG2,
            encode_routine_arg(1), MUL, EX2, RETURN,
        ),
        (encode_routine_arg(0), ABS, encode_constant_bits(EPSILON_BITS), ADD, SQRT, RETURN),
        (encode_routine_arg(0), ABS, encode_constant_bits(EPSILON_BITS), ADD, RSQRT, RETURN),
        (
            encode_routine_arg(0), encode_routine_arg(1), MUL,
            encode_routine_arg(1), encode_routine_arg(1), MUL,
            encode_constant_bits(EPSILON_BITS), ADD, DIV, RETURN,
        ),
        (
            encode_routine_arg(0), encode_routine_arg(0), encode_routine_arg(0), MUL,
            encode_constant_bits(EPSILON_BITS), ADD, DIV, RETURN,
        ),
        (encode_routine_arg(0), LG2, encode_constant_bits(LN2_BITS), MUL, RETURN),
        (encode_routine_arg(0), LG2, encode_constant_bits(LOG10_2_BITS), MUL, RETURN),
        (encode_routine_arg(0), LG2, encode_routine_arg(1), MUL, EX2, RETURN),
    )
)
DEFAULT_ROUTINES.flags.writeable = False
