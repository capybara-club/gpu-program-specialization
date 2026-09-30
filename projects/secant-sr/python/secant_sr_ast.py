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
"""Secant 0.3 expressions, including register-bank leaves and local toggles."""
from __future__ import annotations
from dataclasses import dataclass
import math
import struct

OPCODES = {
    'constant':0x81,'return':0x83,'add':0x85,'sub':0x86,'mul':0x87,'div':0x88,
    'neg':0x89,'sqrt':0x8a,'rcp':0x8b,'abs':0x8c,'min':0x8d,'max':0x8e,
    'sin':0x90,'cos':0x91,'exp2':0x92,'log2':0x93,'rsqrt':0x94,'tanh':0x95,
    'input':0xb6,'exp':0xba,'log':0xbb,'bank':0xbc,'toggle2':0xbd,'toggle4':0xbe,'affine_bank':0xbf,
}
BINARY={'add','sub','mul','div','min','max'}
UNARY={'neg','sqrt','rcp','abs','sin','cos','exp2','log2','rsqrt','tanh','exp','log'}
LEAVES={'input','bank','constant','affine_bank'}

def _index(value, limit, label):
    if not isinstance(value,int) or isinstance(value,bool) or not 0 <= value < limit:
        raise ValueError(f'{label} must be an integer in [0, {limit})')
    return value

@dataclass(frozen=True)
class Expression:
    op: str
    args: tuple['Expression', ...] = ()
    value: float | int | tuple[int,int] | tuple[int,float,float] | None = None

    def __post_init__(self):
        object.__setattr__(self, 'args', tuple(self.args))
        if any(not isinstance(a,Expression) for a in self.args):
            raise ValueError('children must be Expressions')
        if self.op in LEAVES:
            if self.args: raise ValueError('a leaf cannot have children')
            if self.op=='affine_bank':
                if not isinstance(self.value,tuple) or len(self.value)!=3: raise ValueError('affine bank requires slot, scale, offset')
                slot,scale,offset=self.value; _index(slot,128,'bank slot')
                scale=Expression.constant(scale).value; offset=Expression.constant(offset).value
                object.__setattr__(self,'value',(slot,scale,offset))
            elif self.op in {'input','bank'}: _index(self.value,128,self.op)
            else:
                try: v=struct.unpack('<f',struct.pack('<f',float(self.value)))[0]
                except (ValueError,TypeError,OverflowError,struct.error) as error:
                    raise ValueError('literal must be a finite float32') from error
                if not math.isfinite(v): raise ValueError('literal must be a finite float32')
                object.__setattr__(self, 'value', v)
        elif self.op in {'toggle2','toggle4'}:
            n=2 if self.op=='toggle2' else 4
            if len(self.args)!=n or any(x.op not in LEAVES for x in self.args):
                raise ValueError('toggles select only direct input, bank or literal leaves')
            if n==2: _index(self.value,32,'toggle bit')
            else:
                if not isinstance(self.value,tuple) or len(self.value)!=2: raise ValueError('two bit indices required')
                for b in self.value: _index(b,32,'toggle bit')
                if self.value[0]==self.value[1]: raise ValueError('four-way bit indices must differ')
        elif self.op in BINARY|UNARY:
            if len(self.args)!=(2 if self.op in BINARY else 1): raise ValueError('wrong operation arity')
            if self.value is not None: raise ValueError('arithmetic nodes have no leaf value')
        else: raise ValueError(f'unsupported operation {self.op!r}')

    @staticmethod
    def input(index): return Expression('input',value=index)
    @staticmethod
    def bank(slot): return Expression('bank',value=slot)
    @staticmethod
    def constant(value): return Expression('constant',value=float(value))
    @staticmethod
    def affine_bank(slot,scale,offset): return Expression('affine_bank',value=(slot,scale,offset))
    @staticmethod
    def toggle2(a,b,bit): return Expression('toggle2',(a,b),bit)
    @staticmethod
    def toggle4(a,b,c,d,low_bit,high_bit): return Expression('toggle4',(a,b,c,d),(low_bit,high_bit))

    def encode(self):
        output=bytearray()
        def visit(n):
            for child in n.args: visit(child)
            output.append(OPCODES[n.op])
            if n.op=='affine_bank': output.extend(struct.pack('<Bff',*n.value))
            elif n.op in {'input','bank','toggle2'}: output.append(n.value)
            elif n.op=='toggle4': output.extend(n.value)
            elif n.op=='constant': output.extend(struct.pack('<f',float(n.value)))
        visit(self);output.append(OPCODES['return']);return bytes(output)

    @staticmethod
    def decode(data):
        data=bytes(data);at=0;stack=[];names={v:k for k,v in OPCODES.items()}
        while at<len(data):
            op=names.get(data[at]);at+=1
            if op=='return':
                if at!=len(data) or len(stack)!=1: raise ValueError('invalid return/stack')
                return stack[0]
            if op is None: raise ValueError('unknown or retired opcode')
            width=9 if op=='affine_bank' else 4 if op=='constant' else 2 if op=='toggle4' else 1 if op in {'input','bank','toggle2'} else 0
            if at+width>len(data): raise ValueError('truncated instruction')
            value=struct.unpack_from('<Bff',data,at) if op=='affine_bank' else struct.unpack_from('<f',data,at)[0] if op=='constant' else tuple(data[at:at+2]) if op=='toggle4' else data[at] if width else None
            at+=width
            arity=4 if op=='toggle4' else 2 if op=='toggle2' or op in BINARY else 1 if op in UNARY else 0
            if len(stack)<arity: raise ValueError('stack underflow')
            args=tuple(stack[-arity:]) if arity else ()
            if arity: del stack[-arity:]
            stack.append(Expression(op,args,value))
        raise ValueError('missing return')

    def resolve(self, constants, permutation):
        if self.op=='affine_bank':
            slot,scale,offset=self.value
            product=struct.unpack('<f',struct.pack('<f',scale*constants[slot]))[0]
            return Expression.constant(product+offset)
        if self.op=='bank': return Expression.constant(constants[self.value])
        if self.op in {'toggle2','toggle4'}:
            low=self.value if self.op=='toggle2' else self.value[0]
            k=(permutation>>low)&1
            if self.op=='toggle4': k|=((permutation>>self.value[1])&1)<<1
            return self.args[k].resolve(constants,permutation)
        return Expression(self.op,tuple(a.resolve(constants,permutation) for a in self.args),self.value)

    def evaluate(self, inputs, constants=(), permutation=0):
        if self.op=='affine_bank': return self.resolve(constants,permutation).evaluate(inputs)
        if self.op=='input': return float(inputs[self.value])
        if self.op=='bank': return float(constants[self.value])
        if self.op=='constant': return float(self.value)
        if self.op in {'toggle2','toggle4'}: return self.resolve(constants,permutation).evaluate(inputs)
        values=tuple(a.evaluate(inputs,constants,permutation) for a in self.args)
        functions={'add':lambda a,b:a+b,'sub':lambda a,b:a-b,'mul':lambda a,b:a*b,'div':lambda a,b:a/b,
                   'neg':lambda a:-a,'sqrt':math.sqrt,'rcp':lambda a:1/a,'abs':abs,'min':min,'max':max,
                   'sin':math.sin,'cos':math.cos,'exp2':lambda a:2**a,'log2':math.log2,
                   'rsqrt':lambda a:1/math.sqrt(a),'tanh':math.tanh,'exp':math.exp,'log':math.log}
        return float(functions[self.op](*values))

def nguyen1():
    x=Expression.input(0);x2=Expression('mul',(x,x));x3=Expression('mul',(x2,x))
    return Expression('add',(Expression('add',(x3,x2)),x))
