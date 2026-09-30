<!--
SPDX-FileCopyrightText: 2026 Charles Durham
SPDX-License-Identifier: MIT

MIT License

Copyright (c) 2026 Charles Durham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# AST PTX

## Overview
AST PTX is a small C99 header API for compiling a postorder AST instruction array into a single f32 PTX expression. It is meant for cases where the upstream system already has a sane AST and only needs a direct PTX stub, without Stack PTX's mutation-friendly stack-program layer.

The core compiler lives in `ast_ptx.h`. Concrete f32 instruction encodes and the CPU interpreter live in:

- `tools/ast_ptx_instructions.h`
- `tools/ast_ptx_interpreter.h`

Define `AST_PTX_IMPLEMENTATION` in exactly one translation unit:

```c
#define AST_PTX_IMPLEMENTATION
#include <ast_ptx_interpreter.h>
```

## Shape
AST programs are arrays of `AstPtxInstruction` terminated by `ast_ptx_encode_return`. Leaves push inputs, constants, or routine arguments. PTX instruction encodes consume the previous values and produce one value. A top-level return moves that final value into the requested output register.

```c
static const AstPtxInstruction program[] = {
    ast_ptx_encode_input(0u),
    ast_ptx_encode_input(1u),
    ast_ptx_encode_ptx_instruction_mul_ftz_f32,
    ast_ptx_encode_constant(1.0f),
    ast_ptx_encode_ptx_instruction_add_ftz_f32,
    ast_ptx_encode_return
};
```

Use the usual measure-and-allocate flow:

```c
const char* const inputs[] = { "x", "y" }; /* Bare names; AST PTX emits `%`. */

size_t required = 0;
ast_ptx_compile("z", inputs, 2,
                ast_ptx_ptx_instruction_names,
                ast_ptx_ptx_instruction_num_args,
                AST_PTX_PTX_INSTRUCTION_NUM_ENUMS,
                routines, num_routines,
                program,
                NULL, 0, &required);

char* stub = malloc(required + 1);
ast_ptx_compile("z", inputs, 2,
                ast_ptx_ptx_instruction_names,
                ast_ptx_ptx_instruction_num_args,
                AST_PTX_PTX_INSTRUCTION_NUM_ENUMS,
                routines, num_routines,
                program,
                stub, required + 1, &required);
```

The generated stub can be passed to PTX Inject. The interpreter is useful for checking that a generated AST produces the same scalar result on the CPU.

## Routines
Routines are also postorder AST instruction arrays. They read their call arguments with `ast_ptx_encode_routine_arg(idx)` and return one value. This is intended for reusable operations such as `pow`, `log10`, or guarded math routines while keeping the top-level AST compiler simple.
