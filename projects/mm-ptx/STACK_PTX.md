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

# Stack PTX

## Overview
Stack PTX is a single-file, header-only C library that compiles a stack-machine instruction list into valid PTX. It is designed for programmatic generation and mutation of PTX rather than hand-written assembly. The instruction language is inspired by Lee Spector's [Push](https://faculty.hampshire.edu/lspector/push.html).

The library has no dependencies beyond the C standard library and is C99 compliant. Similar to stb-style single headers, define `STACK_PTX_IMPLEMENTATION` in exactly one translation unit:
```c
#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>
```
Define `STACK_PTX_DEBUG` to turn errors into asserts at the call site:
```c
#define STACK_PTX_DEBUG
#define STACK_PTX_IMPLEMENTATION
#include <stack_ptx.h>
```

Each function returns an error code that should be checked (just like the CUDA APIs). A `stackPtxCheck` macro is provided in the examples.

## Quickstart
1. Generate or include a description header that defines instruction encodings (see Type descriptions).
2. Build a `StackPtxInstruction` array and terminate it with `stack_ptx_encode_return`.
3. Provide `StackPtxRegister` entries and a list of requests.
4. Measure workspace with `stack_ptx_compile_workspace_size`, allocate it, then call `stack_ptx_compile`.
5. The output buffer contains PTX text.

For a runnable example, start with `examples/stack_ptx/00_simple/main.c`.

## Examples

- Stack PTX only
  - Overview: [examples/stack_ptx/README.md](examples/stack_ptx/README.md)
  - Quickstart (simplest): [examples/stack_ptx/00_simple/main.c](examples/stack_ptx/00_simple/main.c)

- PTX Inject
  - Readme / concepts: [PTX_INJECT.md](PTX_INJECT.md)

- Combined (Stack PTX + PTX Inject)
  - Integrated examples: [examples/stack_ptx_inject/README.md](examples/stack_ptx_inject/README.md)

- Python bindings for Stack PTX + PTX Inject + examples
  - https://github.com/MetaMachines/mm-ptx-py

- PyTorch customizable hyperparameter semirings
  - https://github.com/MetaMachines/mm-kermac-py

## Why Stack PTX
The main goal is to make it easy to generate valid PTX with code that is safe to mutate. For example, in PTX you might write:
```c
mul.ftz.f32 %f0, 1.0, 2.0;
add.ftz.f32 %f1, %f0, 0.2;
```
With Stack PTX you would express this as:
```c
[ stack_ptx_encode_constant_f32(1.0),
  stack_ptx_encode_constant_f32(2.0),
  stack_ptx_encode_ptx_instruction_mul_ftz_f32,
  stack_ptx_encode_constant_f32(0.2),
  stack_ptx_encode_ptx_instruction_add_ftz_f32,
  stack_ptx_encode_return
]
```
In the above example:
- `stack_ptx_encode_constant_f32(1.0)` pushes `1.0` onto the stack.
- `stack_ptx_encode_constant_f32(2.0)` pushes `2.0` onto the stack.
- `stack_ptx_encode_ptx_instruction_mul_ftz_f32` pops two float values, multiplies them, and pushes the result.
- `stack_ptx_encode_constant_f32(0.2)` pushes `0.2` onto the stack.
- `stack_ptx_encode_ptx_instruction_add_ftz_f32` pops two float values and adds them.

Because this is expressed as stack operations, any operation can be deleted, inserted, or swapped at any position and you will still have a valid series of PTX instructions. The compiler performs dead-code elimination, so operations that are not relevant to the requested stack values do not appear in the output.

When the above code runs, the emitted PTX looks like:
```
{
  .reg .f32 %_a<2>;
  mul.ftz.f32 %_a0, 0f40000000, 0f3F800000;
  add.ftz.f32 %_a1, 0f3E4CCCCD, %_a0;
  mov.f32 %output_register, %_a1;
}
```

PTX generation is fast. Given about 100 instructions, Stack PTX can output valid PTX in single-digit microseconds.

Stack PTX code can be written by humans, but it is much better suited to programmatic generation.

## Core concepts

### Stacks
Stack PTX uses stack data structures to store intermediate values while executing instructions and to build its AST before compiling to PTX. Instructions can push, pop, or manipulate stack values.

### The interface
Stack PTX exposes three functions:
- `stack_ptx_result_to_string`: returns the string representation of a `StackPtxResult`.
- `stack_ptx_compile_workspace_size`: returns the amount of memory `stack_ptx_compile` will need to compile a program.
- `stack_ptx_compile`: compiles an instruction list into PTX.

### Instructions
Stack PTX relies on the `StackPtxInstruction` struct to describe what to do. This struct is fixed-size (8 bytes). Most data needed to compile the instruction is stored in those 8 bytes, except for string representations that appear in the output PTX.

Instructions are fixed-size to make it easy to shuffle, insert, or delete elements without breaking the program. You should be able to throw any set of instructions into the array and expect valid PTX assembly as output.

#### Return
Return is the simplest instruction. It is a sentinel value signaling the end of execution for an array of instructions, avoiding the need to pass a length to `stack_ptx_compile`. Encode it with:
```
stack_ptx_encode_return
```
Instruction lists in [mm-ptx-py](https://github.com/MetaMachines/mm-ptx-py) do not need this instruction, since it is appended automatically before compiling.

#### Constants
A constant instruction pushes a constant value onto the relevant stack. The constant is limited to 32 bits because it is stored inside the instruction itself. For example, with the generated encoding macros:
```c
stack_ptx_encode_constant_f32(1.0f)
stack_ptx_encode_constant_u32(10)
```

#### Inputs
Inputs push externally declared PTX registers onto the relevant stack. They require an index to identify them, which is used to access the register name passed to `stack_ptx_compile` via the `StackPtxRegister* registers` array. For example:
```c
stack_ptx_encode_input(0),
stack_ptx_encode_input(2)
```
The instructions above use `registers[0]` and `registers[2]`. It is common to use enums to keep these indices consistent. See `examples/stack_ptx/00_simple/main.c`.

#### PTX instructions
PTX instruction encodings specify the actual PTX ops that will appear in the output from `stack_ptx_compile`. They are still 8 bytes long and contain information about which stacks to pop from and which stacks to push to. Each instruction can consume up to 4 inputs and produce up to 2 outputs, per the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) spec. Encode them like:
```c
stack_ptx_encode_ptx_instruction_add_u32,
stack_ptx_encode_ptx_instruction_mul_ftz_f32
```
From the description tables, `stack_ptx_encode_ptx_instruction_add_u32` reads two values from the `U32` stack and pushes one value back to the `U32` stack. `stack_ptx_encode_ptx_instruction_mul_ftz_f32` reads two values from the `F32` stack and pushes one value back to the `F32` stack.

Instructions can be more complex because argument descriptors are arg types, not just stack types. For example, `V4_F32` means the instruction should read 4 values from the `F32` stack when used as an input, and push 4 values to the `F32` stack when used as a return. These correspond to PTX vector operands like `{%a0, %a1, %a2, %a3}`. See `examples/stack_ptx/06_mma/main.c` and `examples/stack_ptx_inject/03_mma_sync` for tensor core examples.

#### Special registers
Special-register instructions encode PTX special-register strings into PTX. In the description tables, these appear in the `special_registers` section. For example, `tid.x` pushes its value to the `U32` stack. `tid` uses the arg type `V4_U32` and pushes 4 values to the `U32` stack. You will then find `tid.x`, `tid.y`, `tid.z`, and `tid.w` in the output PTX, representing the equivalents of `threadIdx.x`, `threadIdx.y`, and so on. See `examples/stack_ptx/04_special_registers/main.c`.
```c
stack_ptx_encode_special_register_tid_x,
stack_ptx_encode_special_register_tid
```

#### Meta instructions
Meta instructions manipulate stacks during AST construction. If not enough values are present on the relevant stack, the instruction becomes a no-op. If an indexed meta instruction pops a value off the `meta_stack` and it refers to a depth deeper than available, it is also ignored.

A runnable example is at `examples/stack_ptx/03_meta_instructions`.

- Meta `constant`: pushes an integer onto the special `meta_stack` to be used by other meta instructions.
```c
stack_ptx_encode_meta_constant(2)
```
- Meta `dup`: duplicates the top value of the relevant stack.
```c
stack_ptx_encode_meta_dup(STACK_PTX_STACK_TYPE_F32)
```
- Meta `yank_dup`: duplicates a value from a depth indicated by the top of the `meta_stack`.
```c
stack_ptx_encode_meta_yank_dup(STACK_PTX_STACK_TYPE_F32)
```
- Meta `swap`: swaps the top two values of the relevant stack.
```c
stack_ptx_encode_meta_swap(STACK_PTX_STACK_TYPE_U32)
```
- Meta `swap_with`: swaps the top value with a value at a depth indicated by the `meta_stack`.
```c
stack_ptx_encode_meta_swap_with(STACK_PTX_STACK_TYPE_U32)
```
- Meta `replace`: replaces a value at a depth indicated by the `meta_stack` with the top value.
```c
stack_ptx_encode_meta_replace(STACK_PTX_STACK_TYPE_U32)
```
- Meta `drop`: drops the top N values from the relevant stack, where N comes from the `meta_stack`.
```c
stack_ptx_encode_meta_drop(STACK_PTX_STACK_TYPE_U32)
```
- Meta `clear`: clears all values from the relevant stack.
```c
stack_ptx_encode_meta_clear(STACK_PTX_STACK_TYPE_U32)
```
- Meta `rotate`: moves the top of the stack two positions down (`abc` becomes `bca`).
```c
stack_ptx_encode_meta_rotate(STACK_PTX_STACK_TYPE_U32)
```
- Meta `reverse`: reverses the order of a stack (`abcd` becomes `dcba`).
```c
stack_ptx_encode_meta_reverse(STACK_PTX_STACK_TYPE_U32)
```

#### Store and load
Store and load instructions let you stash values from a stack and recall them later. When `store` runs, it pops one value from the requested stack and writes it to a load/store array at the provided index. `load` with the same index pushes that value back onto the stack. You can call `load` multiple times to reuse the same stored value. If `load` is called for an index that was never stored to, the instruction is ignored.
```c
stack_ptx_encode_store(STACK_PTX_STACK_TYPE_F32, 0)
stack_ptx_encode_load(0)
```
For an example, see `examples/stack_ptx/07_store_load`.

#### Routines
Routines are passed into `stack_ptx_compile` as a parameter. Each routine in the `routines` array is another array of instructions, similar to the main `instructions` array. A routine can be invoked by an instruction in the main array, and routines can call other routines. `StackPtxCompilerInfo.max_frame_depth` sets the maximum call depth. Like the main instruction array, each routine must be terminated with `stack_ptx_encode_return`.
```c
stack_ptx_encode_routine(0)
```
Examples:
- `examples/stack_ptx/01_routines`
- `examples/stack_ptx/02_routine_libraries`

#### Requests
Requests are passed into `stack_ptx_compile`. Whereas `input` instructions allow external registers to appear as arguments in Stack PTX generated PTX, requests allow external registers to be set by Stack PTX generated instructions.

Each request is an index into the `StackPtxRegister* registers` array. For each entry in `requests`, the relevant stack is looked up from `registers`, the stack value is popped, and a `mov` instruction is generated to write into the external register name.

## Type descriptions and generated headers
Stack PTX is designed to be unaware of specific stack types, arg types, and PTX instruction sets. It operates from description tables provided to it to generate PTX assembly. This allows you to customize your setup as needed.

The examples generate encoding headers from `examples/type_descriptions/stack_ptx_descriptions.json` using the Python script `tools/stack_ptx_generate_infos.py`. The CMake helper `mm_add_stack_ptx_header` in `cmake/mm_ptx_codegen.cmake` wraps this and is used in `examples/CMakeLists.txt` to generate headers into the build directory.

To run the generator manually:
```bash
python tools/stack_ptx_generate_infos.py \
  --input examples/type_descriptions/stack_ptx_descriptions.json \
  --output generated/stack_ptx_descriptions.h \
  --lang c
```

## C++
This project is written in C but is compliant with C++. To use C++, generate the C++ version of the generated headers instead of the C version by specifying `--lang cpp`:
```bash
python tools/stack_ptx_generate_infos.py \
  --input examples/type_descriptions/stack_ptx_descriptions.json \
  --output generated/stack_ptx_descriptions.hpp \
  --lang cpp
```
The C++ header uses `constexpr` and namespaces to make the API convenient in C++ code.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
