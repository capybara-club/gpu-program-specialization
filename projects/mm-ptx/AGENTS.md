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

# Agent Notes

This repo is optimized for generated PTX workflows. Keep public docs short for humans, and put mechanical editing contracts here so coding agents can preserve the design.

## Repository Shape

- `ptx_inject.h` parses PTX injection markers and renders final PTX with one stub per inject index.
- `stack_ptx.h` compiles mutation-friendly stack programs into PTX stubs.
- `ast_ptx.h` compiles postorder single-output AST programs into PTX stubs.
- Concrete AST PTX f32 instruction names, arities, and encode helpers live in `tools/ast_ptx_instructions.h`.
- The AST PTX f32 CPU interpreter lives in `tools/ast_ptx_interpreter.h`.
- Do not bump version numbers unless the user explicitly asks.

## Documentation Contract

- Keep `README.md` human-first: overview, one CUDA augmentation example, one Stack PTX stub example, one AST PTX stub example, and links.
- Put API details in `STACK_PTX.md`, `AST_PTX.md`, and `PTX_INJECT.md`.
- Put agent-facing invariants here instead of expanding the README.
- When docs mention generated PTX register names, distinguish bare API register names such as `"x"` from emitted PTX register names such as `%x`.

## PTX Inject Contract

`PTX_INJECT` is the readable CUDA annotation path. For runtime CUDA template generators, prefer emitting equivalent inline asm directly because repeatedly expanding the macro is expensive.

There are two valid generated-asm modes:

- Marker mode: generated inline asm emits the same marker block as `PTX_INJECT`; later `ptx_inject_render_ptx` replaces the marker region with a generated stub.
- Final-asm mode: generated inline asm contains the final stub directly; this can skip PTX Inject parsing/rendering for that site.

For marker mode, preserve this contract:

- Stable internal registers are named `_x0`, `_x1`, etc. In inline asm text they appear as `%%_x0`; in final PTX they appear as `%_x0`.
- Operand order is canonical: all `mod` operands, then all `out` operands, then all `in` operands.
- Declare one stable register per operand.
- Move `mod` and `in` operands from asm operands into stable registers before `PTX_INJECT_START`.
- Move `mod` and `out` stable registers back to asm operands after `PTX_INJECT_END`.
- Marker lines must remain parseable:

```ptx
// PTX_INJECT_START site_name
// _x0 o f32 F32 z
// _x1 i f32 F32 x
// _x2 i f32 F32 y
// PTX_INJECT_END
```

`bench/ptx_inject_emitter.h` is the current generated-inline-asm reference. It can emit marker-mode asm or final-asm mode with a supplied stub. Its stub emitter strips a single outer `{ ... }` block from generated PTX stubs before embedding them inside asm.

## Stack PTX Contract

- `StackPtxInstruction` is fixed-width and mutation-friendly.
- Instruction arrays terminate with `stack_ptx_encode_return`.
- Stack PTX may ignore invalid stack operations and perform dead-code elimination; preserve that behavior unless the user explicitly asks for stricter validation.
- `StackPtxRegister.name` is bare, for example `"x"`; emitted PTX uses `%x`.
- Use the established measure-and-allocate compile pattern, including `stack_ptx_compile_workspace_size`.
- Keep generated description headers as the source of concrete instruction encodes.

## AST PTX Contract

- AST PTX programs are postorder `AstPtxInstruction` arrays terminated by `ast_ptx_encode_return`.
- Every top-level AST and every routine must return exactly one value.
- `AstPtxInstruction.aux` is the number of f32 arguments for PTX instruction calls and routine calls.
- PTX instruction `aux` must match `ast_ptx_ptx_instruction_num_args[idx]`.
- Keep `ast_ptx.h` generic. It must not know concrete f32 instruction names or math semantics.
- Keep public encode helpers compact and Stack PTX-like.
- Keep `instructions` immediately before `buffer` in `ast_ptx_compile`.
- Keep routine arguments explicit with `ast_ptx_encode_routine_arg(idx)`.

## Test Commands

After AST PTX edits:

```sh
cmake --build <build-dir> --target ast_ptx_expected_output_tests ast_ptx_cpp_tests ast_ptx_inject_00_simple_tests
ctest --test-dir <build-dir> -R 'mm_ptx\.ast_ptx' --output-on-failure
```

After broad PTX Inject or Stack PTX edits, run the relevant CTest subset:

```sh
ctest --test-dir <build-dir> -R 'mm_ptx\.(ptx_inject|stack_ptx)' --output-on-failure
```
