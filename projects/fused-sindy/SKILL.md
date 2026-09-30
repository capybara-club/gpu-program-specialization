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

---
name: fused-sindy-c99-api-style
description: Use when editing FusedSINDy C99/CUDA APIs, especially bulk compilation, PTX injection, BinaryAST emission, workspace handling, and robust error-return control flow.
---

# FusedSINDy C99 API Style

## Memory Ownership

- Prefer caller-provided memory buffers for API routines. Do not call `malloc` internally unless the API explicitly documents and requires internally allocated output.
- When allocation is explicitly allowed, prefer one allocation chunk over many small allocations. Slice that chunk with offsets and alignment.
- Thread-local scratch should come from one per-thread scratch region. If a worker needs a PTX-stub buffer and a rendered-PTX buffer, allocate both from that scratch region with offsets.
- Hot loops should assume the supplied workspace is large enough. If it is not, return a workspace-exhausted error code. Do not grow memory, retry internally, or allocate fallback buffers.
- Do not measure every AST stub, allocate storage for it, then render it later. Generate transient AST PTX directly into worker scratch during compilation.
- Do not copy stubs only to NUL-terminate them. APIs that pass PTX stubs to `ptx_inject_render_ptx` must either require NUL-terminated stubs or generate NUL-terminated stubs directly into scratch.

## Error Style

- Robust API routines return explicit error-code enums.
- Prefer `*_ERROR_RET(...)` and `*_CHECK_RET(...)` macro styles already used in the codebase.
- Avoid `goto cleanup`.
- If a function needs cleanup, declare cleanup-owned state in the outer function, call one level deeper into an implementation function that can early-return an error code, then unconditionally clean up in the outer function before returning the saved `error_ret`.
- Keep cleanup unconditional and local to the scope that owns the state.

## API Shape

- Public descriptors should make ownership clear: caller-owned input buffers, caller-owned worker workspace, caller-owned output arrays, and explicitly documented internally allocated outputs when unavoidable.
- Do not introduce structs merely to condense function arguments. Keep arguments direct in the function signature. A struct argument is acceptable only when the struct is real data with its own meaning, lifetime, layout, or storage role.
- Public enum domains should be declared as named `typedef enum` types. Do not
  split them into an integer typedef plus a separate anonymous enum unless the
  fixed-width storage size is an intentional ABI requirement.
- In public headers, place all public struct typedefs, enum typedefs, and
  opaque typedefs before the first public function declaration.
- Prefer sizing helpers for coarse workspace planning. The actual compile path should still return workspace-exhausted if the supplied memory is too small.
- Keep hot-path behavior deterministic: no hidden allocation, no retry loops, no data-dependent heap churn.
- Public C-compatible headers should use a library-specific `*_PUBLIC_DEC`
  macro on exported declarations instead of wrapping the whole header body in
  `extern "C"`.

## Tooling And Dependencies

- When a standard package or tool is missing, prefer asking the user to install
  it instead of spending significant effort on a workaround. Use a workaround
  only when installation would be disruptive, unavailable, or slower than the
  workaround.

## Function Layout

- Public header declarations may be one line when the declaration is short and
  remains easy to scan. Prefer multiline declarations in headers only for long
  API signatures.
- Do not place a blank line between adjacent one-line public header typedefs or
  function declarations. Keep blank lines around multiline declarations.
- Format C/CUDA function definitions with attributes, return type, function name, and argument list split across lines:

```c
__device__ __forceinline__
float
u32_as_f32(
    uint32_t bits
) {
    union {
        uint32_t u;
        float f;
    } value;
    value.u = bits;
    return value.f;
}
```

- Keep direct arguments visible in the signature. Prefer wrapping long function calls over hiding arguments in a one-off descriptor struct.
