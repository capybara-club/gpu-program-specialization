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

# Operators And Shapes

FusedSINDy currently exposes one public AST shape:

```text
depth:        3 binary tree
leaves:       8
binary nodes: 7
unary slots:  15
cohort size:  32 generated features
```

This fixed shape is deliberate. It keeps code size, patch-region size, register
pressure, scratch memory, Gram shape, and solve shape predictable.

## BinaryAST

The public C record is:

```c
typedef struct {
    BinaryAstUnaryOp unary[BINARY_AST_NUM_UNARY_OPS];
    BinaryAstBinaryOp binary[BINARY_AST_NUM_BINARY_OPS];
} BinaryAST;
```

Python exposes the same structure as `BinaryAst`.

The AST stores operation structure only. Leaf bindings live in runtime settings:

```text
leaf_masks: int32 [settings, 32]
leaf_words: int32 [settings, 32, 8]
```

## Simpler Expressions

Depth-2 or smaller expressions can be represented inside the depth-3 shape by
using `KEEP_LEFT`, `KEEP_RIGHT`, identity unary ops, and zero padding ASTs.

For example, a single primitive feature can be represented by keeping one leaf
and ignoring the rest of the tree.

## Unary Operators

Current unary domain:

```text
IDENTITY
SQUARE_F32
CUBE_F32
NEG_FTZ_F32
ABS_FTZ_F32
RCP_APPROX_FTZ_F32
SQRT_APPROX_FTZ_F32
RSQRT_APPROX_FTZ_F32
SIN_APPROX_FTZ_F32
COS_APPROX_FTZ_F32
EX2_APPROX_FTZ_F32
EXP_APPROX_FTZ_F32
LOG2_APPROX_FTZ_F32
LOG10_APPROX_FTZ_F32
SAFE_RCP_F32
SAFE_SQRT_F32
SAFE_RSQRT_F32
SAFE_EX2_F32
SAFE_EXP_F32
SAFE_LOG2_F32
SAFE_LOG10_F32
ZERO_F32
```

The `APPROX` operators map to fast approximate device instructions or short
instruction sequences. The `SAFE` operators use fixed protective behavior for
dangerous domains such as division by zero or logarithms of nonpositive values.

## Binary Operators

Current binary domain:

```text
ADD_FTZ_F32
SUB_FTZ_F32
MUL_FTZ_F32
KEEP_LEFT
KEEP_RIGHT
DIV_APPROX_FTZ_F32
MIN_FTZ_F32
MAX_FTZ_F32
SAFE_DIV_F32
```

## Expression Builder

Python includes a small expression builder:

```python
import fused_sindy as fsindy

u = fsindy.feature(0)
ux = fsindy.feature(1)
expr = fsindy.sin(u) + 0.1 * ux
ast, mask, words = fsindy.expression_to_ast_and_leaf_settings(expr)
```

This is useful for explicit examples and tests. High-throughput searches should
usually generate `BinaryAst` structures and settings tensors directly.

## Operator Changes

The operator set is intentionally narrow while the kernel and patching ABI are
stabilized. For unary/binary option changes, different AST shapes, or direct
Stack PTX exposure for custom operator families, email:

```text
cpdurham@proton.me
```

