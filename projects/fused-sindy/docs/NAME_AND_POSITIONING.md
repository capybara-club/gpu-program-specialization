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

# Name And Positioning

## Public Name

Use:

```text
repo:        fused-sindy
project:     FusedSINDy
python:      fused_sindy
short alias: fsindy
```

Tagline:

```text
CUDA-fused sparse equation discovery without materialized feature matrices.
```

The implementation still contains `implicit_sindy` names for ABI continuity:

```text
Python compatibility import: implicit_sindy
C public header:             implicit_sindy.h
C symbol prefix:             implicit_sindy_*
CMake option prefix:         IMPLICIT_SINDY_*
```

## Why FusedSINDy

The important user-visible behavior is fusion: generated feature evaluation is
fused with Gram/statistics accumulation. The name avoids overloading the term
"implicit SINDy", which has established meanings in the literature.

FusedSINDy is SINDy-style, but the central contribution is a CUDA search engine:
compiled feature programs, runtime leaf settings, direct Gram accumulation, and
sparse solves over compact statistics.

## Compared With PySINDy

PySINDy is a better user-facing scientific package today. It is strongest when
the user has a fixed candidate library and wants mature estimator utilities.

FusedSINDy is stronger when the library is not fixed and the search needs to try
many feature structures and many runtime settings over large row counts.

The best framing is:

```text
PySINDy:    mature sparse-regression workflow around explicit libraries
FusedSINDy: CUDA engine for high-throughput generated-feature scoring
```

## Compared With PDE-FIND

PDE-FIND-style workflows are compelling when the PDE library is known:
derivatives, products, polynomial terms, and sparse regression over a structured
library.

FusedSINDy is useful when:

- primitive columns are many or expensive;
- secondary features should not be materialized for every guess;
- constants inside nonlinearities are part of the search;
- many cohorts need to be scored before committing to a feature library.

## Compared With PySR

PySR is broader symbolic regression. It has a mature evolutionary search,
complexity/accuracy tradeoffs, simplification, and constant optimization.

FusedSINDy is narrower:

- models are linear in 32 generated features after STLSQ;
- AST shape is fixed depth-3 for now;
- throughput is optimized for large row/settings batches;
- expression simplification is not the focus.

The likely overlap is using FusedSINDy as a high-throughput candidate scorer for
problems where a general symbolic-regression loop would spend too much time
evaluating or materializing candidate expressions.

