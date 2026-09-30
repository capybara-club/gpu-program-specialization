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

# GP representation for systems with missing equations

## Genome boundary

A system individual contains one independent post-order AST for each missing
mathematical site:

```text
system genome
  missing site 0 -> post-order AST 0
  missing site 1 -> post-order AST 1
  ...
```

The CUDA specialization site returns all of these roots at once, but they are
not encoded as one multi-output post-order program. `SystemGenome` validates
that each AST references only its assigned contiguous leaf-slot range.

This separation is preferable for an initial GP implementation because:

- ordinary subtree mutation and crossover still have one root;
- a failed or unstable missing term can be replaced independently;
- different sites may use different grammars, node limits, units, and leaf
  counts;
- complexity penalties can be reported per equation and per system; and
- ASTs can be specialized sequentially into the site's output registers with
  no multi-root bytecode semantics.

## Settings

The complete system setting owns one constant bank. Each AST owns a slice of
the global dynamic-leaf binding vector. A binding is one integer indexing:

```text
[current state 0, ..., current state N-1,
 setting constant 0, ..., setting constant K-1]
```

Two ASTs can intentionally share a fitted parameter by selecting the same
constant slot. Otherwise they select disjoint constant slots. The leaf binding
slice should travel with an AST when an entire missing site is exchanged.

## Mutation and crossover

The implemented C99 GP uses fixed-size postorder program slots directly and
caches a compact annotation for every node: byte range, subtree node range,
arity, and depth. This permits bounded subtree edits without allocating tree
objects or reparsing the program for every variation operation.

Recommended operations are:

1. **Within-site subtree mutation:** choose a missing site, replace one subtree,
   and preserve that site's type and unit constraints.
2. **Within-site subtree crossover:** exchange compatible subtrees from the same
   missing-site role in two parents.
3. **Whole-site crossover:** exchange an AST together with its binding slice;
   keep or remap constant slots according to the parameter-sharing policy.
4. **Binding mutation:** change a leaf source between allowed states and
   constants without changing the skeleton.
5. **Constant mutation or optimization:** change the shared constant bank, or
   promote the structure to LM.

System-level selection uses one trajectory fitness, while diagnostics retain
per-state and per-experiment residuals so mutation can target the missing site
most associated with the error.

## Implemented population lifecycle

Two fixed population arenas rotate between current and next generations. Each
genome owns its site-local programs, incumbent constant bank, and complete leaf
binding vector. Winner reduction updates those incumbents before selection, so
a useful leaf configuration or constant estimate can survive and seed nearby
settings in the next generation.

The current search uses explicit elites, tournament selection, same-site
subtree crossover, whole-site crossover, subtree mutation, point mutation, and
random immigrants. It is deliberately a small generational baseline. Typed or
unit-aware grammars, novelty/quality-diversity archives, and residual-directed
site mutation remain policy layers rather than runtime requirements.

## When to add multi-output ASTs

A multi-output bytecode should not be the default. Add it only if measurement
shows that explicitly shared subexpressions across missing equations are common
and valuable. At that point, represent the individual as a typed DAG or as
separate roots calling shared post-order routines. The CUDA ABI can remain a
dynamic list of output registers; only the genome and SASS scheduling layer
need to understand shared values.
