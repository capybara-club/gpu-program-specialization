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

# Secant Storage And Source Snapshots

Generated datasets, prepared inputs, caches, logs, and campaign results live
outside the Git working trees. Each machine uses the following layout under
its own home directory:

```text
~/data/secant-runtime/secant/scratch
~/data/secant-runtime/secant-sr/scratch
```

The repositories retain a `scratch` symlink to these locations so existing
commands and relative paths continue to work. Source backup tools must not be
configured to dereference that symlink.

Create a fresh source-only snapshot of both repositories with:

```sh
secant/scripts/source_snapshot.sh ~/data/source-snapshots
```

The script creates a new UTC-timestamped directory and never overwrites a
previous snapshot. It includes current tracked and untracked source files but
excludes Git metadata recursively, runtime data, build trees, virtual
environments, generated binaries, logs, and the unrelated nested
`fortunes-foundation-game` repository. It verifies that the finished snapshot
contains no `.git` directory.

If `secant-sr` is not a sibling of `secant`, pass its path as the second
argument:

```sh
secant/scripts/source_snapshot.sh ~/data/source-snapshots /path/to/secant-sr
```

Runtime stores are machine-local and are not copied by the source snapshot.
Back them up separately only when the experiment artifacts themselves are
needed.
