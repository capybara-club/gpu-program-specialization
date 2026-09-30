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

# SQLite compiler-cache dependency

Pinned upstream SQLite 3.53.4 amalgamation, unmodified sqlite3.c, sqlite3.h
and sqlite3ext.h. The archive and file checksums are recorded in manifest.json.

- Source: https://www.sqlite.org/2026/sqlite-amalgamation-3530400.zip
- Published archive SHA3-256: 628a44cfe82c66aed1ccbbe85a562d2e33ebe64b3288981ed76285612227934e
- License: SQLite public domain; retain upstream notices.

Imported with explicit user authorization on 2026-09-12.
The build never downloads dependencies or silently selects system SQLite.

Only the frontend/runtime compiler-cache layer links SQLite. The hardened core
library and parser/AST producer do not link it. Compile upstream separately with
thread safety enabled, runtime extension loading disabled, memory-status counters
disabled, double-quoted string literals disabled, and hidden symbols to avoid
interposition with an embedding application's SQLite. A future upgrade must pin
its version/checksums and rerun cache, concurrency and full scoring tests.
