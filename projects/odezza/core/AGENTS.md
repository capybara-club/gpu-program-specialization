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

# Odezza native core boundary

This folder is the hardened C99 kernel and execution API. Keep grammar expansion,
GP, search controllers, JSON service policy and experiment tooling outside it.

Before changing files here, notify the user that the native core will change and
describe the reason and affected interface. **Do not issue that notice when the
user explicitly requested the core change**; their request already provides the
notice and authorization. This is a notice rule, not an additional approval gate.

Use `odezza.h` as the only public API header. Shared implementation declarations
belong in `o_odezza_internal.h`; implementation-only helper headers are private.
Split C99 compilation units by function. Public handles own persistent streams,
events and templates; callers own the current CUDA context and device data.
Do not introduce CUDA context management or search policy into this library.

Validate changed contracts with native callers and validate affected downstream
search integration. Record numerical or execution-path differences explicitly.
