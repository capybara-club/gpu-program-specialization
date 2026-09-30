#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Charles Durham
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2026 Charles Durham
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Embed our CUDA source and its shared wire layout in the C99 executable."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parent
out=Path(sys.argv[1])
sources=[('sr_lm_source',root/'kernels/lm.cu'),('sr_lm_layout',root/'src/lm_layout.h')]
text='/* Generated from repository CUDA source; do not edit. */\n'
for name,path in sources:
    # Integer initializers avoid C99's 4095-character string-literal limit.
    data=path.read_bytes()+b'\0'
    text+='static const char '+name+'[]={\n'
    text+='\n'.join(','.join(map(str,data[i:i+64]))+',' for i in range(0,len(data),64))+'\n};\n'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(text)
