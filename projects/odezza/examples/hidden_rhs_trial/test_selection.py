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
"""Check family coverage through real native report decoding, without a new fit."""
import json
from collections import Counter
from pathlib import Path
import native
if __name__=='__main__':
 root=Path(__file__).resolve().parents[2]/'scratch/recovery28_odezza';r=json.loads((root/'initial-screen-retry/report.json').read_text());captured=[]
 native.load=lambda:(None,None,None)
 native.batch=lambda jobs,*args,**kwargs:captured.extend(jobs)
 native.fit_report(r,'selection-test',32)
 counts=Counter(j[-1]['origin']['family_index'] for j in captured)
 assert len(captured)==32 and set(counts)=={0,1,2,3} and min(counts.values())==8,counts
 print('All four families retained:',dict(counts))
