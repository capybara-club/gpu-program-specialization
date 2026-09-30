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
"""Exact public term catalogue evaluated by NumPy; no Odezza."""
import numpy as np

def term(X,desc,w,grad=False):
 k,u,v=desc;U=X[:,u];V=X[:,v]
 if k==0:
  z=w*U*V;f=np.sin(z);gw=np.cos(z)*U*V;gu=np.cos(z)*w*V;gv=np.cos(z)*w*U
 elif k==1:
  z=w*U+V;f=np.sin(z);gw=np.cos(z)*U;gu=np.cos(z)*w;gv=np.cos(z)
 elif k==2:
  z=w*U*np.sin(V);f=np.sin(z);gw=np.cos(z)*U*np.sin(V);gu=np.cos(z)*w*np.sin(V);gv=np.cos(z)*w*U*np.cos(V)
 else:
  f=np.sin(w*U)*np.sin(V);gw=np.cos(w*U)*U*np.sin(V);gu=np.cos(w*U)*w*np.sin(V);gv=np.sin(w*U)*np.cos(V)
 return (f,gw,gu,gv) if grad else f

def catalogue():return [(k,u,v) for k in range(4) for u in range(6) for v in range(6) if u!=v and (k!=0 or u<v)]
def label(d):
 k,u,v=d;u='x'+str(u);v='x'+str(v)
 return [f'sin(w*{u}*{v})',f'sin(w*{u}+{v})',f'sin(w*{u}*sin({v}))',f'sin(w*{u})*sin({v})'][k]
