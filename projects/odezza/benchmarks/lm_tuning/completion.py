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
"""Completion wording distinguishes early queue failures from wall deadlines."""


def message(states, completed_pairs, *, now, deadline):
    confirmed = all(s.get('finished_at') for s in states.values())
    successful = confirmed and all(s['status'] == 'complete' for s in states.values())
    if successful:
        text = 'Odezza paired LM/curvature tuning finished.'
    elif now < deadline:
        text = 'Odezza tuning stopped early because workers failed.'
    else:
        text = 'Odezza tuning deadline reached with failures or unconfirmed workers.'
    return (text + ' ' + str(completed_pairs) + ' completed pairs. ' +
            ', '.join(lane + ': ' + s['status'] for lane, s in states.items()) +
            '. Results, MSEs, timings and unsupported cases are saved on mac1.')
