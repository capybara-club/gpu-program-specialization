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
from __future__ import annotations

from .model import MissingSite, SystemModel, missing, state


_x0 = state(0)
_x1 = state(1)
ONE_SITE_MODEL = SystemModel(
    name="one_site_control",
    state_names=("position", "velocity"),
    missing_sites=(MissingSite("acceleration", 4),),
    derivatives=(
        _x1,
        missing(0) - 0.1 * _x1,
    ),
    observation_interval=0.25,
)


_s0 = state(0)
_s1 = state(1)
_s2 = state(2)
THREE_SITE_MODEL = SystemModel(
    name="three_site_test",
    state_names=("x", "y", "z"),
    missing_sites=(
        MissingSite("drive_x", 2),
        MissingSite("drive_y", 3),
        MissingSite("drive_z", 4),
    ),
    derivatives=(
        missing(0) - 0.2 * _s0,
        missing(1) + _s0 - 0.3 * _s1,
        missing(2) + _s1 - 0.4 * _s2,
    ),
    observation_interval=0.5,
)
