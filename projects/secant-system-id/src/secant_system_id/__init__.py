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
"""Experimental ODE system-identification kernels."""

from .ast import Expression, Program, constant, input_slot
from .bindings import LeafSource, constant_leaf, state_leaf
from .genome import SystemGenome
from .model import MissingSite, SystemModel, missing, state
from .problem import (
    IntegrationSpec,
    NoiseSpec,
    ObservationSeries,
    ParameterSpec,
    ProblemProvenance,
    RecoveryProblem,
    RecoveryProtocol,
    StructureSearchSpec,
)
from .shape import KernelShape, PackedDispatch, PackedKernelSpec

__all__ = [
    "Expression",
    "KernelShape",
    "LeafSource",
    "Program",
    "MissingSite",
    "NoiseSpec",
    "ObservationSeries",
    "ParameterSpec",
    "PackedDispatch",
    "PackedKernelSpec",
    "SystemGenome",
    "SystemModel",
    "IntegrationSpec",
    "ProblemProvenance",
    "RecoveryProblem",
    "RecoveryProtocol",
    "StructureSearchSpec",
    "constant",
    "constant_leaf",
    "input_slot",
    "missing",
    "state",
    "state_leaf",
]
