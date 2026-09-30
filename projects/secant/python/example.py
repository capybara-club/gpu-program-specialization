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

import numpy as np

import secant
from secant.routines import DEFAULT_ROUTINES


def main() -> None:
    x0 = secant.input(0)
    x1 = secant.input(1)
    programs = (
        secant.Program(secant.sin(x0) * secant.cos(x1)),
        secant.Program(secant.safe_div(x0, x1)),
    )
    recipe = secant.MaterializeRecipe(
        num_kernels=1,
        asts_per_kernel=2,
        num_inputs=2,
        patch_capacity_instructions=64,
    )
    template = secant.compile_materialize(recipe)
    specialized_cubin = template.specialize(programs, routines=DEFAULT_ROUTINES)

    columns = np.random.default_rng(7).uniform(0.5, 3.0, size=(2, 4096)).astype(np.float32)
    expected = secant.materialize(programs, columns, routines=DEFAULT_ROUTINES)
    actual = secant.MaterializeModule(specialized_cubin, recipe).run(columns)

    print(f"arch={template.arch}")
    print(f"template_cubin_bytes={len(template.cubin)}")
    print(f"program_bytecode={[program.bytecode.hex() for program in programs]}")
    print(f"output_shape={actual.shape}")
    print(f"max_abs_error={np.max(np.abs(actual - expected)):.9g}")
    print(f"sample_gpu={actual[:, :4]}")


if __name__ == "__main__":
    main()
