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
# SPDX-FileCopyrightText: 2026 Charles Durham
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ctypes
import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from mm_ptx import get_include_dir, get_ptx_inject_header
from mm_ptx.ptx_inject import PTXInject, MutType


PTX_SAMPLE = (
    "// pre\n"
    "// PTX_INJECT_START func\n"
    "// _x0 i f32 F32 x\n"
    "// _x1 m f32 F32 y\n"
    "// _x2 o f32 F32 z\n"
    "// PTX_INJECT_END\n"
    "// post\n"
)

PTX_SAMPLE_TWO = (
    "// pre\n"
    "// PTX_INJECT_START func_a\n"
    "// _x0 i f32 F32 x\n"
    "// PTX_INJECT_END\n"
    "// mid\n"
    "// PTX_INJECT_START func_b\n"
    "// _x0 o f32 F32 y\n"
    "// PTX_INJECT_END\n"
    "// post\n"
)

EXPECTED_TWO_SITE_RENDER = (
    "// pre\n"
    "/*stub-a*/\n"
    "// mid\n"
    "/*stub-b*/\n"
    "// post\n"
)


class TestPtxInjectParsing(unittest.TestCase):
    def test_parse_and_render(self):
        with PTXInject(PTX_SAMPLE) as inject:
            self.assertEqual(len(inject.injects), 1)
            func = inject["func"]

            self.assertEqual(func["x"].mut_type, MutType.IN)
            self.assertEqual(func["x"].data_type, "F32")
            self.assertEqual(func["x"].register_type, "f32")
            self.assertEqual(func["x"].reg, "_x0")

            self.assertEqual(func["y"].mut_type, MutType.MOD)
            self.assertEqual(func["y"].data_type, "F32")
            self.assertEqual(func["y"].register_type, "f32")
            self.assertEqual(func["y"].reg, "_x1")

            self.assertEqual(func["z"].mut_type, MutType.OUT)
            self.assertEqual(func["z"].data_type, "F32")
            self.assertEqual(func["z"].register_type, "f32")
            self.assertEqual(func["z"].reg, "_x2")

            rendered = inject.render_ptx({"func": "\t// stub\n"})

        self.assertIn("// pre", rendered)
        self.assertIn("// post", rendered)
        self.assertIn("// stub", rendered)
        self.assertNotIn("PTX_INJECT_START", rendered)
        self.assertNotIn("PTX_INJECT_END", rendered)


class TestPtxInjectMultiSite(unittest.TestCase):
    def test_render_two_sites(self):
        with PTXInject(PTX_SAMPLE_TWO) as inject:
            self.assertEqual(len(inject.injects), 2)
            self.assertEqual(inject.injects[0].name, "func_a")
            self.assertEqual(inject.injects[1].name, "func_b")
            rendered = inject.render_ptx({"func_a": "/*stub-a*/", "func_b": "/*stub-b*/"})

        self.assertEqual(rendered, EXPECTED_TWO_SITE_RENDER)


class TestPtxInjectCudaIntegration(unittest.TestCase):
    def test_cuda_output_value(self):
        if os.getenv("MM_PTX_RUN_CUDA_TESTS") != "1":
            self.skipTest("Set MM_PTX_RUN_CUDA_TESTS=1 to run CUDA integration tests.")

        try:
            from cuda.core import Device, Program, ProgramOptions, LaunchConfig, launch, DeviceMemoryResource
        except Exception as exc:
            self.skipTest(f"cuda.core not available: {exc}")

        try:
            from cuda.bindings import runtime as cudart
        except Exception as exc:
            self.skipTest(f"cuda.bindings.runtime not available: {exc}")

        try:
            dev = Device()
            dev.set_current()
        except Exception as exc:
            self.skipTest(f"CUDA device not available: {exc}")

        def _make_program_options(**kwargs):
            include_dirs = kwargs.pop("include_dirs", None)
            if include_dirs:
                for include_key in ("include_path", "include_paths", "include_dirs"):
                    try:
                        return ProgramOptions(**kwargs, **{include_key: include_dirs})
                    except TypeError:
                        continue
            return ProgramOptions(**kwargs)

        capability = dev.compute_capability
        arch = f"sm_{capability.major}{capability.minor}"

        header_path = get_ptx_inject_header().replace("\\", "/")
        cuda_code = f"""
#include "{header_path}"

extern "C"
__global__
void
kernel(float* out) {{
    float x = 5.0f;
    float y = 3.0f;
    float z = 0.0f;
    for (int i = 0; i < 2; i++) {{
        PTX_INJECT("func",
            PTX_IN (F32, x, x),
            PTX_MOD(F32, y, y),
            PTX_OUT(F32, z, z)
        );
    }}
    out[0] = z;
}}
"""

        program_options = _make_program_options(
            std="c++11",
            arch=arch,
            include_dirs=[get_include_dir()],
        )
        prog = Program(cuda_code, code_type="c++", options=program_options)
        mod = prog.compile("ptx", logs=sys.stdout)
        annotated_ptx = mod.code.decode("utf-8")

        with PTXInject(annotated_ptx) as inject:
            func = inject["func"]
            stub = (
                f"\tadd.ftz.f32 %{func['y'].reg}, %{func['x'].reg}, %{func['y'].reg};\n"
                f"\tadd.ftz.f32 %{func['z'].reg}, %{func['x'].reg}, %{func['y'].reg};"
            )
            rendered_ptx = inject.render_ptx({"func": stub})

        prog = Program(rendered_ptx, code_type="ptx", options=ProgramOptions(arch=arch))
        mod = prog.compile("cubin", logs=sys.stdout)
        ker = mod.get_kernel("kernel")

        dmem = DeviceMemoryResource(dev)
        out_buf = dmem.allocate(ctypes.sizeof(ctypes.c_float))

        stream = dev.default_stream
        launch(stream, LaunchConfig(grid=1, block=1), ker, out_buf)
        stream.sync()

        out_host = ctypes.c_float(0.0)
        out_host_ptr = ctypes.addressof(out_host)
        out_ptr = out_buf.handle
        if hasattr(out_ptr, "getPtr"):
            out_ptr = out_ptr.getPtr()
        out_ptr = int(out_ptr)

        err, = cudart.cudaMemcpy(
            out_host_ptr,
            out_ptr,
            ctypes.sizeof(ctypes.c_float),
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        if err != cudart.cudaError_t.cudaSuccess:
            self.fail(f"cudaMemcpy failed: {err}")

        self.assertAlmostEqual(out_host.value, 18.0, places=3)

