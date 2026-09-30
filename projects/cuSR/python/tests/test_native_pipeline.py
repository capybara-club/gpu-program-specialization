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

import unittest

import numpy as np

import cusr
from cusr.routines import DEFAULT_ROUTINES, SAFE_DIV, SAFE_SQRT

try:
    import torch
except ImportError:
    torch = None


class NativeAstTests(unittest.TestCase):
    def test_native_ast_encoding_matches_python(self) -> None:
        cusr.verify_native_abi()

    def test_numpy_runner_handles_postfix_and_routines(self) -> None:
        programs = cusr.pack_programs(
            (
                (cusr.encode_input(0), cusr.SIN, cusr.encode_input(1), cusr.COS, cusr.MUL, cusr.RETURN),
                (cusr.encode_input(2), cusr.encode_input(3), SAFE_DIV, cusr.RETURN),
            )
        )
        inputs = np.array(
            (
                (1.25, 0.5, 3.0, 2.0),
                (0.75, 1.5, 4.0, 5.0),
            ),
            dtype=np.float32,
        )
        actual = cusr.evaluate_programs(programs, inputs, routines=DEFAULT_ROUTINES)
        expected = np.stack(
            (
                np.sin(inputs[:, 0]) * np.cos(inputs[:, 1]),
                inputs[:, 2] * inputs[:, 3] / (inputs[:, 3] * inputs[:, 3] + np.float32(1.0e-8)),
            )
        ).astype(np.float32)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-6)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA-enabled PyTorch is required")
    def test_patch_load_launch_with_pytorch_storage(self) -> None:
        capability_major, capability_minor = torch.cuda.get_device_capability()
        if capability_major not in (8, 9, 10, 12):
            self.skipTest("cuSR supports sm_80 and newer known SASS encodings")

        instantiations = cusr.make_tile_static_mse_instantiations(
            2,
            ast_capacity=8,
            tile_rows=64,
            threads_per_cta=128,
        )
        compiled = cusr.compile_tile_static_mse_cubin(
            instantiations,
            arch=f"sm_{capability_major}{capability_minor}",
        )
        inspection = cusr.inspect_cubin(
            compiled.cubin,
            compiled.lowered_names,
            ast_capacity=8,
            expected_occurrences=1,
        )
        layout = cusr.prepare_patch_layout(inspection)
        programs = cusr.pack_programs(
            (
                (cusr.encode_input(0), cusr.SIN, cusr.encode_input(1), cusr.COS, cusr.MUL, cusr.RETURN),
                (cusr.encode_input(2), cusr.encode_input(3), SAFE_DIV, cusr.RETURN),
                (cusr.encode_input(4), cusr.encode_input(5), cusr.MUL, cusr.encode_input(6), cusr.ADD, cusr.RETURN),
                (cusr.encode_input(7), SAFE_SQRT, cusr.encode_input(0), cusr.TANH, cusr.ADD, cusr.RETURN),
            )
        )
        patched_cubin = bytearray(compiled.cubin)
        stats = cusr.patch_cubin_in_place(layout, programs, patched_cubin, routines=DEFAULT_ROUTINES)

        self.assertNotEqual(patched_cubin, compiled.cubin)
        self.assertEqual(stats.asts_patched, 4)
        self.assertEqual(stats.sites_patched, 2)
        self.assertEqual(layout.max_register_count(patched_cubin), stats.max_patched_register_count)

        rng = np.random.default_rng(0xC057)
        rows = 257
        columns = 8
        settings = 7
        x = rng.uniform(0.5, 3.0, size=(columns, rows)).astype(np.float32)
        target = rng.uniform(-2.0, 2.0, size=rows).astype(np.float32)
        leaf_masks = np.array((0xFF, 0x55, 0xAA, 0x0F, 0xF0, 0x33, 0xCC), dtype=np.uint8)
        column_indices = (
            np.tile(np.arange(8, dtype=np.uint32), (settings, 1))
            + np.arange(settings, dtype=np.uint32)[:, None]
        ) % columns
        constants = rng.uniform(0.5, 2.0, size=(settings, 8)).astype(np.float32)
        leaf_words = constants.view(np.uint32).copy()
        for setting_idx in range(settings):
            for input_idx in range(8):
                if (int(leaf_masks[setting_idx]) >> input_idx) & 1:
                    leaf_words[setting_idx, input_idx] = column_indices[setting_idx, input_idx]

        expected = cusr.evaluate_sse(
            programs,
            x,
            target,
            leaf_masks,
            leaf_words,
            routines=DEFAULT_ROUTINES,
        )
        module = cusr.TileStaticMseModule(
            patched_cubin,
            compiled.lowered_names,
            instantiations,
        )
        output = module(
            torch.from_numpy(x).cuda(),
            torch.from_numpy(target).cuda(),
            torch.from_numpy(leaf_masks).cuda(),
            torch.from_numpy(leaf_words).cuda(),
            asts_per_kernel=2,
        )
        torch.cuda.synchronize()
        actual = output.cpu().numpy()

        np.testing.assert_allclose(actual, expected, rtol=3.0e-4, atol=3.0e-3)

        explicit_stream = torch.cuda.Stream()
        reused_output = torch.full_like(output, 123.0)
        module(
            torch.from_numpy(x).cuda(),
            torch.from_numpy(target).cuda(),
            torch.from_numpy(leaf_masks).cuda(),
            torch.from_numpy(leaf_words).cuda(),
            asts_per_kernel=2,
            output_sse=reused_output,
            stream=explicit_stream,
        )
        explicit_stream.synchronize()
        np.testing.assert_allclose(reused_output.cpu().numpy(), expected, rtol=3.0e-4, atol=3.0e-3)

    @unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA-enabled PyTorch is required")
    def test_patch_eval_epilogue_and_launch_on_multiple_streams(self) -> None:
        capability_major, capability_minor = torch.cuda.get_device_capability()
        if capability_major not in (8, 9, 10, 12):
            self.skipTest("cuSR supports sm_80 and newer known SASS encodings")

        instantiations = cusr.make_tile_static_eval_instantiations(
            2,
            ast_capacity=8,
            tile_rows=64,
            threads_per_cta=128,
        )
        compiled = cusr.compile_tile_static_eval_cubin(
            instantiations,
            arch=f"sm_{capability_major}{capability_minor}",
        )
        inspection = cusr.inspect_cubin(
            compiled.cubin,
            compiled.lowered_names,
            ast_capacity=8,
            expected_occurrences=1,
        )
        layout = cusr.prepare_patch_layout(inspection)
        programs = cusr.pack_programs(
            (
                (cusr.encode_input(0), cusr.SIN, cusr.encode_input(1), cusr.COS, cusr.MUL, cusr.RETURN),
                (cusr.encode_input(2), cusr.encode_input(3), SAFE_DIV, cusr.RETURN),
                (cusr.encode_input(4), cusr.encode_input(5), cusr.MUL, cusr.encode_input(6), cusr.ADD, cusr.RETURN),
                (cusr.encode_input(7), SAFE_SQRT, cusr.encode_input(0), cusr.TANH, cusr.ADD, cusr.RETURN),
            )
        )
        patched_cubin = bytearray(compiled.cubin)
        stats = cusr.patch_cubin_in_place(
            layout,
            programs,
            patched_cubin,
            epilogue=cusr.PatchEpilogue.VALUE,
            routines=DEFAULT_ROUTINES,
        )

        self.assertEqual(stats.asts_patched, 4)
        self.assertEqual(stats.sites_patched, 2)

        rng = np.random.default_rng(0xE7A1)
        rows = 131
        columns = 8
        settings = 7
        x = rng.uniform(0.5, 3.0, size=(columns, rows)).astype(np.float32)
        leaf_masks = np.array((0xFF, 0x55, 0xAA, 0x0F, 0xF0, 0x33, 0xCC), dtype=np.uint8)
        column_indices = (
            np.tile(np.arange(8, dtype=np.uint32), (settings, 1))
            + np.arange(settings, dtype=np.uint32)[:, None]
        ) % columns
        constants = rng.uniform(0.5, 2.0, size=(settings, 8)).astype(np.float32)
        leaf_words = constants.view(np.uint32).copy()
        for setting_idx in range(settings):
            for input_idx in range(8):
                if (int(leaf_masks[setting_idx]) >> input_idx) & 1:
                    leaf_words[setting_idx, input_idx] = column_indices[setting_idx, input_idx]

        expected = cusr.evaluate_values(
            programs,
            x,
            leaf_masks,
            leaf_words,
            routines=DEFAULT_ROUTINES,
        )
        module = cusr.TileStaticEvalModule(
            patched_cubin,
            compiled.lowered_names,
            instantiations,
        )
        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        output = module(
            torch.from_numpy(x).cuda(),
            torch.from_numpy(leaf_masks).cuda(),
            torch.from_numpy(leaf_words).cuda(),
            asts_per_kernel=2,
            streams=streams,
        )
        for stream in streams:
            stream.synchronize()
        actual = output.cpu().numpy()

        np.testing.assert_allclose(actual, expected, rtol=3.0e-4, atol=3.0e-5)


if __name__ == "__main__":
    unittest.main()
