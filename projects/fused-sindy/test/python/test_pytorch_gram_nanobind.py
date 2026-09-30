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

import fused_sindy as isindy


SAFE_EPS = np.float32(1.0e-6)
SAFE_EX2_LO = np.float32(-126.0)
SAFE_EX2_HI = np.float32(126.0)
SAFE_EXP_LO = np.float32(-80.0)
SAFE_EXP_HI = np.float32(80.0)
LOG2_E = np.float32(1.4426950408889634)


def _torch_or_skip(test: unittest.TestCase):
    try:
        import torch
    except Exception as exc:
        test.skipTest(f"PyTorch is not available: {exc}")
    if not torch.cuda.is_available():
        test.skipTest("PyTorch CUDA is not available")
    return torch


def _i32_bits_to_f32(value: np.int32 | int) -> np.float32:
    return np.asarray([value], dtype=np.int32).view(np.float32)[0]


def _apply_unary(op: int, x: np.ndarray) -> np.ndarray:
    if op == isindy.UnaryOp.SQUARE_F32:
        return (x * x).astype(np.float32)
    if op == isindy.UnaryOp.CUBE_F32:
        return (x * x * x).astype(np.float32)
    if op == isindy.UnaryOp.NEG_FTZ_F32:
        return (-x).astype(np.float32)
    if op == isindy.UnaryOp.ABS_FTZ_F32:
        return np.abs(x).astype(np.float32)
    if op == isindy.UnaryOp.RCP_APPROX_FTZ_F32:
        return (np.float32(1.0) / x).astype(np.float32)
    if op == isindy.UnaryOp.SQRT_APPROX_FTZ_F32:
        return np.sqrt(x).astype(np.float32)
    if op == isindy.UnaryOp.RSQRT_APPROX_FTZ_F32:
        return (np.float32(1.0) / np.sqrt(x)).astype(np.float32)
    if op == isindy.UnaryOp.SIN_APPROX_FTZ_F32:
        return np.sin(x).astype(np.float32)
    if op == isindy.UnaryOp.COS_APPROX_FTZ_F32:
        return np.cos(x).astype(np.float32)
    if op == isindy.UnaryOp.EX2_APPROX_FTZ_F32:
        return np.exp2(x).astype(np.float32)
    if op == isindy.UnaryOp.EXP_APPROX_FTZ_F32:
        return np.exp(x).astype(np.float32)
    if op == isindy.UnaryOp.LOG2_APPROX_FTZ_F32:
        return np.log2(x).astype(np.float32)
    if op == isindy.UnaryOp.LOG10_APPROX_FTZ_F32:
        return np.log10(x).astype(np.float32)
    if op == isindy.UnaryOp.SAFE_RCP_F32:
        return (np.float32(1.0) / np.maximum(np.abs(x), SAFE_EPS)).astype(np.float32)
    if op == isindy.UnaryOp.SAFE_SQRT_F32:
        return np.sqrt(np.maximum(np.abs(x), SAFE_EPS)).astype(np.float32)
    if op == isindy.UnaryOp.SAFE_RSQRT_F32:
        return (np.float32(1.0) / np.sqrt(np.maximum(np.abs(x), SAFE_EPS))).astype(np.float32)
    if op == isindy.UnaryOp.SAFE_EX2_F32:
        return np.exp2(np.clip(x, SAFE_EX2_LO, SAFE_EX2_HI)).astype(np.float32)
    if op == isindy.UnaryOp.SAFE_EXP_F32:
        return np.exp2(np.clip(x, SAFE_EXP_LO, SAFE_EXP_HI) * LOG2_E).astype(np.float32)
    if op == isindy.UnaryOp.SAFE_LOG2_F32:
        return np.log2(np.maximum(np.abs(x), SAFE_EPS)).astype(np.float32)
    if op == isindy.UnaryOp.SAFE_LOG10_F32:
        return np.log10(np.maximum(np.abs(x), SAFE_EPS)).astype(np.float32)
    if op == isindy.UnaryOp.ZERO_F32:
        return np.zeros_like(x, dtype=np.float32)
    if op == isindy.UnaryOp.IDENTITY:
        return x.astype(np.float32)
    raise ValueError(f"unsupported unary op {op}")


def _apply_binary(op: int, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if op == isindy.BinaryOp.SUB_FTZ_F32:
        return (lhs - rhs).astype(np.float32)
    if op == isindy.BinaryOp.MUL_FTZ_F32:
        return (lhs * rhs).astype(np.float32)
    if op == isindy.BinaryOp.DIV_APPROX_FTZ_F32:
        return (lhs / rhs).astype(np.float32)
    if op == isindy.BinaryOp.MIN_FTZ_F32:
        return np.minimum(lhs, rhs).astype(np.float32)
    if op == isindy.BinaryOp.MAX_FTZ_F32:
        return np.maximum(lhs, rhs).astype(np.float32)
    if op == isindy.BinaryOp.SAFE_DIV_F32:
        return (lhs / np.maximum(np.abs(rhs), SAFE_EPS)).astype(np.float32)
    if op == isindy.BinaryOp.KEEP_LEFT:
        return lhs.astype(np.float32)
    if op == isindy.BinaryOp.KEEP_RIGHT:
        return rhs.astype(np.float32)
    if op == isindy.BinaryOp.ADD_FTZ_F32:
        return (lhs + rhs).astype(np.float32)
    raise ValueError(f"unsupported binary op {op}")


def _children(ast_idx: int) -> tuple[int, int, int]:
    if 8 <= ast_idx < 12:
        local = ast_idx - 8
        return local, local * 2, local * 2 + 1
    if 12 <= ast_idx < 14:
        local = ast_idx - 12
        return 4 + local, 8 + local * 2, 8 + local * 2 + 1
    if ast_idx == 14:
        return 6, 12, 13
    raise ValueError(f"bad ast_idx {ast_idx}")


def _evaluate_feature(
    primitive: np.ndarray,
    ast: isindy.BinaryAst,
    leaf_mask: np.int32,
    leaf_words: np.ndarray,
) -> np.ndarray:
    rows = primitive.shape[1]
    primitive_cols = primitive.shape[0]
    mask = int(np.uint32(leaf_mask))
    values: list[np.ndarray] = []

    for leaf_idx in range(isindy.NUM_INPUTS):
        word = leaf_words[leaf_idx]
        if ((mask >> leaf_idx) & 1) != 0:
            col = int(word)
            if 0 <= col < primitive_cols:
                leaf = primitive[col].astype(np.float32)
            else:
                leaf = np.zeros((rows,), dtype=np.float32)
        else:
            leaf = np.full((rows,), _i32_bits_to_f32(word), dtype=np.float32)
        values.append(_apply_unary(int(ast.unary[leaf_idx]), leaf))

    for ast_idx in range(isindy.NUM_INPUTS, isindy.NUM_UNARY_OPS):
        binary_idx, lhs_idx, rhs_idx = _children(ast_idx)
        combined = _apply_binary(int(ast.binary[binary_idx]), values[lhs_idx], values[rhs_idx])
        values.append(_apply_unary(int(ast.unary[ast_idx]), combined))

    return values[-1]


def _expected_outputs(
    primitive: np.ndarray,
    targets: np.ndarray,
    leaf_masks: np.ndarray,
    leaf_words: np.ndarray,
    asts: list[isindy.BinaryAst],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    settings = leaf_masks.shape[0]
    rhs_count = targets.shape[0]
    rows = primitive.shape[1]
    features = len(asts)
    gram = np.zeros((settings, features, features), dtype=np.float32)
    x_sum = np.zeros((settings, features), dtype=np.float32)
    xty = np.zeros((settings, rhs_count, features), dtype=np.float32)
    y_sum = np.zeros((settings, rhs_count), dtype=np.float32)
    yy = np.zeros((settings, rhs_count), dtype=np.float32)

    for setting_idx in range(settings):
        panel = np.empty((features, rows), dtype=np.float32)
        for feature_idx, ast in enumerate(asts):
            panel[feature_idx] = _evaluate_feature(
                primitive,
                ast,
                leaf_masks[setting_idx, feature_idx],
                leaf_words[setting_idx, feature_idx],
            )

        gram[setting_idx] = (panel @ panel.T).astype(np.float32)
        x_sum[setting_idx] = np.sum(panel, axis=1, dtype=np.float32)
        for rhs_idx in range(rhs_count):
            xty[setting_idx, rhs_idx] = np.sum(panel * targets[rhs_idx], axis=1, dtype=np.float32)
            y_sum[setting_idx, rhs_idx] = np.sum(targets[rhs_idx], dtype=np.float32)
            yy[setting_idx, rhs_idx] = np.sum(targets[rhs_idx] * targets[rhs_idx], dtype=np.float32)

    return gram, x_sum, xty, y_sum, yy


def _expected_columns(
    primitive: np.ndarray,
    leaf_masks: np.ndarray,
    leaf_words: np.ndarray,
    asts: list[isindy.BinaryAst],
    cohort_index: int = 0,
) -> np.ndarray:
    settings = leaf_masks.shape[0]
    rows = primitive.shape[1]
    columns = np.zeros((settings, 32, rows), dtype=np.float32)
    ast_begin = cohort_index * 32

    for setting_idx in range(settings):
        for feature_idx in range(32):
            ast_idx = ast_begin + feature_idx
            if ast_idx >= len(asts):
                continue
            columns[setting_idx, feature_idx] = _evaluate_feature(
                primitive,
                asts[ast_idx],
                leaf_masks[setting_idx, feature_idx],
                leaf_words[setting_idx, feature_idx],
            )

    return columns


def _make_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[isindy.BinaryAst]]:
    rows = 257
    primitive_cols = 11
    settings = 4
    rhs_count = 2
    asts = isindy.random_ast_cohort(
        32,
        seed=19,
        unary_ops=[
            isindy.UnaryOp.IDENTITY,
            isindy.UnaryOp.SQUARE_F32,
            isindy.UnaryOp.CUBE_F32,
            isindy.UnaryOp.NEG_FTZ_F32,
            isindy.UnaryOp.ABS_FTZ_F32,
        ],
        binary_ops=[
            isindy.BinaryOp.ADD_FTZ_F32,
            isindy.BinaryOp.SUB_FTZ_F32,
            isindy.BinaryOp.MUL_FTZ_F32,
            isindy.BinaryOp.KEEP_LEFT,
            isindy.BinaryOp.KEEP_RIGHT,
        ],
    )
    rng = np.random.default_rng(23)

    row_idx = np.arange(rows, dtype=np.float32)
    primitive = np.empty((primitive_cols, rows), dtype=np.float32)
    for col in range(primitive_cols):
        primitive[col] = (
            np.float32(0.03 * col)
            + np.float32(0.001) * (row_idx - np.float32(128.0))
            + np.float32(0.0007) * ((row_idx + col * 3) % 17)
        ).astype(np.float32)

    targets = np.empty((rhs_count, rows), dtype=np.float32)
    targets[0] = (primitive[0] + np.float32(0.5) * primitive[1] * primitive[2]).astype(np.float32)
    targets[1] = (primitive[3] - np.float32(0.25) * primitive[4]).astype(np.float32)

    leaf_masks = np.zeros((settings, 32), dtype=np.int32)
    leaf_words = np.zeros((settings, 32, isindy.NUM_INPUTS), dtype=np.int32)
    for setting in range(settings):
        for feature in range(32):
            mask = np.uint32(0)
            for leaf in range(isindy.NUM_INPUTS):
                if rng.random() < 0.7:
                    mask |= np.uint32(1 << leaf)
                    leaf_words[setting, feature, leaf] = np.int32(rng.integers(0, primitive_cols))
                else:
                    value = np.float32(rng.uniform(-0.25, 0.25))
                    leaf_words[setting, feature, leaf] = isindy.f32_constant_word(float(value))
            leaf_masks[setting, feature] = np.int32(mask.view(np.int32))

    return primitive, targets, leaf_masks, leaf_words, asts


class PytorchNanobindGramTest(unittest.TestCase):
    def test_compile_and_run_single_ast_columns_matches_numpy(self) -> None:
        torch = _torch_or_skip(self)
        primitive, _targets, leaf_masks, leaf_words, asts = _make_inputs()
        asts = asts[:13]

        primitive_t = torch.tensor(primitive, device="cuda")
        leaf_masks_t = torch.tensor(leaf_masks, device="cuda", dtype=torch.int32)
        leaf_words_t = torch.tensor(leaf_words, device="cuda", dtype=torch.int32)
        expected = _expected_columns(primitive, leaf_masks, leaf_words, asts, 0)[:, :len(asts), :]

        for kernels_per_module in (isindy.KernelsPerModule.ONE, isindy.KernelsPerModule.FOUR, 3):
            with self.subTest(kernels_per_module=kernels_per_module):
                with isindy.compile_single_ast_column_kernels(
                    asts,
                    kernels_per_module=kernels_per_module,
                    worker_count=2,
                ) as compiled:
                    actual_t = compiled.run_columns(
                        primitive_t,
                        leaf_masks_t,
                        leaf_words_t,
                    )
                    torch.cuda.synchronize()
                    actual = actual_t.cpu().numpy()

                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=8.0e-4,
                    rtol=8.0e-4,
                )

    def test_single_ast_column_can_use_explicit_feature_settings_slot(self) -> None:
        torch = _torch_or_skip(self)
        primitive, _targets, leaf_masks, leaf_words, asts = _make_inputs()
        ast_index = 5
        feature_index = 7
        expected = np.empty((leaf_masks.shape[0], primitive.shape[1]), dtype=np.float32)
        for setting_idx in range(leaf_masks.shape[0]):
            expected[setting_idx] = _evaluate_feature(
                primitive,
                asts[ast_index],
                leaf_masks[setting_idx, feature_index],
                leaf_words[setting_idx, feature_index],
            )

        primitive_t = torch.tensor(primitive, device="cuda")
        leaf_masks_t = torch.tensor(leaf_masks, device="cuda", dtype=torch.int32)
        leaf_words_t = torch.tensor(leaf_words, device="cuda", dtype=torch.int32)
        with isindy.compile_single_ast_column_kernels(asts[:8], kernels_per_module=4, worker_count=2) as compiled:
            actual_t = compiled.run_column(
                primitive_t,
                leaf_masks_t,
                leaf_words_t,
                ast_index=ast_index,
                feature_index=feature_index,
            )
            torch.cuda.synchronize()
            actual = actual_t.cpu().numpy()

        np.testing.assert_allclose(
            actual,
            expected,
            atol=8.0e-4,
            rtol=8.0e-4,
        )

    def test_compile_and_run_columns_matches_numpy(self) -> None:
        torch = _torch_or_skip(self)
        primitive, _targets, leaf_masks, leaf_words, asts = _make_inputs()
        asts = asts + isindy.random_ast_cohort(
            13,
            seed=31,
            unary_ops=[
                isindy.UnaryOp.IDENTITY,
                isindy.UnaryOp.SQUARE_F32,
                isindy.UnaryOp.CUBE_F32,
                isindy.UnaryOp.NEG_FTZ_F32,
                isindy.UnaryOp.ABS_FTZ_F32,
            ],
            binary_ops=[
                isindy.BinaryOp.ADD_FTZ_F32,
                isindy.BinaryOp.SUB_FTZ_F32,
                isindy.BinaryOp.MUL_FTZ_F32,
                isindy.BinaryOp.KEEP_LEFT,
                isindy.BinaryOp.KEEP_RIGHT,
            ],
        )

        primitive_t = torch.tensor(primitive, device="cuda")
        leaf_masks_t = torch.tensor(leaf_masks, device="cuda", dtype=torch.int32)
        leaf_words_t = torch.tensor(leaf_words, device="cuda", dtype=torch.int32)

        for kernels_per_module in (isindy.KernelsPerModule.ONE, isindy.KernelsPerModule.FOUR, 3):
            with self.subTest(kernels_per_module=kernels_per_module):
                with isindy.compile_column_kernels(
                    asts,
                    kernels_per_module=kernels_per_module,
                    worker_count=2,
                ) as compiled:
                    self.assertEqual(compiled.num_cohorts, 2)
                    actual0_t = compiled.run_columns(
                        primitive_t,
                        leaf_masks_t,
                        leaf_words_t,
                        cohort_index=0,
                    )
                    actual1_t = compiled.run_columns(
                        primitive_t,
                        leaf_masks_t,
                        leaf_words_t,
                        cohort_index=1,
                    )
                    torch.cuda.synchronize()
                    actual0 = actual0_t.cpu().numpy()
                    actual1 = actual1_t.cpu().numpy()

                np.testing.assert_allclose(
                    actual0,
                    _expected_columns(primitive, leaf_masks, leaf_words, asts, 0),
                    atol=8.0e-4,
                    rtol=8.0e-4,
                )
                np.testing.assert_allclose(
                    actual1,
                    _expected_columns(primitive, leaf_masks, leaf_words, asts, 1),
                    atol=8.0e-4,
                    rtol=8.0e-4,
                )

    def test_compile_and_run_gram_matches_numpy(self) -> None:
        torch = _torch_or_skip(self)
        primitive, targets, leaf_masks, leaf_words, asts = _make_inputs()

        primitive_t = torch.tensor(primitive, device="cuda")
        targets_t = torch.tensor(targets, device="cuda")
        leaf_masks_t = torch.tensor(leaf_masks, device="cuda", dtype=torch.int32)
        leaf_words_t = torch.tensor(leaf_words, device="cuda", dtype=torch.int32)
        expected_gram, expected_x_sum, expected_xty, expected_y_sum, expected_yy = _expected_outputs(
            primitive,
            targets,
            leaf_masks,
            leaf_words,
            asts,
        )

        for kernels_per_module in (isindy.KernelsPerModule.ONE, isindy.KernelsPerModule.FOUR, 3):
            with self.subTest(kernels_per_module=kernels_per_module):
                compiled = isindy.compile_gram_kernels(
                    asts,
                    kernels_per_module=kernels_per_module,
                    worker_count=2,
                )
                outputs = compiled.run_gram(
                    primitive_t,
                    targets_t,
                    leaf_masks_t,
                    leaf_words_t,
                )
                torch.cuda.synchronize()

                actual_gram = outputs.gram.contiguous().cpu().numpy()
                upper = np.triu(np.ones((32, 32), dtype=bool))
                np.testing.assert_allclose(actual_gram[:, upper], expected_gram[:, upper], atol=8.0e-4, rtol=8.0e-4)
                np.testing.assert_allclose(outputs.x_sum.cpu().numpy(), expected_x_sum, atol=8.0e-4, rtol=8.0e-4)
                np.testing.assert_allclose(outputs.xty.cpu().numpy(), expected_xty, atol=8.0e-4, rtol=8.0e-4)
                np.testing.assert_allclose(outputs.y_sum.cpu().numpy(), expected_y_sum, atol=8.0e-5, rtol=8.0e-5)
                np.testing.assert_allclose(outputs.yy.cpu().numpy(), expected_yy, atol=8.0e-5, rtol=8.0e-5)

    def test_extended_expression_ops_match_numpy(self) -> None:
        torch = _torch_or_skip(self)
        rows = 193
        x_np = np.linspace(0.25, 1.0, rows, dtype=np.float32)
        y_np = np.linspace(1.25, 2.0, rows, dtype=np.float32)
        primitive = np.stack([x_np, y_np]).astype(np.float32)
        targets = np.stack([
            (x_np + np.float32(0.25) * y_np).astype(np.float32),
        ]).astype(np.float32)
        x = isindy.feature(0)
        y = isindy.feature(1)
        expressions = [
            isindy.const(1.0),
            x,
            y,
            -x,
            abs(x - y),
            isindy.rcp(y),
            isindy.sqrt(y),
            isindy.rsqrt(y),
            isindy.sin(x),
            isindy.cos(y),
            isindy.ex2(x * 0.25),
            isindy.exp(x * 0.25),
            isindy.log2(y),
            isindy.log10(y),
            x / y,
            isindy.safe_div(x, x - 0.624),
            isindy.minimum(x, y),
            isindy.maximum(x, y),
            isindy.safe_rcp(x - 0.624),
            isindy.safe_sqrt(x - y),
            isindy.safe_rsqrt(x - y),
            isindy.safe_ex2(x * 0.25),
            isindy.safe_exp(x * 0.25),
            isindy.safe_log2(x - y),
            isindy.safe_log10(x - y),
            isindy.square(x + y),
            isindy.cube(x * 0.25),
            isindy.sin(x + y),
            isindy.cos(y - x),
            isindy.exp((x + y) * 0.125),
            isindy.log2(y + 1.0),
            isindy.log10(y + 1.0),
        ]
        asts, leaf_masks, leaf_words = isindy.expressions_to_asts_and_settings(expressions)
        expected_gram, expected_x_sum, expected_xty, expected_y_sum, expected_yy = _expected_outputs(
            primitive,
            targets,
            leaf_masks,
            leaf_words,
            asts,
        )

        primitive_t = torch.tensor(primitive, device="cuda")
        targets_t = torch.tensor(targets, device="cuda")
        leaf_masks_t = torch.tensor(leaf_masks, device="cuda", dtype=torch.int32)
        leaf_words_t = torch.tensor(leaf_words, device="cuda", dtype=torch.int32)
        with isindy.compile_gram_kernels(asts, worker_count=2) as compiled:
            outputs = compiled.run_gram(
                primitive_t,
                targets_t,
                leaf_masks_t,
                leaf_words_t,
            )
        torch.cuda.synchronize()

        actual_gram = outputs.gram.contiguous().cpu().numpy()
        upper = np.triu(np.ones((32, 32), dtype=bool))
        np.testing.assert_allclose(actual_gram[:, upper], expected_gram[:, upper], atol=3.0e-2, rtol=3.0e-3)
        np.testing.assert_allclose(outputs.x_sum.cpu().numpy(), expected_x_sum, atol=3.0e-2, rtol=3.0e-3)
        np.testing.assert_allclose(outputs.xty.cpu().numpy(), expected_xty, atol=3.0e-2, rtol=3.0e-3)
        np.testing.assert_allclose(outputs.y_sum.cpu().numpy(), expected_y_sum, atol=8.0e-5, rtol=8.0e-5)
        np.testing.assert_allclose(outputs.yy.cpu().numpy(), expected_yy, atol=8.0e-5, rtol=8.0e-5)


if __name__ == "__main__":
    unittest.main()
