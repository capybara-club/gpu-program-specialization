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
from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import fused_sindy as isindy


FEATURES = 32
LEAVES = 8
PRIMITIVE_NAMES = ("u", "u_x", "u_xx")
SETTINGS_KERNEL_PATH = Path(__file__).with_name("burgers_pde_settings_philox.cu")


@dataclass
class SearchResult:
    mse: float = math.inf
    objective: float = math.inf
    active_count: int = 0
    cohort_idx: int = -1
    setting_idx: int = -1
    sweep_idx: int = -1
    alpha: float = math.nan
    threshold: float = math.nan
    beta: np.ndarray | None = None
    intercept: float = math.nan
    feature_text: list[str] | None = None


@dataclass
class CompiledBatch:
    batch_idx: int
    start_cohort: int
    batch_cohorts: int
    asts: list[isindy.BinaryAst]
    gram_kernels: object
    compile_ms: float


@dataclass
class PendingCohort:
    global_cohort_idx: int
    asts: list[isindy.BinaryAst]
    settings_np: tuple[np.ndarray, np.ndarray] | None
    leaf_masks_t: object
    leaf_words_t: object
    train_raw: object
    validation_raw: object
    stlsq: object
    validation_mse: object


@dataclass
class GpuPhaseEvents:
    settings: list[tuple[object, object]]
    train_gram: list[tuple[object, object]]
    validation_gram: list[tuple[object, object]]
    solve: list[tuple[object, object]]
    mse: list[tuple[object, object]]


@dataclass
class PendingBatch:
    compiled: CompiledBatch
    stream: object
    start_event: object
    end_event: object
    phase_events: GpuPhaseEvents
    cohorts: list[PendingCohort]
    enqueue_ms: float
    settings_alloc_ms: float
    settings_launch_ms: float
    settings: int
    train_rows: int
    validation_rows: int
    num_sweeps: int


def f32_word(value: float) -> np.int32:
    return np.asarray([value], dtype=np.float32).view(np.int32)[0]


def word_to_f32(value: np.int32 | int) -> np.float32:
    return np.asarray([value], dtype=np.int32).view(np.float32)[0]


def make_burgers_data(
    *,
    seed: int,
    rows: int,
    nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2.0 * np.pi, rows).astype(np.float32)
    t = rng.uniform(0.0, 2.0 * np.pi, rows).astype(np.float32)

    phase0 = x + np.float32(0.31) * t
    phase1 = np.float32(2.0) * x - np.float32(0.17) * t
    phase2 = np.float32(3.0) * x + np.float32(0.11) * t
    u = (
        np.sin(phase0)
        + np.float32(0.45) * np.cos(phase1)
        + np.float32(0.20) * np.sin(phase2)
    ).astype(np.float32)
    ux = (
        np.cos(phase0)
        - np.float32(0.90) * np.sin(phase1)
        + np.float32(0.60) * np.cos(phase2)
    ).astype(np.float32)
    uxx = (
        -np.sin(phase0)
        - np.float32(1.80) * np.cos(phase1)
        - np.float32(1.80) * np.sin(phase2)
    ).astype(np.float32)
    primitive = np.stack([u, ux, uxx]).astype(np.float32)
    target = (-u * ux + np.float32(nu) * uxx).reshape(1, rows).astype(np.float32)
    return primitive, target


def ast_from_expr(expr: object) -> isindy.BinaryAst:
    ast, _mask, _words = isindy.expression_to_ast_and_leaf_settings(expr)
    return ast


def template_asts() -> dict[str, isindy.BinaryAst]:
    a = isindy.feature(0)
    b = isindy.feature(1)
    return {
        "identity": ast_from_expr(a),
        "neg": ast_from_expr(-a),
        "square": ast_from_expr(isindy.square(a)),
        "cube": ast_from_expr(isindy.cube(a)),
        "add": ast_from_expr(a + b),
        "sub": ast_from_expr(a - b),
        "mul": ast_from_expr(a * b),
        "safe_div": ast_from_expr(isindy.safe_div(a, b)),
    }


def make_template_cohort(
    rng: np.random.Generator,
    templates: dict[str, isindy.BinaryAst],
) -> list[isindy.BinaryAst]:
    weighted = (
        ["identity"] * 8
        + ["mul"] * 10
        + ["add"] * 4
        + ["sub"] * 3
        + ["square"] * 3
        + ["cube"] * 1
        + ["neg"] * 1
        + ["safe_div"] * 2
    )
    cohort = [templates["mul"], templates["identity"]]
    while len(cohort) < FEATURES:
        cohort.append(templates[str(rng.choice(weighted))])
    return cohort


def random_leaf_settings(
    rng: np.random.Generator,
    settings: int,
    primitive_cols: int,
    *,
    feature_leaf_probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    masks = np.zeros((settings, FEATURES), dtype=np.int32)
    words = np.empty((settings, FEATURES, LEAVES), dtype=np.int32)
    constants = np.asarray(
        [
            f32_word(-2.0),
            f32_word(-1.0),
            f32_word(-0.5),
            f32_word(0.0),
            f32_word(0.5),
            f32_word(1.0),
            f32_word(2.0),
        ],
        dtype=np.int32,
    )

    for leaf_idx in range(LEAVES):
        use_feature = rng.random((settings, FEATURES)) < feature_leaf_probability
        feature_cols = rng.integers(0, primitive_cols, size=(settings, FEATURES), dtype=np.int32)
        constant_words = rng.choice(constants, size=(settings, FEATURES), replace=True)
        words[:, :, leaf_idx] = np.where(use_feature, feature_cols, constant_words)
        masks |= np.where(use_feature, np.int32(1 << leaf_idx), np.int32(0))
    return masks, words


def seed_burgers_setting(
    leaf_masks: np.ndarray,
    leaf_words: np.ndarray,
) -> None:
    leaf_masks[0, :] = 0
    leaf_words[0, :, :] = f32_word(0.0)
    leaf_masks[0, 0] = np.int32((1 << 0) | (1 << 4))
    leaf_words[0, 0, 0] = np.int32(0)
    leaf_words[0, 0, 4] = np.int32(1)
    leaf_masks[0, 1] = np.int32(1 << 0)
    leaf_words[0, 1, 0] = np.int32(2)


class PyTorchStreamWrapper:
    def __init__(
        self,
        pt_stream,
    ) -> None:
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        return (0, self.pt_stream.cuda_stream)


class CudaPhiloxSettingsGenerator:
    def __init__(
        self,
        torch,
        device,
    ) -> None:
        from cuda.core import Device, Program, ProgramOptions

        self._device = None
        self._program = None
        self._module = None
        self._kernel = None
        self.jit_ms = 0.0

        begin = time.perf_counter()
        source = SETTINGS_KERNEL_PATH.read_text(encoding="utf-8")
        with torch.cuda.device(device):
            torch.empty((), device=device)
            device_index = torch.device(device).index
            if device_index is None:
                device_index = torch.cuda.current_device()
            self._device = Device(device_index)
            self._device.set_current()
            options = ProgramOptions(
                name=SETTINGS_KERNEL_PATH.name,
                std="c++11",
                arch=f"sm_{self._device.arch}",
            )
            self._program = Program(source, code_type="c++", options=options)
            self._module = self._program.compile("cubin")
            self._kernel = self._module.get_kernel("implicit_sindy_generate_leaf_settings_philox")
        self.jit_ms = (time.perf_counter() - begin) * 1000.0

    def close(self) -> None:
        if self._program is not None:
            program = self._program
            self._program = None
            self._module = None
            self._kernel = None
            program.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def generate(
        self,
        leaf_masks_t,
        leaf_words_t,
        stream,
        *,
        primitive_cols: int,
        seed: int,
        start_cohort_seed: int,
        feature_leaf_probability: float,
        seed_compatible_setting: bool,
    ) -> None:
        if len(leaf_masks_t.shape) == 2:
            num_cohorts = 1
            num_settings = int(leaf_masks_t.shape[0])
            leaf_masks_cohort_stride = 0
        else:
            num_cohorts = int(leaf_masks_t.shape[0])
            num_settings = int(leaf_masks_t.shape[1])
            leaf_masks_cohort_stride = int(leaf_masks_t.stride(0))

        if len(leaf_words_t.shape) == 3:
            leaf_words_cohort_stride = 0
            leaf_words_feature_stride = int(leaf_words_t.stride(1))
        else:
            leaf_words_cohort_stride = int(leaf_words_t.stride(0))
            leaf_words_feature_stride = int(leaf_words_t.stride(2))

        block = 128
        grid_x = (num_settings + block - 1) // block
        grid_y = 32
        grid_z = num_cohorts

        from cuda.core import LaunchConfig, launch

        cuda_stream = self._device.create_stream(PyTorchStreamWrapper(stream))
        try:
            config = LaunchConfig(grid=(grid_x, grid_y, grid_z), block=block)
            launch(
                cuda_stream,
                config,
                self._kernel,
                int(leaf_masks_t.data_ptr()),
                int(leaf_words_t.data_ptr()),
                np.int64(num_cohorts),
                np.int64(num_settings),
                np.int64(leaf_masks_cohort_stride),
                np.int64(leaf_words_cohort_stride),
                np.int64(leaf_words_feature_stride),
                np.int32(primitive_cols),
                np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF),
                np.uint64(int(start_cohort_seed) & 0xFFFFFFFFFFFFFFFF),
                np.float32(feature_leaf_probability),
                np.int32(1 if seed_compatible_setting else 0),
            )
        finally:
            cuda_stream.close()


def children(ast_idx: int) -> tuple[int, int, int]:
    if 8 <= ast_idx < 12:
        local = ast_idx - 8
        return local, local * 2, local * 2 + 1
    if 12 <= ast_idx < 14:
        local = ast_idx - 12
        return 4 + local, 8 + local * 2, 8 + local * 2 + 1
    if ast_idx == 14:
        return 6, 12, 13
    raise ValueError(f"node {ast_idx} does not have children")


def apply_unary_text(op: int, text: str) -> str:
    op = int(isindy.UnaryOp(op))
    if op == isindy.UnaryOp.IDENTITY:
        return text
    if op == isindy.UnaryOp.SQUARE_F32:
        return f"({text})^2"
    if op == isindy.UnaryOp.CUBE_F32:
        return f"({text})^3"
    if op == isindy.UnaryOp.NEG_FTZ_F32:
        return f"-({text})"
    if op == isindy.UnaryOp.SAFE_RCP_F32:
        return f"safe_rcp({text})"
    if op == isindy.UnaryOp.SAFE_SQRT_F32:
        return f"safe_sqrt({text})"
    if op == isindy.UnaryOp.SAFE_RSQRT_F32:
        return f"safe_rsqrt({text})"
    if op == isindy.UnaryOp.SAFE_EX2_F32:
        return f"safe_ex2({text})"
    if op == isindy.UnaryOp.SAFE_EXP_F32:
        return f"safe_exp({text})"
    if op == isindy.UnaryOp.SAFE_LOG2_F32:
        return f"safe_log2({text})"
    if op == isindy.UnaryOp.SAFE_LOG10_F32:
        return f"safe_log10({text})"
    return f"{isindy.UnaryOp(op).name.lower()}({text})"


def apply_binary_text(op: int, lhs: str, rhs: str) -> str:
    op = int(isindy.BinaryOp(op))
    if op == isindy.BinaryOp.KEEP_LEFT:
        return lhs
    if op == isindy.BinaryOp.KEEP_RIGHT:
        return rhs
    if op == isindy.BinaryOp.ADD_FTZ_F32:
        return f"({lhs} + {rhs})"
    if op == isindy.BinaryOp.SUB_FTZ_F32:
        return f"({lhs} - {rhs})"
    if op == isindy.BinaryOp.MUL_FTZ_F32:
        return f"({lhs} * {rhs})"
    if op == isindy.BinaryOp.SAFE_DIV_F32:
        return f"safe_div({lhs}, {rhs})"
    if op == isindy.BinaryOp.DIV_APPROX_FTZ_F32:
        return f"({lhs} / {rhs})"
    if op == isindy.BinaryOp.MIN_FTZ_F32:
        return f"min({lhs}, {rhs})"
    if op == isindy.BinaryOp.MAX_FTZ_F32:
        return f"max({lhs}, {rhs})"
    raise ValueError(f"unsupported binary op {op}")


def feature_texts(
    asts: Sequence[isindy.BinaryAst],
    leaf_masks: np.ndarray,
    leaf_words: np.ndarray,
    setting_idx: int,
) -> list[str]:
    out: list[str] = []
    for feature_idx, ast in enumerate(asts):
        mask = int(np.uint32(leaf_masks[setting_idx, feature_idx]))
        values: list[str] = []
        for leaf_idx in range(LEAVES):
            word = leaf_words[setting_idx, feature_idx, leaf_idx]
            if ((mask >> leaf_idx) & 1) != 0:
                text = PRIMITIVE_NAMES[int(word)] if 0 <= int(word) < len(PRIMITIVE_NAMES) else f"p{int(word)}"
            else:
                text = f"{float(word_to_f32(word)):.6g}"
            values.append(apply_unary_text(ast.unary[leaf_idx], text))
        for ast_idx in range(LEAVES, isindy.NUM_UNARY_OPS):
            binary_idx, lhs_idx, rhs_idx = children(ast_idx)
            text = apply_binary_text(ast.binary[binary_idx], values[lhs_idx], values[rhs_idx])
            values.append(apply_unary_text(ast.unary[ast_idx], text))
        out.append(values[-1])
    return out


def raw_beta_and_intercept(solve) -> tuple[np.ndarray, np.ndarray]:
    beta_standardized = solve.beta_standardized.detach().cpu().numpy()
    x_scale = solve.x_scale.detach().cpu().numpy()
    x_mean = solve.x_mean.detach().cpu().numpy()
    y_mean = solve.y_mean.detach().cpu().numpy()
    beta = beta_standardized / x_scale[None, :, None, :]
    intercept = y_mean[None, :, :] - np.sum(x_mean[None, :, None, :] * beta, axis=-1)
    return beta, intercept


def update_best(
    best: SearchResult,
    *,
    cohort_idx: int,
    settings_np: tuple[np.ndarray, np.ndarray] | None,
    leaf_masks_t,
    leaf_words_t,
    asts: Sequence[isindy.BinaryAst],
    stlsq,
    mse,
    solve_info,
    alphas: np.ndarray,
    thresholds: np.ndarray,
    coefficient_tol: float,
    sparsity_penalty: float,
) -> SearchResult:
    beta, intercept = raw_beta_and_intercept(stlsq)
    active = (np.abs(beta) >= coefficient_tol).sum(axis=(2, 3))
    active += (np.abs(intercept) >= coefficient_tol).sum(axis=2)
    total = mse[:, :, 0]
    invalid = solve_info != 0
    total = np.where(invalid, np.inf, total)
    objective = total + sparsity_penalty * active
    flat_idx = int(np.argmin(objective))
    sweep_idx, setting_idx = np.unravel_index(flat_idx, total.shape)
    mse_value = float(total[sweep_idx, setting_idx])
    objective_value = float(objective[sweep_idx, setting_idx])
    if not np.isfinite(mse_value) or objective_value >= best.objective:
        return best

    if settings_np is None:
        leaf_masks = leaf_masks_t[setting_idx : setting_idx + 1].detach().cpu().numpy()
        leaf_words = leaf_words_t[setting_idx : setting_idx + 1].detach().cpu().numpy()
        text_setting_idx = 0
    else:
        leaf_masks, leaf_words = settings_np
        text_setting_idx = setting_idx

    text = feature_texts(asts, leaf_masks, leaf_words, text_setting_idx)
    return SearchResult(
        mse=mse_value,
        objective=objective_value,
        active_count=int(active[sweep_idx, setting_idx]),
        cohort_idx=cohort_idx,
        setting_idx=int(setting_idx),
        sweep_idx=int(sweep_idx),
        alpha=float(alphas[sweep_idx]),
        threshold=float(thresholds[sweep_idx]),
        beta=beta[sweep_idx, setting_idx, 0].copy(),
        intercept=float(intercept[sweep_idx, setting_idx, 0]),
        feature_text=text,
    )


def print_best(best: SearchResult, *, coefficient_tol: float) -> None:
    if best.beta is None or best.feature_text is None:
        print("no finite candidate found")
        return
    print()
    print("best candidate")
    print(f"  cohort={best.cohort_idx} setting={best.setting_idx} sweep={best.sweep_idx}")
    print(f"  alpha={best.alpha:.3g} threshold={best.threshold:.3g}")
    print(f"  validation_mse={best.mse:.8e}")
    print(f"  objective={best.objective:.8e} active_count={best.active_count}")
    terms: list[str] = []
    if abs(best.intercept) >= coefficient_tol:
        terms.append(f"{best.intercept:+.6g}*1")
    for idx, coef in enumerate(best.beta):
        coef_f = float(coef)
        if abs(coef_f) >= coefficient_tol:
            terms.append(f"{coef_f:+.6g}*{best.feature_text[idx]}")
    rhs = " + ".join(terms) if terms else "0"
    print(f"  u_t = {rhs}")
    print("  active features")
    for idx, coef in enumerate(best.beta):
        if abs(float(coef)) >= coefficient_tol:
            print(f"    f{idx:02d}: coef={float(coef):+.8g} expr={best.feature_text[idx]}")


def compile_batch(
    *,
    batch_idx: int,
    start_cohort: int,
    batch_cohorts: int,
    asts: list[isindy.BinaryAst],
    worker_count: int,
) -> CompiledBatch:
    begin = time.perf_counter()
    gram_kernels = isindy.compile_gram_kernels(
        asts,
        kernels_per_module=isindy.KernelsPerModule.FOUR,
        worker_count=worker_count,
    )
    return CompiledBatch(
        batch_idx=batch_idx,
        start_cohort=start_cohort,
        batch_cohorts=batch_cohorts,
        asts=asts,
        gram_kernels=gram_kernels,
        compile_ms=(time.perf_counter() - begin) * 1000.0,
    )


def make_gpu_phase_events() -> GpuPhaseEvents:
    return GpuPhaseEvents(
        settings=[],
        train_gram=[],
        validation_gram=[],
        solve=[],
        mse=[],
    )


def enqueue_timed_phase(
    torch,
    stream,
    phase_events: GpuPhaseEvents,
    phase_name: str,
    fn,
):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    getattr(phase_events, phase_name).append((start, end))
    start.record(stream)
    value = fn()
    end.record(stream)
    return value


def enqueue_batch(
    *,
    torch,
    settings_generator: CudaPhiloxSettingsGenerator,
    solver,
    compiled: CompiledBatch,
    train_primitive_t,
    train_targets_t,
    validation_primitive_t,
    validation_targets_t,
    alphas_t,
    thresholds_t,
    train_rows: int,
    validation_rows: int,
    settings: int,
    seed: int,
    feature_leaf_probability: float,
    seed_compatible_setting: bool,
) -> PendingBatch:
    stream = torch.cuda.Stream(device=compiled.gram_kernels.device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    phase_events = make_gpu_phase_events()
    cohorts: list[PendingCohort] = []
    begin = time.perf_counter()
    settings_alloc_ms = 0.0
    settings_launch_ms = 0.0

    stream.wait_stream(torch.cuda.current_stream(device=compiled.gram_kernels.device))
    with torch.cuda.device(compiled.gram_kernels.device), torch.cuda.stream(stream):
        start_event.record(stream)
        settings_begin = time.perf_counter()
        leaf_masks_batch_t = torch.empty(
            (compiled.batch_cohorts, settings, FEATURES),
            device=compiled.gram_kernels.device,
            dtype=torch.int32,
        )
        leaf_words_batch_t = torch.empty(
            (compiled.batch_cohorts, settings, FEATURES, LEAVES),
            device=compiled.gram_kernels.device,
            dtype=torch.int32,
        )
        settings_alloc_ms += (time.perf_counter() - settings_begin) * 1000.0

        settings_begin = time.perf_counter()
        enqueue_timed_phase(
            torch,
            stream,
            phase_events,
            "settings",
            lambda: settings_generator.generate(
                leaf_masks_batch_t,
                leaf_words_batch_t,
                stream,
                primitive_cols=len(PRIMITIVE_NAMES),
                seed=seed,
                start_cohort_seed=compiled.start_cohort,
                feature_leaf_probability=feature_leaf_probability,
                seed_compatible_setting=seed_compatible_setting,
            ),
        )
        settings_launch_ms += (time.perf_counter() - settings_begin) * 1000.0

        for local_cohort_idx in range(compiled.batch_cohorts):
            global_cohort_idx = compiled.start_cohort + local_cohort_idx
            asts = compiled.asts[local_cohort_idx * FEATURES : (local_cohort_idx + 1) * FEATURES]
            leaf_masks_t = leaf_masks_batch_t[local_cohort_idx]
            leaf_words_t = leaf_words_batch_t[local_cohort_idx]

            train_raw = enqueue_timed_phase(
                torch,
                stream,
                phase_events,
                "train_gram",
                lambda: compiled.gram_kernels.run_gram(
                    train_primitive_t,
                    train_targets_t,
                    leaf_masks_t,
                    leaf_words_t,
                    cohort_index=local_cohort_idx,
                    stream=stream,
                ),
            )
            validation_raw = enqueue_timed_phase(
                torch,
                stream,
                phase_events,
                "validation_gram",
                lambda: compiled.gram_kernels.run_gram(
                    validation_primitive_t,
                    validation_targets_t,
                    leaf_masks_t,
                    leaf_words_t,
                    cohort_index=local_cohort_idx,
                    stream=stream,
                ),
            )
            stlsq = enqueue_timed_phase(
                torch,
                stream,
                phase_events,
                "solve",
                lambda: solver.solve_stlsq(
                    train_raw,
                    alphas_t,
                    thresholds_t,
                    row_count=train_rows,
                    stream=stream,
                ),
            )
            validation_mse = enqueue_timed_phase(
                torch,
                stream,
                phase_events,
                "mse",
                lambda: solver.score_validation_mse(
                    validation_raw,
                    stlsq,
                    validation_row_count=validation_rows,
                    stream=stream,
                ),
            )
            cohorts.append(
                PendingCohort(
                    global_cohort_idx=global_cohort_idx,
                    asts=asts,
                    settings_np=None,
                    leaf_masks_t=leaf_masks_t,
                    leaf_words_t=leaf_words_t,
                    train_raw=train_raw,
                    validation_raw=validation_raw,
                    stlsq=stlsq,
                    validation_mse=validation_mse,
                )
            )
        end_event.record(stream)

    return PendingBatch(
        compiled=compiled,
        stream=stream,
        start_event=start_event,
        end_event=end_event,
        phase_events=phase_events,
        cohorts=cohorts,
        enqueue_ms=(time.perf_counter() - begin) * 1000.0,
        settings_alloc_ms=settings_alloc_ms,
        settings_launch_ms=settings_launch_ms,
        settings=settings,
        train_rows=train_rows,
        validation_rows=validation_rows,
        num_sweeps=int(alphas_t.shape[0]),
    )


def event_pair_elapsed_ms(pairs: list[tuple[object, object]]) -> float:
    total = 0.0
    for start, end in pairs:
        total += float(start.elapsed_time(end))
    return total


def phase_elapsed_ms(phase_events: GpuPhaseEvents) -> dict[str, float]:
    return {
        "settings": event_pair_elapsed_ms(phase_events.settings),
        "train_gram": event_pair_elapsed_ms(phase_events.train_gram),
        "validation_gram": event_pair_elapsed_ms(phase_events.validation_gram),
        "solve": event_pair_elapsed_ms(phase_events.solve),
        "mse": event_pair_elapsed_ms(phase_events.mse),
    }


def evaluate_pending_batch(
    *,
    pending: PendingBatch,
    best: SearchResult,
    alphas_np: np.ndarray,
    thresholds_np: np.ndarray,
    coefficient_tol: float,
    sparsity_penalty: float,
) -> SearchResult:
    previous_objective = best.objective
    wait_begin = time.perf_counter()
    pending.stream.synchronize()
    wait_ms = (time.perf_counter() - wait_begin) * 1000.0
    gpu_ms = float(pending.start_event.elapsed_time(pending.end_event))
    phases_ms = phase_elapsed_ms(pending.phase_events)
    measured_phase_ms = sum(phases_ms.values())
    other_gpu_ms = max(gpu_ms - measured_phase_ms, 0.0)
    module_count = pending.compiled.gram_kernels.num_cubins
    batch_best_mse = math.inf
    cohorts = pending.compiled.batch_cohorts
    total_settings = cohorts * pending.settings
    train_row_settings = total_settings * pending.train_rows
    validation_row_settings = total_settings * pending.validation_rows
    gram_row_settings = train_row_settings + validation_row_settings
    gpu_seconds = max(gpu_ms * 1.0e-3, 1.0e-9)
    gram_seconds = max((phases_ms["train_gram"] + phases_ms["validation_gram"]) * 1.0e-3, 1.0e-9)
    solve_candidates = total_settings * pending.num_sweeps
    host_launch_ms = max(
        pending.enqueue_ms - pending.settings_alloc_ms - pending.settings_launch_ms,
        0.0,
    )

    try:
        for cohort in pending.cohorts:
            mse_np = cohort.validation_mse.detach().cpu().numpy()
            solve_info_np = cohort.stlsq.solve_info.detach().cpu().numpy()
            best = update_best(
                best,
                cohort_idx=cohort.global_cohort_idx,
                settings_np=cohort.settings_np,
                leaf_masks_t=cohort.leaf_masks_t,
                leaf_words_t=cohort.leaf_words_t,
                asts=cohort.asts,
                stlsq=cohort.stlsq,
                mse=mse_np,
                solve_info=solve_info_np,
                alphas=alphas_np,
                thresholds=thresholds_np,
                coefficient_tol=coefficient_tol,
                sparsity_penalty=sparsity_penalty,
            )
            cohort_total = mse_np[:, :, 0]
            cohort_total = np.where(solve_info_np != 0, np.inf, cohort_total)
            batch_best_mse = min(batch_best_mse, float(np.min(cohort_total)))
    finally:
        pending.compiled.gram_kernels.close()

    print(
        f"  batch {pending.compiled.batch_idx:03d}: "
        f"cohorts={pending.compiled.batch_cohorts} modules={module_count} "
        f"compile_ms={pending.compiled.compile_ms:.2f} enqueue_ms={pending.enqueue_ms:.2f} "
        f"settings_alloc_ms={pending.settings_alloc_ms:.2f} "
        f"settings_launch_ms={pending.settings_launch_ms:.2f} "
        f"gpu_ms={gpu_ms:.2f} sync_wait_ms={wait_ms:.2f} "
        f"batch_best_mse={batch_best_mse:.8e} global_best_mse={best.mse:.8e} "
        f"improved={'yes' if best.objective < previous_objective else 'no'}",
        flush=True,
    )
    print(
        f"    host_phase_ms: "
        f"settings_alloc={pending.settings_alloc_ms:.2f} "
        f"settings_kernel_launch={pending.settings_launch_ms:.2f} "
        f"launch_and_python={host_launch_ms:.2f}",
        flush=True,
    )
    print(
        f"    gpu_phase_ms: "
        f"settings={phases_ms['settings']:.2f} "
        f"train_gram={phases_ms['train_gram']:.2f} "
        f"validation_gram={phases_ms['validation_gram']:.2f} "
        f"solve={phases_ms['solve']:.2f} "
        f"mse={phases_ms['mse']:.2f} "
        f"other_stream={other_gpu_ms:.2f}",
        flush=True,
    )
    print(
        f"    throughput: "
        f"gpu_ms_per_cohort={gpu_ms / max(cohorts, 1):.3f} "
        f"settings_per_s={total_settings / gpu_seconds:.3e} "
        f"gram_row_settings_per_s={gram_row_settings / gram_seconds:.3e} "
        f"end_to_end_row_settings_per_s={gram_row_settings / gpu_seconds:.3e} "
        f"solve_candidates_per_s={solve_candidates / gpu_seconds:.3e}",
        flush=True,
    )
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--cohorts", type=int, default=192)
    parser.add_argument("--settings", type=int, default=2048)
    parser.add_argument("--modules-per-compile", type=int, default=48)
    parser.add_argument("--worker-count", type=int, default=0)
    parser.add_argument("--train-rows", type=int, default=65536)
    parser.add_argument("--validation-rows", type=int, default=32768)
    parser.add_argument("--feature-leaf-probability", type=float, default=0.82)
    parser.add_argument("--coefficient-tol", type=float, default=1.0e-3)
    parser.add_argument("--sparsity-penalty", type=float, default=1.0e-8)
    parser.add_argument("--no-seeded-compatible-setting", action="store_true")
    args = parser.parse_args()

    if args.cohorts <= 0 or args.settings <= 0 or args.modules_per_compile <= 0:
        raise ValueError("cohorts, settings, and modules-per-compile must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")

    rng = np.random.default_rng(args.seed)
    templates = template_asts()
    train_primitive, train_targets = make_burgers_data(seed=args.seed + 101, rows=args.train_rows, nu=args.nu)
    validation_primitive, validation_targets = make_burgers_data(seed=args.seed + 202, rows=args.validation_rows, nu=args.nu)
    train_primitive_t = torch.tensor(train_primitive, device="cuda")
    train_targets_t = torch.tensor(train_targets, device="cuda")
    validation_primitive_t = torch.tensor(validation_primitive, device="cuda")
    validation_targets_t = torch.tensor(validation_targets, device="cuda")
    alphas_np = np.asarray([1.0e-8, 1.0e-8, 1.0e-6, 1.0e-6, 1.0e-5, 1.0e-4], dtype=np.float32)
    thresholds_np = np.asarray([1.0e-4, 1.0e-3, 5.0e-3, 1.0e-2, 5.0e-2, 1.0e-1], dtype=np.float32)
    alphas_t = torch.tensor(alphas_np, device="cuda")
    thresholds_t = torch.tensor(thresholds_np, device="cuda")

    kernels_per_module = int(isindy.KernelsPerModule.FOUR)
    cohorts_per_compile = args.modules_per_compile * kernels_per_module
    print("synthetic Burgers PDE search")
    print(f"  target: u_t = -u*u_x + {args.nu:g}*u_xx")
    print(f"  cohorts={args.cohorts} settings/cohort={args.settings}")
    print(
        f"  modules/compile={args.modules_per_compile} cohorts/compile={cohorts_per_compile} "
        f"worker_count={args.worker_count or (os.cpu_count() or 1)}"
    )
    print(f"  train_rows={args.train_rows} validation_rows={args.validation_rows}")

    best = SearchResult()
    start = time.perf_counter()
    generated = 0
    batch_idx = 0
    pending: PendingBatch | None = None

    def compile_next() -> CompiledBatch | None:
        nonlocal generated, batch_idx
        if generated >= args.cohorts:
            return None
        batch_cohorts = min(cohorts_per_compile, args.cohorts - generated)
        asts: list[isindy.BinaryAst] = []
        for _ in range(batch_cohorts):
            asts.extend(make_template_cohort(rng, templates))
        compiled = compile_batch(
            batch_idx=batch_idx,
            start_cohort=generated,
            batch_cohorts=batch_cohorts,
            asts=asts,
            worker_count=args.worker_count,
        )
        generated += batch_cohorts
        batch_idx += 1
        return compiled

    with ExitStack() as stack:
        settings_generator = stack.enter_context(CudaPhiloxSettingsGenerator(torch, train_primitive_t.device))
        print(f"  settings_jit_ms={settings_generator.jit_ms:.2f}")
        start = time.perf_counter()
        solver = stack.enter_context(isindy.RidgeSolver())
        compiled = compile_next()
        if compiled is None:
            raise RuntimeError("no cohorts to compile")
        pending = enqueue_batch(
            torch=torch,
            settings_generator=settings_generator,
            solver=solver,
            compiled=compiled,
            train_primitive_t=train_primitive_t,
            train_targets_t=train_targets_t,
            validation_primitive_t=validation_primitive_t,
            validation_targets_t=validation_targets_t,
            alphas_t=alphas_t,
            thresholds_t=thresholds_t,
            train_rows=args.train_rows,
            validation_rows=args.validation_rows,
            settings=args.settings,
            seed=args.seed,
            feature_leaf_probability=args.feature_leaf_probability,
            seed_compatible_setting=not args.no_seeded_compatible_setting,
        )
        while generated < args.cohorts:
            compiled = compile_next()
            if compiled is None or pending is None:
                raise RuntimeError("internal pipeline state error")
            best = evaluate_pending_batch(
                pending=pending,
                best=best,
                alphas_np=alphas_np,
                thresholds_np=thresholds_np,
                coefficient_tol=args.coefficient_tol,
                sparsity_penalty=args.sparsity_penalty,
            )
            pending = enqueue_batch(
                torch=torch,
                settings_generator=settings_generator,
                solver=solver,
                compiled=compiled,
                train_primitive_t=train_primitive_t,
                train_targets_t=train_targets_t,
                validation_primitive_t=validation_primitive_t,
                validation_targets_t=validation_targets_t,
                alphas_t=alphas_t,
                thresholds_t=thresholds_t,
                train_rows=args.train_rows,
                validation_rows=args.validation_rows,
                settings=args.settings,
                seed=args.seed,
                feature_leaf_probability=args.feature_leaf_probability,
                seed_compatible_setting=not args.no_seeded_compatible_setting,
            )
        if pending is not None:
            best = evaluate_pending_batch(
                pending=pending,
                best=best,
                alphas_np=alphas_np,
                thresholds_np=thresholds_np,
                coefficient_tol=args.coefficient_tol,
                sparsity_penalty=args.sparsity_penalty,
            )

    print(f"elapsed_s={time.perf_counter() - start:.3f}")
    print_best(best, coefficient_tol=args.coefficient_tol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
