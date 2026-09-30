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
"""Prepare official SRBench/PMLB rows for the secant-sr C99 runner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from srbench_v2 import (
    SRBENCH_V2_BLACKBOX_PROTOCOL,
    SRBENCH_V2_PROTOCOL,
    SRBENCH_V2_TRAIN_ROWS,
)


MAGIC = b"SECSRDS\0"
VERSION = 1
HEADER = struct.Struct("<8sIIQQ")


@dataclass(frozen=True)
class PreparedDataset:
    path: Path
    source_path: Path
    source_sha256: str
    num_inputs: int
    num_train_rows: int
    num_validation_rows: int
    split_seed: int
    scale_x: bool
    scale_y: bool
    target_noise: float
    protocol: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_array(output, values: np.ndarray) -> None:
    output.write(np.asarray(values, dtype="<f4", order="C").tobytes(order="C"))


def _dataset_write(
    source_path: Path,
    output_dir: Path,
    source_sha256: str,
    split_seed: int,
    scale_x: bool,
    scale_y: bool,
    target_noise: float,
    protocol: str,
    train_x,
    train_y: np.ndarray,
    validation_x,
    validation_y: np.ndarray,
) -> PreparedDataset:
    train_input = np.asarray(train_x, dtype=np.float32).T.copy(order="C")
    train_target = np.asarray(train_y, dtype=np.float32)
    validation_input = np.asarray(validation_x, dtype=np.float32).T.copy(order="C")
    validation_target = np.asarray(validation_y, dtype=np.float32)
    for name, values in (
        ("train input", train_input),
        ("train target", train_target),
        ("validation input", validation_input),
        ("validation target", validation_target),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{source_path} contains a non-finite {name} value after f32 conversion")

    num_inputs = train_input.shape[0]
    num_train_rows = train_input.shape[1]
    num_validation_rows = validation_input.shape[1]
    protocol_suffix = "" if protocol == "secant" else f"_{protocol}"
    noise_suffix = "" if target_noise == 0.0 and protocol == "secant" else f"_noise{target_noise:.17g}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{source_path.name.removesuffix('.tsv.gz')}_v{VERSION}_seed{split_seed}_"
        f"train{num_train_rows}_validation{num_validation_rows}_sx{int(scale_x)}_sy{int(scale_y)}"
        f"{protocol_suffix}{noise_suffix}_{source_sha256[:16]}.secsr"
    )
    if not output_path.exists():
        temporary_path = output_path.with_suffix(".tmp")
        with temporary_path.open("wb") as output:
            output.write(HEADER.pack(MAGIC, VERSION, num_inputs, num_train_rows, num_validation_rows))
            _write_array(output, train_input)
            _write_array(output, train_target)
            _write_array(output, validation_input)
            _write_array(output, validation_target)
        temporary_path.replace(output_path)

    return PreparedDataset(
        path=output_path,
        source_path=source_path,
        source_sha256=source_sha256,
        num_inputs=num_inputs,
        num_train_rows=num_train_rows,
        num_validation_rows=num_validation_rows,
        split_seed=split_seed,
        scale_x=scale_x,
        scale_y=scale_y,
        target_noise=target_noise,
        protocol=protocol,
    )


def prepare_dataset(
    source_path: Path,
    output_dir: Path,
    split_seed: int,
    max_train_rows: int,
    max_validation_rows: int,
    scale_x: bool = False,
    scale_y: bool = False,
) -> PreparedDataset:
    """Reproduce SRBench ingestion/splitting and emit Secant's f32 transport."""
    if split_seed < 0 or split_seed > np.iinfo(np.uint32).max:
        raise ValueError("split_seed must fit in uint32 for scikit-learn RandomState")
    if max_train_rows < 0 or max_validation_rows < 0:
        raise ValueError("row limits must be nonnegative; zero means all rows")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    source_sha256 = _sha256(source_path)
    frame = pd.read_csv(source_path, sep="\t", compression="gzip")
    frame = frame.rename(columns={name: name.strip().replace(".", "_") for name in frame.columns})
    if "target" not in frame.columns:
        raise ValueError(f"{source_path} has no target column")

    feature_names = [name for name in frame.columns if name != "target"]
    features = frame[feature_names]
    target = frame["target"].to_numpy()

    # This matches experiment/evaluate_model.py in the official SRBench source.
    np.random.seed(split_seed)
    train_x, validation_x, train_y, validation_y = train_test_split(
        features,
        target,
        train_size=0.75,
        test_size=0.25,
        random_state=split_seed,
    )
    if max_train_rows > 0 and len(train_y) > max_train_rows:
        sample_idx = np.random.choice(np.arange(len(train_x)), size=max_train_rows, replace=False)
        train_x = train_x.iloc[sample_idx]
        train_y = train_y[sample_idx]
    if max_validation_rows > 0 and len(validation_y) > max_validation_rows:
        validation_x = validation_x.iloc[:max_validation_rows]
        validation_y = validation_y[:max_validation_rows]
    if scale_x:
        scaler_x = StandardScaler()
        train_x = scaler_x.fit_transform(train_x)
        validation_x = scaler_x.transform(validation_x)
    if scale_y:
        scaler_y = StandardScaler()
        train_y = scaler_y.fit_transform(np.asarray(train_y).reshape(-1, 1)).ravel()
        validation_y = scaler_y.transform(np.asarray(validation_y).reshape(-1, 1)).ravel()

    return _dataset_write(
        source_path, output_dir, source_sha256, split_seed, scale_x, scale_y, 0.0, "secant",
        train_x, train_y, validation_x, validation_y,
    )


def prepare_srbench_v2_groundtruth_dataset(
    source_path: Path,
    output_dir: Path,
    split_seed: int,
    target_noise: float,
) -> PreparedDataset:
    """Reproduce SRBench v2.0's `-sym_data` data path exactly."""
    if split_seed < 0 or split_seed > np.iinfo(np.uint32).max:
        raise ValueError("split_seed must fit in uint32 for scikit-learn RandomState")
    if target_noise < 0.0 or not np.isfinite(target_noise):
        raise ValueError("target_noise must be finite and nonnegative")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    source_sha256 = _sha256(source_path)
    frame = pd.read_csv(source_path, sep="\t", compression="gzip")
    frame = frame.rename(columns={name: name.strip().replace(".", "_") for name in frame.columns})
    if "target" not in frame.columns:
        raise ValueError(f"{source_path} has no target column")

    feature_names = [name for name in frame.columns if name != "target"]
    features = frame[feature_names].to_numpy()
    target = frame["target"].to_numpy()

    # This intentionally follows the v2.0 implementation, including sampling
    # with replacement and testing the complete dataset size before sampling.
    np.random.seed(split_seed)
    train_x, validation_x, train_y, validation_y = train_test_split(
        features,
        target,
        train_size=0.75,
        test_size=0.25,
        random_state=split_seed,
    )
    if len(target) > SRBENCH_V2_TRAIN_ROWS:
        sample_idx = np.random.choice(np.arange(len(train_x)), size=SRBENCH_V2_TRAIN_ROWS)
        train_x = train_x[sample_idx]
        train_y = train_y[sample_idx]
    if target_noise > 0.0:
        train_y = train_y + np.random.normal(
            0.0,
            target_noise * np.sqrt(np.mean(np.square(train_y))),
            size=len(train_y),
        )

    return _dataset_write(
        source_path, output_dir, source_sha256, split_seed, False, False, target_noise,
        SRBENCH_V2_PROTOCOL, train_x, train_y, validation_x, validation_y,
    )


def prepare_srbench_v2_blackbox_dataset(
    source_path: Path,
    output_dir: Path,
    split_seed: int,
) -> PreparedDataset:
    """Reproduce SRBench v2.0's scaled black-box data path.

    SRBench inverse-transforms predictions before reporting scores. Secant
    instead transforms the held-out target with the training-target scaler and
    scores in standardized coordinates. R-squared is invariant to this affine
    transformation, so this preserves the published black-box ranking metric
    without exposing held-out targets during search.
    """
    if split_seed < 0 or split_seed > np.iinfo(np.uint32).max:
        raise ValueError("split_seed must fit in uint32 for scikit-learn RandomState")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    source_sha256 = _sha256(source_path)
    frame = pd.read_csv(source_path, sep="\t", compression="gzip")
    frame = frame.rename(columns={name: name.strip().replace(".", "_") for name in frame.columns})
    if "target" not in frame.columns:
        raise ValueError(f"{source_path} has no target column")

    feature_names = [name for name in frame.columns if name != "target"]
    features = frame[feature_names].to_numpy()
    target = frame["target"].to_numpy()

    np.random.seed(split_seed)
    train_x, validation_x, train_y, validation_y = train_test_split(
        features,
        target,
        train_size=0.75,
        test_size=0.25,
        random_state=split_seed,
    )
    if len(target) > SRBENCH_V2_TRAIN_ROWS:
        # np.random.choice defaults to replacement, matching SRBench v2.0.
        sample_idx = np.random.choice(np.arange(len(train_x)), size=SRBENCH_V2_TRAIN_ROWS)
        train_x = train_x[sample_idx]
        train_y = train_y[sample_idx]

    scaler_x = StandardScaler()
    train_x = scaler_x.fit_transform(train_x)
    validation_x = scaler_x.transform(validation_x)
    scaler_y = StandardScaler()
    train_y = scaler_y.fit_transform(np.asarray(train_y).reshape(-1, 1)).ravel()
    validation_y = scaler_y.transform(np.asarray(validation_y).reshape(-1, 1)).ravel()

    return _dataset_write(
        source_path,
        output_dir,
        source_sha256,
        split_seed,
        True,
        True,
        0.0,
        SRBENCH_V2_BLACKBOX_PROTOCOL,
        train_x,
        train_y,
        validation_x,
        validation_y,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("scratch/srbench-prepared"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-rows", type=int, default=0, help="zero keeps all SRBench training rows")
    parser.add_argument("--validation-rows", type=int, default=0, help="zero keeps the complete test split")
    parser.add_argument("--scale-x", action="store_true")
    parser.add_argument("--scale-y", action="store_true")
    args = parser.parse_args()
    prepared = prepare_dataset(
        args.source,
        args.output_dir,
        args.seed,
        args.train_rows,
        args.validation_rows,
        args.scale_x,
        args.scale_y,
    )
    print(
        f"path={prepared.path} source={prepared.source_path} source_sha256={prepared.source_sha256} "
        f"inputs={prepared.num_inputs} train_rows={prepared.num_train_rows} "
        f"validation_rows={prepared.num_validation_rows} split_seed={prepared.split_seed}"
        f" scale_x={int(prepared.scale_x)} scale_y={int(prepared.scale_y)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
