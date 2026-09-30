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

import argparse
import json
import math
import os
import struct
import subprocess
import sys

try:
    import numpy as np
except Exception:
    np = None


FIRST_RNG_STATE = 0x4D595DF4
F32_ONE_OVER_2_24 = struct.unpack("<f", struct.pack("<f", 1.0 / 16777216.0))[0]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def read_f32(path):
    data = read_bytes(path)
    if np is not None:
        return np.frombuffer(data, dtype="<f4").astype(np.float32, copy=True)
    count = len(data) // 4
    return list(struct.unpack("<" + "f" * count, data))


def read_u8(path):
    data = read_bytes(path)
    if np is not None:
        return np.frombuffer(data, dtype=np.uint8).copy()
    return list(data)


def read_u32(path):
    data = read_bytes(path)
    if np is not None:
        return np.frombuffer(data, dtype="<u4").astype(np.uint32, copy=True)
    count = len(data) // 4
    return list(struct.unpack("<" + "I" * count, data))


class Lcg:
    def __init__(self):
        self.state = FIRST_RNG_STATE

    def rand_u32(self):
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state

    def rand_unit_f32(self):
        return f32(float(self.rand_u32() >> 8) * F32_ONE_OVER_2_24)


def make_data(meta):
    rows = meta["num_rows"]
    columns = meta["num_columns"]
    rng = Lcg()
    x = [0.0] * (rows * columns)

    for column in range(columns):
        for row in range(rows):
            unit = rng.rand_unit_f32()
            value = f32(f32(0.30) + f32(f32(0.90) * unit))
            value = f32(value + f32(f32(0.03) * float(column)))
            x[column * rows + row] = value

    target = [0.0] * rows
    for row in range(rows):
        x0 = x[row]
        x1 = x[rows + row]
        target[row] = f32(f32(0.50) + f32(f32(0.25) * x0) + f32(f32(0.125) * x1))

    return x, target


def make_settings(meta):
    masks = [0xFF, 0x55, 0xAA]
    settings = []
    for setting in range(meta["num_settings"]):
        columns = []
        constants = []
        for input_idx in range(8):
            columns.append((setting * 2 + input_idx * 3) & 7)
            constants.append(f32(1.0 + 0.125 * input_idx))
        settings.append((masks[setting], columns, constants))
    return settings


def ast_eval(program_idx, inputs):
    if program_idx == 0:
        return f32(math.sin(inputs[0]))
    if program_idx == 1:
        return f32(math.cos(inputs[1]))
    if program_idx == 2:
        denom = f32(abs(inputs[3]))
        denom = f32(denom + f32(0.25))
        return f32(inputs[2] / denom)
    if program_idx == 3:
        lhs = f32(math.sin(inputs[4]))
        rhs = f32(math.cos(inputs[5]))
        return f32(lhs * rhs)
    if program_idx == 4:
        return f32(math.sqrt(f32(abs(inputs[6]) + f32(0.25))))
    if program_idx == 5:
        return f32(1.0 / math.sqrt(f32(abs(inputs[7]) + f32(0.75))))
    if program_idx == 6:
        return f32(min(f32(max(inputs[0], inputs[1])), inputs[2]))
    if program_idx == 7:
        if hasattr(math, "fma"):
            return f32(math.fma(inputs[0], inputs[1], inputs[2]))
        return f32(f32(inputs[0] * inputs[1]) + inputs[2])
    if program_idx == 8:
        return f32(math.tanh(inputs[3]))
    raise ValueError("bad program index")


def inputs_for_row(x, rows, setting, row):
    mask, columns, constants = setting
    inputs = []
    for input_idx in range(8):
        if (mask & (1 << input_idx)) != 0:
            inputs.append(x[columns[input_idx] * rows + row])
        else:
            inputs.append(constants[input_idx])
    return inputs


def expected_tile_static_mse(meta, x, target, settings):
    rows = meta["num_rows"]
    out = []

    for program_idx in range(meta["num_programs"]):
        for setting in settings:
            sse = f32(0.0)
            for row in range(rows):
                prediction = ast_eval(program_idx, inputs_for_row(x, rows, setting, row))
                error = f32(prediction - target[row])
                sse = f32(f32(error * error) + sse)
            out.append(f32(sse / float(rows)))
    return out


def as_list(values):
    if np is not None and hasattr(values, "tolist"):
        return values.tolist()
    return values


def compare(name, actual, expected, atol, rtol):
    actual_list = as_list(actual)
    if len(actual_list) != len(expected):
        raise AssertionError(f"{name}: length mismatch actual={len(actual_list)} expected={len(expected)}")

    max_abs = -1.0
    max_rel = 0.0
    worst = 0
    for i, (a, e) in enumerate(zip(actual_list, expected)):
        actual_value = float(a)
        expected_value = float(e)
        if not math.isfinite(actual_value) or not math.isfinite(expected_value):
            raise AssertionError(
                f"{name}: non-finite value at {i} actual={actual_value:.9g} expected={expected_value:.9g}"
            )
        abs_err = abs(actual_value - expected_value)
        rel_err = abs_err / max(abs(expected_value), 1.0)
        if abs_err > max_abs:
            max_abs = abs_err
            max_rel = rel_err
            worst = i

    if max_abs > atol and max_rel > rtol:
        raise AssertionError(
            f"{name}: max_abs={max_abs:.9g} max_rel={max_rel:.9g} worst={worst} "
            f"actual={float(actual_list[worst]):.9g} expected={float(expected[worst]):.9g}"
        )

    print(f"{name}: ok max_abs={max_abs:.9g} max_rel={max_rel:.9g} worst={worst}")


def check_compare_contract():
    try:
        compare("nonfinite_contract", [float("nan")], [1.0], 0.0, 0.0)
    except AssertionError:
        return
    raise AssertionError("compare accepted a non-finite result")


def main():
    check_compare_contract()
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip-producer", action="store_true")
    args = parser.parse_args()

    if not args.skip_producer:
        if args.producer is None:
            raise SystemExit("--producer is required unless --skip-producer is used")
        subprocess.run([args.producer, args.out_dir], check=True)

    with open(os.path.join(args.out_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    generated_x, generated_target = make_data(meta)
    settings = make_settings(meta)

    x = read_f32(os.path.join(args.out_dir, "x_f32.bin"))
    target = read_f32(os.path.join(args.out_dir, "target_f32.bin"))
    mse_128_output = read_f32(os.path.join(args.out_dir, "tile_static_mse_128_output_f32.bin"))
    mse_256_output = read_f32(os.path.join(args.out_dir, "tile_static_mse_256_output_f32.bin"))
    mse_64_output = read_f32(os.path.join(args.out_dir, "tile_static_mse_64_output_f32.bin"))
    mse_partial_output = read_f32(os.path.join(args.out_dir, "tile_static_mse_partial_output_f32.bin"))
    leaf_masks = read_u8(os.path.join(args.out_dir, "leaf_masks_u8.bin"))
    leaf_words = read_u32(os.path.join(args.out_dir, "leaf_words_u32.bin"))

    compare("input_rng", x, generated_x, 1e-7, 1e-7)
    compare("target_rng", target, generated_target, 1e-7, 1e-7)

    expected_masks = [setting[0] for setting in settings]
    compare("leaf_masks", leaf_masks, expected_masks, 0.0, 0.0)
    expected_words = []
    for setting in settings:
        mask, columns, constants = setting
        for input_idx in range(8):
            if mask & (1 << input_idx):
                expected_words.append(columns[input_idx])
            else:
                expected_words.append(struct.unpack("<I", struct.pack("<f", constants[input_idx]))[0])
    compare("leaf_words", leaf_words, expected_words, 0.0, 0.0)

    expected_mse = expected_tile_static_mse(meta, generated_x, generated_target, settings)
    compare("tile_static_mse_128", mse_128_output, expected_mse, 5.0e-4, 5.0e-5)
    compare("tile_static_mse_256", mse_256_output, expected_mse, 5.0e-4, 5.0e-5)
    compare("tile_static_mse_64", mse_64_output, expected_mse, 5.0e-4, 5.0e-5)
    compare("tile_static_mse_128_256", mse_256_output, mse_128_output, 5.0e-4, 5.0e-5)
    compare("tile_static_mse_128_64", mse_64_output, mse_128_output, 5.0e-4, 5.0e-5)
    partial_count = meta["active_programs"] * meta["num_settings"]
    compare("tile_static_mse_partial", mse_partial_output, expected_mse[:partial_count], 5.0e-4, 5.0e-5)

    print("numpy:", "yes" if np is not None else "no, using stdlib fallback")


if __name__ == "__main__":
    main()
