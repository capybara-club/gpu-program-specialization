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

import time
from pathlib import Path
from typing import Sequence

import numpy as np


FEATURES = 32
LEAVES = 8
SETTINGS_KERNEL_PATH = Path(__file__).with_name("map_elites_settings_philox.cu")


def f32_word(value: float) -> np.int32:
    return np.asarray([value], dtype=np.float32).view(np.int32)[0]


def word_to_f32(value: np.int32 | int) -> np.float32:
    return np.asarray([value], dtype=np.int32).view(np.float32)[0]


def active_indices(mask: int) -> list[int]:
    return [idx for idx in range(FEATURES) if ((int(mask) >> idx) & 1) != 0]


class PyTorchStreamWrapper:
    def __init__(
        self,
        pt_stream,
    ) -> None:
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        return (0, self.pt_stream.cuda_stream)


class CudaMapElitesSettingsGenerator:
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
            self._kernel = self._module.get_kernel("implicit_sindy_map_elites_generate_settings_philox")
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
        inherited_masks_t,
        inherited_words_t,
        inherited_terms_t,
        stream,
        *,
        primitive_cols: int,
        seed: int,
        generation: int,
        start_cohort_seed: int,
        feature_leaf_probability: float,
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
        grid_y = FEATURES
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
                int(inherited_masks_t.data_ptr()),
                int(inherited_words_t.data_ptr()),
                int(inherited_terms_t.data_ptr()),
                np.int64(num_cohorts),
                np.int64(num_settings),
                np.int64(leaf_masks_cohort_stride),
                np.int64(leaf_words_cohort_stride),
                np.int64(leaf_words_feature_stride),
                np.int64(inherited_masks_t.stride(0)),
                np.int64(inherited_words_t.stride(0)),
                np.int64(inherited_words_t.stride(1)),
                np.int32(primitive_cols),
                np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF),
                np.uint64(int(generation) & 0xFFFFFFFFFFFFFFFF),
                np.uint64(int(start_cohort_seed) & 0xFFFFFFFFFFFFFFFF),
                np.float32(feature_leaf_probability),
            )
        finally:
            cuda_stream.close()


def inherited_settings_tensors(
    torch,
    candidates: Sequence[object],
    device,
) -> tuple[object, object, object]:
    masks = np.zeros((len(candidates), FEATURES), dtype=np.int32)
    words = np.zeros((len(candidates), FEATURES, LEAVES), dtype=np.int32)
    terms = np.zeros((len(candidates),), dtype=np.int32)
    for idx, candidate in enumerate(candidates):
        inherited_terms = int(candidate.inherited_terms)
        terms[idx] = np.int32(inherited_terms)
        if inherited_terms > 0:
            masks[idx, :inherited_terms] = candidate.inherited_masks[:inherited_terms]
            words[idx, :inherited_terms, :] = candidate.inherited_words[:inherited_terms, :]
    return (
        torch.tensor(masks, device=device, dtype=torch.int32),
        torch.tensor(words, device=device, dtype=torch.int32),
        torch.tensor(terms, device=device, dtype=torch.int32),
    )
