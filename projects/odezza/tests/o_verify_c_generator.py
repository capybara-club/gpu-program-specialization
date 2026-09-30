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
"""Independently verify C99 runtime-scoring manifests, hashes, and CLI shapes.

The historical Python generator uses a different trajectory ABI. Its source is
not a byte-for-byte reference for the C99 runtime scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

DIMENSIONS = ("state_capacity", "constant_capacity", "system_capacity",
              "shared_patch_capacity", "system_patch_capacity")
DEFAULT = (4, 8, 8, 64, 64)
API_TEST = (4, 4, 8, 384, 384)


def verify(source: str, dimensions: tuple[int, ...]) -> str:
    begin = "/* ODEZZA_MANIFEST_BEGIN\n"
    end = "ODEZZA_MANIFEST_END */\n"
    if not source.startswith(begin) or end not in source:
        raise ValueError("missing manifest")
    raw, payload = source[len(begin):].split(end, 1)
    manifest = json.loads(raw)
    if manifest["schema"] != "odezza.cuda-template" or manifest["generator_abi_version"] != 5:
        raise ValueError("unexpected C99 generator ABI")
    if manifest["data"]["trajectory_layout"] != "runtime_ragged_dense_state":
        raise ValueError("unexpected trajectory ABI")
    shape = manifest["shape"]
    expected = dict(zip(DIMENSIONS, dimensions))
    states, constants, systems, shared, patch = dimensions
    expected.update(input_count=states + constants, output_count=states,
                    arena_instruction_count=systems * patch)
    if shape != expected:
        raise ValueError(f"shape mismatch: {shape} != {expected}")
    for name, value in zip(DIMENSIONS[2:], (systems, shared, patch)):
        if manifest["specialization"][name] != value:
            raise ValueError(f"specialization disagrees on {name}")
    for name, value in zip(DIMENSIONS[:3], dimensions[:3]):
        declaration = f"unsigned int odezza_{name} = {value}u;"
        if declaration not in payload:
            raise ValueError(f"missing capacity symbol: {name}")
    source_hash = manifest.pop("source_sha256")
    template_id = manifest.pop("template_id")
    if hashlib.sha256(payload.encode()).hexdigest() != source_hash:
        raise ValueError("payload hash mismatch")
    declaration = re.match(r'extern "C" __device__ __constant__ unsigned int odezza_template_id\[8\] = \{([^}]+)\};', payload)
    if declaration is None:
        raise ValueError("missing template identity symbol")
    words = [int(word.strip().removesuffix("u"), 16) for word in declaration[1].split(",")]
    if len(words) != 8 or b"".join(word.to_bytes(4, "little") for word in words).hex() != template_id:
        raise ValueError("template identity symbol mismatch")
    normalized = payload[:declaration.start(1)] + ", ".join(["0x00000000u"] * 8) + payload[declaration.end(1):]
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    if hashlib.sha256(canonical + b"\0" + normalized.encode()).hexdigest() != template_id:
        raise ValueError("template identity does not match the printed manifest and payload")
    return template_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("c_source", type=Path, nargs="?")
    parser.add_argument("--generator", type=Path)
    args = parser.parse_args()
    if args.c_source is None and args.generator is None:
        parser.error("supply a generated C99 source or --generator executable")
    if args.c_source:
        source = args.c_source.read_text()
        verify(source, API_TEST)
        # Neither semantic nor payload tampering may preserve a valid identity.
        for modified in (source.replace("grid-y selects system", "grid-y selects candidate"), source + "\n"):
            try:
                verify(modified, API_TEST)
            except ValueError:
                pass
            else:
                raise AssertionError("tampered artifact was accepted")
    if args.generator:
        executable = str(args.generator.resolve())
        default = subprocess.check_output([executable], text=True)
        verify(default, DEFAULT)
        ids = set()
        for dimensions in (DEFAULT, API_TEST, (1, 0, 1, 64, 128), (8, 8, 1, 64, 128), (6, 8, 4, 128, 256)):
            command = [executable]
            for name, value in zip(DIMENSIONS, dimensions):
                command.extend(["--" + name.replace("_", "-"), str(value)])
            actual = subprocess.check_output(command, text=True)
            if actual != subprocess.check_output(command, text=True):
                raise AssertionError("generation is not deterministic")
            ids.add(verify(actual, dimensions))
            if dimensions == DEFAULT and actual != default:
                raise AssertionError("CLI defaults differ from explicit defaults")
            if dimensions == API_TEST and args.c_source and actual != source:
                raise AssertionError("CLI and C API artifacts differ for the same shape")
        if len(ids) != 5:
            raise AssertionError("different shapes share an identity")
    print("C99 CUDA artifact: manifest identity, payload hash, and requested shapes verified")


if __name__ == "__main__":
    main()
