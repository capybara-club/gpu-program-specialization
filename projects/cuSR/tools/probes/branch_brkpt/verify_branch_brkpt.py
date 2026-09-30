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
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_ARCHES = (80, 86, 89, 90, 100, 120)
BRANCH_PROBES = {
    "cusr_branch_skip_004": 4,
    "cusr_branch_skip_016": 16,
    "cusr_branch_skip_064": 64,
}

BRA_OPCODE = 0x7947
BPT_WORD0 = 0x000000040000795C
MARKER_BITS = 0x7FC0FFEE


def run(cmd):
    subprocess.run(cmd, check=True)


def branch_word0_sm8(target_delta_instructions):
    branch_target_field = 16 * target_delta_instructions - 16
    return (branch_target_field << 32) | BRA_OPCODE


def branch_word0_sm90_plus(target_delta_instructions):
    branch_target_field = 4 * target_delta_instructions - 4
    return (
        ((branch_target_field & 0xFF) << 16)
        | ((branch_target_field & ~0xFF) << 26)
        | BRA_OPCODE
    )


def expected_branch_word0(arch, target_delta_instructions):
    if 80 <= arch < 90:
        return branch_word0_sm8(target_delta_instructions)
    if arch >= 90:
        return branch_word0_sm90_plus(target_delta_instructions)
    raise ValueError(f"unsupported arch sm_{arch}")


def branch_target_mask(arch):
    if 80 <= arch < 90:
        return 0xFFFFFFFF00000000
    if arch >= 90:
        return 0xFFFFFFFFFFFF0000
    raise ValueError(f"unsupported arch sm_{arch}")


def parse_sass(text):
    functions = {}
    current_function = None
    pending = None

    function_re = re.compile(r"^\s*Function\s*:\s*(\S+)")
    section_re = re.compile(r"^\s*\.section\s+\.text\.([^,\s]+)")
    label_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$")
    instruction_re = re.compile(
        r"/\*([0-9a-fA-F]+)\*/\s*(.*?)\s*/\*\s*(0x[0-9a-fA-F]+)\s*\*/"
    )
    word_re = re.compile(r"^\s*/\*\s*(0x[0-9a-fA-F]+)\s*\*/")

    for line in text.splitlines():
        function_match = function_re.search(line)
        if function_match:
            current_function = function_match.group(1)
            functions.setdefault(current_function, [])
            pending = None
            continue

        section_match = section_re.search(line)
        if section_match:
            current_function = section_match.group(1)
            functions.setdefault(current_function, [])
            pending = None
            continue

        label_match = label_re.search(line)
        if label_match and label_match.group(1).startswith("cusr_"):
            current_function = label_match.group(1)
            functions.setdefault(current_function, [])
            pending = None
            continue

        instruction_match = instruction_re.search(line)
        if current_function and instruction_match:
            address = int(instruction_match.group(1), 16)
            text_part = instruction_match.group(2).strip()
            word0 = int(instruction_match.group(3), 16)
            pending = {
                "address": address,
                "text": text_part,
                "word0": word0,
                "word1": None,
            }
            functions[current_function].append(pending)
            continue

        word_match = word_re.search(line)
        if pending is not None and word_match:
            pending["word1"] = int(word_match.group(1), 16)
            pending = None

    return functions


def is_bpt(instruction):
    return instruction["word0"] == BPT_WORD0 or "BPT.TRAP" in instruction["text"]


def is_branch(instruction):
    return (instruction["word0"] & 0xFFFF) == BRA_OPCODE or "BRA" in instruction["text"]


def marker_immediate(instruction):
    return (instruction["word0"] >> 32) & 0xFFFFFFFF


def find_function(functions, name):
    if name in functions:
        return functions[name]
    for function_name, instructions in functions.items():
        if function_name.endswith(name):
            return instructions
    raise KeyError(name)


def verify_branch_probe(arch, functions, name, expected_bpt_count):
    instructions = find_function(functions, name)
    branch_index = None
    for i, ins in enumerate(instructions):
        if not is_branch(ins):
            continue

        bpt_count_for_branch = 0
        for next_ins in instructions[i + 1 :]:
            if is_bpt(next_ins):
                bpt_count_for_branch += 1
                continue
            break

        if bpt_count_for_branch == expected_bpt_count:
            branch_index = i
            break

    if branch_index is None:
        raise RuntimeError(f"sm_{arch} {name}: unconditional branch with {expected_bpt_count} BPTs not found")

    bpt_count = 0
    for ins in instructions[branch_index + 1 :]:
        if is_bpt(ins):
            bpt_count += 1
            continue
        break

    if bpt_count != expected_bpt_count:
        raise RuntimeError(
            f"sm_{arch} {name}: expected {expected_bpt_count} BPTs after BRA, found {bpt_count}"
        )

    branch = instructions[branch_index]
    target_delta = bpt_count + 1
    expected_word0 = expected_branch_word0(arch, target_delta)
    target_mask = branch_target_mask(arch)
    if (branch["word0"] & target_mask) != (expected_word0 & target_mask):
        sm8_word0 = branch_word0_sm8(target_delta)
        sm90_word0 = branch_word0_sm90_plus(target_delta)
        raise RuntimeError(
            f"sm_{arch} {name}: branch target-field mismatch "
            f"actual=0x{branch['word0']:016x} expected=0x{expected_word0:016x} "
            f"sm8=0x{sm8_word0:016x} sm90+=0x{sm90_word0:016x}"
        )

    for offset in range(bpt_count):
        bpt = instructions[branch_index + 1 + offset]
        if bpt["word0"] != BPT_WORD0:
            raise RuntimeError(
                f"sm_{arch} {name}: BPT word mismatch at +{offset}: 0x{bpt['word0']:016x}"
            )

    return branch, bpt_count


def verify_anchor_probe(arch, functions):
    instructions = find_function(functions, "cusr_brkpt_site_anchor")
    marker_index = next(
        (i for i, ins in enumerate(instructions) if marker_immediate(ins) == MARKER_BITS),
        None,
    )
    if marker_index is None:
        raise RuntimeError(f"sm_{arch} cusr_brkpt_site_anchor: marker FADD not found")

    max_bpt_run = 0
    current_bpt_run = 0
    bpt_words = set()
    for ins in instructions:
        if is_bpt(ins):
            current_bpt_run += 1
            bpt_words.add(ins["word0"])
            if current_bpt_run > max_bpt_run:
                max_bpt_run = current_bpt_run
        else:
            current_bpt_run = 0

    if max_bpt_run < 16:
        raise RuntimeError(
            f"sm_{arch} cusr_brkpt_site_anchor: expected a contiguous BPT pad of at least 16, found {max_bpt_run}"
        )

    if bpt_words != {BPT_WORD0}:
        formatted = ", ".join(f"0x{word:016x}" for word in sorted(bpt_words))
        raise RuntimeError(f"sm_{arch} cusr_brkpt_site_anchor: unexpected BPT words {formatted}")

    return instructions[marker_index], max_bpt_run


def compile_and_disassemble(root, out_dir, arch):
    nvcc = shutil.which("nvcc")
    nvdisasm = shutil.which("nvdisasm")
    if nvcc is None or nvdisasm is None:
        raise RuntimeError("nvcc and nvdisasm must be on PATH")

    source = root / "branch_brkpt_probe.cu"
    arch_dir = out_dir / f"sm_{arch}"
    arch_dir.mkdir(parents=True, exist_ok=True)
    cubin = arch_dir / "branch_brkpt_probe.cubin"
    sass = arch_dir / "branch_brkpt_probe.sass"

    run([
        nvcc,
        "-cubin",
        f"-arch=sm_{arch}",
        "-O3",
        "-Xptxas=--opt-level=1",
        str(source),
        "-o",
        str(cubin),
    ])

    with sass.open("w", encoding="utf-8") as stream:
        subprocess.run([nvdisasm, "-c", "-hex", str(cubin)], check=True, stdout=stream)

    return cubin, sass


def verify_arch(root, out_dir, arch):
    cubin, sass = compile_and_disassemble(root, out_dir, arch)
    functions = parse_sass(sass.read_text(encoding="utf-8"))

    branch_summaries = []
    for name, bpt_count in BRANCH_PROBES.items():
        branch, count = verify_branch_probe(arch, functions, name, bpt_count)
        branch_summaries.append(
            f"{name}:{count}BPT word0=0x{branch['word0']:016x} word1=0x{branch['word1']:016x}"
        )

    marker, marker_bpts = verify_anchor_probe(arch, functions)
    return {
        "arch": arch,
        "cubin": cubin,
        "sass": sass,
        "branches": branch_summaries,
        "marker": f"marker_word0=0x{marker['word0']:016x} marker_word1=0x{marker['word1']:016x} max_bpt_run={marker_bpts}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", action="append", type=int, dest="arches")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = args.out if args.out is not None else root / "out"
    arches = tuple(args.arches) if args.arches else DEFAULT_ARCHES

    ok = True
    for arch in arches:
        try:
            summary = verify_arch(root, out_dir, arch)
            print(f"sm_{arch}: PASS")
            for branch in summary["branches"]:
                print(f"  {branch}")
            print(f"  {summary['marker']}")
            print(f"  sass={summary['sass']}")
        except subprocess.CalledProcessError as exc:
            ok = False
            print(f"sm_{arch}: FAIL command exited {exc.returncode}", file=sys.stderr)
        except Exception as exc:
            ok = False
            print(f"sm_{arch}: FAIL {exc}", file=sys.stderr)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
