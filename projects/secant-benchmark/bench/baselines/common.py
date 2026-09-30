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
import csv
from pathlib import Path


BASELINE_FIELDS = (
    "system",
    "backend",
    "shape",
    "ast_mode",
    "corpus",
    "corpus_hash",
    "seed",
    "rows",
    "asts",
    "workers",
    "execution_mode",
    "best_seconds",
    "median_seconds",
    "asts_per_second",
    "row_evals_per_second",
    "repeats",
    "checksum",
    "notes",
)


def parse_csv_records(text):
    lines = [line for line in text.splitlines() if line.strip()]
    header = ",".join(BASELINE_FIELDS)
    try:
        header_idx = lines.index(header)
    except ValueError as error:
        raise RuntimeError("benchmark output does not contain the baseline CSV header") from error
    rows = list(csv.DictReader(lines[header_idx:]))
    if not rows:
        raise RuntimeError("benchmark output does not contain baseline CSV rows")
    for row in rows:
        require_record(row)
    return rows


def parse_csv_record(text):
    rows = parse_csv_records(text)
    if len(rows) != 1:
        raise RuntimeError(f"expected one CSV record, found {len(rows)}")
    return rows[0]


def require_record(row):
    missing = [field for field in BASELINE_FIELDS if field not in row]
    if missing:
        raise RuntimeError(f"baseline record is missing field: {missing[0]}")
    if row["shape"] not in ("materialize", "sse"):
        raise RuntimeError(f"invalid baseline shape: {row['shape']}")
    if not row["corpus"] or not row["corpus_hash"]:
        raise RuntimeError("baseline record has no corpus identity")
    if int(row["seed"]) < 0:
        raise RuntimeError("baseline seed must be nonnegative")
    if float(row["median_seconds"]) <= 0.0:
        raise RuntimeError("baseline median must be positive")
    if float(row["row_evals_per_second"]) <= 0.0:
        raise RuntimeError("baseline throughput must be positive")


def write_csv(path, rows):
    path = Path(path)
    rows = list(rows)
    if not rows:
        raise RuntimeError("refusing to write an empty baseline CSV")
    for row in rows:
        require_record(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=BASELINE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"baseline CSV is empty: {path}")
    for row in rows:
        require_record(row)
    return rows


def resume_records(path, key_fields, allowed_values):
    path = Path(path)
    if not path.is_file():
        return [], set()

    rows = read_csv(path)
    allowed = {
        field: {str(value) for value in values}
        for field, values in allowed_values.items()
    }
    keys = set()
    for row in rows:
        for field, values in allowed.items():
            if row[field] not in values:
                raise RuntimeError(
                    f"existing baseline CSV has incompatible {field}: "
                    f"{row[field]}")
        key = tuple(row[field] for field in key_fields)
        if key in keys:
            raise RuntimeError(
                "existing baseline CSV contains a duplicate case: "
                + ",".join(key))
        keys.add(key)
    return rows, keys


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise ValueError("value must be positive")
    return value
