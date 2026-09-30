#!/bin/sh
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

set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    printf 'usage: %s SNAPSHOT_PARENT [SECANT_SR_ROOT]\n' "$0" >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
secant_root=$(dirname -- "$script_dir")
secant_sr_root=${2:-"$(dirname -- "$secant_root")/secant-sr"}
snapshot_parent=$1
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
snapshot_root=$snapshot_parent/secant-source-$timestamp

if [ ! -d "$secant_root/.git" ]; then
    printf 'Secant repository not found at %s\n' "$secant_root" >&2
    exit 1
fi

if [ ! -d "$secant_sr_root/.git" ]; then
    printf 'Secant-SR repository not found at %s\n' "$secant_sr_root" >&2
    exit 1
fi

if [ -e "$snapshot_root" ] || [ -L "$snapshot_root" ]; then
    printf 'snapshot destination already exists: %s\n' "$snapshot_root" >&2
    exit 1
fi

snapshot_repo() {
    source_root=$1
    destination_root=$2

    mkdir -p "$destination_root"
    rsync -a \
        --exclude='.git/' \
        --exclude='/scratch' \
        --exclude='/build/' \
        --exclude='/build-*/' \
        --exclude='/build_*/' \
        --exclude='/.venv/' \
        --exclude='/.baseline-*/' \
        --exclude='/fortunes-foundation-game/' \
        --exclude='node_modules/' \
        --exclude='__pycache__/' \
        --exclude='.DS_Store' \
        --exclude='*.pyc' \
        --exclude='*.o' \
        --exclude='*.cubin' \
        --exclude='*.ptx' \
        --exclude='*.sass' \
        --exclude='*.hsaco' \
        --exclude='*.log' \
        "$source_root/" "$destination_root/"
}

snapshot_repo "$secant_root" "$snapshot_root/secant"
snapshot_repo "$secant_sr_root" "$snapshot_root/secant-sr"

history_path=$(find "$snapshot_root" -type d -name .git -print -quit)
if [ -n "$history_path" ]; then
    printf 'snapshot unexpectedly contains Git history: %s\n' "$history_path" >&2
    exit 1
fi

printf 'source snapshot: %s\n' "$snapshot_root"
du -sh "$snapshot_root/secant" "$snapshot_root/secant-sr"
