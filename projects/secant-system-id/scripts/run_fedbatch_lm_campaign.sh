#!/usr/bin/env bash
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

set -u

if [[ $# -ne 7 ]]; then
    echo "usage: $0 HOST_LABEL ARCH SCORE_MODE BUILD_DIRECTORY GENERATIONS SEEDS_CSV EXPECTED_COMPLETION_ET" >&2
    exit 2
fi

host_label=$1
architecture=$2
requested_score_mode=$3
score_mode=$requested_score_mode
build_directory=$4
generations=$5
seeds_csv=$6
expected_completion_et=$7

population=${SECANT_POPULATION:-1024}
settings=${SECANT_SETTINGS:-512}
checkpoint_stride=${SECANT_CHECKPOINT_STRIDE:-25}
lm_promotion_interval=${SECANT_LM_PROMOTION_INTERVAL:-25}
lm_promotion_count=${SECANT_LM_PROMOTION_COUNT:-4}
lm_settings=${SECANT_LM_SETTINGS:-8192}
lm_starts=${SECANT_LM_STARTS:-4}

case "$score_mode" in
    fused)
        score_arguments=()
        ;;
    materialized)
        score_arguments=(--full-mse-gp --materialized-gp)
        ;;
    *)
        echo "score mode must be fused or materialized" >&2
        exit 2
        ;;
esac

repository=$(cd "$(dirname "$0")/.." && pwd)
notify_command=${SECANT_NOTIFY_COMMAND:-/home/cdurham/.local/bin/secant-notify}
progress_path="$repository/$build_directory/progress.json"
report_path="$repository/$build_directory/recovery_report.json"
log_path="$repository/$build_directory/campaign.log"
launch_path="$repository/$build_directory/campaign_launch.json"
mkdir -p "$(dirname "$log_path")"

notify() {
    if [[ -x "$notify_command" ]]; then
        "$notify_command" "$1" || echo "warning: Telegram notification failed" >&2
    fi
}

progress_summary() {
    python3 -c 'import json,math,sys; p=json.load(open(sys.argv[1])); latest=p.get("latest"); searched=len(p.get("searched_seeds", [])); d=(latest or {}).get("lm_diagnostics", {}); s=d.get("specialization", {}); o=d.get("optimization", {}); f=d.get("pressure_fallback", {}); print("" if not latest or not isinstance(latest.get("seed"), int) or not math.isfinite(float(latest.get("best_mse", float("nan")))) or not math.isfinite(float(latest.get("search_seconds", float("nan")))) else "{}|{}|{:.9g}|{:.1f}|{}|{}|{}|{}|{}".format(searched, latest["seed"], latest["best_mse"], latest["search_seconds"], s.get("rejected", 0), s.get("attempted", 0), o.get("accepted_candidates", 0), o.get("evaluated_candidates", 0), f.get("evaluated_candidates_using_fallback", 0)))' "$progress_path"
}

final_summary() {
    python3 -c 'import json,statistics,sys; r=json.load(open(sys.argv[1])); rows=r["seed_results"]; best=min(rows,key=lambda x:x["best"]["gpu_training_mse"]); d=r.get("trajectory_lm_diagnostics", {}); s=d.get("specialization", {}); o=d.get("optimization", {}); f=d.get("pressure_fallback", {}); print("best training MSE {:.9g}; best held-out MSE {:.9g}; median search {:.1f}s; LM accepted {}/{} evaluated; rejected {}/{} selected; pressure fallback {}".format(best["best"]["gpu_training_mse"], best["best"]["held_out_trajectory_mse"], statistics.median(x["search"]["wall_seconds"] for x in rows), o.get("accepted_candidates", 0), o.get("evaluated_candidates", 0), s.get("rejected", 0), s.get("attempted", 0), f.get("evaluated_candidates_using_fallback", 0)))' "$report_path"
}

started_et=$(TZ=America/Detroit date '+%Y-%m-%d %I:%M:%S %p %Z')
repository_commit=$(git -C "$repository" rev-parse HEAD) || exit 1
python3 - "$launch_path" "$repository_commit" "$started_et" "$expected_completion_et" "$host_label" "$architecture" "$requested_score_mode" "$score_mode" "$build_directory" "$generations" "$seeds_csv" "$population" "$settings" "$checkpoint_stride" "$lm_promotion_interval" "$lm_promotion_count" "$lm_settings" "$lm_starts" <<'PY' || exit 1
import json
import socket
import sys
from pathlib import Path

(
    launch_path,
    repository_commit,
    started_et,
    expected_completion_et,
    host_label,
    architecture,
    requested_score_mode,
    effective_score_mode,
    build_directory,
    generations,
    seeds_csv,
    population,
    settings,
    checkpoint_stride,
    lm_promotion_interval,
    lm_promotion_count,
    lm_settings,
    lm_starts,
) = sys.argv[1:]
record = {
    "schema": "secant.system_id.campaign_launch.v1",
    "status": "launched",
    "hostname": socket.gethostname(),
    "host_label": host_label,
    "repository_commit": repository_commit,
    "started_et": started_et,
    "expected_completion_et": expected_completion_et,
    "architecture": architecture,
    "requested_score_mode": requested_score_mode,
    "effective_score_mode": effective_score_mode,
    "build_directory": build_directory,
    "generations": int(generations),
    "seeds": [int(value) for value in seeds_csv.split(",")],
    "population": int(population),
    "settings_per_genome": int(settings),
    "checkpoint_stride": int(checkpoint_stride),
    "lm": {
        "promotion_interval": int(lm_promotion_interval),
        "promotion_count": int(lm_promotion_count),
        "selection_mode": "random",
        "selection_pool": 32,
        "random_trigger_probability": 0.14,
        "binding_settings_per_candidate": int(lm_settings),
        "starts_per_setting": int(lm_starts),
        "loader_workers": 1,
        "maximum_loaded_modules": 1,
    },
    "gp_loader_workers": 1,
    "gp_maximum_loaded_modules": 2,
    "training_design": "product_paired16",
    "trajectories": 16,
    "max_depth": 8,
    "parsimony": 0.0001,
}
Path(launch_path).write_text(json.dumps(record, indent=2) + "\n")
PY
notify "$host_label Secant System ID random-LM campaign started at $started_et. Expected completion: $expected_completion_et. Requested score mode: $requested_score_mode; effective score mode: $score_mode. Population: $population; settings: $settings; generations: $generations; seeds: $seeds_csv; LM interval: $lm_promotion_interval; trigger probability: 14%."

cd "$repository" || exit 1
PYTHONPATH=src python3 -m secant_system_id.cli fedbatch-c99-recovery-run \
    --arch "$architecture" \
    --trajectories 16 \
    --training-design product_paired16 \
    --population "$population" \
    --settings "$settings" \
    --generations "$generations" \
    --seeds "$seeds_csv" \
    --checkpoint-stride "$checkpoint_stride" \
    --workers 1 \
    --loaded-modules 2 \
    --max-depth 8 \
    --parsimony 0.0001 \
    --lm-promotion-interval "$lm_promotion_interval" \
    --lm-promotion-count "$lm_promotion_count" \
    --lm-selection-mode random \
    --lm-selection-pool 32 \
    --lm-random-trigger-probability 0.14 \
    --lm-settings "$lm_settings" \
    --lm-starts "$lm_starts" \
    --lm-workers 1 \
    --lm-loaded-modules 1 \
    --build-directory "$build_directory" \
    "${score_arguments[@]}" > "$log_path" 2>&1 &
campaign_pid=$!

last_completed=0
while kill -0 "$campaign_pid" 2>/dev/null; do
    if [[ -f "$progress_path" ]]; then
        summary=$(progress_summary 2>/dev/null || true)
        if [[ -n "$summary" ]]; then
            IFS='|' read -r completed seed best_mse search_seconds rejected attempted accepted evaluated fallbacks <<< "$summary"
            if [[ "$completed" -gt "$last_completed" ]]; then
                notify "$host_label finished search seed $seed ($completed total): provisional best training MSE $best_mse in ${search_seconds}s; materialized correctness replay is still pending. LM accepted $accepted/$evaluated evaluated; specialization rejected $rejected/$attempted selected; pressure fallback $fallbacks."
                last_completed=$completed
            fi
        fi
    fi
    sleep 30
done

wait "$campaign_pid"
status=$?
finished_et=$(TZ=America/Detroit date '+%Y-%m-%d %I:%M:%S %p %Z')
if [[ $status -eq 0 && -f "$report_path" ]]; then
    summary=$(final_summary)
    notify "$host_label Secant System ID LM campaign completed at $finished_et: $summary."
else
    notify "$host_label Secant System ID LM campaign FAILED at $finished_et with exit status $status. Log: $log_path"
fi
exit "$status"
