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

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import statistics
import struct
from typing import Sequence

from .ast import InstructionType, Program
from .c_runtime import GPCandidate


SELECTION_OBJECTIVE = "objective"
SELECTION_NOVELTY_RANDOM = "novelty-random"
SELECTION_RANDOM = "random"


_LEAF_INSTRUCTIONS = {
    InstructionType.STATIC_COLUMN_INPUT_F32,
    InstructionType.DYNAMIC_COLUMN_INPUT_F32,
    InstructionType.DYNAMIC_CONSTANT_INPUT_F32,
    InstructionType.DYNAMIC_CONSTANT_OR_COLUMN_INPUT_F32,
}


def _normalized_program(program: bytes) -> bytes:
    output = bytearray()
    leaf_names: dict[int, int] = {}
    for instruction in Program(program).instructions():
        output.append(instruction.kind)
        if instruction.kind in _LEAF_INSTRUCTIONS:
            assert instruction.operand is not None
            source = int(instruction.operand)
            if source not in leaf_names:
                leaf_names[source] = len(leaf_names)
            output.append(leaf_names[source])
        elif instruction.kind == InstructionType.CONSTANT_BITS_F32:
            assert instruction.operand is not None
            output.extend(struct.pack("<I", instruction.operand))
        elif instruction.operand is not None:
            output.append(instruction.operand)
    return bytes(output)


def structure_sha256(programs: Sequence[bytes]) -> str:
    digest = hashlib.sha256(b"secant-system-id-structure-v1\0")
    digest.update(len(programs).to_bytes(4, "little"))
    for program in programs:
        normalized = _normalized_program(program)
        digest.update(len(normalized).to_bytes(4, "little"))
        digest.update(normalized)
    return digest.hexdigest()


def _random_rank(seed: int, generation: int, candidate: GPCandidate, structure_id: str) -> bytes:
    digest = hashlib.sha256(b"secant-system-id-lm-random-v1\0")
    digest.update(str(seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(generation).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(candidate.index).encode("ascii"))
    digest.update(b"\0")
    digest.update(structure_id.encode("ascii"))
    return digest.digest()


@dataclass(frozen=True)
class LMSelectedCandidate:
    candidate: GPCandidate
    structure_sha256: str
    structure_novel: bool
    selected_by_novelty: bool
    selected_by_random: bool


@dataclass(frozen=True)
class LMSelection:
    mode: str
    generation: int
    population_candidates: int
    unique_structures: int
    pool_structures: int
    novel_pool_structures: int
    random_trigger_probability: float
    random_triggered: bool
    selected: tuple[LMSelectedCandidate, ...]
    observed_structure_ids: frozenset[str]

    def boundary_record(self) -> dict[str, object]:
        novelty = {item.candidate.index for item in self.selected if item.selected_by_novelty}
        random = {item.candidate.index for item in self.selected if item.selected_by_random}
        return {
            "generation": self.generation,
            "mode": self.mode,
            "population_candidates": self.population_candidates,
            "unique_structures": self.unique_structures,
            "selection_pool_structures": self.pool_structures,
            "novel_pool_structures": self.novel_pool_structures,
            "random_trigger_probability": self.random_trigger_probability,
            "random_triggered": self.random_triggered,
            "novelty_selected": len(novelty),
            "random_selected": len(random),
            "novelty_only": len(novelty - random),
            "random_only": len(random - novelty),
            "overlap": len(novelty & random),
            "union_selected": len(novelty | random),
            "neither": self.pool_structures - len(novelty | random),
            "selected": [
                {
                    "candidate_index": item.candidate.index,
                    "structure_sha256": item.structure_sha256,
                    "structure_novel": item.structure_novel,
                    "selected_by_novelty": item.selected_by_novelty,
                    "selected_by_random": item.selected_by_random,
                }
                for item in self.selected
            ],
        }


def select_lm_candidates(
    candidates: Sequence[GPCandidate],
    seen_structure_ids: set[str],
    mode: str,
    promotion_count: int,
    novelty_count: int,
    random_count: int,
    pool_size: int,
    seed: int,
    generation: int,
    random_trigger_probability: float = 1.0,
) -> LMSelection:
    if mode not in (SELECTION_OBJECTIVE, SELECTION_NOVELTY_RANDOM, SELECTION_RANDOM):
        raise ValueError(f"unknown LM selection mode: {mode}")
    if min(promotion_count, pool_size) <= 0 or novelty_count < 0 or random_count < 0:
        raise ValueError("invalid LM selection dimensions")
    if not 0.0 <= random_trigger_probability <= 1.0:
        raise ValueError("random LM trigger probability must be in [0, 1]")
    ranked = sorted(candidates, key=lambda candidate: (candidate.objective, candidate.index))
    unique: list[tuple[GPCandidate, str]] = []
    observed: set[str] = set()
    for candidate in ranked:
        structure_id = structure_sha256(candidate.programs)
        if structure_id in observed:
            continue
        observed.add(structure_id)
        unique.append((candidate, structure_id))
    pool = unique[:pool_size]
    random_triggered = True
    if mode == SELECTION_OBJECTIVE:
        selected = tuple(
            LMSelectedCandidate(candidate, structure_id, structure_id not in seen_structure_ids, False, False)
            for candidate, structure_id in pool[:promotion_count]
        )
    elif mode == SELECTION_NOVELTY_RANDOM:
        novel = [(candidate, structure_id) for candidate, structure_id in pool if structure_id not in seen_structure_ids]
        novelty_indices = {candidate.index for candidate, _ in novel[:novelty_count]}
        random_ranked = sorted(
            pool,
            key=lambda item: (_random_rank(seed, generation, item[0], item[1]), item[0].index),
        )
        random_indices = {candidate.index for candidate, _ in random_ranked[:random_count]}
        union_indices = novelty_indices | random_indices
        selected = tuple(
            LMSelectedCandidate(
                candidate,
                structure_id,
                structure_id not in seen_structure_ids,
                candidate.index in novelty_indices,
                candidate.index in random_indices,
            )
            for candidate, structure_id in pool
            if candidate.index in union_indices
        )
    else:
        trigger_digest = hashlib.sha256(
            f"secant-system-id-lm-trigger-v1\0{seed}\0{generation}".encode("ascii")
        ).digest()
        trigger_value = int.from_bytes(trigger_digest[:8], "little") / float(1 << 64)
        random_triggered = trigger_value < random_trigger_probability
        random_ranked = sorted(
            pool,
            key=lambda item: (_random_rank(seed, generation, item[0], item[1]), item[0].index),
        )
        random_indices = (
            {candidate.index for candidate, _ in random_ranked[:promotion_count]}
            if random_triggered
            else set()
        )
        selected = tuple(
            LMSelectedCandidate(
                candidate,
                structure_id,
                structure_id not in seen_structure_ids,
                False,
                True,
            )
            for candidate, structure_id in pool
            if candidate.index in random_indices
        )
    return LMSelection(
        mode,
        generation,
        len(candidates),
        len(unique),
        len(pool),
        sum(structure_id not in seen_structure_ids for _, structure_id in pool),
        random_trigger_probability,
        random_triggered,
        selected,
        frozenset(structure_id for _, structure_id in pool),
    )


def _outcome_summary(events: Sequence[dict[str, object]], key: str) -> dict[str, object]:
    selected = [event for event in events if bool(event.get(key))]
    folds = [
        float(event["before_mse"]) / float(event["after_mse"])
        for event in selected
        if float(event["after_mse"]) > 0.0
    ]
    return {
        "evaluated": len(selected),
        "accepted": sum(bool(event["accepted"]) for event in selected),
        "improved_over_1_percent": sum(
            float(event["after_mse"]) < 0.99 * float(event["before_mse"])
            for event in selected
        ),
        "improved_over_10_percent": sum(
            float(event["after_mse"]) < 0.9 * float(event["before_mse"])
            for event in selected
        ),
        "median_improvement_fold": statistics.median(folds) if folds else None,
    }


def lm_selection_summary(
    boundaries: Sequence[dict[str, object]], events: Sequence[dict[str, object]]
) -> dict[str, object]:
    venn_fields = ("novelty_only", "random_only", "overlap", "neither")
    return {
        "boundary_count": len(boundaries),
        "venn": {
            field: sum(int(boundary.get(field, 0)) for boundary in boundaries)
            for field in venn_fields
        },
        "selected": {
            "novelty": sum(int(boundary.get("novelty_selected", 0)) for boundary in boundaries),
            "random": sum(int(boundary.get("random_selected", 0)) for boundary in boundaries),
            "union": sum(int(boundary.get("union_selected", 0)) for boundary in boundaries),
            "eligible_union": sum(int(boundary.get("eligible_union", 0)) for boundary in boundaries),
            "random_triggered_boundaries": sum(bool(boundary.get("random_triggered")) for boundary in boundaries),
        },
        "outcomes": {
            "novelty": _outcome_summary(events, "selected_by_novelty"),
            "random": _outcome_summary(events, "selected_by_random"),
            "novelty_only": _outcome_summary(events, "selected_by_novelty_only"),
            "random_only": _outcome_summary(events, "selected_by_random_only"),
            "overlap": _outcome_summary(events, "selected_by_both"),
        },
    }


def _numeric_summary(values: Sequence[float | int]) -> dict[str, object]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "minimum": min(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "mean": statistics.fmean(finite) if finite else None,
        "maximum": max(finite) if finite else None,
    }


def _rejection_category(reason: str) -> str:
    lowered = reason.lower()
    if "cse" in lowered or "liveness" in lowered or "output register" in lowered:
        return "derivative_output_pressure"
    if "register" in lowered:
        return "register_pressure"
    if "patch" in lowered or "site capacity" in lowered:
        return "patch_capacity"
    if "derivative" in lowered or "differentiat" in lowered or "unsupported" in lowered:
        return "unsupported_derivative"
    return "other"


def lm_run_diagnostics(
    boundaries: Sequence[dict[str, object]], events: Sequence[dict[str, object]]
) -> dict[str, object]:
    """Aggregate selection, specialization, optimization, and kernel health telemetry."""

    selected = sum(int(boundary.get("union_selected", 0)) for boundary in boundaries)
    eligible = sum(int(boundary.get("eligible_union", 0)) for boundary in boundaries)
    skipped_entries: list[dict[str, object]] = []
    skip_reasons: Counter[str] = Counter()
    skip_categories: Counter[str] = Counter()
    rejected_complexities: list[float] = []
    rejected_nodes_by_site: list[list[float]] = []
    for boundary in boundaries:
        generation = boundary.get("generation")
        seed = boundary.get("seed")
        raw_skipped = boundary.get("skipped", ())
        if not isinstance(raw_skipped, (list, tuple)):
            continue
        for raw in raw_skipped:
            if not isinstance(raw, dict):
                continue
            reason = str(raw.get("reason", "unspecified"))
            skip_reasons[reason] += 1
            skip_categories[_rejection_category(reason)] += 1
            complexity = raw.get("candidate_complexity")
            if isinstance(complexity, (int, float)):
                rejected_complexities.append(float(complexity))
            node_counts = raw.get("ast_node_counts", ())
            if isinstance(node_counts, (list, tuple)):
                while len(rejected_nodes_by_site) < len(node_counts):
                    rejected_nodes_by_site.append([])
                for site, count in enumerate(node_counts):
                    if isinstance(count, (int, float)):
                        rejected_nodes_by_site[site].append(float(count))
            if len(skipped_entries) < 16:
                skipped_entries.append(
                    {
                        "seed": seed,
                        "generation": generation,
                        "candidate_index": raw.get("candidate_index"),
                        "candidate_complexity": complexity,
                        "ast_node_counts": list(node_counts)
                        if isinstance(node_counts, (list, tuple))
                        else None,
                        "structure_sha256": raw.get("structure_sha256"),
                        "category": _rejection_category(reason),
                        "reason": reason,
                    }
                )
    recorded_skips = sum(skip_reasons.values())
    reported_skips = sum(int(boundary.get("skipped_count", 0)) for boundary in boundaries)
    rejected = max(recorded_skips, reported_skips, max(0, selected - eligible))

    accepted = sum(bool(event.get("accepted")) for event in events)
    improved = sum(
        isinstance(event.get("before_mse"), (int, float))
        and isinstance(event.get("after_mse"), (int, float))
        and float(event["after_mse"]) < float(event["before_mse"])
        for event in events
    )
    total_fits = sum(int(event.get("fits", 0)) for event in events)
    finite_fits = sum(int(event.get("finite_fits", event.get("fits", 0))) for event in events)
    invalid_fits = sum(int(event.get("invalid_mse_fits", 0)) for event in events)
    fits_with_accepted_steps = sum(int(event.get("fits_with_accepted_steps", 0)) for event in events)
    fits_hitting_iteration_limit = sum(
        int(event.get("fits_hitting_iteration_limit", 0)) for event in events
    )
    total_iterations = sum(int(event.get("total_lm_iterations", 0)) for event in events)
    total_accepted_steps = sum(int(event.get("total_accepted_steps", 0)) for event in events)

    primal_fallbacks: Counter[str] = Counter()
    partial_fallbacks: Counter[str] = Counter()
    fallback_events = 0
    for event in events:
        primal = event.get("primal_pressure_fallback_groups", ())
        partial = event.get("partial_pressure_fallback_groups", ())
        primal = primal if isinstance(primal, (list, tuple)) else ()
        partial = partial if isinstance(partial, (list, tuple)) else ()
        if primal or partial:
            fallback_events += 1
        primal_fallbacks.update(str(group) for group in primal)
        partial_fallbacks.update(str(group) for group in partial)

    batch_records = [boundary.get("lm_batch") for boundary in boundaries]
    batch_records = [record for record in batch_records if isinstance(record, dict)]
    batch_wall_seconds = sum(float(record.get("wall_seconds", 0.0)) for record in batch_records)
    batch_fits = sum(int(record.get("fits", 0)) for record in batch_records)
    timing = {
        "recorded_batches": len(batch_records),
        "missing_batches": max(0, sum(int(boundary.get("eligible_union", 0)) > 0 for boundary in boundaries) - len(batch_records)),
        "wall_seconds": batch_wall_seconds,
        "upload_seconds": sum(float(record.get("upload_seconds", 0.0)) for record in batch_records),
        "download_seconds": sum(float(record.get("download_seconds", 0.0)) for record in batch_records),
        "sum_ticket_seconds": sum(float(record.get("sum_ticket_seconds", 0.0)) for record in batch_records),
        "sum_queue_wait_seconds": sum(float(record.get("sum_queue_wait_seconds", 0.0)) for record in batch_records),
        "sum_image_copy_seconds": sum(float(record.get("sum_image_copy_seconds", 0.0)) for record in batch_records),
        "sum_module_load_seconds": sum(float(record.get("sum_module_load_seconds", 0.0)) for record in batch_records),
        "fits": batch_fits,
        "fits_per_wall_second": batch_fits / batch_wall_seconds if batch_wall_seconds > 0.0 else None,
    }

    population_snapshots = [
        boundary.get("gp_population_snapshot") for boundary in boundaries
    ]
    population_snapshots = [
        snapshot for snapshot in population_snapshots if isinstance(snapshot, dict)
    ]
    population_observations = sum(
        int(snapshot.get("candidates", 0)) for snapshot in population_snapshots
    )
    invalid_population_observations = sum(
        int(snapshot.get("invalid_mse_candidates", 0)) for snapshot in population_snapshots
    )
    snapshot_invalid_rates = [
        int(snapshot.get("invalid_mse_candidates", 0))
        / int(snapshot.get("candidates", 0))
        for snapshot in population_snapshots
        if int(snapshot.get("candidates", 0)) > 0
    ]
    gp_population = {
        "recorded_boundaries": len(population_snapshots),
        "candidate_observations": population_observations,
        "invalid_mse_candidate_observations": invalid_population_observations,
        "invalid_mse_observation_rate": invalid_population_observations / population_observations
        if population_observations
        else None,
        "snapshot_invalid_rate": _numeric_summary(snapshot_invalid_rates),
        "snapshot_median_complexity": _numeric_summary(
            [
                float(snapshot["median_complexity"])
                for snapshot in population_snapshots
                if isinstance(snapshot.get("median_complexity"), (int, float))
            ]
        ),
        "maximum_observed_complexity": max(
            (
                float(snapshot["maximum_complexity"])
                for snapshot in population_snapshots
                if isinstance(snapshot.get("maximum_complexity"), (int, float))
            ),
            default=None,
        ),
    }

    flags: list[dict[str, str]] = []
    if boundaries and selected == 0:
        flags.append(
            {
                "severity": "info",
                "code": "lm_never_selected",
                "detail": "No LM trigger selected a candidate during this run.",
            }
        )
    if selected > 0 and eligible == 0:
        flags.append(
            {
                "severity": "critical",
                "code": "lm_all_specializations_rejected",
                "detail": f"All {selected} selected LM candidates were rejected before evaluation.",
            }
        )
    elif rejected > 0:
        fraction = rejected / selected if selected else 0.0
        flags.append(
            {
                "severity": "warning" if fraction >= 0.10 else "info",
                "code": "lm_specialization_rejections",
                "detail": f"{rejected} of {selected} selected LM candidates were rejected ({fraction:.1%}).",
            }
        )
    if events and accepted == 0:
        flags.append(
            {
                "severity": "warning",
                "code": "lm_no_accepted_promotions",
                "detail": f"None of {len(events)} evaluated LM candidates improved the GP incumbent.",
            }
        )
    if invalid_fits > 0:
        flags.append(
            {
                "severity": "warning",
                "code": "lm_invalid_fit_scores",
                "detail": f"{invalid_fits} of {total_fits} LM fits produced non-finite MSE values.",
            }
        )
    if snapshot_invalid_rates and max(snapshot_invalid_rates) >= 0.50:
        flags.append(
            {
                "severity": "warning",
                "code": "gp_population_mostly_invalid",
                "detail": "At least one recorded GP boundary had 50% or more candidates without a valid MSE.",
            }
        )
    if fallback_events > 0:
        flags.append(
            {
                "severity": "info",
                "code": "lm_pressure_fallback_used",
                "detail": f"The pressure fallback recovered {fallback_events} evaluated candidates.",
            }
        )

    severities = {str(flag["severity"]) for flag in flags}
    status = (
        "not_run"
        if not boundaries
        else "failed"
        if "critical" in severities
        else "degraded"
        if "warning" in severities
        else "healthy"
    )
    register_counts = [
        float(event["register_count"])
        for event in events
        if isinstance(event.get("register_count"), (int, float))
    ]
    successful_complexities = [
        float(event["candidate_complexity"])
        for event in events
        if isinstance(event.get("candidate_complexity"), (int, float))
    ]
    successful_nodes_by_site: list[list[float]] = []
    for event in events:
        node_counts = event.get("ast_node_counts", ())
        if not isinstance(node_counts, (list, tuple)):
            continue
        while len(successful_nodes_by_site) < len(node_counts):
            successful_nodes_by_site.append([])
        for site, count in enumerate(node_counts):
            if isinstance(count, (int, float)):
                successful_nodes_by_site[site].append(float(count))
    primal_instructions = [
        float(sum(event.get("primal_sass_instructions", ())))
        for event in events
        if isinstance(event.get("primal_sass_instructions"), (list, tuple))
    ]
    partial_instructions = [
        float(sum(event.get("partial_sass_instructions", ())))
        for event in events
        if isinstance(event.get("partial_sass_instructions"), (list, tuple))
    ]
    return {
        "status": status,
        "health_flags": flags,
        "selection": {
            "boundaries": len(boundaries),
            "triggered_boundaries": sum(bool(boundary.get("random_triggered")) for boundary in boundaries),
            "selected_candidates": selected,
            "boundaries_with_no_selection": sum(int(boundary.get("union_selected", 0)) == 0 for boundary in boundaries),
            "boundaries_with_all_selected_rejected": sum(
                int(boundary.get("union_selected", 0)) > 0
                and int(boundary.get("eligible_union", 0)) == 0
                for boundary in boundaries
            ),
        },
        "gp_population": gp_population,
        "specialization": {
            "attempted": selected,
            "succeeded": eligible,
            "rejected": rejected,
            "success_rate": eligible / selected if selected else None,
            "rejection_rate": rejected / selected if selected else None,
            "rejection_categories": dict(sorted(skip_categories.items())),
            "rejection_reasons": dict(sorted(skip_reasons.items())),
            "rejected_candidate_complexity": _numeric_summary(rejected_complexities),
            "rejected_ast_nodes_by_site": [
                _numeric_summary(values) for values in rejected_nodes_by_site
            ],
            "rejection_examples": skipped_entries,
        },
        "optimization": {
            "evaluated_candidates": len(events),
            "accepted_candidates": accepted,
            "accepted_rate": accepted / len(events) if events else None,
            "candidates_with_lower_mse": improved,
            "selected_to_evaluated_rate": len(events) / selected if selected else None,
            "selected_to_accepted_rate": accepted / selected if selected else None,
            "fits": total_fits,
            "finite_mse_fits": finite_fits,
            "invalid_mse_fits": invalid_fits,
            "fits_with_accepted_steps": fits_with_accepted_steps,
            "fits_hitting_iteration_limit": fits_hitting_iteration_limit,
            "average_iterations_per_fit": total_iterations / total_fits if total_fits else None,
            "average_accepted_steps_per_fit": total_accepted_steps / total_fits if total_fits else None,
        },
        "pressure_fallback": {
            "evaluated_candidates_using_fallback": fallback_events,
            "candidate_rate": fallback_events / len(events) if events else None,
            "primal_groups": dict(sorted(primal_fallbacks.items())),
            "partial_groups": dict(sorted(partial_fallbacks.items())),
        },
        "kernel": {
            "evaluated_candidate_complexity": _numeric_summary(successful_complexities),
            "evaluated_ast_nodes_by_site": [
                _numeric_summary(values) for values in successful_nodes_by_site
            ],
            "register_count": _numeric_summary(register_counts),
            "primal_sass_instructions": _numeric_summary(primal_instructions),
            "partial_sass_instructions": _numeric_summary(partial_instructions),
        },
        "timing": timing,
    }
