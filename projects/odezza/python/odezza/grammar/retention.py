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
"""Bounded unique winners per resolved binding; provenance attached at report time."""
from __future__ import annotations


def policies(retain):
    return ([retain["global"]] if "global" in retain else []) + \
           ([retain["per_family"]] if "per_family" in retain else []) + list(retain.get("by_tag", {}).values())


def retention_k(retain):
    return max([1] + [p["k"] for p in policies(retain)])


def local_retention_k(retain):
    # A resolved binding contributes just its best coefficient row to structure
    # and variant rankings. Numeric rankings require k distinct rows per binding.
    return max([1] + [p["k"] for p in policies(retain) if p["unit"] == "numeric_candidate"])


def ranked(rows, policy):
    k = policy["k"]
    key = {"numeric_candidate": "numeric_id", "resolved_structure": "structure_id", "variant": "variant_id"}[policy["unit"]]
    seen, result = set(), []
    for row in sorted(rows, key=lambda r: (r["mse"], r["variant_id"], r["configuration_index"], r["id"])):
        if row[key] not in seen:
            seen.add(row[key]); result.append(row)
            if len(result) >= k: break
    return result if k else []


class Ranking:
    """Keep only the best k unique units, including when a unit reappears later."""
    def __init__(self, policy):
        self.k = policy["k"]
        self.unit = {"numeric_candidate": "numeric_id", "resolved_structure": "structure_id", "variant": "variant_id"}[policy["unit"]]
        self.rows, self.worst = {}, None

    @staticmethod
    def order(row):
        return row["mse"], row["variant_id"], row["configuration_index"], row["id"]

    def offer(self, row):
        if not self.k: return
        unit, order = row[self.unit], self.order(row)
        old = self.rows.get(unit)
        if old is not None and self.order(old) <= order: return
        if old is None and len(self.rows) >= self.k and order >= self.worst: return
        self.rows[unit] = row
        if len(self.rows) > self.k:
            self.rows.pop(max(self.rows, key=lambda key: self.order(self.rows[key])))
        self.worst = max(map(self.order, self.rows.values()))

    def results(self):
        return sorted(self.rows.values(), key=self.order)


def make_report(rows, memberships, retain, family_ids, states=None):
    global_rank = Ranking(retain["global"]) if "global" in retain else None
    families = {name: Ranking(retain["per_family"]) for name in family_ids} if "per_family" in retain else {}
    tags = {name: Ranking(policy) for name, policy in retain.get("by_tag", {}).items()}
    for row in rows:
        row = dict(row)
        provenance = memberships.get(row["variant_id"], []) if memberships is not None else row["provenance"]
        row["families"] = sorted({p["family_id"] for p in provenance})
        row["tags"] = sorted({tag for p in provenance for tag in p["tags"]})
        row["provenance"] = provenance
        if global_rank: global_rank.offer(row)
        for name in row["families"]:
            if name in families: families[name].offer(row)
        for name in row["tags"]:
            if name in tags: tags[name].offer(row)
    result = {}
    if "global" in retain:
        result["global"] = global_rank.results()
    if "per_family" in retain:
        result["families"] = {name: ranking.results() for name, ranking in families.items()}
    result["tags"] = {name: ranking.results() for name, ranking in tags.items()}
    if states is not None:
        from .reference import describe
        cache = {}
        for group in ([result["global"]] if "global" in result else []) + list(result.get("families", {}).values()) + list(result["tags"].values()):
            for i, row in enumerate(group):
                if row["id"] not in cache: cache[row["id"]] = describe(row, states)
                group[i] = cache[row["id"]]
    return result
