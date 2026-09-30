<!--
SPDX-FileCopyrightText: 2026 Charles Durham
SPDX-License-Identifier: MIT

MIT License

Copyright (c) 2026 Charles Durham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

**Yes—return compact indices from the GPU, then reconstruct only the retained winners.** I would use **a variant index plus a configuration index**, both tied to an immutable run manifest.

The AST index identifies the equation structure. The variant and configuration identify the particular toggle groups, state assignments, constant values, and RNG draws.

**The GPU result can stay very small.**

For FP32 scoring, a proposed record is:

```cpp
struct ScoreHit {
    float    mse;
    uint32_t variant_index;
    uint64_t configuration_index;
};
```

That is 16 bytes per retained hit with the usual layout. The host’s variant table maps `variant_index` to the skeleton, pool plan, and provenance. You do not need string IDs or complete equations on the GPU.

Once reduction finishes, resolve only those hits.

**The configuration index already provides the right mechanism.**

Our compiler emits an ordered Cartesian product in `pools.pool_axes`, and this existing helper decodes it:

```python
indices = configuration_indices(pools, configuration_index)
```

The last axis varies fastest. For example, with axes:

```text
2 state choices × 3 constant values × 32 RNG draws
```

index `137` resolves to:

```text
state-choice position 1
constant-bank index 1
RNG draw index 9
```

Repeated slots and shared RNG axes retain their intended correlations.

**Use deterministic RNG to recover the sampled coefficients.**

Counter-based generators such as Philox support this approach: their output is a deterministic function of a key and counter, without needing to replay a sequential generator. [Random123 documentation](https://github.com/DEShawResearch/random123)

Preserve these in the run manifest:

- The emitted variant table and referenced skeletons.
- Ordered axes, toggle groups, and constant banks.
- RNG algorithm/version, key mapping, streams, distributions, and transforms.
- Prepared-problem revision, effective solver settings, and numeric precision.

Derive RNG addresses from logical bank/stream/draw coordinates. Thread IDs, launch order, or chunk boundaries should not affect them. Run-scoped banks should retain the same draws across skeletons; skeleton-scoped banks should include the stable skeleton identity.

Also, **persist the emitted ordinal-to-program mapping**. Recovering AST number 900,000 should be a lookup, not a regeneration of the first 899,999 ASTs. A compact file with an offset index is sufficient.

**For durable winners, also save the actual resolved values.**

After selecting top-k, I would materialize and store:

| Information | Purpose |
|---|---|
| Complete resolved RHS/postorder | Inspect or export the model |
| Selected state bindings | Explain what the toggles chose |
| Actual fixed and transformed RNG values | Replay without depending entirely on RNG regeneration |
| Final fitted parameters, if LM ran | Preserve the optimization result |
| Score, execution profile, and provenance | Interpret and reproduce the evaluation |

This costs very little for tens or hundreds of winners. For scoring, a small finalization kernel could regenerate the winners’ transformed coefficients using the same GPU implementation.

**LM is the important exception:** its initialization can be recovered from indices and RNG, but its optimized parameters must be retained. Keep the final fitted vector alongside retained LM hits or in a referenced buffer. Rerunning LM would be another computation, not simple reconstruction.

Store numerical values in execution precision; rounded coefficients in a displayed equation should not be the replay representation.

**For MCP, return candidates once and reference them from each leaderboard.**

A proposed response shape—not an implemented endpoint—would be:

```json
{
  "run_id": "run_123",
  "status": "completed",
  "ranking": {
    "metric": "mse",
    "unit": "resolved_structure"
  },
  "global": ["candidate_42"],
  "by_family": {
    "negative_feedback": ["candidate_42"]
  },
  "by_tag": {
    "saturating_removal": ["candidate_42"]
  },
  "candidates": {
    "candidate_42": {
      "mse": 0.00123,
      "phase": "score",
      "origin": {
        "variant_index": 271,
        "configuration_index": "137"
      },
      "model_ref": "model_42",
      "snapshot_ref": "snapshot_42"
    }
  }
}
```

The score and identifiers above are illustrative. For small top-k responses, include readable equations and coefficient values directly as well. References can carry exact snapshots and requested trajectories. Encode potentially large 64-bit indices as decimal strings in JSON.

Two ranking details matter:

- **Preserve each requested leaderboard during reduction.** A global shortlist alone can discard every candidate needed for a family’s top-k.
- **Apply the requested uniqueness rule before truncating.** Overlapping toggle groups can otherwise fill the shortlist with duplicate resolved candidates. Taking raw top-k and deduplicating afterward does not guarantee top-k distinct models.

For the LLM-facing default, I recommend **distinct resolved structures**, profiling over their numerical configurations. Offer numerical-candidate ranking when the researcher wants parameter alternatives.

This gives you a small GPU output, inexpensive reconstruction, and self-contained winners that remain useful after the original run is gone.