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

# Validation results

Historical validation record from this conversation. See `../HANDOFF_VERIFICATION.txt` for the fresh handoff checks. The million-program host planning run below was not repeated merely to package this archive.

The package was checked with the Python standard library test runner:

```bash
python -m unittest discover -s tests -v
```

All 83 tests passed. The tests cover independent and shared grammar expansions, recursive bounds and deadlines, safe expression parsing, hygienic local bindings, postorder evaluation, exhaustive binary/quad coverage, deterministic per-leaf and joint sampling, numeric Cartesian indexing, RNG scope and transforms, tagged provenance after deduplication, integration compatibility, separate LM limits, SQLite deduplication, atomic CLI output, and compact-stream round trips.

End-to-end example planning produced these counts:

| Example | Skeletons | Toggle or pool variants | Configuration visits |
| --- | ---: | ---: | ---: |
| oscillator | 3 | 3 | 33 |
| toggles_all | 2 | 100 | 800 |
| toggles_sample | 1 | 12 | 96 |
| toggles_joint_sample | 1 | 7 | 56 |
| constant_rng_product | 1 | 1 | 384 |
| future_stiff with annotation-only flag | 1 | 1 | 2 |

The separate LM example validated two fitted scalar parameters. The recursive and million examples also passed bounded CLI checks at 100 skeletons.

The full million example was additionally traversed with:

```bash
python -m odegrammar plan examples/million.json --max-seconds 300
```

It emitted and deduplicated 1,000,000 skeleton records and 1,000,000 variant records through the planning iterator, representing 1,000,000 configuration visits. No duplicate programs were encountered. This run took 108.633 seconds in the provided Python environment and performed 4,333,329 grammar expansion steps. The summary's stop reason was `max_derivations`, because that request's one-million derivation cap was reached. It conservatively did not assert iterator exhaustion beyond the cap.

That timing includes host grammar expansion, lowering, pool planning, and exact in-memory ID deduplication. It does not include serializing a million-program JSONL file, GPU compilation, RNG bank generation on a GPU, ODE integration, scoring, or LM. It is a functional scale check, not a benchmark claim about the user's execution engine.

Compact-stream tests verify exact reconstruction of inline variants, including RNG bank lengths, original identifiers, tagged provenance, and configuration counts. The Python RNG helpers intentionally use the explicitly named `sha256_reference_v1` algorithm and are not a Philox implementation. They validate addressing and transformation semantics; the execution adapter supplies the production GPU generator.
