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

# Source history and workspace backups

The September 10 checkpoint closes the uncommitted-work gap from September 2.
Source checkpoint: `16244c541b05fbddf33d53521a0fd693d000dd66`.
It records the existing core, adapters, search experiments, generators, tests,
examples, and reports without changing their numerical behavior. The separate
backup-documentation commit follows it.

## What is preserved

Checkpoint directory: `20260910T153631Z-full-workspace-and-source`.

- `workspace.tar.gz`: the entire mac1 Odezza directory before committing,
  including `.git`, the original staged index, untracked experiments, ignored
  files, and retained data. No workspace paths were excluded. Symlinks are
  preserved, not followed into directories outside the workspace.
- `workspace-manifest.json`: file hashes and metadata for the capture. The
  archive contains 87,233 entries, including 75,673 regular files. Capture
  inventories matched before and after; every regular archive member was read
  back and checked against its hash.
- `history-before.bundle`, staged/unstaged binary patches, and Git status:
  additional ways to recover or inspect the original state.
- `history.bundle`: Git history after the source and documentation commits.
- `post-commit-overlay.tar.gz`: the committed files and final `.git` directory,
  to apply over a new restoration of the pre-commit workspace. This retains the
  archived experiment data while updating source and Git state to the committed
  checkpoint. Use `git-state.json` to identify that commit.
- `SHA256-before.json` and `SHA256-final.json`: transfer verification manifests.
  Copy receipts and `restore-verification.json` describe the checks performed.

The full archive is **3,420,219,080 bytes**. Its SHA256 is
`f43386a2d665f3b0539ed95e72438cc8283d3817df2a547967046b368de63256`.
All four off-host copies of the full archive and its initial manifest matched.

## Locations

| Machine | Checkpoint parent | Separate Git backup repository |
|---|---|---|
| mac1 | `/Users/cdurham/code/.odezza-checkpoints/` | Working repository in `/Users/cdurham/code/odezza` |
| mac3 | `/Users/cdurham/odezza-checkpoints/` | `/Users/cdurham/odezza-checkpoints/source-history.git` |
| rack1 | `/home/cdurham/odezza-checkpoints/` | History bundles in the checkpoint; no Git executable installed |
| rohini | `/home/cdurham/odezza-checkpoints/` | `/home/cdurham/odezza-checkpoints/source-history.git` |
| ada | `/home/cdurham/odezza-checkpoints/` | `/home/cdurham/odezza-checkpoints/source-history.git` |

Append the checkpoint directory name above to each checkpoint parent. These are
private directories. Existing working checkouts and deployed workers on the
other hosts are not replaced by the backup repositories.

## What belongs in Git

Reusable code, tests, example requests, protocols, and Markdown reports are
versioned. Large generated banks, trajectories, logs, binaries, and campaign
records stay in workspace checkpoints. `.gitignore` records these exclusions;
`benchmarks/lm_tuning/runs/` remains excluded by its existing policy. An ignored
file is not backed up automatically merely because source has been pushed.

The source selection was scanned for common credential formats. This was a
pattern check, not a guarantee that arbitrary text contains no sensitive data.
The full workspace backup retains its original contents and should stay private.
Existing formatting was preserved, including Markdown hard line breaks and a
few blank lines at EOF reported by the staged whitespace check.

## Continuing safely

The mac1 repository has `backup-mac3`, `backup-rohini`, and `backup-ada` remotes.
Those remotes describe the historical September 10 topology. For the current
September 12 topology, use mac3 and rack1 as described below; Rohini and Ada may
be offline or reformatted. The earlier push commands are retained for history:

```sh
git push --all backup-mac3
git push --tags backup-mac3
git push --all backup-rohini
git push --tags backup-rohini
git push --all backup-ada
git push --tags backup-ada
```

Take another workspace checkpoint when new experiment artifacts matter. This is
a manual checkpoint, not an automatic backup schedule or an off-site guarantee.
It covers mac1's Odezza workspace; machine configuration, credentials outside
that folder, and results existing only on other hosts are outside this archive.

To restore the exact pre-commit state, verify the manifest and extract only
`workspace.tar.gz` into a new empty directory. To restore the committed state
with retained artifacts, extract the full archive and then its post-commit
overlay into that same new directory. Before applying the overlay, move the
restored `odezza/.git` directory outside `odezza` (for example, to
`pre-commit-git` beside it). The overlay supplies the final complete Git
directory; replacing it as a unit avoids overwriting read-only Git objects.
The first rehearsal caught this issue; the retained restore diagnostic records
it and the repeated full rehearsal validates this corrected procedure.
Do not extract over a live checkout. For
source history alone, clone `history.bundle` or one of the bare backup
repositories and select the commit recorded in `git-state.json`.

The overnight study remains unstarted. Its frozen execution source hashes are
unchanged by this checkpoint.

## 2026-09-12 hardware move and service pools

Rohini and Ada have verified full `/home/cdurham` backups at
`rack1:/home/cdurham/odezza-checkpoints/20260912-hardware-move/`. Ada's archive is
23.3 GB; Rohini's is 186.0 GB. These include Git/uncommitted work, experiments,
results, datasets and user settings; they are not OS disk images. Both machines
were idle and are cleared for shutdown. Rack1 retains approximately 1.27 TB free.

Ada's capture matched its source completely. Rohini's only source-comparison
changes were live Codex logs/model-cache files; consistent SQLite online-backup
snapshots and a current model-cache overlay are included. Full destination
checksums, archive integrity and overlay hashes passed. Follow the sidecar-removal
instructions when restoring those SQLite snapshots. The private backup record is
[on mac1](/Users/cdurham/code/.odezza-checkpoints/20260912-hardware-move/README.md)
and accompanies the archives on rack1.

The full current mac1 Odezza workspace is also preserved under
`20260912-service-pools` in the existing checkpoint roots on mac1, mac3 and rack1.
Its 3,253,213,129-byte archive includes `.git` and uncommitted changes; all 78,649
regular files were compared to source and off-host archive checksums matched.
Apply `post-validation.tar.gz` after restoring `workspace.tar.gz` to recover
subsequent validation/documentation updates. This is a private file checkpoint,
not a new commit or reviewed release.

The service's active GPU host is rack1. Its direct connection is owned by mac3,
so neither mac1 nor the two machines being rearranged is a service dependency.

## CUB integration checkpoint — 2026-09-12

The CUB-only core/runtime integration and its exact service validation are saved
under `20260912-cub-integration` on all three machines:

- mac1: `/Users/cdurham/code/.odezza-checkpoints/20260912-cub-integration`
- mac3: `/Users/cdurham/code/.odezza-checkpoints/20260912-cub-integration`
- rack1: `/home/cdurham/odezza-checkpoints/20260912-cub-integration`

`before.tar.gz` / `after.tar.gz` preserve the affected mac1 source, tests and
validation artifacts. `before-live.tar.gz` / `after-live.tar.gz` preserve the
rack1 core/runtime/worker source and native binaries. `SHA256SUMS` gives each
archive's digest; verify it before restore. The native runtime binary embeds its
original live library directory in its runtime search path: restore there or
rebuild using the preserved Makefiles. Stop only the verified supervisor/child
before replacing loaded libraries. Backups do not restore RAM-only queued jobs
or a broker generation. The existing mac3 broker generation was kept throughout
this update. Repository-wide commit/release review remains separate.

Implementation and test evidence:
[the CUB integration record](../benchmarks/score_topk/integration/README.md).

## SQLite cache and retention checkpoint — 2026-09-12

Items 3 and 4 are preserved under `20260912-cache-retention` in the same three
checkpoint roots (mac1/mac3 `/Users/cdurham/code/.odezza-checkpoints`, rack1
`/home/cdurham/odezza-checkpoints`). `after-source.tar.gz` contains the working
source, vendored SQLite, tests and validation records. `before-live.tar.gz` and
`after-live.tar.gz` preserve native binaries and service sources;
`before-cache.tar.gz` is optional disposable-cache rollback data. Verify all
archives against `SHA256SUMS`. No customer jobs/results, Git commit or release tag
are created by this checkpoint. Restore live binaries to their recorded library
paths or rebuild, and drain the supervisor before replacing them.

[Implementation, test and deployment record](../benchmarks/cache_retention/README.md).


## Source commit and history backup — 2026-09-12

Checkpoint: `20260912-source-commit` on mac1/mac3 under
`/Users/cdurham/code/.odezza-checkpoints/`, and on rack1 under
`/home/cdurham/odezza-checkpoints/`. The current source history is also pushed to
mac3’s existing bare backup repository. No live worker checkout is reset by this
operation; no contact with Rohini or Ada is required.

Git includes native/frontend/service code, the separate Python grammar tools,
tests, examples, documentation, small provenance summaries and pinned SQLite.
Generated logs, profiler traces and bulk run/ranking reports are ignored.
The small synthetic request/reference fixtures consumed by regression tests are
explicit exceptions. Archived comparison scripts can require their historical
reference reports: restore those from checkpoints before reproducing old/new
comparisons. The standalone generators and ordinary tests retain their source
inputs in Git.

- `history.bundle`: complete Git refs/history at the checkpoint.
- `source.tar.gz`: exact committed HEAD, without Git internals or run data.
- `uncommitted-run-artifacts.tar.gz`: the generated output excluded from this
  commit, with its file inventory and SHA256 hashes. This supplements the existing
  full-workspace backups; it does not replace their older campaign data.
- `commit.json`: precise HEAD/branch and verification information.
- `SHA256SUMS`: archive/bundle checksums verified on all three machines.

Verify checksums before restore. Clone `history.bundle` into an empty directory
and check out the HEAD recorded in `commit.json` to restore source history, or
extract `source.tar.gz` into an empty directory for source only. Restore the run
artifact archive separately if those historical comparisons are needed. The
checkpoint is a source commit and backup, not a tagged production release.
