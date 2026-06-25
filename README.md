# trajlens

The quality and synthesis layer for the open robot-learning data ecosystem.

ruff for robot data — lint, fix, and generate clean LeRobotDataset datasets.

## Status

Pre-v0.1 (`0.1.0.dev0`), under active development. Not yet on PyPI.

`lint` is implemented and audited against the public Hub (see [Real-world audit](#real-world-audit-of-the-hub) below).
`fix` and `web` are stubs reserved for the v0.2 milestone.

## Install (dev)

```bash
git clone https://github.com/<your-username>/trajlens
cd trajlens
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,hub]"
```

The `[hub]` extra pulls in `huggingface_hub`; it's only required to lint datasets by Hub repo id rather than local path.

## Usage

```bash
trajlens lint <path-or-org/dataset>          # human-readable terminal report
trajlens lint <path-or-org/dataset> --json   # machine-readable JSON report
trajlens lint <path-or-org/dataset> --report out.html
trajlens lint <path-or-org/dataset> --sarif out.sarif   # SARIF 2.1.0, for CI annotations
trajlens lint <path-or-org/dataset> --deep   # also decode video and verify per-frame stats
```

Exit codes follow lint-tool convention: `0` = clean, `1` = WARN present, `2` = FAIL or load ERROR — so `trajlens lint` composes directly into CI gates.

By default, checks that require materializing a lot of data over the network (full video decode, per-frame stats reconciliation) are skipped for Hub datasets and reported as INFO/skipped rather than run. Pass `--deep` to force them; expect this to be significantly slower and to fetch the full dataset.

## What it checks

trajlens validates a [LeRobotDataset](https://github.com/huggingface/lerobot) (v2.0, v2.1, or v3.0) against its own declared metadata, independent of any particular consumer's assumptions. Checks are grouped by category and run as a check engine over each dataset:

| Category | Check | Severity | What it catches |
|---|---|---|---|
| STRUCTURAL | `VERSION_DETECTED` | INFO | Reports the detected `codebase_version`. |
| STRUCTURAL | `SCHEMA_CONSISTENCY` | FAIL | Parquet column dtypes/widths disagree with `info.json`'s declared feature shapes. |
| STRUCTURAL | `INDEX_CONTINUITY` | FAIL | Gaps or duplicates in `frame_index`/`episode_index`/global `index` columns. |
| STRUCTURAL | `METADATA_DATA_AGREEMENT` | FAIL | Declared episode lengths/`from`-`to` boundaries disagree with actual Parquet row counts (catches [#2401](https://github.com/huggingface/lerobot/issues/2401)-class corruption). |
| STRUCTURAL | `PATH_TEMPLATE_RESOLVES` | FAIL | A declared shard path (data or video) doesn't resolve to a readable file. |
| SEMANTIC | `FEATURE_DIMENSIONALITY` | FAIL | A feature's actual column width doesn't match its declared `shape`. |
| SEMANTIC | `TASK_INTEGRITY` | FAIL | A `task_index` reference has no corresponding, non-empty task description. |
| SEMANTIC | `LANGUAGE_PRESENT` | WARN | An episode has no non-empty language/task description. |
| SEMANTIC | `CAMERA_INTRINSICS_PLAUSIBLE` | INFO | Advisory; skipped where the LeRobot format carries no intrinsics field. |
| TEMPORAL | `TIMESTAMP_MONOTONIC` | FAIL | Timestamps are not strictly increasing within an episode. |
| TEMPORAL | `TIMESTAMP_SPACING` | WARN | Timestamp spacing is inconsistent with declared `fps` beyond decoder tolerance. |
| STATISTICAL | `STATS_MATCH_DATA` | FAIL | Recomputed global Welford stats diverge from `meta/stats.json`. Skipped over Hub HTTP by default — too slow without `--deep`. |
| STATISTICAL | `PER_EPISODE_STATS_MATCH` | WARN | Same, per-episode. Skipped over Hub HTTP by default. |
| STATISTICAL | `VALUE_SANITY` | WARN | Out-of-range or NaN/Inf values in numeric features. Skipped over Hub HTTP by default. |
| VIDEO | `DECODABLE_SPOTCHECK` | FAIL | A sampled video segment fails to decode. |
| KNOWNBUG | `TIMESTAMP_DRIFT` | FAIL | Cumulative timestamp drift matching the known lerobot [#3177](https://github.com/huggingface/lerobot/issues/3177) bug pattern. |

Every check's full result — message, severity, and structured `details` — is included in the JSON/HTML/SARIF report; the table above is the summary.

## Real-world audit of the Hub

`scripts/audit_hub.py` runs `trajlens lint --json` against a random sample of public Hub datasets tagged `lerobot`, each in an isolated subprocess with a 60s timeout, and aggregates the results. It's how this project validates itself against the actual long tail of community datasets rather than only its own fixtures.

A 100-dataset run (2026-06-24) produced:

| Status | Count | Meaning |
|---|---|---|
| PASS | 19 | No issues found. |
| WARN | 0 | — |
| FAIL | 13 | A real check fired — schema mismatch, metadata/data disagreement, missing language, etc. |
| ERROR | 47 | Dataset failed to *load* (unsupported v2.x Hub streaming, malformed/missing `meta/`, mistagged or deleted repos) — never reached the check engine. |
| TIMEOUT | 21 | Exceeded the 60s per-dataset budget. |

These figures are from a single 100-dataset random sample (raw results: see the `v0.1.0` release assets); `audit_hub.py` samples a fresh random subset of `lerobot`-tagged Hub datasets on each run, so rerunning it will produce a similarly-shaped but not identical distribution.

Of the 47 load-time ERRORs, none are trajlens bugs: about half (24) are the documented v0.1 limitation that v2.x Hub datasets can't be lazily streamed (shard paths are implicit and require a local filesystem to glob), and the rest are dead/mistagged Hub references, repos that aren't actually LeRobotDatasets (no `meta/` directory at the repo root), or genuinely malformed `meta/info.json` (wrong dtype, missing required fields) on the dataset's side.

TIMEOUTs were investigated as a possible performance bug rather than accepted as an inherent network ceiling: profiling two small, previously-timing-out datasets (`abdul004/so101_multi_task_v1`, 125 episodes; `Elvinky/pick_green_block_into_box`, 102 episodes) found that loading a dataset's metadata over Hub HTTP was issuing dozens of small, separately-latency-bound reads per Parquet shard, and downloading the `meta/` file tree one file at a time. Fixing both (single whole-shard fetch instead of scattered reads; parallelized `meta/` download) brought those two datasets from 60s+ timeouts down to 33s and 11s respectively, and cut the audit's overall TIMEOUT count and mean per-dataset duration by roughly a third in before/after sampling. The remaining TIMEOUTs are concentrated in genuinely large multi-thousand-episode shards, where 60s is a real infra ceiling rather than a fixable inefficiency.

### Launch audit findings

Of the 81 datasets that reached a grade (excluding ERROR/TIMEOUT, where no check ever ran), two known upstream `lerobot` bugs accounted for a meaningful share of the failures:

| Known bug | Prevalence (of successfully-linted datasets) |
|---|---|
| `KNOWNBUG.TIMESTAMP_DRIFT` ([#3177](https://github.com/huggingface/lerobot/issues/3177)) | 3.1% |
| `STRUCTURAL.METADATA_DATA_AGREEMENT` ([#2401](https://github.com/huggingface/lerobot/issues/2401)) | 18.8% |

`audit_hub.py` resamples a fresh random subset of `lerobot`-tagged Hub datasets on every run, so these are not a fixed, reproducible distribution — rerunning the audit will not return the same percentages, only a similarly-shaped one. Raw per-dataset results behind these specific numbers are attached to the `v0.1.0` GitHub release as `audit_results_100.json` and `audit_summary_100.txt`.

## Performance note: Hub vs. local

Linting a 100-episode dataset locally takes under 30 seconds.

Linting a Hub dataset directly (`trajlens lint org/dataset`) streams metadata and data shards over HTTP. It will inherently be slower than a local copy — typically under a minute for small-to-medium datasets, more for very large ones — because of unavoidable network round trips. For repeated linting, downloading the dataset locally first is still faster.

## License

Apache-2.0
