# Changelog

All notable changes to trajlens are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

## [0.3.0] - 2026-07-21

### Added
- Per-episode findings (#37): `CheckResult` gains an optional `per_episode: dict[int, str] | None`
  field, populated by the three checks with independent per-episode signal --
  `STRUCTURAL.METADATA_DATA_AGREEMENT`, `TEMPORAL.TIMESTAMP_MONOTONIC`,
  `STATISTICAL.PER_EPISODE_STATS_MATCH`. `trajlens lint --json` gains an
  additive `episodes` key (worst-5 episodes ranked by trust contribution);
  the web dashboard renders a worst-episodes table via the existing external
  `app.js`/`style.css`
- `trajlens lint --share` / `--share-out <file>` (#37): redacted single-file
  JSON summary (trust score, grade, format version, per-check finding
  counts, worst-5 episodes) for pasting into a GitHub issue. Never includes
  `message`/`details`/`per_episode` free text, since check-authored strings
  can embed local paths. Follow-up in this release adds a `dataset_ref`
  field: the full Hub repo id for Hub datasets, or the local dataset
  directory's basename only -- never a parent path component -- for local
  paths
- `REPAIR.TASK_INDEX_REPAIR` fixer (#38): repairs `SEMANTIC.TASK_INTEGRITY`'s
  dangling-`task_index` finding using a nearest-valid-task heuristic
  (`argmin(|defined_index - N|)`); refuses with `RepairError` on equidistant
  candidates or an empty task table rather than guessing. v3.0 only
- `REPAIR.VIDEO_METADATA_SYNC` fixer (#39): rewrites `info.json`'s declared
  global `fps` to match the video container's own frame rate (read via PyAV,
  the same `av.open()` entry point `VIDEO.DECODABLE_SPOTCHECK` already uses).
  Targets `VIDEO.RESOLUTION_FPS_MATCH`, which remains catalog-only and
  unimplemented as a standalone check -- this fixer's ground-truth read
  bypasses it directly, and `--only` is the only way to invoke the fixer
  since the WARN+ auto-selection threshold can never reach it
- `STRUCTURAL.ORPHAN_SHARD` check (WARN) and `REPAIR.ORPHAN_SHARD_REPORT`
  fixer (#40): detects v3.0 data/video shards on disk that no episode record
  references (the reverse of `STRUCTURAL.PATH_TEMPLATE_RESOLVES`). Report-only
  by default; `--quarantine` relocates orphans to
  `<output>/.trajlens-quarantine/` with a manifest instead of just reporting
  them. Deletion is never implemented anywhere in the module
- `trajlens fix --only`/`--except` (#41): comma-separated fixer id(s) to
  force-run or exclude, validated against the known fixer set before any
  work starts; an unknown id or a same-id conflict between the two flags
  exits 2 with a typed message. All six fixers (including the three added
  in this release) are now wired into `trajlens fix`'s fixed execution order
  -- they previously were not
- Redaction-safe `dataset_ref` in issue reports and three GitHub issue forms
  (`false_positive.yml`, `bug_report.yml`, `new_corruption_class.yml`) plus a
  README "Found something?" section pointing to them

### Fixed
- `ArrowInvalid` typed-error wrap at the loader boundary (#41): a corrupt or
  non-Parquet shard previously let a raw `pyarrow.lib.ArrowInvalid` traceback
  escape from `open_parquet_shard` and `model/adapters.py`'s reads; now wrapped
  in `DatasetFormatError` naming the file. `open_hub_parquet_shard`'s
  not-found and found-but-corrupt cases are now distinguished with separate
  messages instead of one conflated error

### Changed
- `repair/protocol.py`'s `FrameChange.old_value`/`new_value` widened from
  `float` to `int | float` to correctly represent `task_index` (int64,
  #38); zero behavior change to `timestamp_dedrift`, its only prior user
- `uv.lock` is now committed (#41); previously gitignored, where the
  on-disk copy had silently drifted a release behind `pyproject.toml`

## [0.2.0] - 2026-07-05

### Added
- `src/trajlens/repair/`: repair engine with three fixers behind a common
  `Fixer` protocol (dry-run by default, copy-on-write, never mutates the
  original) —
  - `timestamp_dedrift.py`: rewrites frame timestamps to
    `quantize(frame_index / fps)` for `KNOWNBUG.TIMESTAMP_DRIFT`
  - `stats_recompute.py`: recomputes global + per-episode stats from data
    and rewrites `meta/stats.json` for `STATISTICAL.STATS_MATCH_DATA`
  - `episode_reindex.py`: rebuilds `dataset_from_index`/`dataset_to_index`
    boundaries from each shard's ground-truth `index` column for
    `STRUCTURAL.METADATA_DATA_AGREEMENT` (#2401); fails closed
    (`RepairError`, zero output) on internally-inconsistent `index` data
    with no independent ground truth to repair against
- `trajlens fix <ref>`: CLI command wiring all three fixers via a new
  `repair/orchestrator.py` (ADR-001). Dry-run by default; `--apply --out
  <path>` writes. Fixed composition order (`episode_reindex` ->
  `timestamp_dedrift` -> `stats_recompute`) via N temp-copy chaining, since
  the `Fixer` protocol has no in-place-rewrite mode. Hub refs are refused
  before any fixer runs (Hub `SourceHandle.root` only ever contains
  `meta/**` locally). Exit codes `0`/`1`/`2` = nothing to fix / fixed /
  could not fix
- `trajlens web <ref>`: read-only local dashboard over the lint report
  (optional `[web]` extra: FastAPI + uvicorn). Binds to 127.0.0.1 only, no
  flag to widen the bind; strict CSP/security headers on every response; no
  route accepts a path, ref, or dataset id from the browser
  (`06_SECURITY_AND_THREAT_MODEL.md` T10). Dashboard shell, CSS, and JS are
  served as external files (`static/index.html`, `static/style.css`,
  `static/app.js`) under a package-fixed `/static` mount, alongside
  `GET /api/report`
- `report/json_report.py`: `results[].details` now included in `--json`
  output (previously computed but dropped), giving the dashboard's
  drill-down access to the same structured detail the checks already
  produce

### Fixed
- Dashboard CSP release-blocker: index.html's inline `<script>`/`<style>`
  and `style=""` attributes were silently dropped by real browser CSP
  enforcement (`script-src 'self'; style-src 'self'`, no `unsafe-inline`),
  leaving the dashboard stuck at "Loading report..." with no styling.
  `TestClient` never enforces CSP so the test suite gave no signal. Fixed
  by externalizing CSS/JS rather than loosening the policy

## [0.1.0] - 2026-06-24

### Added
- Project scaffold: `pyproject.toml`, repo layout, CI pipeline
- `errors.py`: typed exception hierarchy
- `logging.py`: structlog configuration with secret redaction (T6 mitigation)
- `cli.py`: Typer CLI skeleton with `--version`; `lint`/`fix`/`web` stubs
- `SECURITY.md`, `CONTRIBUTING.md`, `LICENSE` (Apache-2.0)
- GitHub Actions: lint, typecheck, test, security, build CI jobs
- `sources/`: `safe_join` path-traversal defense, resource bounds, `info.json`
  parsing, v2.0/v2.1/v3.0 version detection, lazy Parquet/video shard handles,
  Hub streaming via `HfApi`/`HfFileSystem`
- `model/`: `CanonicalDataset` and per-version adapters over LeRobotDataset
  v2.0, v2.1, and v3.0
- `checks/`: `Check` protocol, `CheckEngine`, `CheckRegistry`, and 16 checks
  across STRUCTURAL, TEMPORAL, KNOWNBUG, VIDEO, SEMANTIC, and STATISTICAL
  categories, including `STRUCTURAL.METADATA_DATA_AGREEMENT` (catches
  lerobot's v2.1->v3.0 conversion corruption, [#2401](https://github.com/huggingface/lerobot/issues/2401))
  and `KNOWNBUG.TIMESTAMP_DRIFT` ([#3177](https://github.com/huggingface/lerobot/issues/3177))
- `report/`: trust score formula, terminal/JSON/HTML/SARIF renderers, CI exit
  codes (0/1/2 = PASS/WARN/FAIL)
- `scripts/audit_hub.py`: real-world audit harness against public Hub
  datasets; see README for the launch 100-dataset run results

### Fixed
- `KNOWNBUG.TIMESTAMP_DRIFT` false positive from float32 quantization at
  episode boundaries
- `SEMANTIC.FEATURE_DIMENSIONALITY` false positive on dict-shaped `names`
  feature metadata
- O(n_episodes) per-file metadata download and scattered Parquet shard reads
  causing Hub dataset timeouts
- `frame_index` namespacing crash on multi-camera Hub datasets
