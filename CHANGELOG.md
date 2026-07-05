# Changelog

All notable changes to trajlens are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

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
