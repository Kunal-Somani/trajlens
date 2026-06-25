# Changelog

All notable changes to trajlens are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

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
