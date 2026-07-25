# v0.4.0 release checklist results

## Corpus regression gate (07 §5)

- `bash scripts/run_tests.sh`: 638 passed, 1 deselected, coverage 93.22%.
- `tests/unit/test_corpus_fixtures.py -v --no-cov`: 60 passed.
- `tests/unit/test_scale.py -v --no-cov -k determinism`:
  `test_serial_and_parallel_produce_identical_results` passed — zero
  verdict differences between serial and parallel execution.

## Pre-release security checklist (06 §6)

- `pip-audit`: no known vulnerabilities found in audited (PyPI)
  dependencies. Non-PyPI ROS2 message packages present in the local
  environment are not auditable by pip-audit and are not project
  dependencies (pre-existing environment condition, see STATUS.md).
- `bandit -ll -r src/trajlens/`: no medium/high issues (5 low-confidence
  informational only).
- Redaction test: `tests/unit/test_scaffold.py` covers `_redact_secrets`
  for `token`, `password`, `secret`, `authorization`, `bearer` keys,
  case-insensitive; passed in full suite run.
- Path-traversal fixtures rejected: `tests/unit/test_sources_paths.py::
  test_rejects_symlink_escaping_root` passed.
- Allocation-bomb fixtures produce ERROR: `tests/unit/test_sources_bounds.py::
  test_over_bound_raises` passed (declared-size ceiling enforced,
  raises before allocation).
- `fix` round-trip tests green: all round-trip test modules
  (timestamp_dedrift, video_metadata_sync, stats_recompute,
  episode_reindex, orphan_shard_report, task_index_repair,
  sources_loader, sources_handles, per_episode_findings) passed in
  full suite run.

## v0.4-specific checks

- T1 (GitHub Action entrypoint): `.github/actions/lint/entrypoint.sh`
  builds `args=("lint" "${INPUT_DATASET_REF}")` as an array; dataset-ref
  is never interpolated into a shell string.
- T2 (baseline read path): `src/trajlens/baseline.py` calls
  `BaselineFile.model_validate(raw)` (Pydantic) after a
  `schema_version` check and before any field access.
- T4 (watch mode path containment): `src/trajlens/watch.py` routes
  every candidate path through `safe_join(self._root, *relative.parts)`
  before use.
- T3 (worker isolation): `src/trajlens/checks/engine.py` runs
  thread-safe checks via `ProcessPoolExecutor`, submitting one future
  per check with results written to per-check indexed slots; no shared
  mutable state passed between workers.
