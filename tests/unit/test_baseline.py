"""Unit tests for BaselineStore: load/save, diff, and identity-key invariants."""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from trajlens.baseline import BASELINE_SCHEMA_VERSION, BaselineStore, IdentityKey, _result_identity
from trajlens.checks.protocol import CheckResult, Severity
from trajlens.errors import DatasetFormatError


def _r(check_id: str, severity: Severity = Severity.FAIL) -> CheckResult:
    return CheckResult(check_id=check_id, severity=severity, message="test")


class TestDiffHappyPath:
    def test_finding_in_both_is_unchanged(self, tmp_path: Path) -> None:
        result = _r("STRUCTURAL.X")
        store = BaselineStore.from_results([result])
        diff = store.diff([result])
        assert diff.unchanged == [result]
        assert diff.new == []
        assert diff.resolved == []

    def test_finding_only_in_current_is_new(self) -> None:
        store = BaselineStore.from_results([])
        result = _r("STRUCTURAL.X")
        diff = store.diff([result])
        assert diff.new == [result]
        assert diff.unchanged == []
        assert diff.resolved == []

    def test_finding_only_in_baseline_is_resolved(self) -> None:
        result = _r("STRUCTURAL.X")
        store = BaselineStore.from_results([result])
        diff = store.diff([])
        assert diff.resolved[0].check_id == "STRUCTURAL.X"
        assert diff.new == []
        assert diff.unchanged == []


class TestUpdateBaseline:
    def test_save_writes_exactly_current_findings(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        results = [_r("STRUCTURAL.X"), _r("TEMPORAL.Y")]
        BaselineStore.from_results(results).save(path)

        loaded = BaselineStore.load(path)
        assert {f.check_id for f in loaded.findings} == {"STRUCTURAL.X", "TEMPORAL.Y"}
        assert len(loaded.findings) == 2

    def test_save_overwrites_not_appends(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        BaselineStore.from_results([_r("A.ONE")]).save(path)
        BaselineStore.from_results([_r("B.TWO")]).save(path)

        loaded = BaselineStore.load(path)
        assert [f.check_id for f in loaded.findings] == ["B.TWO"]


class TestCorruptedBaseline:
    def test_missing_file_raises_dataset_format_error(self, tmp_path: Path) -> None:
        try:
            BaselineStore.load(tmp_path / "nonexistent.json")
            raise AssertionError("expected DatasetFormatError")
        except DatasetFormatError:
            pass

    def test_invalid_json_raises_dataset_format_error(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        path.write_text("{not valid json", encoding="utf-8")
        try:
            BaselineStore.load(path)
            raise AssertionError("expected DatasetFormatError")
        except DatasetFormatError:
            pass

    def test_missing_required_field_raises_dataset_format_error(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        path.write_text(
            json.dumps({"schema_version": BASELINE_SCHEMA_VERSION, "findings": [{}]}),
            encoding="utf-8",
        )
        try:
            BaselineStore.load(path)
            raise AssertionError("expected DatasetFormatError")
        except DatasetFormatError:
            pass


class TestSchemaVersionMismatch:
    def test_mismatch_names_both_versions(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        path.write_text(
            json.dumps({"schema_version": "999", "findings": []}),
            encoding="utf-8",
        )
        try:
            BaselineStore.load(path)
            raise AssertionError("expected DatasetFormatError")
        except DatasetFormatError as exc:
            message = str(exc)
            assert "999" in message
            assert BASELINE_SCHEMA_VERSION in message


class TestDeterminism:
    def test_relinting_unchanged_dataset_produces_zero_new(self, tmp_path: Path) -> None:
        path = tmp_path / "baseline.json"
        results = [_r("STRUCTURAL.X"), _r("TEMPORAL.Y", Severity.WARN)]
        BaselineStore.from_results(results).save(path)

        for _ in range(3):
            loaded = BaselineStore.load(path)
            diff = loaded.diff(results)
            assert diff.new == []


_check_id_strategy = st.sampled_from(["STRUCTURAL.A", "TEMPORAL.B", "STATISTICAL.C"])
_episode_strategy = st.one_of(st.none(), st.integers(min_value=0, max_value=100))
_shard_strategy = st.one_of(st.none(), st.sampled_from(["data/chunk-000/ep0.parquet", None]))


@st.composite
def _finding_sets(draw: st.DrawFn) -> tuple[list[CheckResult], list[CheckResult]]:
    identities = draw(
        st.lists(
            st.tuples(_check_id_strategy, _episode_strategy, _shard_strategy),
            min_size=0,
            max_size=8,
            unique=True,
        )
    )
    results = [
        CheckResult(
            check_id=check_id,
            severity=Severity.FAIL,
            message="test",
            per_episode={episode: "x"} if episode is not None else None,
            details={"shard_path": shard} if shard is not None else {},
        )
        for check_id, episode, shard in identities
    ]
    indices = draw(
        st.lists(st.integers(min_value=0, max_value=len(results) - 1), unique=True)
        if results
        else st.just([])
    )
    baseline_subset = [results[i] for i in indices]
    return results, baseline_subset


class TestDiffProperty:
    @given(_finding_sets())
    def test_new_and_resolved_match_set_difference(
        self, data: tuple[list[CheckResult], list[CheckResult]]
    ) -> None:
        current, baseline_subset = data
        store = BaselineStore.from_results(baseline_subset)
        diff = store.diff(current)

        baseline_identities: set[IdentityKey] = {_result_identity(r) for r in baseline_subset}
        current_identities: set[IdentityKey] = {_result_identity(r) for r in current}

        assert {_result_identity(r) for r in diff.new} == current_identities - baseline_identities
        assert {f.identity() for f in diff.resolved} == baseline_identities - current_identities
