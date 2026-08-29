"""Tests for the format-neutral FrameSource/FrameBatch machinery (v0.5 M1-B).

Per 00_AGENT_OPERATING_MANUAL.md §5: happy path, at least two failure modes,
and one edge case for this unit of work.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.fixtures.builders import build_v2_dataset, build_v3_dataset
from trajlens.errors import DatasetFormatError
from trajlens.model import build_canonical_dataset
from trajlens.model.lerobot_v2 import _feature_spec_from_arrow
from trajlens.sources.loader import SourceLoader


def _resolve(tmp_path: Path):  # type: ignore[no-untyped-def]
    return SourceLoader().resolve(str(tmp_path))


class TestHappyPathV3:
    def test_frame_source_num_rows_matches_episode_length(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=3)
        ds = build_canonical_dataset(_resolve(tmp_path))
        ep = ds.episode(1)

        source = ds.frame_source_for_episode(ep)

        assert source.num_rows() == ep.length

    def test_iter_batches_only_yields_this_episodes_rows(self, tmp_path: Path) -> None:
        """v3.0 shards can hold multiple episodes' frames concatenated -- the
        FrameSource must filter to just this episode's rows, not the whole shard."""
        build_v3_dataset(tmp_path, num_episodes=3)
        ds = build_canonical_dataset(_resolve(tmp_path))
        ep = ds.episode(1)

        source = ds.frame_source_for_episode(ep)
        batches = list(source.iter_batches(columns=["episode_index"]))

        total_rows = sum(b.num_rows for b in batches)
        assert total_rows == ep.length
        for batch in batches:
            assert set(batch.columns["episode_index"].tolist()) == {ep.episode_index}

    def test_schema_maps_scalar_columns(self, tmp_path: Path) -> None:
        build_v3_dataset(tmp_path, num_episodes=1)
        ds = build_canonical_dataset(_resolve(tmp_path))
        ep = ds.episode(0)

        schema = ds.frame_source_for_episode(ep).schema()

        assert schema["timestamp"].dtype == "float32"
        assert schema["timestamp"].shape == (1,)
        assert schema["episode_index"].dtype == "int64"


class TestHappyPathV2:
    def test_frame_source_num_rows_and_schema(self, tmp_path: Path) -> None:
        build_v2_dataset(tmp_path, codebase_version="v2.1", num_episodes=2)
        ds = build_canonical_dataset(_resolve(tmp_path))
        ep = ds.episode(0)

        source = ds.frame_source_for_episode(ep)

        assert source.num_rows() == ep.length
        schema = source.schema()
        assert schema["timestamp"].dtype == "float32"

        batches = list(source.iter_batches())
        assert sum(b.num_rows for b in batches) == ep.length


class TestFailureModes:
    def test_unsupported_scalar_arrow_type_raises_format_error(self) -> None:
        with pytest.raises(DatasetFormatError, match="unsupported Arrow type"):
            _feature_spec_from_arrow("weird", pa.date32())

    def test_unsupported_fixed_size_list_element_type_raises_format_error(self) -> None:
        with pytest.raises(DatasetFormatError, match="unsupported fixed-size-list"):
            _feature_spec_from_arrow("weird_vec", pa.list_(pa.date32(), 3))


class TestEdgeCase:
    def test_fixed_size_list_column_maps_to_shaped_feature(self, tmp_path: Path) -> None:
        """Edge case: a vector-valued (fixed-size-list) column, which none of
        the standard fixture builders produce, still maps to a shaped FeatureSpec."""
        table = pa.table(
            {
                "state": pa.array(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    type=pa.list_(pa.float32(), 3),
                )
            }
        )
        shard_path = tmp_path / "shard.parquet"
        pq.write_table(table, shard_path)
        pf = pq.ParquetFile(shard_path)

        spec = _feature_spec_from_arrow("state", pf.schema_arrow.field("state").type)

        assert spec.dtype == "float32"
        assert spec.shape == (3,)
