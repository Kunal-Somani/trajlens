"""Utilities for check implementations."""

from __future__ import annotations

from typing import Any

from trajlens.model.canonical import CanonicalDataset, EpisodeRecord


class ShardColumnCache:
    """Reads distinct Parquet shards exactly once to avoid O(N^2) HTTP reads.

    In Hub-hosted datasets with multi-episode shards, calling pf.read() for
    each episode incurs an extreme penalty because the same column is re-read
    from the network for every episode sharing the shard.

    This cache reads the required columns once per shard, memoizes the entire
    shard's columns in memory as Python lists, and precomputes the slice bounds
    for every episode contained within that shard in O(K) time, enabling O(1)
    slice extraction for all subsequent episodes in the shard.
    """

    def __init__(self, columns: list[str]):
        # We implicitly require episode_index to group rows.
        if "episode_index" not in columns:
            self.columns = [*columns, "episode_index"]
        else:
            self.columns = columns

        self._cached_episodes: set[int] = set()
        self._cols_pylist: dict[str, list[Any]] = {}
        self._ep_bounds: dict[int, tuple[int, int]] = {}

    def get_episode_data(
        self, ds: CanonicalDataset, episode: EpisodeRecord
    ) -> dict[str, list[Any]]:
        """Return the sliced column data for the given episode.

        If the episode is not in the currently cached shard, the appropriate
        shard is read and cached. Returns a dictionary mapping column name
        to a list of values belonging to the requested episode.
        """
        if episode.episode_index not in self._cached_episodes:
            pf = ds.parquet_shard_for_episode(episode)
            table = pf.read(columns=self.columns)  # type: ignore[no-untyped-call]
            self._cols_pylist = {col: table.column(col).to_pylist() for col in self.columns}

            # Precompute episode boundaries for O(1) slicing
            self._ep_bounds.clear()
            ep_mask = self._cols_pylist["episode_index"]

            current_ep = None
            start_idx = 0
            for i, v in enumerate(ep_mask):
                if v != current_ep:
                    if current_ep is not None:
                        self._ep_bounds[current_ep] = (start_idx, i)
                    current_ep = v
                    start_idx = i
            if current_ep is not None:
                self._ep_bounds[current_ep] = (start_idx, len(ep_mask))

            self._cached_episodes = set(ep_mask)

        start_i, end_i = self._ep_bounds.get(episode.episode_index, (0, 0))
        return {col: self._cols_pylist[col][start_i:end_i] for col in self.columns}
