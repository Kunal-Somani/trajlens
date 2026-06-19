"""Tests for resource bound enforcement (T2 — allocation/iteration bombs)."""

from __future__ import annotations

import pytest

from trajlens.errors import ResourceBoundError
from trajlens.sources.bounds import MAX_DECLARED_EPISODES, check_resource_bound


class TestCheckResourceBound:
    def test_under_bound_does_not_raise(self) -> None:
        check_resource_bound(10, max_value=MAX_DECLARED_EPISODES, what="episode count")

    def test_at_bound_does_not_raise(self) -> None:
        check_resource_bound(
            MAX_DECLARED_EPISODES, max_value=MAX_DECLARED_EPISODES, what="episode count"
        )

    def test_over_bound_raises(self) -> None:
        with pytest.raises(ResourceBoundError, match="10000000000"):
            check_resource_bound(
                10_000_000_000, max_value=MAX_DECLARED_EPISODES, what="episode count"
            )

    def test_error_message_states_ceiling(self) -> None:
        with pytest.raises(ResourceBoundError, match=str(MAX_DECLARED_EPISODES)):
            check_resource_bound(
                MAX_DECLARED_EPISODES + 1, max_value=MAX_DECLARED_EPISODES, what="episode count"
            )
