"""Hard ceilings on dataset-declared counts (T2 in the threat model).

A broken or hostile dataset can declare an absurd episode/frame count.
check_resource_bound() is the single place that decision gets made: exceed
the ceiling and we stop with ResourceBoundError, we never attempt to
iterate. The ceilings are a security policy choice, not a format fact —
generous enough for any real public LeRobot dataset, small enough to fail
fast on a hostile or corrupt declaration.
"""

from __future__ import annotations

from typing import Final

from trajlens.errors import ResourceBoundError

MAX_DECLARED_EPISODES: Final[int] = 1_000_000
MAX_DECLARED_FRAMES: Final[int] = 500_000_000


def check_resource_bound(value: int, *, max_value: int, what: str) -> None:
    """Raise ResourceBoundError if a dataset-declared count exceeds max_value."""
    if value > max_value:
        raise ResourceBoundError(
            f"declared {what} is {value}, which exceeds the hard ceiling of "
            f"{max_value}. Refusing to process — this dataset is either "
            f"corrupt or hostile."
        )
