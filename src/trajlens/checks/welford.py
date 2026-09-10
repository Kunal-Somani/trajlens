"""Welford's online algorithm for streaming mean and variance (05 §6: stream, don't slurp).

Implements the single-pass algorithm from:
  B. P. Welford (1962). "Note on a method for calculating corrected sums of
  squares and products." Technometrics 4(3): 419-420.

This is the only correct approach for large datasets that may not fit in RAM —
loading everything into memory and calling numpy.mean/std would violate the
streaming memory discipline required by 05_ENGINEERING_STANDARDS.md §6 and
06_SECURITY_AND_THREAT_MODEL.md T2.

Per-feature accumulators are used by STATISTICAL.STATS_MATCH_DATA and
STATISTICAL.PER_EPISODE_STATS_MATCH.  Each accumulator tracks one scalar
stream; callers maintain one WelfordAccumulator per feature column.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class WelfordAccumulator:
    """Single-pass online mean/variance accumulator for a scalar stream.

    Maintains count, running mean (M1), and running sum of squared deviations
    from the mean (M2) using Welford's recurrence.  Call update() for each
    scalar value, then query mean/variance/std/min/max after streaming all data.

    NaN handling: a NaN observation bumps ``count`` (the population size a
    caller reports, e.g. into stats.json's "count" field, must still include
    every frame seen, NaN or not) but must NEVER be folded into the Welford
    recurrence itself. Welford's recurrence divides by the number of
    observations incorporated so far (``delta / n``) -- if a NaN observation
    bumped that same denominator without contributing a value, every
    subsequent valid observation would be silently under-weighted, biasing
    mean and variance for the rest of the stream. A separate internal
    ``_valid_count`` is the recurrence's actual denominator; ``count`` and
    ``_valid_count`` coincide whenever no NaN has been seen.
    """

    _count: int = field(default=0, init=False)
    _valid_count: int = field(default=0, init=False)
    _mean: float = field(default=0.0, init=False)
    _m2: float = field(default=0.0, init=False)
    _min: float = field(default=math.inf, init=False)
    _max: float = field(default=-math.inf, init=False)

    def update(self, value: float) -> None:
        """Incorporate one new scalar observation."""
        self._count += 1
        if math.isnan(value):
            # Counted above (the population size includes every frame seen),
            # but never folded into the mean/variance recurrence -- see the
            # class docstring for why bumping the recurrence's own
            # denominator here would poison every later observation, not
            # just this one.
            return
        self._valid_count += 1
        delta = value - self._mean
        self._mean += delta / self._valid_count
        delta2 = value - self._mean
        self._m2 += delta * delta2
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        """Mean over non-NaN observations. Returns 0.0 if none have been fed."""
        return self._mean

    @property
    def variance(self) -> float:
        """Population variance over non-NaN observations. Returns 0.0 for n < 2."""
        if self._valid_count < 2:
            return 0.0
        return self._m2 / self._valid_count

    @property
    def std(self) -> float:
        """Population standard deviation over non-NaN observations. Returns 0.0 for n < 2."""
        return math.sqrt(self.variance)

    @property
    def min(self) -> float:
        """Observed minimum. Returns +inf if no observations."""
        return self._min

    @property
    def max(self) -> float:
        """Observed maximum. Returns -inf if no observations."""
        return self._max
