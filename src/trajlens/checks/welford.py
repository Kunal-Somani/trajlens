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
    """

    _count: int = field(default=0, init=False)
    _mean: float = field(default=0.0, init=False)
    _m2: float = field(default=0.0, init=False)
    _min: float = field(default=math.inf, init=False)
    _max: float = field(default=-math.inf, init=False)

    def update(self, value: float) -> None:
        """Incorporate one new scalar observation."""
        if math.isnan(value):
            # NaN is tracked via count so the NaN-scan check knows it appeared,
            # but we cannot fold it into mean/variance without poisoning both.
            # Callers that need NaN counts should count separately; here we
            # still bump count so the population size stays correct.
            self._count += 1
            return
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
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
        """Population mean. Returns 0.0 if no observations have been fed."""
        return self._mean

    @property
    def variance(self) -> float:
        """Population variance (divides by N). Returns 0.0 for n < 2."""
        if self._count < 2:
            return 0.0
        return self._m2 / self._count

    @property
    def std(self) -> float:
        """Population standard deviation. Returns 0.0 for n < 2."""
        return math.sqrt(self.variance)

    @property
    def min(self) -> float:
        """Observed minimum. Returns +inf if no observations."""
        return self._min

    @property
    def max(self) -> float:
        """Observed maximum. Returns -inf if no observations."""
        return self._max
