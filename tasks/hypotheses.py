"""Hypothesis space for the number game (Tenenbaum 1999, Bigelow & Piantadosi 2016).

Each hypothesis is a frozen dataclass with a name, human-readable description,
membership predicate, and kind ("rule" or "interval").  The HypothesisRegistry
supports composable transforms: bases × transforms → deduplicated rule set.
"""

import math
from dataclasses import dataclass
from typing import Callable

from tasks.prompts import interval_label


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def _is_power(base: int, x: int) -> bool:
    """True if x is a positive integer power of base (base^1, base^2, ...)."""
    if x < base:
        return False
    val = base
    while val < x:
        val *= base
    return val == x


@dataclass(frozen=True)
class Hypothesis:
    name: str
    description: str
    predicate: Callable[[int], bool]
    kind: str  # "rule" | "interval"

    def extension(self, domain: range) -> frozenset[int]:
        """Compute the set of domain values satisfying the predicate."""
        return frozenset(x for x in domain if self.predicate(x))


class HypothesisRegistry:
    """Composable hypothesis builder: bases × transforms → deduplicated rules."""

    def __init__(self, max_integer: int = 100):
        self.max_integer = max_integer
        self.domain = range(1, max_integer + 1)
        self.bases: list[Hypothesis] = []
        self.transforms: dict[str, tuple[str, Callable[[int], int]]] = {}

    def add_base(self, name: str, description: str, predicate: Callable[[int], bool]) -> None:
        self.bases.append(Hypothesis(name=name, description=description, predicate=predicate, kind="rule"))

    def add_transform(self, name: str, description: str, func: Callable[[int], int]) -> None:
        self.transforms[name] = (description, func)

    def compose(self, base: Hypothesis, tname: str) -> Hypothesis:
        """Apply transform tname to base: extension = {f(n) : n in base.extension} ∩ domain."""
        desc, func = self.transforms[tname]
        base_ext = base.extension(self.domain)
        # Precompute the transformed extension within domain
        ext = frozenset(v for v in (func(n) for n in base_ext) if 1 <= v <= self.max_integer)
        if tname == "identity":
            name = base.name
            full_desc = base.description
        else:
            name = f"{base.name}_{tname}"
            full_desc = f"{base.description} ({desc})"
        return Hypothesis(
            name=name, description=full_desc,
            predicate=lambda x, _ext=ext: x in _ext, kind="rule",
        )

    def build(self, transforms: list[str] | tuple[str, ...] = ("identity",),
              natural_intervals: bool = False,
              ) -> tuple[list[Hypothesis], list[Hypothesis], dict[str, list[str]] | None]:
        """Compose all bases × transforms, deduplicate by extension, filter len < 3.

        Returns (rules, intervals, rule_families).
        rule_families maps base description → surviving composed rules when the
        prompt uses transformed rule families; identity-only spaces return None.
        """
        seen: dict[frozenset[int], Hypothesis] = {}
        rule_families: dict[str, list[str]] = {}
        for base in self.bases:
            for tname in transforms:
                h = self.compose(base, tname)
                ext = h.extension(self.domain)
                if len(ext) < 3:
                    continue
                # First-seen wins for deduplication
                if ext not in seen:
                    seen[ext] = h
                    rule_families.setdefault(base.description, []).append(h.description)
        rules = list(seen.values())
        intervals = _build_intervals(self.domain, natural=natural_intervals)
        if all(tname == "identity" for tname in transforms):
            return rules, intervals, None
        return rules, intervals, rule_families


def _build_intervals(domain: range, natural: bool = False) -> list[Hypothesis]:
    """All [a, b] interval hypotheses over domain.

    natural=True: endpoints are multiples of 5, starting from 0.
    """
    max_int = domain.stop - 1
    if natural:
        starts = range(0, max_int + 1, 5)
        pairs = ((a, b) for a in starts for b in range(max(a, 5), max_int + 1, 5))
    else:
        pairs = ((a, b) for a in domain for b in range(a, max_int + 1))
    return [Hypothesis(name=f"[{a},{b}]", description=interval_label(a, b),
                       predicate=lambda x, _a=a, _b=b: _a <= x <= _b, kind="interval")
            for a, b in pairs]


def aggregate_by_rule_family(
    posteriors: dict[str, float], rule_families: dict[str, list[str]],
) -> dict[str, float]:
    """Aggregate per-rule posteriors to base-level by summing over transforms."""
    result = {}
    for base_desc, rule_descs in rule_families.items():
        result[base_desc] = sum(posteriors.get(d, 0.0) for d in rule_descs)
    if "other" in posteriors:
        result["other"] = posteriors["other"]
    return result


def default_registry(max_integer: int = 100, preset: str = "all") -> HypothesisRegistry:
    """Build hypothesis registry with bases selected by preset.

    preset="all": Tenenbaum99 — all 34 math-rule bases (primordials + powers,
                  ending-in-d, 5n+k).
    preset="bigelow": B&P 2016 primordial sets only (odd, even, squares, cubes,
                      primes, multiples of 3..12).
    """
    r = HypothesisRegistry(max_integer)

    r.add_base("odd", "odd numbers", lambda x: x % 2 == 1)
    r.add_base("even", "even numbers", lambda x: x % 2 == 0)
    r.add_base("squares", "perfect squares", lambda x: math.isqrt(x) ** 2 == x)
    r.add_base("cubes", "perfect cubes", lambda x: round(x ** (1/3)) ** 3 == x)
    r.add_base("primes", "prime numbers", is_prime)
    for n in range(3, 13):
        r.add_base(f"mult_{n}", f"multiples of {n}", lambda x, _n=n: x % _n == 0)

    if preset == "all":
        for base in range(2, 11):
            r.add_base(f"pow_{base}", f"powers of {base}", lambda x, _b=base: _is_power(_b, x))
        for d in range(10):
            r.add_base(f"end_{d}", f"numbers ending in {d}", lambda x, _d=d: x % 10 == _d)
        for k in range(1, 5):
            r.add_base(f"5n+{k}", f"numbers of the form 5n+{k}", lambda x, _k=k: x % 5 == _k % 5)

    r.add_transform("identity", "identity", lambda n: n)
    r.add_transform("n+1", "n+1", lambda n: n + 1)
    r.add_transform("n-1", "n-1", lambda n: n - 1)
    r.add_transform("n+2", "n+2", lambda n: n + 2)
    r.add_transform("n-2", "n-2", lambda n: n - 2)
    r.add_transform("2n", "2n", lambda n: 2 * n)
    r.add_transform("3n", "3n", lambda n: 3 * n)
    r.add_transform("2n+1", "2n+1", lambda n: 2 * n + 1)
    r.add_transform("3n-1", "3n-1", lambda n: 3 * n - 1)
    r.add_transform("3n+1", "3n+1", lambda n: 3 * n + 1)
    r.add_transform("2^n", "2^n", lambda n: 2 ** n)
    r.add_transform("2^(n+1)", "2^(n+1)", lambda n: 2 ** (n + 1))
    r.add_transform("2^n+1", "2^n+1", lambda n: 2 ** n + 1)
    r.add_transform("2^n-1", "2^n-1", lambda n: 2 ** n - 1)

    return r
