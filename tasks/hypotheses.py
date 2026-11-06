"""Hypothesis space for the number game (Tenenbaum 1999, Bigelow & Piantadosi 2016).

Each hypothesis is a frozen dataclass with a name, human-readable description,
and membership predicate; rules and intervals are kept in separate lists.  The HypothesisRegistry
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


def is_square(x: int) -> bool:
    root = math.isqrt(x)
    return root * root == x


def is_cube(x: int) -> bool:
    return any(n ** 3 == x for n in range(1, math.isqrt(x) + 2))


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

    def support(self, domain: range) -> frozenset[int]:
        """Compute the set of domain values satisfying the predicate."""
        return frozenset(x for x in domain if self.predicate(x))


class HypothesisRegistry:
    """Composable hypothesis builder: bases × transforms → deduplicated rules."""

    def __init__(self, domain: int = 100):
        self.domain = domain
        self.numbers = range(1, domain + 1)
        self.bases: list[Hypothesis] = []
        self.transforms: dict[str, Callable[[int], int]] = {}

    def add_base(self, name: str, description: str, predicate: Callable[[int], bool]) -> None:
        self.bases.append(Hypothesis(name=name, description=description, predicate=predicate))

    def add_transform(self, name: str, func: Callable[[int], int]) -> None:
        self.transforms[name] = func

    def compose(self, base: Hypothesis, tname: str) -> Hypothesis:
        """Apply transform tname to base: support = {f(n) : n in base.support} ∩ domain."""
        func = self.transforms[tname]
        support = frozenset(v for v in (func(n) for n in base.support(self.numbers)) if 1 <= v <= self.domain)
        if tname == "identity":
            name = base.name
            description = base.description
        else:
            name = f"{base.name}_{tname}"
            description = f"{base.description} ({tname})"
        return Hypothesis(
            name=name, description=description,
            predicate=lambda x, _support=support: x in _support,
        )

    def build(self, transforms: list[str] | tuple[str, ...] = ("identity",),
              ) -> tuple[list[Hypothesis], list[Hypothesis], dict[str, list[str]] | None]:
        """Compose all bases × transforms, deduplicate by support, filter len < 3.

        Returns (rules, intervals, rule_families).
        rule_families maps base description → surviving composed rules when the
        prompt uses transformed rule families; identity-only spaces return None.
        """
        seen: dict[frozenset[int], Hypothesis] = {}
        rule_families: dict[str, list[str]] = {}
        for base in self.bases:
            for tname in transforms:
                h = self.compose(base, tname)
                support = h.support(self.numbers)
                if len(support) < 3:
                    continue
                # First-seen wins for deduplication
                if support not in seen:
                    seen[support] = h
                    rule_families.setdefault(base.description, []).append(h.description)
        rules = list(seen.values())
        intervals = _build_intervals(self.domain)
        if all(tname == "identity" for tname in transforms):
            return rules, intervals, None
        return rules, intervals, rule_families


def _build_intervals(domain: int) -> list[Hypothesis]:
    """Natural [a, b] interval hypotheses: endpoints are multiples of 5, starting from 0."""
    pairs = ((a, b) for a in range(0, domain + 1, 5) for b in range(max(a, 5), domain + 1, 5))
    return [Hypothesis(name=f"[{a},{b}]", description=interval_label(a, b),
                       predicate=lambda x, _a=a, _b=b: _a <= x <= _b)
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


def default_registry(domain: int = 100, preset: str = "all") -> HypothesisRegistry:
    """Build hypothesis registry with bases selected by preset.

    preset="all": Tenenbaum99 — all 34 math-rule bases (primordials + powers,
                  ending-in-d, 5n+k).
    preset="bigelow": B&P 2016 primordial sets only (odd, even, squares, cubes,
                      primes, multiples of 3..12).
    """
    r = HypothesisRegistry(domain)

    r.add_base("odd", "odd numbers", lambda x: x % 2 == 1)
    r.add_base("even", "even numbers", lambda x: x % 2 == 0)
    r.add_base("squares", "perfect squares", is_square)
    r.add_base("cubes", "perfect cubes", is_cube)
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

    r.add_transform("identity", lambda n: n)
    r.add_transform("n+1", lambda n: n + 1)
    r.add_transform("n-1", lambda n: n - 1)
    r.add_transform("n+2", lambda n: n + 2)
    r.add_transform("n-2", lambda n: n - 2)
    r.add_transform("2n", lambda n: 2 * n)
    r.add_transform("3n", lambda n: 3 * n)
    r.add_transform("2n+1", lambda n: 2 * n + 1)
    r.add_transform("3n-1", lambda n: 3 * n - 1)
    r.add_transform("3n+1", lambda n: 3 * n + 1)
    r.add_transform("2^n", lambda n: 2 ** n)
    r.add_transform("2^(n+1)", lambda n: 2 ** (n + 1))
    r.add_transform("2^n+1", lambda n: 2 ** n + 1)
    r.add_transform("2^n-1", lambda n: 2 ** n - 1)

    return r
