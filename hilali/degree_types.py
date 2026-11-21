"""Degree types of elliptic minimal Sullivan algebras.

A *degree type* is a pair ``(evens, odds)`` of nondecreasing tuples, the
degrees of the even and of the odd generators. By Friedlander and Halperin,
a degree type is realised by an elliptic minimal Sullivan algebra if and only
if it satisfies the *strong arithmetic condition* (SAC): for every subset
``S`` of the even generators, there are at least ``|S|`` odd generators ``y``
such that ``|y| + 1`` is a sum of at least two degrees of elements of ``S``
(repetitions allowed). The elliptic algebras of a type have formal dimension

    N = sum(odds) - sum(e - 1 for e in evens),

and satisfy ``sum(evens) <= N`` and ``sum(odds) <= 2N - 1``.
"""

from __future__ import annotations

import itertools
from typing import List, Sequence, Tuple

DegreeType = Tuple[Tuple[int, ...], Tuple[int, ...]]


def is_decomposable_degree(target: int, degrees: Sequence[int]) -> bool:
    """``target`` is a sum of at least two elements of ``degrees``, with repetition."""
    gens = sorted(set(degrees))
    if not gens or target <= 0:
        return False
    counts: List[set] = [set() for _ in range(target + 1)]
    counts[0].add(0)
    for v in range(1, target + 1):
        for g in gens:
            if g <= v:
                counts[v].update(min(c + 1, 2) for c in counts[v - g])
    return 2 in counts[target]


def satisfies_sac(evens: Sequence[int], odds: Sequence[int]) -> bool:
    """The strong arithmetic condition of Friedlander and Halperin."""
    r = len(evens)
    if len(odds) < r:
        return False
    for k in range(1, r + 1):
        for subset in itertools.combinations(range(r), k):
            degs = [evens[i] for i in subset]
            if sum(1 for y in odds if is_decomposable_degree(y + 1, degs)) < k:
                return False
    return True


def formal_dimension(evens: Sequence[int], odds: Sequence[int]) -> int:
    return sum(odds) - sum(e - 1 for e in evens)


def _partitions(parts: Sequence[int], total_max: int) -> List[Tuple[int, ...]]:
    """All nonincreasing tuples of elements of ``parts`` with sum at most ``total_max``."""
    parts = sorted(set(parts), reverse=True)
    out: List[Tuple[int, ...]] = []

    def extend(prefix: List[int], largest: int, remaining: int) -> None:
        out.append(tuple(prefix))
        for q in parts:
            if q <= largest and q <= remaining:
                extend(prefix + [q], q, remaining - q)

    extend([], max(parts, default=0), total_max)
    return out


def elliptic_degree_types(n_max: int, n_min: int = 2) -> List[Tuple[Tuple[int, ...], Tuple[int, ...], int]]:
    """All degree types satisfying the SAC with ``n_min <= N <= n_max``.

    Returns triples ``(evens, odds, N)`` with sorted tuples. Types with no
    generators at all are excluded.
    """
    even_parts = list(range(2, n_max + 1, 2))
    odd_parts = list(range(3, 2 * n_max, 2))
    out = []
    for evens in _partitions(even_parts, n_max):
        shift = sum(e - 1 for e in evens)
        for odds in _partitions(odd_parts, 2 * n_max - 1):
            if not odds or len(odds) < len(evens):
                continue
            n = sum(odds) - shift
            if n < n_min or n > n_max or sum(evens) > n or sum(odds) > 2 * n - 1:
                continue
            if satisfies_sac(evens, odds):
                out.append((tuple(sorted(evens)), tuple(sorted(odds)), n))
    return out


def basis_sizes(degrees: Sequence[int], n_max: int) -> List[int]:
    """``dim (Lambda V)^n`` for ``0 <= n <= n_max``, from the degrees alone."""
    counts = [0] * (n_max + 1)
    counts[0] = 1
    for g in degrees:
        if g % 2:
            for k in range(n_max, g - 1, -1):
                counts[k] += counts[k - g]
        else:
            for k in range(g, n_max + 1):
                counts[k] += counts[k - g]
    return counts


def max_basis_size(degrees: Sequence[int], offset: int = 1) -> int:
    """Largest ``dim (Lambda V)^n`` for ``n <= N + offset``, a proxy for the cost of a computation."""
    evens = [g for g in degrees if g % 2 == 0]
    odds = [g for g in degrees if g % 2]
    n = formal_dimension(evens, odds)
    return max(basis_sizes(degrees, max(n + offset, 0)))
