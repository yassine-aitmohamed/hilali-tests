"""Weight gradings and the weighted Euler characteristic.

A weight grading of rank ``r`` assigns to each generator ``v`` a weight in
``Z^r`` such that ``d`` preserves the induced weight on ``Lambda V``. The
cohomology then splits as ``H = sum H^n_gamma`` and, for an elliptic
algebra, the weighted Euler characteristic is the Laurent polynomial

    chi_T(H) = sum_gamma ( sum_n (-1)^n dim H^n_gamma ) t^gamma.

Clearly ``dim H >= ||chi_T(H)||_1``, the sum of the absolute values of the
coefficients. If no odd generator has weight zero, then no even generator
has weight zero either, and

    chi_T(H) * prod_i (1 - t^{alpha_i}) = prod_j (1 - t^{beta_j}),

where ``alpha_i`` are the weights of the even and ``beta_j`` those of the
odd generators. Hence ``||chi_T(H)||_1`` can be computed from the degrees
and weights of the generators alone.

Laurent polynomials are dictionaries ``{exponent tuple: coefficient}``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .linalg import rank
from .sullivan import Monomial, SullivanAlgebra

Weight = Tuple[int, ...]
Laurent = Dict[Weight, int]


def monomial_weight(monomial: Monomial, weights: Sequence[Weight]) -> Weight:
    r = len(weights[0])
    return tuple(sum(e * weights[i][k] for i, e in enumerate(monomial)) for k in range(r))


def is_weight_grading(algebra: SullivanAlgebra, weights: Sequence[Weight]) -> bool:
    """``d`` preserves the weights: every term of ``d(v)`` has the weight of ``v``."""
    if len(weights) != algebra.m:
        raise ValueError("one weight is needed for each generator")
    return all(
        monomial_weight(a, weights) == tuple(weights[i])
        for i in range(algebra.m)
        for a in algebra.dgen[i]
    )


def homogeneous_weights(algebra: SullivanAlgebra) -> Optional[List[Weight]]:
    """The weights ``(l - 1)|v| - 1`` when ``d`` has a single word length ``l >= 2``."""
    lengths = algebra.word_lengths()
    if len(lengths) != 1:
        return None
    ell = lengths[0]
    return [((ell - 1) * g - 1,) for g in algebra.degrees]


def weighted_betti_numbers(
    algebra: SullivanAlgebra,
    weights: Sequence[Weight],
    nmax: Optional[int] = None,
    prime: Optional[int] = None,
) -> Dict[Tuple[int, Weight], int]:
    """Nonzero dimensions ``dim H^n_gamma``, keyed by ``(n, gamma)``.

    Ranks are exact by default, or computed modulo ``prime``.
    """
    if not is_weight_grading(algebra, weights):
        raise ValueError("the weights are not preserved by d")
    if nmax is None:
        nmax = algebra.formal_dimension
    ranks = {}
    for n in range(nmax + 1):
        ranks.update(_weighted_ranks(algebra, weights, n, prime))
    out = {}
    for n in range(nmax + 1):
        counts: Dict[Weight, int] = defaultdict(int)
        for a in algebra.basis(n):
            counts[monomial_weight(a, weights)] += 1
        for w, c in counts.items():
            h = c - ranks.get((n, w), 0) - ranks.get((n - 1, w), 0)
            if h:
                out[(n, w)] = h
    return out


def _weighted_ranks(
    algebra: SullivanAlgebra, weights: Sequence[Weight], n: int, prime: Optional[int]
) -> Dict[Tuple[int, Weight], int]:
    source = algebra.basis(n)
    blocks: Dict[Weight, List[int]] = defaultdict(list)
    for j, a in enumerate(source):
        blocks[monomial_weight(a, weights)].append(j)
    out = {}
    for w, columns in blocks.items():
        rows: Dict[Monomial, Dict[int, int]] = defaultdict(dict)
        for jj, j in enumerate(columns):
            for b, c in algebra.d_monomial(source[j]).items():
                rows[b][jj] = c
        if not rows:
            continue
        matrix = np.zeros((len(rows), len(columns)), dtype=np.int64)
        for i, entries in enumerate(rows.values()):
            for jj, c in entries.items():
                matrix[i, jj] = c
        out[(n, w)] = rank(matrix, algebra.modulus or prime)
    return out


def euler_characteristic(betti: Dict[Tuple[int, Weight], int]) -> Laurent:
    """``chi_T(H)`` from the output of :func:`weighted_betti_numbers`."""
    out: Laurent = defaultdict(int)
    for (n, w), h in betti.items():
        out[w] += (-1) ** n * h
    return {w: c for w, c in out.items() if c}


def l1_norm(poly: Laurent) -> int:
    return sum(abs(c) for c in poly.values())


def laurent_multiply(f: Laurent, g: Laurent) -> Laurent:
    out: Laurent = defaultdict(int)
    for a, c in f.items():
        for b, e in g.items():
            out[tuple(x + y for x, y in zip(a, b))] += c * e
    return {k: v for k, v in out.items() if v}


def _one_minus(w: Weight) -> Laurent:
    zero = (0,) * len(w)
    if tuple(w) == zero:
        return {}
    return {zero: 1, tuple(w): -1}


def divide_by_one_minus(f: Laurent, alpha: Weight) -> Optional[Laurent]:
    """Exact quotient ``f / (1 - t^alpha)``, or ``None`` if it is not a Laurent polynomial."""
    if not any(alpha):
        return None
    alpha = tuple(alpha)
    rem: Laurent = dict(f)
    quotient: Laurent = {}

    def key(w: Weight) -> Tuple[int, ...]:
        # order along alpha, ties broken lexicographically
        return (sum(x * y for x, y in zip(w, alpha)),) + tuple(w)

    top = max((key(w)[0] for w in f), default=0)
    while rem:
        w = min(rem, key=key)
        if key(w)[0] > top:
            return None
        c = rem.pop(w)
        quotient[w] = quotient.get(w, 0) + c
        shifted = tuple(x + y for x, y in zip(w, alpha))
        rem[shifted] = rem.get(shifted, 0) + c
        if rem[shifted] == 0:
            del rem[shifted]
    return {k: v for k, v in quotient.items() if v}


def euler_characteristic_from_weights(
    algebra: SullivanAlgebra, weights: Sequence[Weight]
) -> Optional[Laurent]:
    """``prod_j (1 - t^{beta_j}) / prod_i (1 - t^{alpha_i})``, computed exactly.

    Returns ``None`` if some even generator has weight zero or the quotient
    is not a Laurent polynomial (then the algebra is not elliptic, or some
    odd generator has weight zero).
    """
    r = len(weights[0])
    numerator: Laurent = {(0,) * r: 1}
    for j in algebra.odd_indices:
        numerator = laurent_multiply(numerator, _one_minus(tuple(weights[j])))
    for i in algebra.even_indices:
        if numerator == {}:
            return None
        numerator = divide_by_one_minus(numerator, tuple(weights[i]))
        if numerator is None:
            return None
    return numerator


def satisfies_product_identity(
    chi: Laurent, algebra: SullivanAlgebra, weights: Sequence[Weight]
) -> bool:
    """``chi * prod_i (1 - t^{alpha_i}) == prod_j (1 - t^{beta_j})``."""
    r = len(weights[0])
    left: Laurent = dict(chi)
    right: Laurent = {(0,) * r: 1}
    for i in algebra.even_indices:
        left = laurent_multiply(left, _one_minus(tuple(weights[i])))
    for j in algebra.odd_indices:
        right = laurent_multiply(right, _one_minus(tuple(weights[j])))
    return left == right


def weight_bound(algebra: SullivanAlgebra, weights: Sequence[Weight]) -> Optional[int]:
    """The lower bound ``||chi_T(H)||_1 <= dim H`` computed from the weights."""
    if any(not any(weights[j]) for j in algebra.odd_indices):
        return None
    chi = euler_characteristic_from_weights(algebra, weights)
    return None if chi is None else l1_norm(chi)
