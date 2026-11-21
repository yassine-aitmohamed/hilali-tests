"""The associated pure algebra, the lower degree and the derivation delta.

Write ``V = Q + P`` with ``Q = V^even`` (dimension ``q``) and ``P = V^odd``
(dimension ``p``), and ``R = Q[V^even]``. The *lower degree* of an element of
``R (x) Lambda^k P`` is ``k``. The differential splits as
``d = d_{-1} + d_1 + d_3 + ...``, where ``d_i`` raises the lower degree by
``i``. The component ``d_{-1} = d_sigma`` is the differential of the
associated pure algebra: ``d_sigma(x_i) = 0`` and ``d_sigma(y_j) = f_j``, the
part of ``d(y_j)`` in ``R``.

The homology of ``(Lambda V, d_sigma)`` is the Koszul homology
``H_k = H_k(f_1, ..., f_p; R)``, with ``H_0 = A = R/(f_1, ..., f_p)``. When
``chi_pi = q - p = -1`` and the algebra is elliptic, only ``H_0`` and
``H_1`` are nonzero. The odd spectral sequence then has a single possibly
nonzero differential, the derivation ``delta: A -> H_1`` induced by ``d_1``,
and

    dim H^even(Lambda V, d) = dim ker delta,
    dim H^odd(Lambda V, d)  = dim coker delta.

This module computes these spaces degree by degree.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .linalg import P1, nullspace_mod, rank
from .sullivan import Monomial, Polynomial, SullivanAlgebra


def pure_model(algebra: SullivanAlgebra) -> SullivanAlgebra:
    """The associated pure algebra ``(Lambda V, d_sigma)``."""
    odd = algebra.odd_indices
    values = []
    for i in range(algebra.m):
        if algebra.odd[i]:
            values.append(
                {a: c for a, c in algebra.dgen[i].items() if all(a[j] == 0 for j in odd)}
            )
        else:
            values.append({})
    return SullivanAlgebra(algebra.degrees, values, algebra.names, algebra.modulus)


def is_pure(algebra: SullivanAlgebra) -> bool:
    """The even generators are closed and ``d(V^odd)`` lies in ``Q[V^even]``."""
    pure = pure_model(algebra)
    return all(algebra.dgen[i] == pure.dgen[i] for i in range(algebra.m))


def lower_degree(algebra: SullivanAlgebra, monomial: Monomial) -> int:
    """Number of odd factors of a monomial."""
    return algebra.odd_length(monomial)


class Bigraded:
    """``Lambda V`` graded by degree ``n`` and lower degree ``k``.

    Ranks are computed modulo ``prime``, or over Q if ``prime`` is ``None``.
    """

    def __init__(self, algebra: SullivanAlgebra, prime: Optional[int] = P1) -> None:
        self.algebra = algebra
        self.prime = algebra.modulus or prime
        self._bases: Dict[Tuple[int, int], Tuple[List[Monomial], Dict[Monomial, int]]] = {}

    def basis(self, n: int, k: int) -> Tuple[List[Monomial], Dict[Monomial, int]]:
        """Monomials of degree ``n`` and lower degree ``k``, with their positions."""
        key = (n, k)
        if key not in self._bases:
            monos = [a for a in self.algebra.basis(n) if self.algebra.odd_length(a) == k]
            self._bases[key] = (monos, {a: j for j, a in enumerate(monos)})
        return self._bases[key]

    def component(self, n: int, k: int, shift: int) -> np.ndarray:
        """Matrix of the component of ``d`` from bidegree ``(n, k)`` to ``(n+1, k+shift)``."""
        source, _ = self.basis(n, k)
        target, index = self.basis(n + 1, k + shift)
        matrix = np.zeros((len(target), len(source)), dtype=np.int64)
        for j, a in enumerate(source):
            for b, c in self.algebra.d_monomial(a).items():
                if self.algebra.odd_length(b) == k + shift:
                    matrix[index[b], j] = c
        return matrix

    def rank(self, matrix: np.ndarray) -> int:
        return rank(matrix, self.prime)

    def sigma_rank(self, n: int, k: int) -> int:
        """Rank of ``d_sigma: (n, k) -> (n+1, k-1)``."""
        if k <= 0:
            return 0
        return self.rank(self.component(n, k, -1))

    def koszul_homology(self, n: int, k: int) -> int:
        """``dim H_k^n``, the Koszul homology in degree ``n`` and lower degree ``k``."""
        size = len(self.basis(n, k)[0])
        if size == 0:
            return 0
        return size - self.sigma_rank(n, k) - self.sigma_rank(n - 1, k + 1)

    def delta_rank(self, n: int) -> int:
        """Rank of ``delta: A^n -> H_1^{n+1}``, the map induced by ``d_1`` on ``R^n``."""
        d1 = self.component(n, 0, 1)
        if d1.size == 0 or not d1.any():
            return 0
        boundaries = self.component(n, 2, -1)
        if boundaries.size == 0:
            return self.rank(d1)
        return self.rank(np.concatenate([d1, boundaries], axis=1)) - self.rank(boundaries)


def delta_analysis(
    algebra: SullivanAlgebra,
    prime: Optional[int] = P1,
    compare: bool = True,
) -> dict:
    """Koszul homology and the derivation ``delta`` of an elliptic algebra.

    The result contains, for ``0 <= n <= N + 1``:

    * ``A``: ``dim A^n`` (``H_0`` in even degrees);
    * ``H1``: ``dim H_1^n`` (odd degrees);
    * ``rank_delta``: rank of ``delta: A^n -> H_1^{n+1}``;
    * ``kernel`` and ``cokernel`` of ``delta`` by degree;

    and the totals ``dim_A``, ``dim_H1``, ``dim_higher`` (sum of ``dim H_k``
    for ``k >= 2``, which vanishes when ``chi_pi = -1``), ``rank``,
    ``dim_kernel``, and ``dim_H_sigma`` (``dim H(Lambda V, d_sigma)``).

    The predicted Betti numbers are ``kernel`` in even and ``cokernel`` in odd
    degrees. They are correct when ``chi_pi = -1``. With ``compare=True``, the
    Betti numbers of ``(Lambda V, d)`` are computed directly and compared.
    """
    bg = Bigraded(algebra, prime)
    top = algebra.formal_dimension + 1
    p = algebra.num_odd
    dims_a = [bg.koszul_homology(n, 0) if n % 2 == 0 else 0 for n in range(top + 1)]
    dims_h1 = [bg.koszul_homology(n, 1) if n % 2 == 1 else 0 for n in range(top + 1)]
    higher = sum(
        bg.koszul_homology(n, k) for n in range(top + 1) for k in range(2, p + 1)
    )
    ranks = [bg.delta_rank(n) if n % 2 == 0 and dims_a[n] else 0 for n in range(top + 1)]
    kernel = [dims_a[n] - ranks[n] if n % 2 == 0 else 0 for n in range(top + 1)]
    cokernel = [dims_h1[n] - ranks[n - 1] if n % 2 == 1 else 0 for n in range(top + 1)]
    result = {
        "N": algebra.formal_dimension,
        "q": algebra.num_even,
        "p": p,
        "A": dims_a,
        "H1": dims_h1,
        "rank_delta": ranks,
        "kernel": kernel,
        "cokernel": cokernel,
        "dim_A": sum(dims_a),
        "dim_H1": sum(dims_h1),
        "dim_higher": higher,
        "rank": sum(ranks),
        "dim_kernel": sum(kernel),
        "dim_H_sigma": sum(dims_a) + sum(dims_h1) + higher,
    }
    predicted = [kernel[n] + cokernel[n] for n in range(top)]
    result["predicted_betti"] = predicted
    if compare:
        method = "exact" if prime is None and not algebra.modulus else "modular"
        primes = (prime,) if prime else (P1,)
        betti = algebra.betti_numbers(method=method, primes=primes)
        result["betti"] = betti
        result["match"] = predicted == betti
    return result


def has_nonclosed_even(algebra: SullivanAlgebra) -> bool:
    """Some even generator ``x`` has ``d(x) != 0``."""
    return any(algebra.dgen[i] for i in algebra.even_indices)


def has_chain_level_d1(algebra: SullivanAlgebra) -> bool:
    """Some even generator has a nonzero component ``d_1(x)`` of lower degree one."""
    return any(
        algebra.odd_length(a) == 1 for i in algebra.even_indices for a in algebra.dgen[i]
    )


# ---------------------------------------------------------------------------
# The space of candidate derivations A -> H_1 of degree one


def _homology_representatives(
    bg: Bigraded, n: int, admissible: bool
) -> np.ndarray:
    """Columns: cycles representing a basis of ``H_1^n`` modulo ``bg.prime``.

    With ``admissible=True``, only cycles with no term of word length one
    (that is, with coefficients in the maximal ideal of ``R``) are used.
    """
    prime = bg.prime
    monos, _ = bg.basis(n, 1)
    if not monos:
        return np.zeros((0, 0), dtype=np.int64)
    sigma = bg.component(n, 1, -1)
    if sigma.size and sigma.any():
        cycles = nullspace_mod(sigma, prime).T
    else:
        cycles = np.eye(len(monos), dtype=np.int64)
    if admissible:
        linear = [j for j, a in enumerate(monos) if sum(a) == 1]
        constraint = cycles[linear, :] if linear else None
        if constraint is not None and constraint.any():
            kernel = nullspace_mod(constraint, prime).T
            cycles = (cycles @ kernel) % prime if kernel.size else cycles[:, :0]
    boundaries = bg.component(n - 1, 2, -1)
    current = boundaries if boundaries.size else np.zeros((len(monos), 0), dtype=np.int64)
    current_rank = rank(current, prime)
    chosen = []
    for c in range(cycles.shape[1]):
        extended = np.concatenate([current, cycles[:, c:c + 1]], axis=1)
        new_rank = rank(extended, prime)
        if new_rank > current_rank:
            chosen.append(cycles[:, c])
            current, current_rank = extended, new_rank
    if not chosen:
        return np.zeros((len(monos), 0), dtype=np.int64)
    return np.array(chosen).T


def derivation_space_dimension(
    algebra: SullivanAlgebra, prime: int = P1, admissible: bool = False
) -> Tuple[int, int]:
    """Dimension of the space of derivations ``A -> H_1`` of degree one, over ``F_prime``.

    Such a derivation is determined by ``v_i = delta(x_i)`` in
    ``H_1^{|x_i|+1}``, subject to the conditions
    ``sum_i (df_j/dx_i) v_i = 0`` in ``H_1`` for all ``j``. Returns the pair
    ``(dimension, number of free parameters before the conditions)``. With
    ``admissible=True`` each ``v_i`` is represented by a cycle with
    coefficients in the maximal ideal.
    """
    base = pure_model(algebra)
    pure = SullivanAlgebra(base.degrees, base.dgen, base.names, algebra.modulus or prime)
    bg = Bigraded(pure, prime)
    even = pure.even_indices
    reps = {i: _homology_representatives(bg, pure.degrees[i] + 1, admissible) for i in even}
    unknowns = sum(reps[i].shape[1] for i in even)
    if unknowns == 0:
        return 0, 0
    images, boundary_blocks = [], []
    for j in pure.odd_indices:
        block = _condition_block(pure, bg, j, even, reps)
        if block is not None:
            images.append(block[0])
            boundary_blocks.append(block[1])
    rows = sum(m.shape[0] for m in images)
    if rows == 0:
        return unknowns, unknowns
    image = np.concatenate(images, axis=0)
    boundaries = _block_diagonal(boundary_blocks, rows)
    image_rank = rank(np.concatenate([image, boundaries], axis=1), bg.prime) - rank(
        boundaries, bg.prime
    )
    return unknowns - image_rank, unknowns


def _condition_block(
    pure: SullivanAlgebra,
    bg: Bigraded,
    j: int,
    even: Sequence[int],
    reps: Dict[int, np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Matrix of ``(v_i) -> sum_i (df_j/dx_i) v_i`` and the boundaries ``B_1``."""
    f = pure.dgen[j]
    n = pure.degrees[j] + 2
    target, index = bg.basis(n, 1)
    if not target:
        return None
    columns = []
    for i in even:
        derivative = _partial_derivative(pure, f, i)
        sources, _ = bg.basis(pure.degrees[i] + 1, 1)
        for c in range(reps[i].shape[1]):
            cycle = {sources[t]: int(reps[i][t, c]) for t in np.flatnonzero(reps[i][:, c])}
            product = pure.multiply(derivative, cycle) if derivative else {}
            column = np.zeros(len(target), dtype=np.int64)
            for a, v in product.items():
                column[index[a]] = (column[index[a]] + v) % bg.prime
            columns.append(column)
    return np.array(columns).T, bg.component(n - 1, 2, -1)


def _partial_derivative(algebra: SullivanAlgebra, f: Polynomial, i: int) -> Polynomial:
    out: Polynomial = {}
    for a, c in f.items():
        if a[i]:
            b = list(a)
            b[i] -= 1
            out[tuple(b)] = out.get(tuple(b), 0) + c * a[i]
    return algebra.add({}, out)


def _block_diagonal(blocks: Sequence[np.ndarray], rows: int) -> np.ndarray:
    width = sum(b.shape[1] for b in blocks if b.size)
    out = np.zeros((rows, width), dtype=np.int64)
    r = c = 0
    for b in blocks:
        h = b.shape[0]
        if b.size:
            out[r:r + h, c:c + b.shape[1]] = b
            c += b.shape[1]
        r += h
    return out


# ---------------------------------------------------------------------------
# Linkage to a complete intersection


def linkage_test(
    algebra: SullivanAlgebra, rng: Optional[random.Random] = None, prime: int = P1
) -> List[bool]:
    """Test whether ``I = (f_1, ..., f_p)`` is linked to a complete intersection.

    For each ``j0``, the relations ``f_j`` with ``j != j0`` are replaced by
    random elements ``g_j`` of ``I`` of the same degrees, and the test checks
    whether ``g_j`` is a regular sequence, that is, whether the pure algebra
    with ``d(y'_j) = g_j`` is elliptic. Returns one boolean for each ``j0``.
    This is a randomized test over ``F_prime``: ``True`` is conclusive over
    ``F_prime``, ``False`` may come from an unlucky choice.
    """
    rng = rng or random.Random(0)
    ring = algebra.even_ring()
    ring = SullivanAlgebra(ring.degrees, None, ring.names, prime)
    relations = [f for f in algebra.pure_relations()]
    degrees = [ring.monomial_degree(next(iter(f))) if f else None for f in relations]
    out = []
    for j0 in range(len(relations)):
        others = [k for k in range(len(relations)) if k != j0]
        if any(degrees[k] is None for k in others):
            out.append(False)
            continue
        generic = [_generic_ideal_element(ring, relations, degrees[k], rng, prime) for k in others]
        if any(not g for g in generic):
            out.append(False)
            continue
        out.append(_pure_algebra(ring, generic, [degrees[k] - 1 for k in others], prime).is_elliptic())
    return out


def _generic_ideal_element(
    ring: SullivanAlgebra,
    relations: Sequence[Polynomial],
    n: int,
    rng: random.Random,
    prime: int,
) -> Polynomial:
    out: Polynomial = {}
    for f in relations:
        if not f:
            continue
        df = ring.monomial_degree(next(iter(f)))
        if df > n:
            continue
        for mono in ring.basis(n - df):
            c = rng.randrange(1, prime)
            for a, v in f.items():
                b = tuple(x + y for x, y in zip(a, mono))
                out[b] = (out.get(b, 0) + c * v) % prime
    return {a: c for a, c in out.items() if c}


def _pure_algebra(
    ring: SullivanAlgebra, relations: Sequence[Polynomial], odd_degrees: Sequence[int], prime: int
) -> SullivanAlgebra:
    q = ring.m
    s = len(relations)
    values = [{} for _ in range(q)]
    values += [{a + (0,) * s: c for a, c in g.items()} for g in relations]
    return SullivanAlgebra(list(ring.degrees) + list(odd_degrees), values, modulus=prime)
