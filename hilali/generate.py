"""Random minimal Sullivan algebras, built generator by generator.

The generators are added in nondecreasing degree. When a generator ``v`` of
degree ``k`` is added to an algebra ``(Lambda W, d)`` built from generators
of degree at most ``k``, its differential ``d(v)`` must be a cocycle of degree
``k + 1`` in ``Lambda W``. Every element of degree ``k + 1`` of ``Lambda W``
is decomposable, because all generators of ``W`` have degree at most ``k``.
Hence any such choice gives a minimal Sullivan algebra, and ``d^2 = 0``
holds automatically.

Up to isomorphism only the class of ``d(v)`` in ``H^{k+1}(Lambda W)``
matters. The integral generator :func:`random_minimal_model` therefore
chooses ``d(v)`` as a random integral combination of cocycles representing a
basis of ``H^{k+1}``. The generator :func:`generic_model` works over a prime
field and chooses ``d(v)`` as a random combination of all cocycles, which
models the generic point of the variety of differentials.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

import numpy as np

from .linalg import P1, P31, kernel_exact, lift_vector, nullspace_mod, rank
from .sullivan import Polynomial, SullivanAlgebra

#: Above this size, exact kernels are not attempted when a lift fails.
_EXACT_KERNEL_LIMIT = 400


def integral_cocycles(algebra: SullivanAlgebra, n: int) -> Optional[List[Polynomial]]:
    """Integral basis of the cocycles ``Z^n`` of an integral algebra.

    The kernel of ``d`` is computed modulo a 31-bit prime and lifted by
    rational reconstruction. Every lifted vector is checked to be a cocycle
    over Z. If a lift fails, the kernel is computed exactly when the matrix
    is small enough, and ``None`` is returned otherwise.
    """
    basis = algebra.basis(n)
    if not basis:
        return []
    matrix = algebra.differential_matrix(n) if algebra.basis(n + 1) else None
    if matrix is None or not matrix.any():
        return [{a: 1} for a in basis]
    cocycles = _lifted_kernel(algebra, matrix, basis)
    if cocycles is None and max(matrix.shape) <= _EXACT_KERNEL_LIMIT:
        cocycles = [_as_polynomial(basis, v) for v in kernel_exact(matrix)]
    return cocycles


def _lifted_kernel(
    algebra: SullivanAlgebra, matrix: np.ndarray, basis: Sequence
) -> Optional[List[Polynomial]]:
    out = []
    for v in nullspace_mod(matrix, P31):
        lifted = lift_vector(v, P31)
        if lifted is None:
            return None
        poly = _as_polynomial(basis, lifted)
        if algebra.d(poly):
            return None
        out.append(poly)
    return out


def _as_polynomial(basis: Sequence, vector: Sequence[int]) -> Polynomial:
    return {basis[i]: int(c) for i, c in enumerate(vector) if c}


def cohomology_representatives(
    algebra: SullivanAlgebra, n: int, prime: int = P1
) -> Optional[List[Polynomial]]:
    """Integral cocycles whose classes form a basis of ``H^n``.

    Cocycles are selected greedily, keeping those that are independent of
    the coboundaries and of the cocycles already kept. Independence is
    tested modulo ``prime``.
    """
    cocycles = integral_cocycles(algebra, n)
    if not cocycles:
        return cocycles
    index = algebra.index(n)
    if algebra.basis(n - 1):
        current = algebra.differential_matrix(n - 1)
    else:
        current = np.zeros((len(index), 0), dtype=np.int64)
    current_rank = rank(current, prime)
    chosen = []
    for z in cocycles:
        column = np.zeros((len(index), 1), dtype=np.int64)
        for a, c in z.items():
            column[index[a], 0] = c
        extended = np.concatenate([current, column], axis=1)
        new_rank = rank(extended, prime)
        if new_rank > current_rank:
            chosen.append(z)
            current, current_rank = extended, new_rank
    return chosen


def _pad(poly: Polynomial, m: int) -> Polynomial:
    return {a + (0,) * (m - len(a)): c for a, c in poly.items()}


def random_minimal_model(
    degrees: Sequence[int],
    rng: random.Random,
    density: float = 0.6,
    coefficients: Sequence[int] = (1, -1, 2, -2, 3),
    zero_probability: float = 0.0,
) -> Optional[SullivanAlgebra]:
    """A random integral minimal Sullivan algebra on generators of the given degrees.

    For each new generator ``v``, a basis of ``H^{|v|+1}`` of the algebra
    built so far is represented by integral cocycles. Each representative is
    kept with probability ``density`` (at least one is kept) and multiplied
    by a coefficient drawn from ``coefficients``. With probability
    ``zero_probability`` the generator is closed instead.

    Returns ``None`` in the rare case where integral cocycles cannot be lifted.
    """
    degrees = sorted(degrees)
    m = len(degrees)
    values: List[Polynomial] = []
    for i, k in enumerate(degrees):
        if i == 0 or rng.random() < zero_probability:
            values.append({})
            continue
        partial = SullivanAlgebra(degrees[:i], [_pad(v, i) for v in values])
        reps = cohomology_representatives(partial, k + 1)
        if reps is None:
            return None
        values.append(_random_combination(partial, reps, rng, density, coefficients))
    return SullivanAlgebra(degrees, [_pad(v, m) for v in values])


def _random_combination(
    partial: SullivanAlgebra,
    reps: Sequence[Polynomial],
    rng: random.Random,
    density: float,
    coefficients: Sequence[int],
) -> Polynomial:
    if not reps:
        return {}
    selected = [z for z in reps if rng.random() < density] or [rng.choice(reps)]
    value: Polynomial = {}
    for z in selected:
        value = partial.add(value, z, rng.choice(coefficients))
    return value


def random_elliptic_model(
    degrees: Sequence[int],
    rng: random.Random,
    attempts: int = 20,
    **options,
) -> Optional[SullivanAlgebra]:
    """Draw :func:`random_minimal_model` until an elliptic algebra is found."""
    for _ in range(attempts):
        model = random_minimal_model(degrees, rng, **options)
        if model is not None and model.is_elliptic():
            return model
    return None


def generic_model(
    degrees: Sequence[int],
    rng: random.Random,
    pattern: Optional[str] = None,
    prime: int = P1,
) -> SullivanAlgebra:
    """A random minimal Sullivan algebra over ``F_prime``.

    For each generator ``v``, ``d(v)`` is a combination of a basis of the
    cocycles ``Z^{|v|+1}`` of the algebra built so far. The string
    ``pattern`` gives one letter per generator, in increasing degree order:

    * ``g``: all basis cocycles with random nonzero coefficients (generic);
    * ``s``: a random half of the basis cocycles (sparse);
    * ``z``: ``d(v) = 0``;
    * ``p``: a random polynomial in the even generators (pure). This is a
      cocycle only if the even generators are closed, which is required.

    By default all letters are ``g``.
    """
    degrees = sorted(degrees)
    m = len(degrees)
    pattern = pattern or "g" * m
    if len(pattern) != m:
        raise ValueError("the pattern needs one letter per generator")
    values: List[Polynomial] = []
    for i, k in enumerate(degrees):
        letter = pattern[i]
        if i == 0 or letter == "z":
            values.append({})
            continue
        partial = SullivanAlgebra(degrees[:i], [_pad(v, i) for v in values], modulus=prime)
        if letter == "p":
            values.append(_pure_value(partial, k + 1, rng, prime))
        else:
            values.append(_generic_value(partial, k + 1, rng, prime, sparse=letter == "s"))
    return SullivanAlgebra(degrees, [_pad(v, m) for v in values], modulus=prime)


def _generic_value(
    partial: SullivanAlgebra, n: int, rng: random.Random, prime: int, sparse: bool
) -> Polynomial:
    basis = partial.basis(n)
    if not basis:
        return {}
    if partial.basis(n + 1):
        matrix = partial.differential_matrix(n)
        kernel = nullspace_mod(matrix, prime) if matrix.any() else np.eye(len(basis), dtype=np.int64)
    else:
        kernel = np.eye(len(basis), dtype=np.int64)
    if kernel.shape[0] == 0:
        return {}
    rows = range(kernel.shape[0])
    if sparse:
        rows = [r for r in rows if rng.random() < 0.5] or [rng.randrange(kernel.shape[0])]
    vector = np.zeros(len(basis), dtype=np.int64)
    for r in rows:
        vector = (vector + rng.randrange(1, prime) * kernel[r]) % prime
    return {basis[t]: int(vector[t]) for t in np.flatnonzero(vector)}


def _pure_value(partial: SullivanAlgebra, n: int, rng: random.Random, prime: int) -> Polynomial:
    if any(partial.dgen[i] for i in partial.even_indices):
        raise ValueError("pattern 'p' requires closed even generators")
    return {
        a: rng.randrange(1, prime)
        for a in partial.basis(n)
        if partial.odd_length(a) == 0
    }
