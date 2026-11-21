"""Linear algebra over prime fields and over the rationals.

All matrices are integer matrices stored as ``numpy`` arrays of ``int64``.

Modular ranks use primes below ``2**21``. Entries are reduced to ``[0, p)``,
so a product of two entries is below ``2**42`` and fits in ``int64``, and a
dot product of length at most ``2**11`` is below ``2**53``, so it is computed
exactly in ``float64``. This allows blocked elimination with BLAS products.

For an integer matrix ``M`` and any prime ``p``, ``rank_p(M) <= rank_Q(M)``.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd, isqrt, lcm
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Two primes close to ``2**21`` used for modular ranks.
P1: int = 2097143
P2: int = 2097133

#: A 31-bit prime used for kernels and rational reconstruction.
P31: int = 2**31 - 1

_MAX_FLOAT_PRIME = 2**21
_BLOCK = 48


def _inverse_mod(a: int, p: int) -> int:
    return pow(int(a) % p, p - 2, p)


def _mulmod(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    """Return ``(a @ b) mod p`` for reduced matrices and a short inner dimension."""
    c = a.astype(np.float64) @ b.astype(np.float64)
    return np.fmod(c, p).astype(np.int64)


def _panel_pivots(panel: np.ndarray, p: int) -> List[Tuple[int, int]]:
    """Gaussian elimination on a panel, returning the pivot positions.

    The input is not modified. The pivots ``(row, column)`` form a set of
    independent rows and columns of maximal size within the panel.
    """
    s = panel.copy()
    m, b = s.shape
    pivots: List[Tuple[int, int]] = []
    alive = np.ones(m, dtype=bool)
    for c in range(b):
        rows = np.flatnonzero((s[:, c] != 0) & alive)
        if rows.size == 0:
            continue
        r = rows[0]
        pivots.append((int(r), c))
        alive[r] = False
        if c + 1 < b and rows.size > 1:
            pivot_row = (s[r, c + 1:] * _inverse_mod(s[r, c], p)) % p
            others = rows[1:]
            factors = s[others, c].reshape(-1, 1)
            s[others, c + 1:] = (s[others, c + 1:] - (factors * pivot_row) % p) % p
    return pivots


def _inverse_matrix_mod(a: np.ndarray, p: int) -> np.ndarray:
    """Inverse of an invertible square matrix modulo ``p``."""
    k = a.shape[0]
    m = np.concatenate([a % p, np.eye(k, dtype=np.int64)], axis=1)
    for c in range(k):
        r = c + int(np.flatnonzero(m[c:, c])[0])
        if r != c:
            m[[c, r]] = m[[r, c]]
        m[c] = (m[c] * _inverse_mod(m[c, c], p)) % p
        factors = m[:, c].copy()
        factors[c] = 0
        nz = np.flatnonzero(factors)
        if nz.size:
            m[nz] = (m[nz] - (factors[nz].reshape(-1, 1) * m[c]) % p) % p
    return m[:, k:]


def rank_mod(matrix: np.ndarray, p: int = P1, block: int = _BLOCK) -> int:
    """Rank of an integer matrix modulo a prime ``p < 2**21``.

    The elimination proceeds by panels of ``block`` columns. Each panel is
    reduced, and the Schur complement of the remaining columns is updated by
    two matrix products modulo ``p``.
    """
    if p >= _MAX_FLOAT_PRIME:
        return rank_mod_plain(matrix, p)
    a = np.asarray(matrix, dtype=np.int64) % p
    if a.size == 0:
        return 0
    if a.shape[0] < a.shape[1]:
        a = a.T
    a = np.ascontiguousarray(a)
    rank = 0
    while a.shape[0] and a.shape[1]:
        m, n = a.shape
        b = min(block, n)
        pivots = _panel_pivots(a[:, :b], p)
        if not pivots:
            a = a[:, b:]
            continue
        rank += len(pivots)
        prow = np.array([r for r, _ in pivots])
        pcol = np.array([c for _, c in pivots])
        rest = np.ones(m, dtype=bool)
        rest[prow] = False
        if not rest.any() or b == n:
            break
        inv = _inverse_matrix_mod(a[np.ix_(prow, pcol)], p)
        mult = _mulmod(a[rest][:, pcol], inv, p)
        a = (a[rest, b:] - _mulmod(mult, a[prow, b:], p)) % p
    return rank


def rank_mod_plain(matrix: np.ndarray, p: int = P31) -> int:
    """Rank modulo a prime ``p < 2**31`` by unblocked row reduction."""
    a = np.asarray(matrix, dtype=np.int64) % p
    m, n = a.shape
    r = 0
    for c in range(n):
        if r == m:
            break
        nz = np.flatnonzero(a[r:, c])
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            a[[r, i]] = a[[i, r]]
        a[r] = (a[r] * _inverse_mod(a[r, c], p)) % p
        below = r + 1 + np.flatnonzero(a[r + 1:, c])
        if below.size:
            a[below] = (a[below] - (a[below, c].reshape(-1, 1) * a[r]) % p) % p
        r += 1
    return r


def rank_exact(matrix: np.ndarray) -> int:
    """Rank over the rationals, by sparse elimination with ``Fraction`` entries.

    The rows are stored as dictionaries. At each step the sparsest remaining
    row is used as pivot row, with the pivot in its least populated column.
    """
    a = np.asarray(matrix)
    rows: List[Dict[int, Fraction]] = []
    for i in range(a.shape[0]):
        nz = np.flatnonzero(a[i])
        if nz.size:
            rows.append({int(j): Fraction(int(a[i, j])) for j in nz})
    columns: Dict[int, set] = {}
    for i, row in enumerate(rows):
        for j in row:
            columns.setdefault(j, set()).add(i)
    alive = set(range(len(rows)))
    rank = 0
    while alive:
        best = min(alive, key=lambda i: len(rows[i]))
        alive.discard(best)
        pivot_row = rows[best]
        if not pivot_row:
            continue
        j = min(pivot_row, key=lambda jj: len(columns[jj]))
        pivot = pivot_row[j]
        rank += 1
        for i in list(columns[j]):
            if i not in alive:
                continue
            row = rows[i]
            factor = row[j] / pivot
            for jj, v in pivot_row.items():
                new = row.get(jj, 0) - factor * v
                if new == 0:
                    if jj in row:
                        del row[jj]
                        columns[jj].discard(i)
                else:
                    if jj not in row:
                        columns[jj].add(i)
                    row[jj] = new
        for jj in pivot_row:
            columns[jj].discard(best)
    return rank


def rank(matrix: np.ndarray, p: Optional[int] = P1) -> int:
    """Rank modulo ``p``, or over the rationals when ``p`` is ``None``."""
    a = np.asarray(matrix)
    if a.size == 0 or not a.any():
        return 0
    if p is None:
        return rank_exact(a)
    return rank_mod(a, p)


def nullspace_mod(matrix: np.ndarray, p: int = P31) -> np.ndarray:
    """Basis of the right kernel modulo ``p``, as the rows of the returned array.

    The basis is the one read off the reduced row echelon form, with a one in
    a free column and zeros in the other free columns.
    """
    a = np.asarray(matrix, dtype=np.int64) % p
    m, n = a.shape
    pivot_columns: List[int] = []
    r = 0
    for c in range(n):
        if r == m:
            break
        nz = np.flatnonzero(a[r:, c])
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            a[[r, i]] = a[[i, r]]
        a[r] = (a[r] * _inverse_mod(a[r, c], p)) % p
        others = np.flatnonzero(a[:, c])
        others = others[others != r]
        if others.size:
            a[others] = (a[others] - (a[others, c].reshape(-1, 1) * a[r]) % p) % p
        pivot_columns.append(c)
        r += 1
    pivot_set = set(pivot_columns)
    free = [c for c in range(n) if c not in pivot_set]
    basis = np.zeros((len(free), n), dtype=np.int64)
    for k, f in enumerate(free):
        basis[k, f] = 1
        for i, c in enumerate(pivot_columns):
            basis[k, c] = (-a[i, f]) % p
    return basis


def kernel_exact(matrix: np.ndarray) -> List[List[int]]:
    """Integral basis of the right kernel over the rationals (dense ``Fraction`` RREF)."""
    a = [[Fraction(int(x)) for x in row] for row in np.asarray(matrix)]
    m = len(a)
    n = np.asarray(matrix).shape[1]
    pivot_columns: List[int] = []
    r = 0
    for c in range(n):
        i = next((k for k in range(r, m) if a[k][c] != 0), None)
        if i is None:
            continue
        a[r], a[i] = a[i], a[r]
        inv = 1 / a[r][c]
        a[r] = [x * inv for x in a[r]]
        for k in range(m):
            if k != r and a[k][c] != 0:
                f = a[k][c]
                a[k] = [x - f * y for x, y in zip(a[k], a[r])]
        pivot_columns.append(c)
        r += 1
        if r == m:
            break
    pivot_set = set(pivot_columns)
    basis = []
    for f in (c for c in range(n) if c not in pivot_set):
        v = [Fraction(0)] * n
        v[f] = Fraction(1)
        for i, c in enumerate(pivot_columns):
            v[c] = -a[i][f]
        basis.append(_clear_denominators(v))
    return basis


def _clear_denominators(v: Sequence[Fraction]) -> List[int]:
    den = 1
    for x in v:
        den = lcm(den, x.denominator)
    w = [int(x * den) for x in v]
    g = 0
    for x in w:
        g = gcd(g, x)
    return [x // g for x in w] if g > 1 else w


def rational_reconstruction(a: int, p: int = P31) -> Optional[Fraction]:
    """Return ``r/s`` with ``r = a*s mod p`` and ``|r|, s <= sqrt(p/2)``, if it exists."""
    a %= p
    bound = isqrt(p // 2)
    r0, r1 = p, a
    s0, s1 = 0, 1
    while r1 > bound:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        s0, s1 = s1, s0 - q * s1
    if s1 == 0 or abs(s1) > bound:
        return None
    if s1 < 0:
        r1, s1 = -r1, -s1
    return Fraction(r1, s1)


def lift_vector(v: np.ndarray, p: int = P31) -> Optional[List[int]]:
    """Lift a vector modulo ``p`` to a primitive integer vector, if possible."""
    fractions = []
    for x in v:
        if x == 0:
            fractions.append(Fraction(0))
            continue
        q = rational_reconstruction(int(x), p)
        if q is None:
            return None
        fractions.append(q)
    return _clear_denominators(fractions)
