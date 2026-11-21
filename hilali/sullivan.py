"""Minimal Sullivan algebras and their cohomology.

A Sullivan algebra ``(Lambda V, d)`` is described by a finite list of
generators ``v_0, ..., v_{m-1}`` with degrees ``>= 2`` and by the values
``d(v_i)``. The differential is extended to ``Lambda V`` as a derivation of
degree one.

Elements of ``Lambda V`` are represented as follows.

* A *monomial* is a tuple ``a`` of exponents of length ``m``. Odd generators
  have exponent 0 or 1. The monomial ``a`` stands for the product of the
  generators in increasing index order, each raised to its exponent.
* A *polynomial* is a dictionary ``{monomial: coefficient}`` with nonzero
  integer coefficients.

Coefficients are integers. An algebra may carry a ``modulus``, in which case
all coefficients are reduced modulo this prime and the algebra is read over
the prime field.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .linalg import P1, P2, rank

Monomial = Tuple[int, ...]
Polynomial = Dict[Monomial, int]

_TOKEN = re.compile(r"\s*(?:(\d+)|([A-Za-z_][A-Za-z_0-9]*)(?:\^(\d+))?|([-+*]))")


class SullivanAlgebra:
    """A free graded commutative algebra with a derivation of degree one.

    Parameters
    ----------
    degrees:
        Degrees of the generators, all at least two.
    differential:
        The values ``d(v_i)`` as a list of polynomials, or a dictionary
        ``{i: polynomial}``. Missing generators are closed.
    names:
        Optional names of the generators. By default the even generators are
        called ``x1, x2, ...`` and the odd ones ``y1, y2, ...``.
    modulus:
        Optional prime. If given, the algebra is defined over ``F_p``.
    """

    def __init__(
        self,
        degrees: Sequence[int],
        differential: Optional[Sequence[Polynomial] | Mapping[int, Polynomial]] = None,
        names: Optional[Sequence[str]] = None,
        modulus: Optional[int] = None,
    ) -> None:
        self.degrees: List[int] = [int(g) for g in degrees]
        if any(g < 2 for g in self.degrees):
            raise ValueError("generators must have degree at least 2")
        self.m: int = len(self.degrees)
        self.odd: List[bool] = [g % 2 == 1 for g in self.degrees]
        self.modulus: Optional[int] = modulus
        self.names: List[str] = list(names) if names else _default_names(self.odd)
        if differential is None:
            values: List[Polynomial] = [{} for _ in range(self.m)]
        elif isinstance(differential, Mapping):
            values = [dict(differential.get(i, {})) for i in range(self.m)]
        else:
            values = [dict(v) for v in differential]
        if len(values) != self.m:
            raise ValueError("one value of d is needed for each generator")
        self.dgen: List[Polynomial] = [self._normalize(v) for v in values]
        self._dcache: Dict[Monomial, Polynomial] = {}
        self._basis: Dict[int, List[Monomial]] = {}
        self._index: Dict[int, Dict[Monomial, int]] = {}
        self._basis_top = -1

    # ------------------------------------------------------------ construction
    @classmethod
    def from_products(
        cls,
        degrees: Sequence[int],
        spec: Mapping[int, Iterable[Tuple[int, Sequence[int]]]],
        names: Optional[Sequence[str]] = None,
        modulus: Optional[int] = None,
    ) -> "SullivanAlgebra":
        """Build an algebra from ``spec = {i: [(c, [j1, j2, ...]), ...]}``.

        Each pair ``(c, [j1, j2, ...])`` is the term ``c * v_j1 * v_j2 * ...``,
        with the product taken in the given order, so that signs coming from
        odd generators are handled automatically.
        """
        free = cls(degrees, names=names, modulus=modulus)
        values = []
        for i in range(free.m):
            value: Polynomial = {}
            for c, factors in spec.get(i, []):
                term = free.constant(c)
                for j in factors:
                    term = free.multiply(term, free.generator(j))
                value = free.add(value, term)
            values.append(value)
        return cls(degrees, values, names=names, modulus=modulus)

    @classmethod
    def from_strings(
        cls,
        generators: Sequence[Tuple[str, int]],
        differential: Mapping[str, str],
        modulus: Optional[int] = None,
    ) -> "SullivanAlgebra":
        """Build an algebra from named generators and string formulas.

        Example::

            SullivanAlgebra.from_strings(
                [("a", 3), ("b", 3), ("c", 5)], {"c": "a*b"})

        A formula is a sum of terms such as ``2*x^3``, ``-a*b`` or ``x1*y2``.
        Products are read from left to right, so ``b*a`` equals ``-a*b`` for
        odd ``a`` and ``b``.
        """
        names = [name for name, _ in generators]
        degrees = [deg for _, deg in generators]
        position = {name: i for i, name in enumerate(names)}
        if len(position) != len(names):
            raise ValueError("generator names must be distinct")
        free = cls(degrees, names=names, modulus=modulus)
        values: List[Polynomial] = [{} for _ in names]
        for name, formula in differential.items():
            if name not in position:
                raise ValueError(f"unknown generator {name!r}")
            values[position[name]] = free.parse(formula)
        return cls(degrees, values, names=names, modulus=modulus)

    def parse(self, formula: str) -> Polynomial:
        """Parse a polynomial written in the names of the generators."""
        position = {name: i for i, name in enumerate(self.names)}
        result: Polynomial = {}
        term: Optional[Polynomial] = None
        sign = 1
        pos = 0
        formula = formula.strip()
        while pos < len(formula):
            match = _TOKEN.match(formula, pos)
            if not match or match.end() == pos:
                raise ValueError(f"cannot parse {formula!r} at position {pos}")
            pos = match.end()
            number, name, power, op = match.groups()
            if op is not None and op in "+-":
                if term is not None:
                    result = self.add(result, term, sign)
                    term = None
                    sign = 1
                sign *= -1 if op == "-" else 1
                continue
            if op == "*":
                continue
            if number is not None:
                factor = self.constant(int(number))
            else:
                if name not in position:
                    raise ValueError(f"unknown generator {name!r} in {formula!r}")
                factor = self.constant(1)
                for _ in range(int(power) if power else 1):
                    factor = self.multiply(factor, self.generator(position[name]))
            term = factor if term is None else self.multiply(term, factor)
        if term is not None:
            result = self.add(result, term, sign)
        return result

    # ------------------------------------------------------------ serialization
    def to_dict(self) -> dict:
        """Serializable description, inverse of :meth:`from_dict`."""
        return {
            "generators": [[n, g] for n, g in zip(self.names, self.degrees)],
            "differential": [
                [[list(a), int(c)] for a, c in sorted(v.items())] for v in self.dgen
            ],
            "modulus": self.modulus,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "SullivanAlgebra":
        names = [n for n, _ in data["generators"]]
        degrees = [g for _, g in data["generators"]]
        values = [{tuple(a): c for a, c in v} for v in data["differential"]]
        return cls(degrees, values, names=names, modulus=data.get("modulus"))

    # ------------------------------------------------------------ basic data
    @property
    def even_indices(self) -> List[int]:
        return [i for i in range(self.m) if not self.odd[i]]

    @property
    def odd_indices(self) -> List[int]:
        return [i for i in range(self.m) if self.odd[i]]

    @property
    def num_even(self) -> int:
        """``q = dim V^even``."""
        return len(self.even_indices)

    @property
    def num_odd(self) -> int:
        """``p = dim V^odd``."""
        return len(self.odd_indices)

    @property
    def homotopy_euler_characteristic(self) -> int:
        """``chi_pi = dim V^even - dim V^odd``."""
        return self.num_even - self.num_odd

    @property
    def formal_dimension(self) -> int:
        """``N = sum |y_j| - sum (|x_i| - 1)``, the formal dimension if elliptic."""
        return sum(g for g in self.degrees if g % 2) - sum(
            g - 1 for g in self.degrees if g % 2 == 0
        )

    def __repr__(self) -> str:
        gens = ", ".join(f"{n}:{g}" for n, g in zip(self.names, self.degrees))
        return f"SullivanAlgebra({gens})"

    def __str__(self) -> str:
        return "\n".join(
            f"d {self.names[i]} = {self.format(self.dgen[i])}" for i in range(self.m)
        )

    def format(self, poly: Polynomial) -> str:
        """Human readable form of a polynomial."""
        if not poly:
            return "0"
        out = []
        for a, c in sorted(poly.items(), reverse=True):
            factors = [
                self.names[i] if e == 1 else f"{self.names[i]}^{e}"
                for i, e in enumerate(a)
                if e
            ]
            mono = "*".join(factors)
            if mono and abs(c) == 1:
                body = mono
            else:
                body = f"{abs(c)}*{mono}" if mono else str(abs(c))
            out.append(("- " if c < 0 else "+ ") + body)
        text = " ".join(out)
        return text[2:] if text.startswith("+ ") else "-" + text[2:]

    # ------------------------------------------------------------ arithmetic
    def _normalize(self, poly: Polynomial) -> Polynomial:
        p = self.modulus
        out = {}
        for a, c in poly.items():
            c = int(c) % p if p else int(c)
            if c:
                out[tuple(a)] = c
        return out

    def constant(self, c: int) -> Polynomial:
        return self._normalize({(0,) * self.m: c})

    def generator(self, i: int) -> Polynomial:
        return {self.unit_monomial(i): 1}

    def unit_monomial(self, i: int) -> Monomial:
        a = [0] * self.m
        a[i] = 1
        return tuple(a)

    def monomial_degree(self, a: Monomial) -> int:
        return sum(e * g for e, g in zip(a, self.degrees))

    def odd_length(self, a: Monomial) -> int:
        """Lower degree of a monomial: its number of odd factors."""
        return sum(a[i] for i in range(self.m) if self.odd[i])

    def multiply_monomials(self, a: Monomial, b: Monomial) -> Tuple[int, Optional[Monomial]]:
        """Return ``(sign, c)`` with ``a * b = sign * c``, or ``(0, None)`` if zero."""
        swaps = 0
        for j in range(self.m):
            if self.odd[j] and b[j]:
                if a[j]:
                    return 0, None
                for i in range(j + 1, self.m):
                    if self.odd[i] and a[i]:
                        swaps += 1
        return (-1 if swaps & 1 else 1), tuple(x + y for x, y in zip(a, b))

    def multiply(self, f: Polynomial, g: Polynomial) -> Polynomial:
        out: Polynomial = {}
        for a, ca in f.items():
            for b, cb in g.items():
                sign, c = self.multiply_monomials(a, b)
                if sign:
                    out[c] = out.get(c, 0) + sign * ca * cb
        return self._normalize(out)

    def add(self, f: Polynomial, g: Polynomial, c: int = 1) -> Polynomial:
        out = dict(f)
        for a, v in g.items():
            out[a] = out.get(a, 0) + c * v
        return self._normalize(out)

    # ------------------------------------------------------------ differential
    def d_monomial(self, a: Monomial) -> Polynomial:
        """Differential of a monomial, using ``d(v w) = dv w + (-1)^|v| v dw``."""
        cached = self._dcache.get(a)
        if cached is not None:
            return cached
        i = next((k for k in range(self.m) if a[k]), None)
        if i is None:
            return {}
        rest = list(a)
        rest[i] -= 1
        rest_t = tuple(rest)
        if not any(rest_t):
            result = self.dgen[i]
        else:
            result = self.multiply(self.dgen[i], {rest_t: 1})
            d_rest = self.d_monomial(rest_t)
            if d_rest:
                second = self.multiply(self.generator(i), d_rest)
                result = self.add(result, second, -1 if self.odd[i] else 1)
        self._dcache[a] = result
        return result

    def d(self, f: Polynomial) -> Polynomial:
        out: Polynomial = {}
        for a, c in f.items():
            for b, cb in self.d_monomial(a).items():
                out[b] = out.get(b, 0) + c * cb
        return self._normalize(out)

    # ------------------------------------------------------------ bases
    def basis(self, n: int) -> List[Monomial]:
        """Sorted list of the monomials of degree ``n``."""
        if n < 0:
            return []
        if n > self._basis_top:
            self._extend_basis(n + 4)
        return self._basis[n]

    def index(self, n: int) -> Dict[Monomial, int]:
        """Position of each monomial of degree ``n`` in :meth:`basis`."""
        self.basis(n)
        return self._index.get(n, {})

    def _extend_basis(self, top: int) -> None:
        layers: List[List[Monomial]] = [[] for _ in range(top + 1)]
        layers[0] = [()]
        for i, g in enumerate(self.degrees):
            new: List[List[Monomial]] = [[] for _ in range(top + 1)]
            max_exp = 1 if self.odd[i] else top // g
            for deg, monos in enumerate(layers):
                for e in range(max_exp + 1):
                    target = deg + e * g
                    if target > top:
                        break
                    new[target].extend(a + (e,) for a in monos)
            layers = new
        for k in range(top + 1):
            ordered = sorted(layers[k])
            self._basis[k] = ordered
            self._index[k] = {a: j for j, a in enumerate(ordered)}
        self._basis_top = top

    def differential_matrix(self, n: int) -> np.ndarray:
        """Integer matrix of ``d: (Lambda V)^n -> (Lambda V)^{n+1}``.

        Columns are indexed by :meth:`basis` ``(n)`` and rows by
        :meth:`basis` ``(n+1)``.
        """
        source = self.basis(n)
        target = self.index(n + 1)
        matrix = np.zeros((len(target), len(source)), dtype=np.int64)
        for j, a in enumerate(source):
            for b, c in self.d_monomial(a).items():
                matrix[target[b], j] = c
        return matrix

    # ------------------------------------------------------------ checks
    def is_square_zero(self) -> bool:
        """``d^2 = 0``. Since ``d^2`` is a derivation, it suffices to test generators."""
        return all(not self.d(v) for v in self.dgen)

    def has_valid_degrees(self) -> bool:
        """Every ``d(v_i)`` is homogeneous of degree ``|v_i| + 1``."""
        return all(
            self.monomial_degree(a) == self.degrees[i] + 1
            for i in range(self.m)
            for a in self.dgen[i]
        )

    def is_minimal(self) -> bool:
        """``d(V)`` is contained in the decomposables ``Lambda^{>=2} V``."""
        return all(sum(a) >= 2 for v in self.dgen for a in v)

    def satisfies_sullivan_order(self) -> bool:
        """``d(v_i)`` only involves generators of degree smaller than ``|v_i|``."""
        return all(
            self.degrees[j] < self.degrees[i]
            for i in range(self.m)
            for a in self.dgen[i]
            for j in range(self.m)
            if a[j]
        )

    def is_minimal_sullivan(self) -> bool:
        """All the conditions of a minimal Sullivan algebra on generators of degree >= 2."""
        return (
            self.has_valid_degrees()
            and self.is_minimal()
            and self.satisfies_sullivan_order()
            and self.is_square_zero()
        )

    # ------------------------------------------------------------ cohomology
    def _check_method(self, method: str) -> None:
        if method not in ("modular", "exact"):
            raise ValueError("method must be 'modular' or 'exact'")
        if method == "exact" and self.modulus:
            raise ValueError("exact ranks are not available over a prime field")

    def rank_d(self, n: int, prime: Optional[int] = P1) -> int:
        """Rank of ``d`` in degree ``n``, modulo ``prime`` or over Q if ``prime`` is None."""
        if self.modulus:
            prime = self.modulus
        if n < 0 or not self.basis(n) or not self.basis(n + 1):
            return 0
        return rank(self.differential_matrix(n), prime)

    def betti_numbers(
        self,
        nmax: Optional[int] = None,
        method: str = "modular",
        primes: Sequence[int] = (P1, P2),
    ) -> List[int]:
        """Dimensions of ``H^n(Lambda V, d)`` for ``0 <= n <= nmax``.

        With ``method='exact'`` the ranks are computed over Q. With
        ``method='modular'`` they are computed modulo each prime of ``primes``
        and the smallest value is kept in each degree. Since ranks modulo a
        prime never exceed rational ranks, modular Betti numbers are upper
        bounds for the rational ones. Over a prime field, the modulus is used.

        By default ``nmax`` is the formal dimension.
        """
        self._check_method(method)
        if nmax is None:
            nmax = self.formal_dimension
        if nmax < 0:
            return []
        if self.modulus:
            prime_list: List[Optional[int]] = [self.modulus]
        elif method == "exact":
            prime_list = [None]
        else:
            prime_list = list(primes)
        best: Optional[List[int]] = None
        for prime in prime_list:
            ranks = [self.rank_d(n, prime) for n in range(nmax + 1)]
            betti = [
                len(self.basis(n)) - ranks[n] - (ranks[n - 1] if n else 0)
                for n in range(nmax + 1)
            ]
            best = betti if best is None else [min(x, y) for x, y in zip(best, betti)]
        return best or []

    def cohomology_dimension(self, method: str = "modular") -> int:
        """``dim H(Lambda V, d)`` for an elliptic algebra (degrees up to ``N``)."""
        return sum(self.betti_numbers(method=method))

    # ------------------------------------------------------------ ellipticity
    def pure_relations(self) -> List[Polynomial]:
        """The polynomials ``f_j``: the parts of ``d(y_j)`` in ``Q[V^even]``.

        They are returned as polynomials of :meth:`even_ring`, one for each
        odd generator, in the order of :attr:`odd_indices`.
        """
        even = self.even_indices
        out = []
        for j in self.odd_indices:
            out.append(
                {
                    tuple(a[i] for i in even): c
                    for a, c in self.dgen[j].items()
                    if all(a[k] == 0 for k in self.odd_indices)
                }
            )
        return out

    def even_ring(self) -> "SullivanAlgebra":
        """The polynomial ring ``Q[V^even]`` with zero differential."""
        even = self.even_indices
        return SullivanAlgebra(
            [self.degrees[i] for i in even],
            names=[self.names[i] for i in even],
            modulus=self.modulus,
        )

    def pure_quotient_dimension(self, n: int, prime: Optional[int] = P1) -> int:
        """``dim (Q[V^even] / (f_1, ..., f_p))^n``, computed modulo ``prime`` or over Q.

        A modular value is an upper bound for the rational one.
        """
        ring = self.even_ring()
        if self.modulus:
            prime = self.modulus
        return _quotient_dimension(ring, self.pure_relations(), n, prime)

    def is_elliptic(self, prime: Optional[int] = P1) -> bool:
        """Halperin's criterion for ellipticity.

        ``(Lambda V, d)`` is elliptic if and only if ``A = Q[V^even]/(f_j)`` is
        finite-dimensional. If elliptic, ``A`` vanishes above ``N``. If ``A``
        vanishes in ``D`` consecutive degrees, where ``D`` is the largest even
        degree, then it vanishes in all higher degrees. So the algebra is
        elliptic if and only if ``A`` vanishes in degrees ``N+1, ..., N+D``.

        With a prime, the dimensions of ``A`` are upper bounds, so a positive
        answer is rigorous over Q. Use ``prime=None`` for an exact answer.
        """
        even = self.even_indices
        if not even:
            return True
        top = max(self.degrees[i] for i in even)
        n0 = self.formal_dimension
        ring = self.even_ring()
        relations = self.pure_relations()
        if self.modulus:
            prime = self.modulus
        return all(
            _quotient_dimension(ring, relations, n, prime) == 0
            for n in range(n0 + 1, n0 + top + 1)
        )

    # ------------------------------------------------------------ sub-structures
    def word_length_part(self, k: int) -> "SullivanAlgebra":
        """The derivation ``d_k`` keeping only the terms of word length ``k``."""
        values = [{a: c for a, c in v.items() if sum(a) == k} for v in self.dgen]
        return SullivanAlgebra(self.degrees, values, self.names, self.modulus)

    def word_lengths(self) -> List[int]:
        """Sorted list of the word lengths occurring in ``d(V)``."""
        return sorted({sum(a) for v in self.dgen for a in v})

    def restrict(self, indices: Sequence[int]) -> "SullivanAlgebra":
        """Sub-algebra on the given generators, assumed closed under ``d``."""
        idx = list(indices)
        values = []
        for i in idx:
            if any(a[j] for a in self.dgen[i] for j in range(self.m) if j not in idx):
                raise ValueError("the generators do not span a sub-algebra")
            values.append({tuple(a[j] for j in idx): c for a, c in self.dgen[i].items()})
        return SullivanAlgebra(
            [self.degrees[i] for i in idx], values, [self.names[i] for i in idx], self.modulus
        )


def _default_names(odd: Sequence[bool]) -> List[str]:
    names = []
    counts = {False: 0, True: 0}
    for o in odd:
        counts[o] += 1
        names.append(("y" if o else "x") + str(counts[o]))
    return names


def _quotient_dimension(
    ring: SullivanAlgebra, relations: Sequence[Polynomial], n: int, prime: Optional[int]
) -> int:
    """Dimension in degree ``n`` of ``ring / (relations)`` for a polynomial ring."""
    basis = ring.basis(n)
    if not basis:
        return 0
    index = ring.index(n)
    columns = []
    for f in relations:
        if not f:
            continue
        df = ring.monomial_degree(next(iter(f)))
        if df > n:
            continue
        for mono in ring.basis(n - df):
            col: Dict[int, int] = {}
            for a, c in f.items():
                k = index[tuple(x + y for x, y in zip(a, mono))]
                col[k] = col.get(k, 0) + c
            columns.append(col)
    if not columns:
        return len(basis)
    matrix = np.zeros((len(basis), len(columns)), dtype=np.int64)
    for j, col in enumerate(columns):
        for k, c in col.items():
            matrix[k, j] = c
    return len(basis) - rank(matrix, prime)


def is_poincare_duality_sequence(betti: Sequence[int]) -> bool:
    """``b_n = b_{N-n}`` and ``b_N = 1``, where ``N = len(betti) - 1``."""
    return bool(betti) and betti[-1] == 1 and list(betti) == list(reversed(betti))


__all__ = [
    "Monomial",
    "Polynomial",
    "SullivanAlgebra",
    "is_poincare_duality_sequence",
    "P1",
    "P2",
]
