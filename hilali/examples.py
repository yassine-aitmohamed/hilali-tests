"""Standard examples of elliptic minimal Sullivan algebras.

Each function returns a :class:`~hilali.sullivan.SullivanAlgebra`. The
dictionary :data:`EXAMPLES` lists them with their rational Betti numbers and,
when one is natural, a weight grading.
"""

from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from .sullivan import SullivanAlgebra


def even_sphere(n: int = 2) -> SullivanAlgebra:
    """``S^n`` for even ``n``: ``dx = 0``, ``dy = x^2``."""
    return SullivanAlgebra.from_strings([("x", n), ("y", 2 * n - 1)], {"y": "x^2"})


def odd_sphere(n: int = 3) -> SullivanAlgebra:
    """``S^n`` for odd ``n``."""
    return SullivanAlgebra.from_strings([("y", n)], {})


def complex_projective_space(n: int) -> SullivanAlgebra:
    """``CP^n``: ``dy = x^{n+1}``."""
    return SullivanAlgebra.from_strings([("x", 2), ("y", 2 * n + 1)], {"y": f"x^{n + 1}"})


def quaternionic_projective_space(n: int) -> SullivanAlgebra:
    """``HP^n``: ``dy = x^{n+1}`` with ``|x| = 4``."""
    return SullivanAlgebra.from_strings([("x", 4), ("y", 4 * n + 3)], {"y": f"x^{n + 1}"})


def product_of_odd_spheres(*degrees: int) -> SullivanAlgebra:
    return SullivanAlgebra.from_strings([(f"y{k + 1}", g) for k, g in enumerate(degrees)], {})


def heisenberg() -> SullivanAlgebra:
    """The Heisenberg model ``Lambda(a, b, c)``, ``|a| = |b| = 3``, ``|c| = 5``, ``dc = ab``."""
    return SullivanAlgebra.from_strings([("a", 3), ("b", 3), ("c", 5)], {"c": "a*b"})


def s2_times_s2() -> SullivanAlgebra:
    return SullivanAlgebra.from_strings(
        [("x1", 2), ("x2", 2), ("y1", 3), ("y2", 3)], {"y1": "x1^2", "y2": "x2^2"}
    )


def grassmannian_2_4() -> SullivanAlgebra:
    """``Gr_2(C^4)``: ``Q[c1, c2]`` modulo the complete symmetric functions ``h3, h4``."""
    return SullivanAlgebra.from_strings(
        [("c1", 2), ("c2", 4), ("y5", 5), ("y7", 7)],
        {"y5": "c1^3 - 2*c1*c2", "y7": "c1^4 - 3*c1^2*c2 + c2^2"},
    )


def flag_manifold_3() -> SullivanAlgebra:
    """``U(3)/T``: ``Q[t1, t2, t3]`` modulo the symmetric functions.

    The linear relation is used to eliminate ``t3 = -t1 - t2``. The remaining
    relations are the complete symmetric functions ``h2(t1, t2)`` and
    ``h3(t1, t2)``.
    """
    return SullivanAlgebra.from_strings(
        [("t1", 2), ("t2", 2), ("y3", 3), ("y5", 5)],
        {"y3": "t1^2 + t1*t2 + t2^2", "y5": "t1^3 + t1^2*t2 + t1*t2^2 + t2^3"},
    )


_TWO_COMPONENT_GENERATORS = [
    ("a", 3), ("b", 3), ("c", 5), ("e", 5), ("u", 9), ("f", 13), ("g", 13),
]


def two_component_first() -> SullivanAlgebra:
    """A generic point of the first component for the degrees ``3,3,5,5,9,13,13``.

    It is isomorphic to the product of two Heisenberg models and ``S^9``.
    """
    return SullivanAlgebra.from_strings(
        _TWO_COMPONENT_GENERATORS,
        {
            "c": "2*a*b",
            "e": "3*a*b",
            "f": "15*c*u - 10*e*u",
            "g": "21*c*u - 14*e*u",
        },
    )


def two_component_second() -> SullivanAlgebra:
    """A generic point of the second component for the degrees ``3,3,5,5,9,13,13``.

    It is the product of ``Lambda(a, b)`` with the Chevalley-Eilenberg algebra
    of the free 3-step nilpotent Lie algebra on two generators.
    """
    return SullivanAlgebra.from_strings(
        _TWO_COMPONENT_GENERATORS,
        {
            "u": "c*e",
            "f": "2*c*u + 3*e*u",
            "g": "5*c*u - e*u",
        },
    )


class Example(NamedTuple):
    """A named model with its total Betti number, and optionally its Betti numbers and a weight grading."""

    name: str
    build: Callable[[], SullivanAlgebra]
    dim_H: int
    betti: Optional[Tuple[int, ...]] = None
    weights: Optional[Sequence[Tuple[int, ...]]] = None


def poincare_series(*factors: Dict[int, int]) -> Tuple[int, ...]:
    """Betti numbers of a tensor product, from the Betti numbers ``{degree: dim}`` of the factors."""
    total: Dict[int, int] = {0: 1}
    for factor in factors:
        product: Dict[int, int] = {}
        for a, x in total.items():
            for b, y in factor.items():
                product[a + b] = product.get(a + b, 0) + x * y
        total = product
    top = max(total)
    return tuple(total.get(n, 0) for n in range(top + 1))


def _heisenberg_betti(a: int, b: int, c: int) -> Dict[int, int]:
    """Betti numbers of ``Lambda(a, b, c)`` with ``dc = ab``: classes ``1, a, b, ac, bc, abc``."""
    out: Dict[int, int] = {}
    for n in (0, a, b, a + c, b + c, a + b + c):
        out[n] = out.get(n, 0) + 1
    return out


def _sphere_betti(n: int) -> Dict[int, int]:
    return {0: 1, n: 1}


def _example(name, build, factors, weights=None) -> Example:
    betti = poincare_series(*factors)
    return Example(name, build, sum(betti), betti, weights)


EXAMPLES: List[Example] = [
    _example("S^2", even_sphere, [_sphere_betti(2)], [(1,), (2,)]),
    _example("S^4", lambda: even_sphere(4), [_sphere_betti(4)], [(1,), (2,)]),
    _example(
        "CP^3",
        lambda: complex_projective_space(3),
        [{0: 1, 2: 1, 4: 1, 6: 1}],
        [(1,), (4,)],
    ),
    _example(
        "HP^2",
        lambda: quaternionic_projective_space(2),
        [{0: 1, 4: 1, 8: 1}],
        [(1,), (3,)],
    ),
    _example(
        "S^3 x S^3",
        lambda: product_of_odd_spheres(3, 3),
        [_sphere_betti(3), _sphere_betti(3)],
        [(1,), (1,)],
    ),
    _example(
        "Heisenberg (3,3,5)",
        heisenberg,
        [_heisenberg_betti(3, 3, 5)],
        [(1,), (1,), (2,)],
    ),
    _example(
        "S^2 x S^2",
        s2_times_s2,
        [_sphere_betti(2), _sphere_betti(2)],
        [(1,), (-1,), (2,), (-2,)],
    ),
    _example(
        "Gr_2(C^4)",
        grassmannian_2_4,
        [{0: 1, 2: 1, 4: 2, 6: 1, 8: 1}],
        [(1,), (2,), (3,), (4,)],
    ),
    _example(
        "U(3)/T",
        flag_manifold_3,
        [{0: 1, 2: 2, 4: 2, 6: 1}],
        [(1,), (1,), (2,), (3,)],
    ),
    _example(
        "(3,3,5,5,9,13,13), first component",
        two_component_first,
        [_heisenberg_betti(3, 3, 5), _heisenberg_betti(5, 9, 13), _sphere_betti(13)],
    ),
    _example(
        "(3,3,5,5,9,13,13), second component",
        two_component_second,
        [_sphere_betti(3), _sphere_betti(3), {0: 1, 5: 2, 18: 3, 27: 3, 40: 2, 45: 1}],
    ),
]
