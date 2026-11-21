"""Summary records of models, as written to the JSON lines output files."""

from __future__ import annotations

from typing import Dict, Optional

from .classify import structural_flags
from .linalg import P1, P2
from .pure import pure_model
from .sullivan import SullivanAlgebra, is_poincare_duality_sequence


def model_record(
    algebra: SullivanAlgebra,
    exact: str = "auto",
    with_pure: bool = True,
    with_flags: bool = True,
) -> Dict[str, object]:
    """Invariants of a model, as a JSON-serializable dictionary.

    The Betti numbers are computed modulo the primes ``P1`` and ``P2``. They
    are recomputed over Q when ``exact='always'``, or when ``exact='auto'``
    and one of the two excesses is small (``dim H - dim V <= 1`` or
    ``dim H - 2 dim V^odd <= 0``), so that every extremal value is exact.
    See ``data/README.md`` for the description of the fields.
    """
    record: Dict[str, object] = {
        "degrees": algebra.degrees,
        "N": algebra.formal_dimension,
        "dim_V": algebra.m,
        "q": algebra.num_even,
        "p": algebra.num_odd,
        "chi_pi": algebra.homotopy_euler_characteristic,
    }
    elliptic = algebra.is_elliptic(P1) or algebra.is_elliptic(P2)
    record["elliptic"] = elliptic
    if not elliptic:
        return record
    betti = algebra.betti_numbers(primes=(P1, P2))
    record.update(_cohomology_fields(algebra, betti, exact_checked=False))
    if _needs_exact(record, exact):
        betti = algebra.betti_numbers(method="exact")
        record.update(_cohomology_fields(algebra, betti, exact_checked=True))
    if with_pure:
        record["dim_H_pure"] = sum(pure_model(algebra).betti_numbers(primes=(P1, P2)))
    if with_flags:
        record["flags"] = structural_flags(algebra)
    record["model"] = algebra.to_dict()
    return record


def _cohomology_fields(algebra: SullivanAlgebra, betti, exact_checked: bool) -> Dict[str, object]:
    dim_h = sum(betti)
    return {
        "betti": betti,
        "dim_H": dim_h,
        "excess": dim_h - algebra.m,
        "strong_excess": dim_h - 2 * algebra.num_odd,
        "poincare_duality": is_poincare_duality_sequence(betti),
        "exact": exact_checked,
    }


def _needs_exact(record: Dict[str, object], exact: str) -> bool:
    if exact == "always":
        return True
    if exact == "never":
        return False
    return record["excess"] <= 1 or record["strong_excess"] <= 0


def load_model(record: Dict[str, object]) -> Optional[SullivanAlgebra]:
    """The model stored in a record, if any."""
    data = record.get("model")
    return SullivanAlgebra.from_dict(data) if data else None
