#!/usr/bin/env python3
"""Cohomology of the standard examples.

For each example, prints the formal dimension, the generators, the rational
Betti numbers (exact), the two excesses ``dim H - dim V`` and
``dim H - 2 dim V^odd``, a check against the expected Betti numbers, and the
weight bound ``||chi_T||_1`` when a weight grading is available.

Usage: python3 scripts/examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hilali.examples import EXAMPLES  # noqa: E402
from hilali.sullivan import is_poincare_duality_sequence  # noqa: E402
from hilali.weights import homogeneous_weights, weight_bound  # noqa: E402


def describe(example) -> bool:
    algebra = example.build()
    betti = algebra.betti_numbers(method="exact")
    modular = algebra.betti_numbers()
    dim_h = sum(betti)
    weights = example.weights or homogeneous_weights(algebra)
    bound = weight_bound(algebra, weights) if weights else None
    expected = example.betti is None or betti == list(example.betti)
    ok = (
        algebra.is_minimal_sullivan()
        and algebra.is_elliptic(prime=None)
        and is_poincare_duality_sequence(betti)
        and dim_h == example.dim_H
        and expected
        and modular == betti
    )
    nonzero = ", ".join(f"b{n}={b}" for n, b in enumerate(betti) if b)
    print(f"{example.name}")
    print(
        f"  degrees {algebra.degrees}  N={algebra.formal_dimension}  "
        f"dim V={algebra.m}  p={algebra.num_odd}"
    )
    for line in str(algebra).splitlines():
        if not line.endswith("= 0"):
            print(f"  {line}")
    print(f"  {nonzero}")
    print(
        f"  dim H={dim_h}  dim H - dim V={dim_h - algebra.m}  "
        f"dim H - 2p={dim_h - 2 * algebra.num_odd}  weight bound={bound}  "
        f"{'OK' if ok else 'MISMATCH'}"
    )
    return ok


def main() -> int:
    results = [describe(example) for example in EXAMPLES]
    print(f"\n{sum(results)}/{len(results)} examples agree with the expected cohomology")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
