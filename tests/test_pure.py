"""Tests of the associated pure algebra and of the derivation delta."""

import random
import unittest

from . import context  # noqa: F401
from hilali.generate import generic_model
from hilali.pure import (
    delta_analysis,
    derivation_space_dimension,
    has_chain_level_d1,
    has_nonclosed_even,
    is_pure,
    linkage_test,
    pure_model,
)
from hilali.sullivan import SullivanAlgebra


def nonclosed_example() -> SullivanAlgebra:
    """chi_pi = -1, with an even generator w such that d_1(w) != 0."""
    return SullivanAlgebra.from_strings(
        [("x", 2), ("y", 3), ("z", 3), ("w", 6), ("u", 11)],
        {"y": "x^2", "z": "2*x^2", "w": "2*x^2*y - x^2*z", "u": "w^2 + 2*w*y*z + x^6"},
    )


class PureTest(unittest.TestCase):
    def test_pure_model(self):
        algebra = nonclosed_example()
        pure = pure_model(algebra)
        self.assertTrue(is_pure(pure))
        self.assertFalse(is_pure(algebra))
        self.assertEqual(pure.dgen[algebra.names.index("u")], pure.parse("w^2 + x^6"))
        self.assertEqual(sum(pure.betti_numbers(method="exact")), 8)

    def test_delta_analysis(self):
        algebra = nonclosed_example()
        self.assertTrue(algebra.is_minimal_sullivan())
        self.assertTrue(has_nonclosed_even(algebra))
        self.assertTrue(has_chain_level_d1(algebra))
        result = delta_analysis(algebra, prime=None)
        self.assertTrue(result["match"])
        self.assertEqual(result["rank"], 0)
        self.assertEqual(result["dim_kernel"], 4)
        self.assertEqual(result["dim_higher"], 0)
        self.assertEqual(result["betti"], [1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1])
        self.assertEqual(2 * result["dim_kernel"], sum(result["betti"]))

    def test_derivations_and_linkage(self):
        algebra = nonclosed_example()
        self.assertEqual(derivation_space_dimension(algebra), (0, 1))
        self.assertEqual(derivation_space_dimension(algebra, admissible=True), (0, 0))
        self.assertTrue(any(linkage_test(algebra, random.Random(0))))

    def test_generic_chi_minus_one(self):
        rng = random.Random(11)
        checked = 0
        for _ in range(10):
            model = generic_model([2, 3, 3, 6, 11], rng)
            if not has_nonclosed_even(model) or not model.is_elliptic():
                continue
            result = delta_analysis(model)
            self.assertTrue(result["match"])
            self.assertGreaterEqual(result["dim_kernel"], model.num_even + 1)
            checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
