"""Tests of the algebra, the cohomology computations and the known examples."""

import unittest

import numpy as np

from . import context  # noqa: F401
from hilali.examples import EXAMPLES, heisenberg
from hilali.linalg import P1, P2, P31, rank_exact, rank_mod, rank_mod_plain
from hilali.sullivan import SullivanAlgebra, is_poincare_duality_sequence


class ExampleTest(unittest.TestCase):
    def test_known_examples(self):
        for example in EXAMPLES:
            with self.subTest(example.name):
                algebra = example.build()
                self.assertTrue(algebra.is_minimal_sullivan())
                self.assertTrue(algebra.is_elliptic(prime=None))
                exact = algebra.betti_numbers(method="exact")
                self.assertEqual(sum(exact), example.dim_H)
                if example.betti is not None:
                    self.assertEqual(exact, list(example.betti))
                self.assertEqual(algebra.betti_numbers(primes=(P1,)), exact)
                self.assertEqual(algebra.betti_numbers(primes=(P2,)), exact)
                self.assertTrue(is_poincare_duality_sequence(exact))

    def test_specific_values(self):
        values = {example.name: example.dim_H for example in EXAMPLES}
        self.assertEqual(values["S^2"], 2)
        self.assertEqual(values["CP^3"], 4)
        self.assertEqual(values["S^3 x S^3"], 4)
        self.assertEqual(values["Heisenberg (3,3,5)"], 6)
        self.assertEqual(values["S^2 x S^2"], 4)
        self.assertEqual(values["Gr_2(C^4)"], 6)
        self.assertEqual(values["U(3)/T"], 6)
        self.assertEqual(values["(3,3,5,5,9,13,13), first component"], 72)
        self.assertEqual(values["(3,3,5,5,9,13,13), second component"], 48)

    def test_cohomology_vanishes_above_formal_dimension(self):
        algebra = heisenberg()
        n = algebra.formal_dimension
        betti = algebra.betti_numbers(n + 4, method="exact")
        self.assertEqual(betti[n + 1:], [0, 0, 0, 0])


class AlgebraTest(unittest.TestCase):
    def test_signs_of_products(self):
        algebra = SullivanAlgebra.from_strings([("a", 3), ("b", 3), ("x", 2)], {})
        self.assertEqual(algebra.parse("b*a"), algebra.parse("-a*b"))
        self.assertEqual(algebra.parse("a*a"), {})
        self.assertEqual(algebra.parse("x*a - a*x"), {})
        self.assertEqual(algebra.parse("2*x^2 - x*x"), algebra.parse("x^2"))

    def test_leibniz_rule(self):
        algebra = SullivanAlgebra.from_strings(
            [("x", 2), ("a", 3), ("y", 5)], {"y": "x^3"}
        )
        f, g = algebra.parse("x*a"), algebra.parse("y + x*a")
        left = algebra.d(algebra.multiply(f, g))
        right = algebra.add(
            algebra.multiply(algebra.d(f), g), algebra.multiply(f, algebra.d(g)), -1
        )
        self.assertEqual(left, right)

    def test_square_zero_and_minimality(self):
        bad = SullivanAlgebra.from_strings([("x", 2), ("y", 3), ("z", 4)], {"y": "x^2", "z": "x*y"})
        self.assertFalse(bad.is_square_zero())
        linear = SullivanAlgebra.from_strings([("a", 3), ("b", 4)], {"b": "a"})
        self.assertFalse(linear.is_minimal())
        wrong_degree = SullivanAlgebra.from_strings([("x", 2), ("y", 5)], {"y": "x^2"})
        self.assertFalse(wrong_degree.has_valid_degrees())

    def test_invariants(self):
        algebra = SullivanAlgebra.from_strings([("x", 2), ("y", 3), ("z", 3)], {"y": "x^2"})
        self.assertEqual(algebra.formal_dimension, 5)
        self.assertEqual(algebra.homotopy_euler_characteristic, -1)
        self.assertEqual((algebra.num_even, algebra.num_odd), (1, 2))

    def test_non_elliptic(self):
        self.assertFalse(SullivanAlgebra([2, 3]).is_elliptic())
        self.assertFalse(SullivanAlgebra([2, 3]).is_elliptic(prime=None))
        two_evens = SullivanAlgebra.from_strings(
            [("x1", 2), ("x2", 2), ("y1", 3), ("y2", 3)], {"y1": "x1^2", "y2": "x1*x2"}
        )
        self.assertFalse(two_evens.is_elliptic(prime=None))
        self.assertTrue(SullivanAlgebra([3, 5, 7]).is_elliptic())

    def test_serialization(self):
        algebra = heisenberg()
        copy = SullivanAlgebra.from_dict(algebra.to_dict())
        self.assertEqual(copy.dgen, algebra.dgen)
        self.assertEqual(copy.names, algebra.names)

    def test_prime_field(self):
        algebra = SullivanAlgebra.from_strings([("x", 2), ("y", 3)], {"y": "x^2"}, modulus=7)
        self.assertEqual(algebra.betti_numbers(), [1, 0, 1])
        with self.assertRaises(ValueError):
            algebra.betti_numbers(method="exact")


class RankTest(unittest.TestCase):
    def test_modular_and_exact_ranks(self):
        rng = np.random.default_rng(1)
        for _ in range(25):
            m, n = rng.integers(1, 120, size=2)
            r = int(rng.integers(0, min(m, n) + 1))
            matrix = rng.integers(-3, 4, (m, r)) @ rng.integers(-3, 4, (r, n))
            expected = int(np.linalg.matrix_rank(matrix.astype(float)))
            self.assertEqual(rank_mod(matrix, P1), expected)
            self.assertEqual(rank_mod(matrix, P2), expected)
            self.assertEqual(rank_mod_plain(matrix, P31), expected)
            self.assertEqual(rank_exact(matrix), expected)

    def test_blocked_elimination_on_wide_matrix(self):
        rng = np.random.default_rng(2)
        left = rng.integers(-2, 3, (300, 140))
        matrix = left @ rng.integers(-2, 3, (140, 500))
        self.assertEqual(rank_mod(matrix, P1), rank_mod_plain(matrix, P31))

    def test_modular_rank_is_a_lower_bound(self):
        matrix = np.array([[P1, 0], [0, 1]], dtype=np.int64)
        self.assertEqual(rank_mod(matrix, P1), 1)
        self.assertEqual(rank_exact(matrix), 2)


if __name__ == "__main__":
    unittest.main()
