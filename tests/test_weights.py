"""Tests of the weighted Euler characteristic."""

import unittest

from . import context  # noqa: F401
from hilali.examples import EXAMPLES, heisenberg
from hilali.sullivan import SullivanAlgebra
from hilali.weights import (
    divide_by_one_minus,
    euler_characteristic,
    euler_characteristic_from_weights,
    homogeneous_weights,
    is_weight_grading,
    l1_norm,
    satisfies_product_identity,
    weight_bound,
    weighted_betti_numbers,
)

E6 = SullivanAlgebra.from_products(
    [2, 4, 3, 3, 5, 7],
    {2: [(1, [0, 0])], 1: [(1, [0, 3])], 4: [(1, [0, 1]), (-1, [2, 3])], 5: [(1, [1, 1]), (2, [3, 4])]},
)


def e6_weights(wx, wb):
    return [(wx,), (wx + wb,), (2 * wx,), (wb,), (2 * wx + wb,), (2 * wx + 2 * wb,)]


class WeightTest(unittest.TestCase):
    def check(self, algebra, weights, dim_h, bound):
        self.assertTrue(is_weight_grading(algebra, weights))
        betti = weighted_betti_numbers(algebra, weights)
        chi = euler_characteristic(betti)
        self.assertEqual(sum(betti.values()), dim_h)
        self.assertTrue(satisfies_product_identity(chi, algebra, weights))
        self.assertEqual(euler_characteristic_from_weights(algebra, weights), chi)
        self.assertEqual(weight_bound(algebra, weights), bound)
        self.assertLessEqual(l1_norm(chi), dim_h)

    def test_heisenberg(self):
        self.check(heisenberg(), [(1,), (1,), (2,)], 6, 6)
        self.check(heisenberg(), [(1,), (-2,), (-1,)], 6, 6)
        self.check(heisenberg(), [(1, 0), (0, 1), (1, 1)], 6, 6)

    def test_e6(self):
        self.assertTrue(E6.is_minimal_sullivan())
        self.check(E6, [(1, 0), (1, 1), (2, 0), (0, 1), (2, 1), (2, 2)], 8, 8)
        self.check(E6, e6_weights(1, 2), 8, 8)
        self.check(E6, e6_weights(1, 1), 8, 4)
        self.check(E6, e6_weights(2, -1), 8, 4)

    def test_examples(self):
        for example in EXAMPLES:
            algebra = example.build()
            weights = example.weights or homogeneous_weights(algebra)
            if weights:
                with self.subTest(example.name):
                    self.assertLessEqual(weight_bound(algebra, weights), example.dim_H)

    def test_rejects_wrong_weights(self):
        self.assertFalse(is_weight_grading(heisenberg(), [(1,), (1,), (1,)]))

    def test_division(self):
        self.assertEqual(divide_by_one_minus({(0,): 1, (3,): -1}, (1,)), {(0,): 1, (1,): 1, (2,): 1})
        self.assertIsNone(divide_by_one_minus({(0,): 1, (3,): 1}, (1,)))


if __name__ == "__main__":
    unittest.main()
