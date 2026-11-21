"""Tests of the Friedlander-Halperin criterion."""

import unittest

from . import context  # noqa: F401
from hilali.degree_types import (
    basis_sizes,
    elliptic_degree_types,
    is_decomposable_degree,
    satisfies_sac,
)


class DegreeTypeTest(unittest.TestCase):
    def test_decomposable(self):
        self.assertTrue(is_decomposable_degree(4, [2]))
        self.assertFalse(is_decomposable_degree(2, [2]))
        self.assertFalse(is_decomposable_degree(6, [4]))
        self.assertTrue(is_decomposable_degree(6, [2, 4]))

    def test_sac(self):
        self.assertTrue(satisfies_sac([2], [3]))
        self.assertTrue(satisfies_sac([2], [5]))
        self.assertFalse(satisfies_sac([4], [5]))
        self.assertTrue(satisfies_sac([2, 4], [5, 7]))
        self.assertFalse(satisfies_sac([2, 2], [3]))
        self.assertFalse(satisfies_sac([2, 6], [3, 5]))

    def test_enumeration(self):
        types = elliptic_degree_types(12)
        self.assertEqual(len(types), 123)
        self.assertIn(((2,), (3,), 2), types)
        self.assertIn(((), (3, 3, 5), 11), types)
        for evens, odds, n in types:
            self.assertTrue(satisfies_sac(evens, odds))
            self.assertLessEqual(sum(evens), n)
            self.assertLessEqual(sum(odds), 2 * n - 1)

    def test_basis_sizes(self):
        self.assertEqual(basis_sizes([2, 3], 6), [1, 0, 1, 1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
