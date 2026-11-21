"""Tests of the random generators and of the summary records."""

import random
import unittest

from . import context  # noqa: F401
from hilali.classify import structural_flags
from hilali.generate import cohomology_representatives, generic_model, random_minimal_model
from hilali.examples import heisenberg
from hilali.records import load_model, model_record
from hilali.sullivan import SullivanAlgebra, is_poincare_duality_sequence

DEGREES = [
    [3, 3, 5],
    [2, 3, 3, 5],
    [2, 2, 3, 5, 7],
    [2, 4, 3, 5, 7, 9],
    [3, 3, 5, 5, 7, 9],
    [2, 3, 3, 6, 11],
]


class GeneratorTest(unittest.TestCase):
    def test_random_models_are_minimal(self):
        rng = random.Random(7)
        elliptic = 0
        for degrees in DEGREES:
            for _ in range(5):
                model = random_minimal_model(degrees, rng, density=rng.choice([0.5, 1.0]))
                self.assertIsNotNone(model)
                self.assertTrue(model.is_minimal_sullivan())
                if model.is_elliptic():
                    elliptic += 1
                    exact = model.betti_numbers(method="exact")
                    self.assertEqual(model.betti_numbers(), exact)
                    self.assertTrue(is_poincare_duality_sequence(exact))
        self.assertGreater(elliptic, 0)

    def test_cohomology_representatives(self):
        algebra = heisenberg()
        reps = cohomology_representatives(algebra, 8)
        self.assertEqual(len(reps), 2)
        for z in reps:
            self.assertEqual(algebra.d(z), {})

    def test_generic_model(self):
        rng = random.Random(3)
        model = generic_model([2, 2, 3, 3, 5, 7], rng)
        self.assertTrue(model.is_minimal_sullivan())
        self.assertEqual(model.modulus, 2097143)
        pure = generic_model([2, 2, 3, 3], rng, pattern="zzpp")
        self.assertEqual(pure.betti_numbers(), [1, 0, 2, 0, 1])


class RecordTest(unittest.TestCase):
    def test_record_fields(self):
        record = model_record(heisenberg())
        self.assertTrue(record["elliptic"])
        self.assertEqual(record["dim_H"], 6)
        self.assertEqual(record["excess"], 3)
        self.assertEqual(record["strong_excess"], 0)
        self.assertTrue(record["exact"])
        self.assertTrue(record["poincare_duality"])
        self.assertEqual(load_model(record).dgen, heisenberg().dgen)

    def test_non_elliptic_record(self):
        record = model_record(SullivanAlgebra([2, 3]))
        self.assertFalse(record["elliptic"])
        self.assertNotIn("betti", record)

    def test_flags(self):
        flags = structural_flags(heisenberg())
        self.assertTrue(flags["coformal"])
        self.assertTrue(flags["known_class"])
        mixed = SullivanAlgebra.from_strings(
            [("x", 2), ("y", 3), ("z", 3), ("w", 6), ("u", 11)],
            {"y": "x^2", "z": "2*x^2", "w": "2*x^2*y - x^2*z", "u": "w^2 + 2*w*y*z + x^6"},
        )
        self.assertFalse(structural_flags(mixed)["pure"])


if __name__ == "__main__":
    unittest.main()
