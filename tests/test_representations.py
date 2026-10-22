import math
import random
import unittest
from ga.representations import (BinaryRepresentation, IntegerRepresentation, FloatRepresentation,
                                generate_permutation, validate_permutation)


class RepresentationTests(unittest.TestCase):
    def test_binary(self):
        rep = BinaryRepresentation(8)
        self.assertTrue(rep.validate(rep.generate(random.Random(42))))
        for bad in [(0,) * 7, (0,) * 7 + (2,), (0,) * 7 + (1.0,), (True,) * 8]:
            self.assertFalse(rep.validate(bad))

    def test_integer(self):
        rep = IntegerRepresentation(5, -2, 2)
        self.assertTrue(rep.validate(rep.generate(random.Random(42))))
        self.assertTrue(rep.validate((1,) * 5))
        for bad in [(3,) * 5, (1.0,) * 5, (True,) * 5]:
            self.assertFalse(rep.validate(bad))

    def test_float(self):
        rep = FloatRepresentation(3, (-1., 0., 10.), (1., 2., 20.))
        for seed in range(20):
            self.assertTrue(rep.validate(rep.generate(random.Random(seed))))
        for bad in [(math.nan, 1., 12.), (0., math.inf, 12.), (0., 1., 21.)]:
            self.assertFalse(rep.validate(bad))
        self.assertTrue(rep.validate((-1., 2., 20.)))

    def test_invalid_configuration(self):
        for constructor in [lambda: BinaryRepresentation(0), lambda: IntegerRepresentation(3, 4, 1),
                            lambda: FloatRepresentation(2, math.nan, 2),
                            lambda: FloatRepresentation(2, (0.,), 2.),
                            lambda: generate_permutation((1, 1), random.Random())]:
            with self.assertRaises(ValueError):
                constructor()

    def test_permutation(self):
        ids = (7, 2, 15, 9)
        self.assertTrue(validate_permutation(generate_permutation(ids, random.Random(1)), ids))
        for bad in [(7, 2, 15), (7, 2, 15, 15), (7, 2, 15, 99), (7., 2, 15, 9)]:
            self.assertFalse(validate_permutation(bad, ids))
