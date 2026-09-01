import unittest

import numpy as np

from pyparticles.pset.particles_set import ParticlesSet


class ParticlesSetCompatibilityTests(unittest.TestCase):
    def test_default_construction_with_numpy_2(self):
        pset = ParticlesSet(8, dtype=np.float32)

        self.assertEqual(pset.size, 8)
        self.assertEqual(pset.dim, 3)
        self.assertEqual(pset.X.shape, (8, 3))
        self.assertEqual(pset.V.shape, (8, 3))
        self.assertEqual(pset.M.shape, (8, 1))
        self.assertEqual(pset.X.dtype, np.float32)

    def test_property_names_are_a_normal_python_collection(self):
        pset = ParticlesSet(4, dtype=np.float32)
        names = pset.get_properties_names()

        self.assertIsInstance(names, list)
        self.assertEqual(set(names), {"X", "V", "M"})

    def test_resize_updates_numpy_properties(self):
        pset = ParticlesSet(4, dtype=np.float32)
        pset.X[:] = np.arange(12, dtype=np.float32).reshape(4, 3)
        pset.resize(7)

        self.assertEqual(pset.size, 7)
        self.assertEqual(pset.X.shape, (7, 3))
        self.assertEqual(pset.V.shape, (7, 3))
        self.assertEqual(pset.M.shape, (7, 1))


if __name__ == "__main__":
    unittest.main()
