import unittest
import weakref

import numpy as np

from pyparticles.animation import animated_ogl_compat
from pyparticles.forces.electromagnetic_field import ElectromagneticField
from pyparticles.forces.electrostatic import Electrostatic
from pyparticles.forces.gravity import Gravity, GravityOCL
from pyparticles.forces.lennard_jones import LenardJones
from pyparticles.forces.pseudo_bubble import PseudoBubble, PseudoBubbleOCL
from pyparticles.ogl.draw_particles_ogl_compat import DrawParticlesGL
from pyparticles.pset.constrained_force_interactions import ConstrainedForceInteractions
from pyparticles.pset.constrained_x import ConstrainedX
from pyparticles.pset.logger import Logger
from pyparticles.pset.opencl_context import OCLC_A, OCLC_M, OCLC_X, OpenCLcontext
from pyparticles.pset.particles_set import ParticlesSet
from pyparticles.pset.rand_cluster import RandCluster
from pyparticles.utils.pypart_global import test_pyopencl


class DemoCompatibilityTests(unittest.TestCase):
    def test_official_demos_use_compat_animation(self):
        import pyparticles.demo.bubble as bubble
        import pyparticles.demo.electromagnetic_demo as electromagnetic
        import pyparticles.demo.electrostatic_demo as electrostatic
        import pyparticles.demo.fountain as fountain
        import pyparticles.demo.gas_lennard_jones as gas_lj
        import pyparticles.demo.gravity_clusters as galaxy
        import pyparticles.demo.solar_system as solar
        import pyparticles.demo.springs as springs
        import pyparticles.demo.springs_constr as springs_constr

        modules = (
            bubble,
            electromagnetic,
            electrostatic,
            fountain,
            gas_lj,
            galaxy,
            solar,
            springs,
            springs_constr,
        )
        for module in modules:
            with self.subTest(module=module.__name__):
                self.assertIs(module.aogl.AnimatedGl, animated_ogl_compat.AnimatedGl)

    def test_animation_trajectory_step_property(self):
        animation = animated_ogl_compat.AnimatedGl()
        animation.trajectory_step = 3
        self.assertEqual(animation.trajectory_step, 3)
        self.assertEqual(animation.draw_particles.trajectory_step, 3)

    def test_cleanup_callbacks_release_captured_objects(self):
        class Captured(object):
            pass

        animation = animated_ogl_compat.AnimatedGl()
        captured = Captured()
        captured_ref = weakref.ref(captured)

        def make_cleanup(obj):
            def cleanup(_animation):
                self.assertIsNotNone(obj)
            return cleanup

        animation.add_cleanup_callback(make_cleanup(captured))
        del captured

        animation.cleanup_resources()
        self.assertIsNone(captured_ref())

        # Cleanup remains idempotent after the callback list has been consumed.
        animation.cleanup_resources()

    def test_draw_compat_properties_work_without_gl_context(self):
        draw = DrawParticlesGL()
        color_fun = lambda pset, i: (1.0, 1.0, 1.0, 1.0)
        vect_fun = lambda rgba, pset: rgba.fill(1.0)
        draw.color_fun = color_fun
        draw.vect_color_fun = vect_fun
        self.assertIs(draw.color_fun, color_fun)
        self.assertIs(draw.vect_color_fun, vect_fun)

    def test_draw_render_benchmark_modes_work_without_gl_context(self):
        draw = DrawParticlesGL()
        expected = {
            "legacy",
            "no_msaa",
            "no_fog",
            "no_blend",
            "no_alpha",
            "no_depth",
            "fast_points",
        }
        self.assertEqual(set(draw.render_benchmark_modes), expected)
        for mode in sorted(expected):
            draw.render_benchmark_mode = mode
            self.assertEqual(draw.render_benchmark_mode, mode)
        with self.assertRaises(ValueError):
            draw.render_benchmark_mode = "not-a-mode"

    def test_logger_enabled_flags_are_real_booleans(self):
        pset = ParticlesSet(3)
        logger = Logger(pset, 8, log_X=True, log_V=False)
        self.assertIs(logger.log_X_enabled, True)
        self.assertIs(logger.log_V_enabled, False)
        logger.log()
        self.assertEqual(logger.log_size, 1)

    def test_rand_cluster_accepts_numpy_mass_and_velocity_arrays(self):
        pset = ParticlesSet(16)
        cluster = RandCluster()
        cluster.insert3(
            pset.X,
            M=pset.M,
            V=pset.V,
            n=pset.size,
            radius=2.0,
            mass_rng=(0.5, 1.0),
            vel_rng=(0.1, 0.2),
            vel_mdl="const",
            vel_dir=(1.0, 0.0, 0.0),
        )
        self.assertTrue(np.isfinite(pset.X).all())
        self.assertTrue(np.isfinite(pset.M).all())
        self.assertTrue(np.isfinite(pset.V).all())
        self.assertTrue((pset.M > 0.0).all())

    def test_lennard_jones_constructs_on_python3(self):
        pset = ParticlesSet(4)
        pset.X[:] = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.5, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
        pset.M[:] = 1.0
        force = LenardJones(pset.size, pset.dim, pset.M, Consts=(1.0, 0.05))
        acceleration = force.update_force(pset)
        self.assertEqual(acceleration.shape, (4, 3))
        self.assertTrue(np.isfinite(acceleration).all())

    def test_constraints_accept_demo_style_arrays(self):
        pset = ParticlesSet(8)
        constrained = ConstrainedX(pset)
        constrained.add_x_constraint(
            np.array([0, 7]),
            np.array([[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]),
        )
        free = constrained.get_cx_free_indicies()
        self.assertIsInstance(free, slice)
        self.assertEqual((free.start, free.stop), (1, 7))

        interactions = ConstrainedForceInteractions(pset)
        interactions.add_connections(np.array([[0.0, 1.0], [1.0, 2.0]]))
        dense = interactions.dense
        self.assertEqual(dense.dtype, np.bool_)
        self.assertTrue(dense[0, 1])
        self.assertTrue(dense[1, 2])

    def test_electromagnetic_field_accepts_numpy_constructor_arrays(self):
        pset = ParticlesSet(4, charge=True)
        pset.M[:] = 2.0
        pset.Q[:] = np.array([[1.0], [-1.0], [1.0], [-1.0]])
        pset.V[:] = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        )

        force = ElectromagneticField(pset.size, m=pset.M, q=pset.Q)
        force.append_electric_field(lambda E, X: E.__setitem__(slice(None), (1.0, 2.0, 3.0)))
        force.append_magnetic_field(lambda B, X: B.__setitem__(slice(None), (0.0, 0.0, 2.0)))
        acceleration = force.update_force(pset)
        self.assertTrue(np.isfinite(acceleration).all())

    def test_electrostatic_constructor_accepts_numpy_arrays(self):
        pset = ParticlesSet(3, charge=True)
        pset.X[:] = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        pset.M[:] = 1.0
        pset.Q[:] = np.array([[1.0], [-1.0], [1.0]])
        force = Electrostatic(pset.size, m=pset.M, q=pset.Q, Consts=1.0)
        acceleration = force.update_force(pset)
        self.assertTrue(np.isfinite(acceleration).all())

    def test_pseudo_bubble_cpu(self):
        pset = ParticlesSet(4)
        pset.X[:] = np.array(
            [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.3, 0.0], [2.0, 0.0, 0.0]]
        )
        pset.M[:] = np.array([[1.0], [1.5], [2.0], [1.0]])
        force = PseudoBubble(pset.size, m=pset.M, Consts=(0.5, 2.0))
        acceleration = force.update_force(pset)
        self.assertTrue(np.isfinite(acceleration).all())

    @unittest.skipUnless(test_pyopencl(), "OpenCL device unavailable")
    def test_pseudo_bubble_opencl_matches_cpu(self):
        pset = ParticlesSet(4, dtype=np.float32)
        pset.X[:] = np.array(
            [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.3, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        pset.M[:] = np.array([[1.0], [1.5], [2.0], [1.0]], dtype=np.float32)

        cpu = PseudoBubble(pset.size, m=pset.M, Consts=(0.5, 2.0))
        expected = cpu.update_force(pset).astype(np.float32)

        ctx = OpenCLcontext(pset.size, 3, OCLC_X | OCLC_A | OCLC_M)
        gpu = PseudoBubbleOCL(
            pset.size,
            m=pset.M,
            Consts=(0.5, 2.0),
            ocl_context=ctx,
        )
        actual = gpu.update_force(pset)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    @unittest.skipUnless(test_pyopencl(), "OpenCL device unavailable")
    def test_gravity_opencl_matches_cpu(self):
        pset = ParticlesSet(5, dtype=np.float32)
        pset.X[:] = np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [2.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        )
        pset.M[:] = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)

        cpu = Gravity(pset.size, m=pset.M, Consts=0.25)
        expected = cpu.update_force(pset).astype(np.float32)

        ctx = OpenCLcontext(pset.size, 3, OCLC_X | OCLC_A | OCLC_M)
        gpu = GravityOCL(pset.size, m=pset.M, Consts=0.25, ocl_context=ctx)
        actual = gpu.update_force(pset)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(gpu.F, actual * pset.M, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
