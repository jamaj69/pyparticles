import unittest

import numpy as np

from pyparticles.forces.fused_const_drag import FusedConstDragOCL
from pyparticles.ode.euler_solver import EulerSolverOCL
from pyparticles.pset.opencl_context import (
    OCLC_A,
    OCLC_M,
    OCLC_V,
    OCLC_X,
    OpenCLcontext,
)
from pyparticles.pset.particles_set import ParticlesSet
from pyparticles.utils.pypart_global import test_pyopencl


@unittest.skipUnless(test_pyopencl(), "No usable OpenCL device")
class OpenCLComputeOnlyTests(unittest.TestCase):
    def test_fused_compute_only_has_zero_hot_path_transfers(self):
        rng = np.random.default_rng(12345)
        n = 256
        steps = 20
        dt = 0.005

        pset = ParticlesSet(n, dtype=np.float32)
        pset.X[:] = rng.normal(0.0, 2.0, size=(n, 3)).astype(np.float32)
        pset.V[:] = rng.normal(0.0, 5.0, size=(n, 3)).astype(np.float32)
        pset.M[:] = 0.1

        ctx = OpenCLcontext(n, 3, OCLC_X | OCLC_V | OCLC_A | OCLC_M)
        force = FusedConstDragOCL(
            n,
            m=pset.M,
            u_force=(0.0, 0.0, -10.0),
            drag_const=0.01,
            ocl_context=ctx,
        )
        solver = EulerSolverOCL(
            force,
            pset,
            dt,
            ocl_context=ctx,
            sync_velocity=False,
            sync_positions=False,
        )

        ctx.set_from_host("X", pset.X)
        ctx.set_from_host("V", pset.V)
        ctx.reset_transfer_stats()

        for _ in range(steps):
            solver.step()

        stats = ctx.transfer_stats
        self.assertEqual(stats["h2d_calls"], 0)
        self.assertEqual(stats["d2h_calls"], 0)
        self.assertEqual(stats["h2d_bytes"], 0)
        self.assertEqual(stats["d2h_bytes"], 0)

        solver.sync_to_host(velocity=True)
        self.assertEqual(ctx.transfer_stats["by_buffer"]["X"]["d2h_calls"], 1)
        self.assertEqual(ctx.transfer_stats["by_buffer"]["V"]["d2h_calls"], 1)

    def test_fountain_boundary_respawns_entirely_on_device(self):
        n = 64
        dt = 0.005
        bounds = (-100.0, 100.0, -100.0, 100.0, 0.0, 100.0)

        pset = ParticlesSet(n, dtype=np.float32)
        pset.M[:] = 0.1
        pset.X[:] = 0.0
        pset.X[:, 2] = -1.0  # every particle starts outside the fountain floor
        pset.V[:] = 0.0

        ctx = OpenCLcontext(n, 3, OCLC_X | OCLC_V | OCLC_A | OCLC_M)
        force = FusedConstDragOCL(
            n,
            m=pset.M,
            u_force=(0.0, 0.0, -10.0),
            drag_const=0.01,
            ocl_context=ctx,
            fountain_bounds=bounds,
        )
        solver = EulerSolverOCL(
            force,
            pset,
            dt,
            ocl_context=ctx,
            sync_velocity=False,
            sync_positions=False,
        )

        ctx.set_from_host("X", pset.X)
        ctx.set_from_host("V", pset.V)
        ctx.reset_transfer_stats()
        solver.step()

        stats = ctx.transfer_stats
        self.assertEqual(stats["h2d_calls"], 0)
        self.assertEqual(stats["d2h_calls"], 0)

        solver.sync_to_host(velocity=True)

        self.assertTrue(np.all(pset.X >= 0.0))
        self.assertTrue(np.all(pset.X < 0.01))
        self.assertTrue(np.all(pset.V[:, 2] > 0.0))
        self.assertTrue(np.all(np.isfinite(pset.X)))
        self.assertTrue(np.all(np.isfinite(pset.V)))

    def test_fused_fountain_render_target_matches_canonical_positions(self):
        rng = np.random.default_rng(9876)
        n = 128
        dt = 0.005
        bounds = (-100.0, 100.0, -100.0, 100.0, 0.0, 100.0)

        pset = ParticlesSet(n, dtype=np.float32)
        pset.M[:] = 0.1
        pset.X[:] = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
        pset.X[:, 2] += 2.0
        pset.V[:] = rng.normal(0.0, 2.0, size=(n, 3)).astype(np.float32)

        ctx = OpenCLcontext(n, 3, OCLC_X | OCLC_V | OCLC_A | OCLC_M)
        ctx.add_array_by_name("render_X", size=n, dim=3, dtype=np.float32)
        render_x = ctx.get_by_name("render_X")

        force = FusedConstDragOCL(
            n,
            m=pset.M,
            u_force=(0.0, 0.0, -10.0),
            drag_const=0.01,
            ocl_context=ctx,
            fountain_bounds=bounds,
        )
        solver = EulerSolverOCL(
            force,
            pset,
            dt,
            ocl_context=ctx,
            sync_velocity=False,
            sync_positions=False,
        )

        ctx.set_from_host("X", pset.X)
        ctx.set_from_host("V", pset.V)
        force.set_render_target(render_x.data)
        solver.step()

        canonical = ctx.X_cla.get(queue=ctx.CL_queue)
        mirrored = render_x.get(queue=ctx.CL_queue)
        np.testing.assert_array_equal(mirrored, canonical)

        # The render target is intentionally one-shot: a normal next step must
        # not overwrite it unless the caller explicitly arms another target.
        frozen = mirrored.copy()
        solver.step()
        np.testing.assert_array_equal(render_x.get(queue=ctx.CL_queue), frozen)


if __name__ == "__main__":
    unittest.main()
