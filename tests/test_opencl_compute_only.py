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

        # Explicit synchronization still produces the final host state.
        solver.sync_to_host(velocity=True)
        self.assertEqual(ctx.transfer_stats["by_buffer"]["X"]["d2h_calls"], 1)
        self.assertEqual(ctx.transfer_stats["by_buffer"]["V"]["d2h_calls"], 1)


if __name__ == "__main__":
    unittest.main()
