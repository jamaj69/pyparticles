import unittest

import numpy as np

from pyparticles.forces.const_force import ConstForceOCL
from pyparticles.forces.drag import DragOCL
from pyparticles.forces.multiple_force import MultipleForceOCL
from pyparticles.ode.euler_solver import EulerSolverOCL
from pyparticles.pset.default_boundary import DefaultBoundary
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
class OpenCLResidencyTests(unittest.TestCase):
    def _make_fountain_like_solver(self, n=128):
        rng = np.random.default_rng(77)
        pset = ParticlesSet(n, dtype=np.float32)
        pset.X[:] = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
        pset.X[:, 2] += 10.0
        pset.V[:] = rng.normal(0.0, 0.5, size=(n, 3)).astype(np.float32)
        pset.M[:] = 0.1

        ctx = OpenCLcontext(n, 3, OCLC_X | OCLC_V | OCLC_A | OCLC_M)
        gravity = ConstForceOCL(
            n,
            u_force=(0.0, 0.0, -10.0),
            ocl_context=ctx,
        )
        drag = DragOCL(n, Consts=0.01, ocl_context=ctx)
        force = MultipleForceOCL(n, ocl_context=ctx)
        force.append_force(gravity)
        force.append_force(drag)
        force.set_masses(pset.M)

        solver = EulerSolverOCL(
            force,
            pset,
            0.005,
            ocl_context=ctx,
            sync_velocity=False,
        )
        return pset, ctx, solver

    def test_inactive_boundary_keeps_velocity_on_device(self):
        pset, ctx, solver = self._make_fountain_like_solver()
        pset.set_boundary(
            DefaultBoundary(
                (-100.0, 100.0, -100.0, 100.0, 0.0, 100.0),
                dim=3,
                defualt_pos=lambda p, i: None,
            )
        )

        ctx.set_from_host("X", pset.X)
        ctx.set_from_host("V", pset.V)
        ctx.reset_transfer_stats()

        solver.step()
        solver.step()

        stats = ctx.transfer_stats
        self.assertEqual(stats["h2d_calls"], 0)
        self.assertEqual(stats["by_buffer"]["X"]["d2h_calls"], 2)
        self.assertEqual(stats["by_buffer"]["V"]["d2h_calls"], 0)
        self.assertEqual(stats["by_buffer"]["A"]["d2h_calls"], 0)
        self.assertEqual(stats["by_buffer"]["A"]["h2d_calls"], 0)

    def test_boundary_hit_syncs_velocity_then_reuploads_changed_state(self):
        n = 8
        pset = ParticlesSet(n, dtype=np.float32)
        pset.M[:] = 1.0
        pset.X[:] = 0.0
        pset.V[:] = 0.0
        pset.X[0, 2] = -1.0

        ctx = OpenCLcontext(n, 3, OCLC_X | OCLC_V | OCLC_A | OCLC_M)
        zero_force = ConstForceOCL(n, u_force=(0.0, 0.0, 0.0), ocl_context=ctx)
        force = MultipleForceOCL(n, ocl_context=ctx)
        force.append_force(zero_force)
        force.set_masses(pset.M)

        def reset_particle(p, indices):
            p.X[indices, :] = 0.0
            p.V[indices, :] = 0.0

        pset.set_boundary(
            DefaultBoundary(
                (-10.0, 10.0, -10.0, 10.0, 0.0, 10.0),
                dim=3,
                defualt_pos=reset_particle,
            )
        )

        solver = EulerSolverOCL(
            force,
            pset,
            0.01,
            ocl_context=ctx,
            sync_velocity=False,
        )

        ctx.set_from_host("X", pset.X)
        ctx.set_from_host("V", pset.V)
        ctx.reset_transfer_stats()

        solver.step()
        first = ctx.transfer_stats
        self.assertEqual(first["by_buffer"]["X"]["d2h_calls"], 1)
        self.assertEqual(first["by_buffer"]["V"]["d2h_calls"], 1)
        self.assertEqual(first["h2d_calls"], 0)

        solver.step()
        second = ctx.transfer_stats
        self.assertEqual(second["by_buffer"]["X"]["h2d_calls"], 1)
        self.assertEqual(second["by_buffer"]["V"]["h2d_calls"], 1)
        self.assertEqual(second["by_buffer"]["V"]["d2h_calls"], 1)


if __name__ == "__main__":
    unittest.main()
