import unittest

import numpy as np

from pyparticles.forces.const_force import ConstForce, ConstForceOCL
from pyparticles.forces.drag import Drag, DragOCL
from pyparticles.forces.fused_const_drag import FusedConstDragOCL
from pyparticles.forces.gravity import GravityOCL
from pyparticles.forces.multiple_force import MultipleForce, MultipleForceOCL
from pyparticles.ode.euler_solver import EulerSolver, EulerSolverOCL
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
class OpenCLRegressionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def test_context_rejects_float64(self):
        with self.assertRaises(TypeError):
            OpenCLcontext(16, 3, dtype=np.float64)

    def test_drag_matches_cpu(self):
        n = 512
        pset = ParticlesSet(n, dtype=np.float32)
        pset.M[:] = self.rng.uniform(0.1, 2.0, size=(n, 1)).astype(np.float32)
        pset.V[:] = self.rng.normal(0.0, 5.0, size=(n, 3)).astype(np.float32)

        cpu = Drag(n, dim=3, Consts=0.01)
        cpu.set_masses(pset.M)
        a_cpu = cpu.update_force(pset).copy()

        ctx = OpenCLcontext(n, 3, OCLC_V | OCLC_A | OCLC_M, dtype=np.float32)
        gpu = DragOCL(n, dim=3, Consts=0.01, ocl_context=ctx)
        gpu.set_masses(pset.M)
        a_gpu = gpu.update_force(pset).copy()

        np.testing.assert_allclose(a_gpu, a_cpu, rtol=2e-5, atol=2e-6)

    def test_euler_matches_cpu_after_many_steps(self):
        n = 512
        steps = 100
        dt = 0.005

        x0 = self.rng.normal(0.0, 10.0, size=(n, 3)).astype(np.float32)
        v0 = self.rng.normal(0.0, 5.0, size=(n, 3)).astype(np.float32)

        p_cpu = ParticlesSet(n, dtype=np.float32)
        p_cpu.X[:] = x0
        p_cpu.V[:] = v0
        p_cpu.M[:] = 1.0

        p_gpu = ParticlesSet(n, dtype=np.float32)
        p_gpu.X[:] = x0
        p_gpu.V[:] = v0
        p_gpu.M[:] = 1.0

        f_cpu = ConstForce(n, dim=3, u_force=(0.25, -0.50, -9.81))
        f_gpu = ConstForce(n, dim=3, u_force=(0.25, -0.50, -9.81))

        s_cpu = EulerSolver(f_cpu, p_cpu, dt)
        ctx = OpenCLcontext(n, 3, OCLC_X | OCLC_V | OCLC_A, dtype=np.float32)
        s_gpu = EulerSolverOCL(f_gpu, p_gpu, dt, ocl_context=ctx)

        for _ in range(steps):
            s_cpu.step()
            s_gpu.step()

        np.testing.assert_allclose(p_gpu.X, p_cpu.X, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(p_gpu.V, p_cpu.V, rtol=2e-5, atol=2e-5)

    def _build_cpu_fountain(self, x0, v0, dt):
        n = len(x0)
        pset = ParticlesSet(n, dtype=np.float32)
        pset.X[:] = x0
        pset.V[:] = v0
        pset.M[:] = 0.1

        grav = ConstForce(n, dim=3, u_force=(0.0, 0.0, -10.0))
        drag = Drag(n, dim=3, Consts=0.01)
        force = MultipleForce(n, dim=3)
        force.append_force(grav)
        force.append_force(drag)
        force.set_masses(pset.M)
        return pset, EulerSolver(force, pset, dt)

    def test_fountain_force_pipeline_matches_cpu(self):
        n = 512
        steps = 50
        dt = 0.005

        x0 = self.rng.normal(0.0, 2.0, size=(n, 3)).astype(np.float32)
        v0 = self.rng.normal(0.0, 5.0, size=(n, 3)).astype(np.float32)
        p_cpu, solver_cpu = self._build_cpu_fountain(x0, v0, dt)

        p_gpu = ParticlesSet(n, dtype=np.float32)
        p_gpu.X[:] = x0
        p_gpu.V[:] = v0
        p_gpu.M[:] = 0.1

        ctx = OpenCLcontext(
            n,
            3,
            OCLC_X | OCLC_V | OCLC_A | OCLC_M,
            dtype=np.float32,
        )
        grav_gpu = ConstForceOCL(
            n,
            dim=3,
            u_force=(0.0, 0.0, -10.0),
            ocl_context=ctx,
        )
        drag_gpu = DragOCL(n, dim=3, Consts=0.01, ocl_context=ctx)
        force_gpu = MultipleForceOCL(n, dim=3, ocl_context=ctx)
        force_gpu.append_force(grav_gpu)
        force_gpu.append_force(drag_gpu)
        force_gpu.set_masses(p_gpu.M)
        solver_gpu = EulerSolverOCL(force_gpu, p_gpu, dt, ocl_context=ctx)

        for _ in range(steps):
            solver_cpu.step()
            solver_gpu.step()

        np.testing.assert_allclose(p_gpu.X, p_cpu.X, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(p_gpu.V, p_cpu.V, rtol=1e-4, atol=1e-4)

    def test_fused_fountain_matches_cpu_and_stays_resident(self):
        n = 512
        steps = 50
        dt = 0.005

        x0 = self.rng.normal(0.0, 2.0, size=(n, 3)).astype(np.float32)
        v0 = self.rng.normal(0.0, 5.0, size=(n, 3)).astype(np.float32)
        p_cpu, solver_cpu = self._build_cpu_fountain(x0, v0, dt)

        p_gpu = ParticlesSet(n, dtype=np.float32)
        p_gpu.X[:] = x0
        p_gpu.V[:] = v0
        p_gpu.M[:] = 0.1
        ctx = OpenCLcontext(n, 3, OCLC_X | OCLC_V | OCLC_A | OCLC_M)
        force = FusedConstDragOCL(
            n,
            m=p_gpu.M,
            u_force=(0.0, 0.0, -10.0),
            drag_const=0.01,
            ocl_context=ctx,
        )
        solver_gpu = EulerSolverOCL(
            force,
            p_gpu,
            dt,
            ocl_context=ctx,
            sync_velocity=False,
        )

        ctx.set_from_host("X", p_gpu.X)
        ctx.set_from_host("V", p_gpu.V)
        ctx.reset_transfer_stats()

        for _ in range(steps):
            solver_cpu.step()
            solver_gpu.step()

        stats = ctx.transfer_stats
        solver_gpu.sync_to_host(velocity=True)

        np.testing.assert_allclose(p_gpu.X, p_cpu.X, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(p_gpu.V, p_cpu.V, rtol=1e-4, atol=1e-4)
        self.assertEqual(stats["h2d_calls"], 0)
        self.assertEqual(stats["d2h_calls"], steps)
        self.assertEqual(stats["by_buffer"]["X"]["d2h_calls"], steps)
        self.assertEqual(stats["by_buffer"]["V"]["d2h_calls"], 0)
        self.assertEqual(stats["by_buffer"]["A"]["h2d_calls"], 0)
        self.assertEqual(stats["by_buffer"]["A"]["d2h_calls"], 0)

    def test_resident_fountain_pipeline_avoids_hot_path_uploads(self):
        n = 128
        dt = 0.005
        pset = ParticlesSet(n, dtype=np.float32)
        pset.X[:] = self.rng.normal(size=(n, 3)).astype(np.float32)
        pset.V[:] = self.rng.normal(size=(n, 3)).astype(np.float32)
        pset.M[:] = 0.1

        ctx = OpenCLcontext(n, 3, OCLC_X | OCLC_V | OCLC_A | OCLC_M)
        grav = ConstForceOCL(n, u_force=(0.0, 0.0, -10.0), ocl_context=ctx)
        drag = DragOCL(n, Consts=0.01, ocl_context=ctx)
        force = MultipleForceOCL(n, ocl_context=ctx)
        force.append_force(grav)
        force.append_force(drag)
        force.set_masses(pset.M)
        solver = EulerSolverOCL(force, pset, dt, ocl_context=ctx)

        # Preload the initial state, then measure only the steady-state hot path.
        ctx.set_from_host("X", pset.X)
        ctx.set_from_host("V", pset.V)
        ctx.reset_transfer_stats()

        solver.step()
        solver.step()

        stats = ctx.transfer_stats
        self.assertEqual(stats["h2d_calls"], 0)
        self.assertEqual(stats["by_buffer"]["A"]["h2d_calls"], 0)
        self.assertEqual(stats["by_buffer"]["A"]["d2h_calls"], 0)
        self.assertEqual(stats["by_buffer"]["X"]["d2h_calls"], 2)
        self.assertEqual(stats["by_buffer"]["V"]["d2h_calls"], 2)

    def test_galaxy_mode_copies_only_positions_to_host(self):
        n = 32
        dt = 0.01
        pset = ParticlesSet(n, dtype=np.float32)
        pset.X[:] = self.rng.normal(0.0, 1.0, size=(n, 3)).astype(np.float32)
        pset.V[:] = self.rng.normal(0.0, 0.1, size=(n, 3)).astype(np.float32)
        pset.M[:] = self.rng.uniform(0.5, 2.0, size=(n, 1)).astype(np.float32)

        ctx = OpenCLcontext(n, 3, OCLC_X | OCLC_V | OCLC_A | OCLC_M)
        force = GravityOCL(n, Consts=1e-4, ocl_context=ctx)
        force.set_masses(pset.M)
        solver = EulerSolverOCL(
            force,
            pset,
            dt,
            ocl_context=ctx,
            sync_velocity=False,
        )

        ctx.set_from_host("X", pset.X)
        ctx.set_from_host("V", pset.V)
        ctx.reset_transfer_stats()

        solver.step()
        solver.step()

        stats = ctx.transfer_stats
        self.assertEqual(stats["h2d_calls"], 0)
        self.assertEqual(stats["d2h_calls"], 2)
        self.assertEqual(stats["by_buffer"]["X"]["d2h_calls"], 2)
        self.assertEqual(stats["by_buffer"]["V"]["d2h_calls"], 0)
        self.assertEqual(stats["by_buffer"]["A"]["d2h_calls"], 0)

        # Explicit synchronization remains available for host consumers.
        solver.sync_to_host(velocity=True)
        self.assertEqual(ctx.transfer_stats["by_buffer"]["V"]["d2h_calls"], 1)


if __name__ == "__main__":
    unittest.main()
