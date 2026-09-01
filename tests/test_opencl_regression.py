import unittest

import numpy as np

from pyparticles.forces.const_force import ConstForce
from pyparticles.forces.drag import Drag, DragOCL
from pyparticles.forces.multiple_force import MultipleForce
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

    def test_fountain_force_pipeline_matches_cpu(self):
        n = 512
        steps = 50
        dt = 0.005

        x0 = self.rng.normal(0.0, 2.0, size=(n, 3)).astype(np.float32)
        v0 = self.rng.normal(0.0, 5.0, size=(n, 3)).astype(np.float32)

        p_cpu = ParticlesSet(n, dtype=np.float32)
        p_cpu.X[:] = x0
        p_cpu.V[:] = v0
        p_cpu.M[:] = 0.1

        grav_cpu = ConstForce(n, dim=3, u_force=(0.0, 0.0, -10.0))
        drag_cpu = Drag(n, dim=3, Consts=0.01)
        force_cpu = MultipleForce(n, dim=3)
        force_cpu.append_force(grav_cpu)
        force_cpu.append_force(drag_cpu)
        force_cpu.set_masses(p_cpu.M)
        solver_cpu = EulerSolver(force_cpu, p_cpu, dt)

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
        grav_gpu = ConstForce(n, dim=3, u_force=(0.0, 0.0, -10.0))
        drag_gpu = DragOCL(n, dim=3, Consts=0.01, ocl_context=ctx)
        force_gpu = MultipleForce(n, dim=3)
        force_gpu.append_force(grav_gpu)
        force_gpu.append_force(drag_gpu)
        force_gpu.set_masses(p_gpu.M)
        solver_gpu = EulerSolverOCL(force_gpu, p_gpu, dt, ocl_context=ctx)

        for _ in range(steps):
            solver_cpu.step()
            solver_gpu.step()

        np.testing.assert_allclose(p_gpu.X, p_cpu.X, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(p_gpu.V, p_cpu.V, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
