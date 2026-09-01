#!/usr/bin/env python3
"""Benchmark the fountain physics core on CPU and resident OpenCL buffers."""

import argparse
from pathlib import Path
import sys
import time

# Allow direct execution as ``python benchmarks/opencl_fountain.py`` from a
# source checkout without requiring an editable install or PYTHONPATH tweak.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from pyparticles.forces.const_force import ConstForce, ConstForceOCL
from pyparticles.forces.drag import Drag, DragOCL
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


def build_initial_state(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 2.0, size=(n, 3)).astype(np.float32)
    v = rng.normal(0.0, 5.0, size=(n, 3)).astype(np.float32)
    return x, v


def build_cpu(x0, v0, dt):
    n = len(x0)
    pset = ParticlesSet(n, dtype=np.float32)
    pset.X[:] = x0
    pset.V[:] = v0
    pset.M[:] = 0.1

    gravity = ConstForce(n, u_force=(0.0, 0.0, -10.0))
    drag = Drag(n, Consts=0.01)
    force = MultipleForce(n)
    force.append_force(gravity)
    force.append_force(drag)
    force.set_masses(pset.M)
    return pset, EulerSolver(force, pset, dt)


def build_gpu(x0, v0, dt):
    n = len(x0)
    pset = ParticlesSet(n, dtype=np.float32)
    pset.X[:] = x0
    pset.V[:] = v0
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
        dt,
        ocl_context=ctx,
        sync_velocity=False,
    )

    # Remove initial state upload from the steady-state transfer accounting.
    ctx.set_from_host("X", pset.X)
    ctx.set_from_host("V", pset.V)
    ctx.reset_transfer_stats()
    return pset, solver, ctx


def run_steps(solver, steps):
    start = time.perf_counter()
    for _ in range(steps):
        solver.step()
    return time.perf_counter() - start


def mib(value):
    return value / (1024.0 * 1024.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--particles", type=int, default=4096)
    parser.add_argument("-s", "--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    x0, v0 = build_initial_state(args.particles, args.seed)
    p_cpu, cpu = build_cpu(x0, v0, args.dt)
    p_gpu, gpu, ctx = build_gpu(x0, v0, args.dt)

    cpu_wall = run_steps(cpu, args.steps)
    gpu_wall = run_steps(gpu, args.steps)
    hot_stats = ctx.transfer_stats

    # V intentionally remains resident during the timed region. Synchronize it
    # only now for the numerical comparison.
    gpu.sync_to_host(velocity=True)

    print("Particles :", args.particles)
    print("Steps     :", args.steps)
    print("Sim time  :", args.steps * args.dt)
    print()
    print("CPU wall  : %.6f s" % cpu_wall)
    print("GPU wall  : %.6f s" % gpu_wall)
    print("CPU/GPU   : %.3fx" % (cpu_wall / gpu_wall))
    print()
    print("X max abs :", np.max(np.abs(p_gpu.X - p_cpu.X)))
    print("X mean abs:", np.mean(np.abs(p_gpu.X - p_cpu.X)))
    print("X allclose:", np.allclose(p_gpu.X, p_cpu.X, rtol=1e-4, atol=1e-4))
    print()
    print("V max abs :", np.max(np.abs(p_gpu.V - p_cpu.V)))
    print("V mean abs:", np.mean(np.abs(p_gpu.V - p_cpu.V)))
    print("V allclose:", np.allclose(p_gpu.V, p_cpu.V, rtol=1e-4, atol=1e-4))
    print()
    print("Steady-state OpenCL transfers during timed region:")
    print("  H2D calls :", hot_stats["h2d_calls"])
    print("  H2D MiB   : %.3f" % mib(hot_stats["h2d_bytes"]))
    print("  D2H calls :", hot_stats["d2h_calls"])
    print("  D2H MiB   : %.3f" % mib(hot_stats["d2h_bytes"]))
    print("  X D2H     :", hot_stats["by_buffer"]["X"]["d2h_calls"])
    print("  V D2H     :", hot_stats["by_buffer"]["V"]["d2h_calls"])
    print("  A H2D/D2H : %d/%d" % (
        hot_stats["by_buffer"]["A"]["h2d_calls"],
        hot_stats["by_buffer"]["A"]["d2h_calls"],
    ))


if __name__ == "__main__":
    main()
