#!/usr/bin/env python3
"""Benchmark CPU, resident OpenCL, fused OpenCL, and compute-only OpenCL."""

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
from pyparticles.forces.fused_const_drag import FusedConstDragOCL
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


def make_pset(x0, v0):
    pset = ParticlesSet(len(x0), dtype=np.float32)
    pset.X[:] = x0
    pset.V[:] = v0
    pset.M[:] = 0.1
    return pset


def build_cpu(x0, v0, dt):
    n = len(x0)
    pset = make_pset(x0, v0)

    gravity = ConstForce(n, u_force=(0.0, 0.0, -10.0))
    drag = Drag(n, Consts=0.01)
    force = MultipleForce(n)
    force.append_force(gravity)
    force.append_force(drag)
    force.set_masses(pset.M)
    return pset, EulerSolver(force, pset, dt)


def preload(ctx, pset):
    ctx.set_from_host("X", pset.X)
    ctx.set_from_host("V", pset.V)
    ctx.reset_transfer_stats()


def build_gpu_generic(x0, v0, dt):
    n = len(x0)
    pset = make_pset(x0, v0)
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
        sync_positions=True,
    )
    preload(ctx, pset)
    return pset, solver, ctx


def build_gpu_fused(x0, v0, dt, sync_positions=True):
    n = len(x0)
    pset = make_pset(x0, v0)
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
        sync_positions=sync_positions,
    )
    preload(ctx, pset)
    return pset, solver, ctx


def run_steps(solver, steps):
    start = time.perf_counter()
    for _ in range(steps):
        solver.step()
    return time.perf_counter() - start


def mib(value):
    return value / (1024.0 * 1024.0)


def print_accuracy(label, pset, reference):
    print(label)
    print("  X max abs :", np.max(np.abs(pset.X - reference.X)))
    print("  X mean abs:", np.mean(np.abs(pset.X - reference.X)))
    print("  X allclose:", np.allclose(pset.X, reference.X, rtol=1e-4, atol=1e-4))
    print("  V max abs :", np.max(np.abs(pset.V - reference.V)))
    print("  V mean abs:", np.mean(np.abs(pset.V - reference.V)))
    print("  V allclose:", np.allclose(pset.V, reference.V, rtol=1e-4, atol=1e-4))


def print_transfers(label, stats):
    print(label)
    print("  H2D calls :", stats["h2d_calls"])
    print("  H2D MiB   : %.3f" % mib(stats["h2d_bytes"]))
    print("  D2H calls :", stats["d2h_calls"])
    print("  D2H MiB   : %.3f" % mib(stats["d2h_bytes"]))
    print("  X D2H     :", stats["by_buffer"]["X"]["d2h_calls"])
    print("  V D2H     :", stats["by_buffer"]["V"]["d2h_calls"])
    print("  A H2D/D2H : %d/%d" % (
        stats["by_buffer"]["A"]["h2d_calls"],
        stats["by_buffer"]["A"]["d2h_calls"],
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--particles", type=int, default=4096)
    parser.add_argument("-s", "--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    x0, v0 = build_initial_state(args.particles, args.seed)
    p_cpu, cpu = build_cpu(x0, v0, args.dt)
    p_generic, generic, generic_ctx = build_gpu_generic(x0, v0, args.dt)
    p_fused, fused, fused_ctx = build_gpu_fused(x0, v0, args.dt, sync_positions=True)
    p_compute, compute, compute_ctx = build_gpu_fused(
        x0, v0, args.dt, sync_positions=False
    )

    cpu_wall = run_steps(cpu, args.steps)
    generic_wall = run_steps(generic, args.steps)
    generic_stats = generic_ctx.transfer_stats
    fused_wall = run_steps(fused, args.steps)
    fused_stats = fused_ctx.transfer_stats
    compute_wall = run_steps(compute, args.steps)
    compute_stats = compute_ctx.transfer_stats

    # Resident V, and compute-only X, are synchronized only after timing so the
    # numerical comparison does not contaminate hot-path transfer statistics.
    generic.sync_to_host(velocity=True)
    fused.sync_to_host(velocity=True)
    compute.sync_to_host(velocity=True)

    print("Particles   :", args.particles)
    print("Steps       :", args.steps)
    print("Sim time    :", args.steps * args.dt)
    print()
    print("CPU wall       : %.6f s" % cpu_wall)
    print("GPU generic    : %.6f s" % generic_wall)
    print("GPU fused      : %.6f s" % fused_wall)
    print("GPU compute-only: %.6f s" % compute_wall)
    print("CPU/generic    : %.3fx" % (cpu_wall / generic_wall))
    print("CPU/fused      : %.3fx" % (cpu_wall / fused_wall))
    print("CPU/compute    : %.3fx" % (cpu_wall / compute_wall))
    print("Generic/fused  : %.3fx" % (generic_wall / fused_wall))
    print("Fused/compute  : %.3fx" % (fused_wall / compute_wall))
    print()
    print_accuracy("Generic accuracy vs CPU:", p_generic, p_cpu)
    print()
    print_accuracy("Fused accuracy vs CPU:", p_fused, p_cpu)
    print()
    print_accuracy("Compute-only accuracy vs CPU:", p_compute, p_cpu)
    print()
    print_transfers("Generic steady-state transfers:", generic_stats)
    print()
    print_transfers("Fused steady-state transfers:", fused_stats)
    print()
    print_transfers("Compute-only steady-state transfers:", compute_stats)


if __name__ == "__main__":
    main()
