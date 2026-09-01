#!/usr/bin/env python3
"""Benchmark naive and tiled OpenCL gravity kernels."""

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from pyparticles.forces.gravity import GravityOCL
from pyparticles.pset.opencl_context import OCLC_A, OCLC_M, OCLC_X, OpenCLcontext
from pyparticles.pset.particles_set import ParticlesSet


def make_state(n, seed):
    rng = np.random.default_rng(seed)
    pset = ParticlesSet(n, dtype=np.float32)
    pset.X[:] = rng.normal(0.0, 10.0, size=(n, 3)).astype(np.float32)
    pset.M[:] = rng.uniform(0.5, 2.0, size=(n, 1)).astype(np.float32)
    return pset


def build_force(pset, mode, tile_size):
    ctx = OpenCLcontext(pset.size, 3, OCLC_X | OCLC_A | OCLC_M)
    force = GravityOCL(
        pset.size,
        Consts=1e-4,
        ocl_context=ctx,
        kernel_mode=mode,
        tile_size=tile_size,
    )
    force.set_masses(pset.M)
    ctx.set_from_host("X", pset.X)
    ctx.reset_transfer_stats()
    return force, ctx


def run_force(force, ctx, pset, iterations):
    # Warm up compilation/caches and make sure X is resident.
    force.update_force_device(pset)
    ctx.CL_queue.finish()

    start = time.perf_counter()
    for _ in range(iterations):
        force.update_force_device(pset)
    ctx.CL_queue.finish()
    wall = time.perf_counter() - start

    out = np.empty_like(pset.X)
    ctx.sync_to_host("A", out)
    return wall, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--particles", type=int, default=2000)
    parser.add_argument("-i", "--iterations", type=int, default=20)
    parser.add_argument("--tile", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    pset = make_state(args.particles, args.seed)
    naive, naive_ctx = build_force(pset, "naive", args.tile)
    tiled, tiled_ctx = build_force(pset, "tiled", args.tile)

    naive_wall, a_naive = run_force(naive, naive_ctx, pset, args.iterations)
    tiled_wall, a_tiled = run_force(tiled, tiled_ctx, pset, args.iterations)

    print("Particles    :", args.particles)
    print("Iterations   :", args.iterations)
    print("Tile size    :", tiled.tile_size)
    print()
    print("Naive wall   : %.6f s" % naive_wall)
    print("Tiled wall   : %.6f s" % tiled_wall)
    print("Naive/tiled  : %.3fx" % (naive_wall / tiled_wall))
    print("Naive/iter   : %.6f ms" % (naive_wall * 1000.0 / args.iterations))
    print("Tiled/iter   : %.6f ms" % (tiled_wall * 1000.0 / args.iterations))
    print()
    print("A max abs    :", np.max(np.abs(a_tiled - a_naive)))
    print("A mean abs   :", np.mean(np.abs(a_tiled - a_naive)))
    print("A allclose   :", np.allclose(a_tiled, a_naive, rtol=2e-5, atol=2e-6))
    print()
    print("Naive hot transfers:", naive_ctx.transfer_stats)
    print("Tiled hot transfers:", tiled_ctx.transfer_stats)


if __name__ == "__main__":
    main()
