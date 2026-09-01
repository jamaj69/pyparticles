# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np

import pyparticles.pset.particles_set as ps
import pyparticles.pset.opencl_context as occ
import pyparticles.pset.default_boundary as db

import pyparticles.ode.euler_solver as els

import pyparticles.forces.const_force as cf
import pyparticles.forces.drag as dr
import pyparticles.forces.multiple_force as mf

import pyparticles.animation.animated_ogl_compat as aogl

from pyparticles.utils.pypart_global import test_pyopencl


def default_pos(pset, indx):
    t = default_pos.sim_time.time

    pset.X[indx, :] = 0.01 * np.random.rand(len(indx), pset.dim).astype(pset.dtype)

    fs = 1.0 / (1.0 + np.exp(-(t * 4.0 - 2.0)))
    alpha = 2.0 * np.pi * np.random.rand(len(indx)).astype(pset.dtype)

    pset.V[indx, 0] = 2.0 * fs * np.cos(alpha)
    pset.V[indx, 1] = 2.0 * fs * np.sin(alpha)
    pset.V[indx, 2] = 10.0 * fs + fs * np.random.rand(len(indx))


def fountain():
    """Fountain demo."""
    steps = 10000000
    dt = 0.005
    pcnt = 100000
    ocl_ok = test_pyopencl()

    if ocl_ok:
        print("OpenCL is installed and enabled")
        print("Try, at least, 200000 particles")

        while True:
            try:
                print("")
                pcnt = int(input("How many particles: "))
            except ValueError:
                print("Please insert a number!")
            else:
                break

    pset = ps.ParticlesSet(pcnt, dtype=np.float32)
    pset.M[:] = 0.1
    pset.X[:, 2] = 0.7 * np.random.rand(pset.size)

    if ocl_ok:
        occx = occ.OpenCLcontext(
            pset.size,
            pset.dim,
            occ.OCLC_X | occ.OCLC_V | occ.OCLC_A | occ.OCLC_M,
        )
        grav = cf.ConstForceOCL(
            pset.size,
            dim=pset.dim,
            u_force=(0.0, 0.0, -10.0),
            ocl_context=occx,
        )
        drag = dr.DragOCL(
            pset.size,
            dim=pset.dim,
            Consts=0.01,
            ocl_context=occx,
        )
        multi = mf.MultipleForceOCL(
            pset.size,
            dim=pset.dim,
            ocl_context=occx,
        )
    else:
        occx = None
        grav = cf.ConstForce(
            pset.size,
            dim=pset.dim,
            u_force=(0.0, 0.0, -10.0),
        )
        drag = dr.Drag(pset.size, dim=pset.dim, Consts=0.01)
        multi = mf.MultipleForce(pset.size, dim=pset.dim)

    multi.append_force(grav)
    multi.append_force(drag)
    multi.set_masses(pset.M)

    if ocl_ok:
        solver = els.EulerSolverOCL(multi, pset, dt, ocl_context=occx)
    else:
        solver = els.EulerSolver(multi, pset, dt)

    default_pos.sim_time = solver.get_sim_time()

    bd = (-100.0, 100.0, -100.0, 100.0, 0.0, 100.0)
    pset.set_boundary(db.DefaultBoundary(bd, dim=3, defualt_pos=default_pos))

    a = aogl.AnimatedGl()
    a.ode_solver = solver
    a.pset = pset
    a.steps = steps
    a.draw_particles.set_draw_model(a.draw_particles.DRAW_MODEL_VECTOR)
    a.init_rotation(-80, [0.7, 0.05, 0])
    a.build_animation()
    a.start()
