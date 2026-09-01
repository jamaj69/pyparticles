# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import gc

import numpy as np

import pyparticles.pset.particles_set as ps
import pyparticles.pset.opencl_context as occ
import pyparticles.pset.default_boundary as db

import pyparticles.ode.euler_solver as els

import pyparticles.forces.const_force as cf
import pyparticles.forces.drag as dr
import pyparticles.forces.multiple_force as mf
from pyparticles.forces.fused_const_drag import FusedConstDragOCL

import pyparticles.animation.animated_ogl_compat as aogl
from pyparticles.ogl.opencl_gl_vbo import OpenCLGLPositionBuffer

from pyparticles.utils.pypart_global import test_pyopencl

try:
    import pyopencl as cl
except ImportError:
    cl = None


def default_pos(pset, indx):
    """Historical host fallback for systems without CL/GL sharing."""
    t = default_pos.sim_time.time

    pset.X[indx, :] = 0.01 * np.random.rand(len(indx), pset.dim).astype(pset.dtype)

    fs = 1.0 / (1.0 + np.exp(-(t * 4.0 - 2.0)))
    alpha = 2.0 * np.pi * np.random.rand(len(indx)).astype(pset.dtype)

    pset.V[indx, 0] = 2.0 * fs * np.cos(alpha)
    pset.V[indx, 1] = 2.0 * fs * np.sin(alpha)
    pset.V[indx, 2] = 10.0 * fs + fs * np.random.rand(len(indx))


def fountain():
    """Fountain demo with resident/fused OpenCL and optional CL/GL interop."""
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

    bd = (-100.0, 100.0, -100.0, 100.0, 0.0, 100.0)

    if ocl_ok:
        # Reliable fallback: fused/resident compute with host X synchronization.
        occx = occ.OpenCLcontext(
            pset.size,
            pset.dim,
            occ.OCLC_X | occ.OCLC_V | occ.OCLC_A | occ.OCLC_M,
        )
        force = FusedConstDragOCL(
            pset.size,
            dim=pset.dim,
            m=pset.M,
            u_force=(0.0, 0.0, -10.0),
            drag_const=0.01,
            ocl_context=occx,
        )
        solver = els.EulerSolverOCL(
            force,
            pset,
            dt,
            ocl_context=occx,
            sync_velocity=False,
            sync_positions=True,
        )
    else:
        grav = cf.ConstForce(
            pset.size,
            dim=pset.dim,
            u_force=(0.0, 0.0, -10.0),
        )
        drag = dr.Drag(pset.size, dim=pset.dim, Consts=0.01)
        force = mf.MultipleForce(pset.size, dim=pset.dim)
        force.append_force(grav)
        force.append_force(drag)
        force.set_masses(pset.M)
        solver = els.EulerSolver(force, pset, dt)

    default_pos.sim_time = solver.get_sim_time()
    pset.set_boundary(db.DefaultBoundary(bd, dim=3, defualt_pos=default_pos))

    a = aogl.AnimatedGl()
    a.ode_solver = solver
    a.pset = pset
    a.steps = steps
    a.draw_particles.set_draw_model(a.draw_particles.DRAW_MODEL_VECTOR)
    a.init_rotation(-80, [0.7, 0.05, 0])

    gl_interop_possible = (
        ocl_ok
        and cl is not None
        and hasattr(cl, "have_gl")
        and cl.have_gl()
    )

    if gl_interop_possible:
        def enable_cl_gl_interop(animation):
            bridge = None
            try:
                shared_ctx = occ.OpenCLcontext(
                    pset.size,
                    pset.dim,
                    occ.OCLC_X | occ.OCLC_V | occ.OCLC_A | occ.OCLC_M,
                    gl_sharing=True,
                )
                bridge = OpenCLGLPositionBuffer(
                    shared_ctx,
                    pset,
                    animation.draw_particles,
                )

                shared_force = FusedConstDragOCL(
                    pset.size,
                    dim=pset.dim,
                    m=pset.M,
                    u_force=(0.0, 0.0, -10.0),
                    drag_const=0.01,
                    ocl_context=shared_ctx,
                    fountain_bounds=bd,
                )
                shared_solver = els.EulerSolverOCL(
                    shared_force,
                    pset,
                    dt,
                    ocl_context=shared_ctx,
                    sync_velocity=False,
                    sync_positions=False,
                )

                shared_ctx.set_from_host("X", pset.X)
                shared_ctx.set_from_host("V", pset.V)

                # Commit the new path only after all GL/CL resources and
                # kernels have been created successfully.
                animation.ode_solver = shared_solver
                pset.set_boundary(None)  # boundary/reset is fused on the GPU
                animation.set_post_step_callback(
                    lambda _animation: bridge.update_from_device()
                )

                def cleanup_interop(_animation, _bridge=bridge, _fallback=solver):
                    # Drop every reference chain leading to the GL-sharing
                    # context while the GLX context is still current.  On the
                    # NVIDIA driver, leaving shared_solver alive until after
                    # glutLeaveMainLoop() can crash native teardown.
                    _animation.set_post_step_callback(None)
                    _bridge.close()
                    _animation.ode_solver = _fallback
                    gc.collect()

                animation.add_cleanup_callback(cleanup_interop)

                print(
                    "OpenCL/OpenGL interop enabled: "
                    "positions render without host copies"
                )
                print("Interop device:", shared_ctx.device.name)
            except Exception as exc:
                if bridge is not None:
                    try:
                        bridge.close()
                    except Exception:
                        pass
                print(
                    "OpenCL/OpenGL interop unavailable; "
                    "using host-sync renderer: %s" % exc
                )

        a.set_gl_context_ready_callback(enable_cl_gl_interop)

    a.build_animation()
    a.start()
