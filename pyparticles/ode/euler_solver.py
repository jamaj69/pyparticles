# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np

import pyparticles.ode.ode_solver as os
import pyparticles.pset.opencl_context as occ

try:
    import pyopencl as cl
except ImportError:
    cl = None


class EulerSolver(os.OdeSolver):
    def __init__(self, force, p_set, dt):
        super(EulerSolver, self).__init__(force, p_set, dt)

    def __step__(self, dt):
        self.force.update_force(self.pset)
        self.pset.V[:] = self.pset.V + self.force.A * dt
        self.pset.X[:] = self.pset.X + self.pset.V * dt
        self.pset.update_boundary()


class EulerSolverOCL(os.OdeSolver):
    """Euler integrator with persistent OpenCL X/V/A buffers.

    ``sync_velocity`` defaults to True to preserve the historical API where
    ``pset.V`` is current immediately after every step.  Rendering-only demos
    that never inspect host velocities may disable it and keep V exclusively
    on the device between steps.
    """

    def __init__(self, force, p_set, dt, ocl_context=None, sync_velocity=True):
        if cl is None:
            raise RuntimeError("PyOpenCL is required for EulerSolverOCL")
        if p_set.dim != 3:
            raise ValueError("EulerSolverOCL currently supports only 3 dimensions")

        super(EulerSolverOCL, self).__init__(force, p_set, dt)

        if ocl_context is None:
            self.__occ = occ.OpenCLcontext(
                self.pset.size,
                self.pset.dim,
                occ.OCLC_X | occ.OCLC_V | occ.OCLC_A,
            )
        else:
            self.__occ = ocl_context

        self.__sync_velocity = bool(sync_velocity)
        self.__init_prog_cl()

    def __init_prog_cl(self):
        source = r"""
        __kernel void euler(
            __global       float *V,
            __global const float *A,
            __global       float *X,
                           float  dt)
        {
            int i = get_global_id(0);
            int i0 = 3*i;
            int i1 = i0 + 1;
            int i2 = i0 + 2;

            V[i0] = V[i0] + A[i0]*dt;
            V[i1] = V[i1] + A[i1]*dt;
            V[i2] = V[i2] + A[i2]*dt;

            X[i0] = X[i0] + V[i0]*dt;
            X[i1] = X[i1] + V[i1]*dt;
            X[i2] = X[i2] + V[i2]*dt;
        }
        """
        self.__cl_program = cl.Program(self.__occ.CL_context, source).build()
        self.__euler_kernel = cl.Kernel(self.__cl_program, "euler")

    def _has_device_force(self):
        return (
            hasattr(self.force, "update_force_device")
            and getattr(self.force, "ocl_context", None) is self.__occ
        )

    def __step__(self, dt):
        if self._has_device_force():
            # Device forces consume the resident X/V buffers directly and
            # leave the complete acceleration in the shared A buffer.
            self.force.update_force_device(self.pset)
        else:
            # A CPU force must see current host state. This path preserves
            # compatibility with arbitrary force implementations.
            self.__occ.sync_to_host("X", self.pset.X)
            self.__occ.sync_to_host("V", self.pset.V)
            self.force.update_force(self.pset)
            self.__occ.set_from_host("A", self.force.A)

        self.__occ.sync_to_device("X", self.pset.X)
        self.__occ.sync_to_device("V", self.pset.V)

        self.__euler_kernel(
            self.__occ.CL_queue,
            (self.pset.size,),
            None,
            self.__occ.V_cla.data,
            self.__occ.A_cla.data,
            self.__occ.X_cla.data,
            np.float32(dt),
        )
        self.__occ.mark_device_modified("X")
        self.__occ.mark_device_modified("V")

        # Positions are required by the current OpenGL renderer each frame.
        self.__occ.sync_to_host("X", self.pset.X)

        boundary = self.pset.boundary
        need_host_velocity = (
            self.__sync_velocity
            or boundary is not None
            or self.pset.log_V_enabled
        )
        if need_host_velocity:
            self.__occ.sync_to_host("V", self.pset.V)

        if boundary is not None:
            changed = boundary.boundary(self.pset)
            # Old/custom boundaries may not return a flag. Be conservative in
            # that case and assume they changed host X/V.
            changed = True if changed is None else bool(changed)
            if changed:
                self.__occ.mark_host_modified("X")
                self.__occ.mark_host_modified("V")

    def sync_to_host(self, velocity=True):
        """Explicitly synchronize resident state for external host consumers."""
        self.__occ.sync_to_host("X", self.pset.X)
        if velocity:
            self.__occ.sync_to_host("V", self.pset.V)
        return self.pset

    def notify_host_modified(self, positions=True, velocities=True):
        """Tell the solver that external code changed host X and/or V."""
        if positions:
            self.__occ.mark_host_modified("X")
        if velocities:
            self.__occ.mark_host_modified("V")

    def get_ocl_context(self):
        return self.__occ

    ocl_context = property(get_ocl_context)

    def get_sync_velocity(self):
        return self.__sync_velocity

    def set_sync_velocity(self, value):
        self.__sync_velocity = bool(value)

    sync_velocity = property(get_sync_velocity, set_sync_velocity)
