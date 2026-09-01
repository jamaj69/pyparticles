# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np

import pyparticles.forces.force as fr
import pyparticles.pset.opencl_context as occ

try:
    import pyopencl as cl
except ImportError:
    cl = None


class ConstForce(fr.Force):
    """Constant acceleration field."""

    def __init__(self, size, dim=3, m=None, u_force=(0, 0, 0), Consts=1.0):
        self.__dim = int(dim)
        self.__size = int(size)
        self.__G = Consts
        self.__UF = np.asarray(u_force, dtype=float)
        self.__A = np.zeros((size, dim))
        self.__M = np.zeros((size, 1))
        self.__A[:] = self.__UF

        if m is not None:
            self.set_masses(m)

    def set_masses(self, m):
        self.__M[:] = m

    def update_force(self, p_set):
        return self.__A

    def getA(self):
        return self.__A

    A = property(getA)

    def getF(self):
        return self.__A * self.__M

    F = property(getF)


class ConstForceOCL(fr.Force):
    """Constant acceleration field accumulated directly in an OpenCL A buffer."""

    def __init__(self, size, dim=3, m=None, u_force=(0, 0, 0), Consts=1.0, ocl_context=None):
        if cl is None:
            raise RuntimeError("PyOpenCL is required for ConstForceOCL")
        if int(dim) != 3:
            raise ValueError("ConstForceOCL currently supports only 3 dimensions")

        self.__size = int(size)
        self.__dim = int(dim)
        self.__occ = ocl_context or occ.OpenCLcontext(size, dim, occ.OCLC_A | occ.OCLC_M)
        self.__dtype = self.__occ.dtype
        self.__UF = np.asarray(u_force, dtype=self.__dtype).reshape(3)
        self.__A = np.zeros((size, dim), dtype=self.__dtype)
        self.__M = np.zeros((size, 1), dtype=self.__dtype)

        source = r"""
        __kernel void const_force(
            float ax,
            float ay,
            float az,
            int accumulate,
            __global float *A)
        {
            int i = get_global_id(0);
            int i0 = 3*i;
            int i1 = i0 + 1;
            int i2 = i0 + 2;

            if (accumulate)
            {
                A[i0] += ax;
                A[i1] += ay;
                A[i2] += az;
            }
            else
            {
                A[i0] = ax;
                A[i1] = ay;
                A[i2] = az;
            }
        }
        """
        self.__program = cl.Program(self.__occ.CL_context, source).build()
        self.__kernel = cl.Kernel(self.__program, "const_force")

        if m is not None:
            self.set_masses(m)

    def set_masses(self, m):
        self.__M[:] = np.asarray(m, dtype=self.__dtype)
        if self.__occ.M_cla is not None:
            self.__occ.set_from_host("M", self.__M)

    def update_force_device(self, p_set=None, accumulate=False, host_authoritative=False):
        self.__kernel(
            self.__occ.CL_queue,
            (self.__size,),
            None,
            self.__dtype(self.__UF[0]),
            self.__dtype(self.__UF[1]),
            self.__dtype(self.__UF[2]),
            np.int32(bool(accumulate)),
            self.__occ.A_cla.data,
        )
        self.__occ.mark_device_modified("A")
        return self.__occ.A_cla

    def update_force(self, p_set):
        self.update_force_device(p_set, accumulate=False)
        self.__occ.sync_to_host("A", self.__A)
        return self.__A

    def getA(self):
        return self.__A

    A = property(getA)

    def getF(self):
        self.__occ.sync_to_host("A", self.__A)
        return self.__A * self.__M

    F = property(getF)

    def get_ocl_context(self):
        return self.__occ

    ocl_context = property(get_ocl_context)
