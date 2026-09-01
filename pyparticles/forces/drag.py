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


class Drag(fr.Force):
    """Quadratic drag force."""

    def __init__(self, size, dim=3, m=None, Consts=1.0):
        self.__dim = int(dim)
        self.__size = int(size)
        self.__G = np.zeros((size, 1))
        self.__G[:] = Consts
        self.__A = np.zeros((size, dim))
        self.__F = np.zeros((size, dim))
        self.__V = np.zeros((size, 1))
        self.__M = np.zeros((size, 1))

        if m is not None:
            self.set_masses(m)

    def set_masses(self, m):
        self.__M[:] = m

    def update_force(self, pset):
        self.__V[:, 0] = np.sqrt(np.sum(pset.V**2, axis=1))
        self.__F[:] = -0.5 * self.__V * pset.V * self.__G
        self.__A[:] = self.__F / self.__M
        return self.__A

    def getA(self):
        return self.__A

    A = property(getA)

    def getF(self):
        return self.__A * self.__M

    F = property(getF)


class DragOCL(fr.Force):
    """OpenCL implementation of 3-D quadratic drag."""

    def __init__(self, size, dim=3, m=None, Consts=1.0, ocl_context=None):
        if cl is None:
            raise RuntimeError("PyOpenCL is required for DragOCL")
        if int(dim) != 3:
            raise ValueError("DragOCL currently supports only 3 dimensions")

        self.__dim = int(dim)
        self.__size = int(size)

        if ocl_context is None:
            self.__occ = occ.OpenCLcontext(
                size,
                dim,
                occ.OCLC_V | occ.OCLC_A | occ.OCLC_M,
            )
        else:
            self.__occ = ocl_context

        self.__dtype = self.__occ.dtype
        self.__K = self.__dtype(Consts)
        self.__A = np.zeros((size, dim), dtype=self.__dtype)
        self.__M = np.zeros((size, 1), dtype=self.__dtype)

        self.__init_prog_cl()
        if m is not None:
            self.set_masses(m)

    def __init_prog_cl(self):
        source = r"""
        __kernel void drag(
            __global const float *V,
            __global const float *M,
                           float  K,
            __global       float *A)
        {
            int i = get_global_id(0);
            int i0 = 3*i;
            int i1 = i0 + 1;
            int i2 = i0 + 2;

            float speed = sqrt(
                V[i0]*V[i0] + V[i1]*V[i1] + V[i2]*V[i2]
            );

            A[i0] = (-0.5f * K * speed * V[i0]) / M[i];
            A[i1] = (-0.5f * K * speed * V[i1]) / M[i];
            A[i2] = (-0.5f * K * speed * V[i2]) / M[i];
        }
        """
        self.__cl_program = cl.Program(self.__occ.CL_context, source).build()
        self.__drag_kernel = cl.Kernel(self.__cl_program, "drag")

    def set_masses(self, m):
        self.__M[:] = np.asarray(m, dtype=self.__dtype)
        self.__occ.M_cla.set(self.__M, queue=self.__occ.CL_queue)

    def update_force(self, pset):
        self.__occ.V_cla.set(
            np.asarray(pset.V, dtype=self.__dtype),
            queue=self.__occ.CL_queue,
        )

        self.__drag_kernel(
            self.__occ.CL_queue,
            (self.__size,),
            None,
            self.__occ.V_cla.data,
            self.__occ.M_cla.data,
            self.__K,
            self.__occ.A_cla.data,
        )

        self.__occ.A_cla.get(self.__occ.CL_queue, self.__A)
        return self.__A

    def getA(self):
        return self.__A

    A = property(getA)

    def getF(self):
        return self.__A * self.__M

    F = property(getF)
