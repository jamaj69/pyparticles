# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np
import scipy.spatial.distance as dist

import pyparticles.forces.force as fr
import pyparticles.pset.opencl_context as occ

try:
    import pyopencl as cl
except ImportError:
    cl = None


class Gravity(fr.Force):
    """Compute pairwise Newtonian gravitational acceleration."""

    def __init__(self, size, dim=3, m=None, Consts=1.0):
        self.__dim = int(dim)
        self.__size = int(size)
        self.__G = Consts
        self.__A = np.zeros((size, dim))
        self.__Fm = np.zeros((size, size))
        self.__V = np.zeros((size, size))
        self.__D = np.zeros((size, size))
        self.__M = np.zeros((size, size))

        if m is not None:
            self.set_masses(m)

    def set_masses(self, m):
        self.__M[:, :] = m

    def update_force(self, p_set):
        self.__D[:] = dist.squareform(dist.pdist(p_set.X, "euclidean"))

        with np.errstate(divide="ignore", invalid="ignore"):
            self.__Fm[:] = -self.__G * self.__M / self.__D**3.0
        np.fill_diagonal(self.__Fm, 0.0)

        for i in range(self.__dim):
            self.__V[:, :] = p_set.X[:, i]
            self.__V[:, :] = (self.__V.T - p_set.X[:, i]).T
            self.__A[:, i] = np.sum(self.__Fm * self.__V, axis=0)

        return self.__A

    def getA(self):
        return self.__A

    A = property(getA)

    def getF(self):
        return self.__A * self.__M[:, 0:1]

    F = property(getF)


class GravityOCL(fr.Force):
    """OpenCL implementation of the 3-D pairwise gravity kernel."""

    def __init__(self, size, dim=3, m=None, Consts=1.0, ocl_context=None):
        if cl is None:
            raise RuntimeError("PyOpenCL is required for GravityOCL")
        if int(dim) != 3:
            raise ValueError("GravityOCL currently supports only 3 dimensions")

        self.__dim = int(dim)
        self.__size = int(size)

        if ocl_context is None:
            self.__occ = occ.OpenCLcontext(
                size,
                dim,
                occ.OCLC_X | occ.OCLC_A | occ.OCLC_M,
            )
        else:
            self.__occ = ocl_context

        self.__dtype = self.__occ.dtype
        self.__G = self.__dtype(Consts)
        self.__A = np.zeros((size, dim), dtype=self.__dtype)
        self.__M = np.zeros((size, 1), dtype=self.__dtype)

        self.__init_prog_cl()
        if m is not None:
            self.set_masses(m)

    def __init_prog_cl(self):
        source = r"""
        __kernel void gravity(
            __global const float *X,
            __global const float *M,
                           float  G,
                             int  accumulate,
            __global       float *A)
        {
            int i = get_global_id(0);
            int sz = get_global_size(0);

            int i0 = 3*i;
            int i1 = i0 + 1;
            int i2 = i0 + 2;

            float4 at = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
            float4 u  = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

            for (int n = 0; n < sz; ++n)
            {
                if (n == i) continue;

                u.x = X[i0] - X[3*n];
                u.y = X[i1] - X[3*n+1];
                u.z = X[i2] - X[3*n+2];

                float d = length(u);
                if (d == 0.0f) continue;

                float f = -G * M[n] / pown(d, 3);
                at.x += f * u.x;
                at.y += f * u.y;
                at.z += f * u.z;
            }

            if (accumulate)
            {
                A[i0] += at.x;
                A[i1] += at.y;
                A[i2] += at.z;
            }
            else
            {
                A[i0] = at.x;
                A[i1] = at.y;
                A[i2] = at.z;
            }
        }
        """
        self.__cl_program = cl.Program(self.__occ.CL_context, source).build()
        self.__kernel = cl.Kernel(self.__cl_program, "gravity")

    def set_masses(self, m):
        self.__M[:] = np.asarray(m, dtype=self.__dtype)
        self.__occ.set_from_host("M", self.__M)

    def update_force_device(self, p_set, accumulate=False, host_authoritative=False):
        if host_authoritative:
            self.__occ.mark_host_modified("X")
        self.__occ.sync_to_device("X", p_set.X)

        self.__kernel(
            self.__occ.CL_queue,
            (self.__size,),
            None,
            self.__occ.X_cla.data,
            self.__occ.M_cla.data,
            self.__G,
            np.int32(bool(accumulate)),
            self.__occ.A_cla.data,
        )
        self.__occ.mark_device_modified("A")
        return self.__occ.A_cla

    def update_force(self, p_set):
        self.update_force_device(p_set, accumulate=False, host_authoritative=True)
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
