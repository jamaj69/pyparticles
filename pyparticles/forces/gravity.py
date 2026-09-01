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
    """OpenCL implementation of 3-D pairwise gravity.

    ``kernel_mode='tiled'`` is the optimized default. Each work-group loads a
    tile of positions and masses into OpenCL local memory and reuses it for all
    target particles in the group. ``kernel_mode='naive'`` preserves the
    original global-memory algorithm for regression and benchmarking.
    """

    def __init__(
        self,
        size,
        dim=3,
        m=None,
        Consts=1.0,
        ocl_context=None,
        kernel_mode="tiled",
        tile_size=128,
    ):
        if cl is None:
            raise RuntimeError("PyOpenCL is required for GravityOCL")
        if int(dim) != 3:
            raise ValueError("GravityOCL currently supports only 3 dimensions")
        if kernel_mode not in ("tiled", "naive"):
            raise ValueError("kernel_mode must be 'tiled' or 'naive'")

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
        self.__kernel_mode = kernel_mode
        self.__tile_size = self._choose_tile_size(tile_size)

        self.__init_prog_cl()
        if m is not None:
            self.set_masses(m)

    def _choose_tile_size(self, requested):
        requested = max(1, int(requested))
        device = self.__occ.CL_queue.device
        max_wg = int(device.max_work_group_size)

        # One float4 position plus one float mass per work-item.
        max_by_local_mem = max(1, int(device.local_mem_size) // 20)
        limit = min(requested, max_wg, max_by_local_mem)

        # Prefer a power of two no larger than the requested/device limit.
        tile = 1
        while tile * 2 <= limit:
            tile *= 2
        return tile

    def __init_prog_cl(self):
        source = r"""
        inline void store_acceleration(
            int i,
            int accumulate,
            float ax,
            float ay,
            float az,
            __global float *A)
        {
            int i0 = 3*i;
            if (accumulate)
            {
                A[i0]   += ax;
                A[i0+1] += ay;
                A[i0+2] += az;
            }
            else
            {
                A[i0]   = ax;
                A[i0+1] = ay;
                A[i0+2] = az;
            }
        }

        __kernel void gravity_naive(
            __global const float *X,
            __global const float *M,
                           float  G,
                             int  count,
                             int  accumulate,
            __global       float *A)
        {
            int i = get_global_id(0);
            if (i >= count) return;

            int i0 = 3*i;
            float xi = X[i0];
            float yi = X[i0+1];
            float zi = X[i0+2];
            float ax = 0.0f;
            float ay = 0.0f;
            float az = 0.0f;

            for (int n = 0; n < count; ++n)
            {
                if (n == i) continue;

                float dx = xi - X[3*n];
                float dy = yi - X[3*n+1];
                float dz = zi - X[3*n+2];
                float r2 = dx*dx + dy*dy + dz*dz;
                if (r2 == 0.0f) continue;

                float inv_r = rsqrt(r2);
                float inv_r3 = inv_r * inv_r * inv_r;
                float f = -G * M[n] * inv_r3;
                ax += f * dx;
                ay += f * dy;
                az += f * dz;
            }

            store_acceleration(i, accumulate, ax, ay, az, A);
        }

        __kernel void gravity_tiled(
            __global const float *X,
            __global const float *M,
                           float  G,
                             int  count,
                             int  accumulate,
            __local        float4 *tile_pos,
            __local        float  *tile_mass,
            __global       float *A)
        {
            int i = get_global_id(0);
            int lid = get_local_id(0);
            int lsize = get_local_size(0);

            float xi = 0.0f;
            float yi = 0.0f;
            float zi = 0.0f;
            if (i < count)
            {
                int i0 = 3*i;
                xi = X[i0];
                yi = X[i0+1];
                zi = X[i0+2];
            }

            float ax = 0.0f;
            float ay = 0.0f;
            float az = 0.0f;

            for (int base = 0; base < count; base += lsize)
            {
                int j = base + lid;
                if (j < count)
                {
                    int j0 = 3*j;
                    tile_pos[lid] = (float4)(X[j0], X[j0+1], X[j0+2], 0.0f);
                    tile_mass[lid] = M[j];
                }
                else
                {
                    tile_pos[lid] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
                    tile_mass[lid] = 0.0f;
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                if (i < count)
                {
                    int tile_count = min(lsize, count - base);
                    for (int k = 0; k < tile_count; ++k)
                    {
                        int source_index = base + k;
                        if (source_index == i) continue;

                        float4 p = tile_pos[k];
                        float dx = xi - p.x;
                        float dy = yi - p.y;
                        float dz = zi - p.z;
                        float r2 = dx*dx + dy*dy + dz*dz;
                        if (r2 == 0.0f) continue;

                        float inv_r = rsqrt(r2);
                        float inv_r3 = inv_r * inv_r * inv_r;
                        float f = -G * tile_mass[k] * inv_r3;
                        ax += f * dx;
                        ay += f * dy;
                        az += f * dz;
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (i < count)
                store_acceleration(i, accumulate, ax, ay, az, A);
        }
        """
        self.__cl_program = cl.Program(self.__occ.CL_context, source).build()
        self.__naive_kernel = cl.Kernel(self.__cl_program, "gravity_naive")
        self.__tiled_kernel = cl.Kernel(self.__cl_program, "gravity_tiled")

    def set_masses(self, m):
        self.__M[:] = np.asarray(m, dtype=self.__dtype)
        self.__occ.set_from_host("M", self.__M)

    def update_force_device(self, p_set, accumulate=False, host_authoritative=False):
        if host_authoritative:
            self.__occ.mark_host_modified("X")
        self.__occ.sync_to_device("X", p_set.X)

        if self.__kernel_mode == "naive":
            self.__naive_kernel(
                self.__occ.CL_queue,
                (self.__size,),
                None,
                self.__occ.X_cla.data,
                self.__occ.M_cla.data,
                self.__G,
                np.int32(self.__size),
                np.int32(bool(accumulate)),
                self.__occ.A_cla.data,
            )
        else:
            tile = self.__tile_size
            global_size = ((self.__size + tile - 1) // tile) * tile
            self.__tiled_kernel(
                self.__occ.CL_queue,
                (global_size,),
                (tile,),
                self.__occ.X_cla.data,
                self.__occ.M_cla.data,
                self.__G,
                np.int32(self.__size),
                np.int32(bool(accumulate)),
                cl.LocalMemory(tile * 16),
                cl.LocalMemory(tile * 4),
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

    def get_kernel_mode(self):
        return self.__kernel_mode

    kernel_mode = property(get_kernel_mode)

    def get_tile_size(self):
        return self.__tile_size

    tile_size = property(get_tile_size)
