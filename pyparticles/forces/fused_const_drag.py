# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# Modern fused OpenCL force/integration helper.

import numpy as np

import pyparticles.forces.force as fr
import pyparticles.pset.opencl_context as occ

try:
    import pyopencl as cl
except ImportError:
    cl = None


class FusedConstDragOCL(fr.Force):
    """Constant acceleration plus quadratic drag with a fused Euler step.

    The normal Force API is retained through ``update_force`` and
    ``update_force_device``.  ``EulerSolverOCL`` may additionally call
    ``euler_step_device`` to evaluate both forces and integrate X/V in one
    OpenCL kernel launch.
    """

    def __init__(
        self,
        size,
        dim=3,
        m=None,
        u_force=(0.0, 0.0, 0.0),
        drag_const=1.0,
        ocl_context=None,
    ):
        if cl is None:
            raise RuntimeError("PyOpenCL is required for FusedConstDragOCL")
        if int(dim) != 3:
            raise ValueError("FusedConstDragOCL currently supports only 3 dimensions")

        self.__size = int(size)
        self.__dim = int(dim)
        self.__occ = ocl_context or occ.OpenCLcontext(
            size,
            dim,
            occ.OCLC_X | occ.OCLC_V | occ.OCLC_A | occ.OCLC_M,
        )
        self.__dtype = self.__occ.dtype
        self.__UF = np.asarray(u_force, dtype=self.__dtype).reshape(3)
        self.__K = self.__dtype(drag_const)
        self.__A = np.zeros((size, dim), dtype=self.__dtype)
        self.__M = np.zeros((size, 1), dtype=self.__dtype)

        self.__build_program()
        if m is not None:
            self.set_masses(m)

    def __build_program(self):
        source = r"""
        inline float3 const_drag_accel(
            float3 v,
            float mass,
            float3 constant_a,
            float K)
        {
            float speed = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
            float scale = (-0.5f * K * speed) / mass;
            return constant_a + scale * v;
        }

        __kernel void const_drag_force(
            __global const float *V,
            __global const float *M,
                           float  cax,
                           float  cay,
                           float  caz,
                           float  K,
                             int  accumulate,
            __global       float *A)
        {
            int i = get_global_id(0);
            int i0 = 3*i;
            float3 v = (float3)(V[i0], V[i0+1], V[i0+2]);
            float3 ca = (float3)(cax, cay, caz);
            float3 a = const_drag_accel(v, M[i], ca, K);

            if (accumulate)
            {
                A[i0]   += a.x;
                A[i0+1] += a.y;
                A[i0+2] += a.z;
            }
            else
            {
                A[i0]   = a.x;
                A[i0+1] = a.y;
                A[i0+2] = a.z;
            }
        }

        __kernel void const_drag_euler(
            __global       float *V,
            __global const float *M,
                           float  cax,
                           float  cay,
                           float  caz,
                           float  K,
                           float  dt,
            __global       float *X)
        {
            int i = get_global_id(0);
            int i0 = 3*i;

            float3 v = (float3)(V[i0], V[i0+1], V[i0+2]);
            float3 ca = (float3)(cax, cay, caz);
            float3 a = const_drag_accel(v, M[i], ca, K);

            v += a * dt;
            V[i0]   = v.x;
            V[i0+1] = v.y;
            V[i0+2] = v.z;

            X[i0]   += v.x * dt;
            X[i0+1] += v.y * dt;
            X[i0+2] += v.z * dt;
        }
        """
        self.__program = cl.Program(self.__occ.CL_context, source).build()
        self.__force_kernel = cl.Kernel(self.__program, "const_drag_force")
        self.__euler_kernel = cl.Kernel(self.__program, "const_drag_euler")

    def set_masses(self, m):
        self.__M[:] = np.asarray(m, dtype=self.__dtype)
        self.__occ.set_from_host("M", self.__M)

    def _common_args(self):
        return (
            self.__dtype(self.__UF[0]),
            self.__dtype(self.__UF[1]),
            self.__dtype(self.__UF[2]),
            self.__K,
        )

    def update_force_device(self, pset, accumulate=False, host_authoritative=False):
        if host_authoritative:
            self.__occ.mark_host_modified("V")
        self.__occ.sync_to_device("V", pset.V)

        cax, cay, caz, drag_const = self._common_args()
        self.__force_kernel(
            self.__occ.CL_queue,
            (self.__size,),
            None,
            self.__occ.V_cla.data,
            self.__occ.M_cla.data,
            cax,
            cay,
            caz,
            drag_const,
            np.int32(bool(accumulate)),
            self.__occ.A_cla.data,
        )
        self.__occ.mark_device_modified("A")
        return self.__occ.A_cla

    def euler_step_device(self, pset, dt):
        """Evaluate force and advance X/V in one device kernel launch."""
        cax, cay, caz, drag_const = self._common_args()
        self.__euler_kernel(
            self.__occ.CL_queue,
            (self.__size,),
            None,
            self.__occ.V_cla.data,
            self.__occ.M_cla.data,
            cax,
            cay,
            caz,
            drag_const,
            self.__dtype(dt),
            self.__occ.X_cla.data,
        )
        self.__occ.mark_device_modified("X")
        self.__occ.mark_device_modified("V")
        return self.__occ.X_cla, self.__occ.V_cla

    def update_force(self, pset):
        self.update_force_device(pset, accumulate=False, host_authoritative=True)
        self.__occ.sync_to_host("A", self.__A)
        return self.__A

    def getA(self):
        self.__occ.sync_to_host("A", self.__A)
        return self.__A

    A = property(getA)

    def getF(self):
        self.__occ.sync_to_host("A", self.__A)
        return self.__A * self.__M

    F = property(getF)

    def get_ocl_context(self):
        return self.__occ

    ocl_context = property(get_ocl_context)
