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

    ``fountain_bounds`` optionally enables the fountain's reset boundary in
    the same kernel launch.  The reset follows the original demo's probability
    distributions but uses a small deterministic device-side hash PRNG, so no
    boundary detection or random state needs to cross PCIe.
    """

    def __init__(
        self,
        size,
        dim=3,
        m=None,
        u_force=(0.0, 0.0, 0.0),
        drag_const=1.0,
        ocl_context=None,
        fountain_bounds=None,
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
        self.__fountain_bounds = None
        self.__last_step_event = None
        if fountain_bounds is not None:
            bounds = np.asarray(fountain_bounds, dtype=self.__dtype).reshape(-1)
            if bounds.size != 6:
                raise ValueError("fountain_bounds must contain 6 values")
            self.__fountain_bounds = bounds

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

        inline uint hash_u32(uint x)
        {
            x ^= x >> 16;
            x *= 0x7feb352du;
            x ^= x >> 15;
            x *= 0x846ca68bu;
            x ^= x >> 16;
            return x;
        }

        inline float random01(uint x)
        {
            return (float)(hash_u32(x) & 0x00ffffffu) * (1.0f / 16777216.0f);
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

        __kernel void const_drag_euler_fountain(
            __global       float *V,
            __global const float *M,
                           float  cax,
                           float  cay,
                           float  caz,
                           float  K,
                           float  dt,
                           float  sim_time,
                            uint  step,
                           float  xmin,
                           float  xmax,
                           float  ymin,
                           float  ymax,
                           float  zmin,
                           float  zmax,
            __global       float *X)
        {
            int i = get_global_id(0);
            int i0 = 3*i;

            float3 x = (float3)(X[i0], X[i0+1], X[i0+2]);
            float3 v = (float3)(V[i0], V[i0+1], V[i0+2]);
            float3 ca = (float3)(cax, cay, caz);
            float3 a = const_drag_accel(v, M[i], ca, K);

            v += a * dt;
            x += v * dt;

            int outside = (
                x.x < xmin || x.x > xmax ||
                x.y < ymin || x.y > ymax ||
                x.z < zmin || x.z > zmax
            );

            if (outside)
            {
                uint seed = ((uint)i + 1u) * 747796405u ^ (step + 1u) * 2891336453u;
                float rx = random01(seed ^ 0x68bc21ebu);
                float ry = random01(seed ^ 0x02e5be93u);
                float rz = random01(seed ^ 0x967a889bu);
                float ra = random01(seed ^ 0x4f1bbcdcu);
                float rv = random01(seed ^ 0x85ebca6bu);

                x = (float3)(0.01f*rx, 0.01f*ry, 0.01f*rz);

                float fs = 1.0f / (1.0f + exp(-(sim_time*4.0f - 2.0f)));
                float alpha = 6.2831853071795864769f * ra;
                v.x = 2.0f * fs * cos(alpha);
                v.y = 2.0f * fs * sin(alpha);
                v.z = 10.0f * fs + fs * rv;
            }

            V[i0]   = v.x;
            V[i0+1] = v.y;
            V[i0+2] = v.z;
            X[i0]   = x.x;
            X[i0+1] = x.y;
            X[i0+2] = x.z;
        }
        """
        self.__program = cl.Program(self.__occ.CL_context, source).build()
        self.__force_kernel = cl.Kernel(self.__program, "const_drag_force")
        self.__euler_kernel = cl.Kernel(self.__program, "const_drag_euler")
        self.__fountain_kernel = cl.Kernel(
            self.__program, "const_drag_euler_fountain"
        )

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

    def euler_step_device(self, pset, dt, sim_time=None, step=None):
        """Evaluate force and advance X/V in one device kernel launch."""
        cax, cay, caz, drag_const = self._common_args()

        if self.__fountain_bounds is None:
            self.__last_step_event = self.__euler_kernel(
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
        else:
            b = self.__fountain_bounds
            self.__last_step_event = self.__fountain_kernel(
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
                self.__dtype(0.0 if sim_time is None else sim_time),
                np.uint32(0 if step is None else step),
                self.__dtype(b[0]),
                self.__dtype(b[1]),
                self.__dtype(b[2]),
                self.__dtype(b[3]),
                self.__dtype(b[4]),
                self.__dtype(b[5]),
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

    def get_fountain_bounds(self):
        if self.__fountain_bounds is None:
            return None
        return tuple(float(value) for value in self.__fountain_bounds)

    fountain_bounds = property(get_fountain_bounds)

    def get_last_step_event(self):
        """Return the most recently enqueued fused Euler OpenCL event."""
        return self.__last_step_event

    last_step_event = property(get_last_step_event)
