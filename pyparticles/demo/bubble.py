# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

import pyparticles.pset.particles_set as ps
import pyparticles.pset.opencl_context as occ
import pyparticles.pset.rand_cluster as rc
import pyparticles.pset.rebound_boundary as rb

import pyparticles.forces.pseudo_bubble as pb
import pyparticles.forces.const_force as cf
import pyparticles.forces.drag as dr
import pyparticles.forces.multiple_force as mf

import pyparticles.ode.stormer_verlet_solver as svs

import pyparticles.animation.animated_ogl as aogl

from pyparticles.utils.pypart_global import test_pyopencl


def bubble():
    """Pseudo bubble simulation."""
    ocl_ok = test_pyopencl()

    if ocl_ok:
        pcnt = 9000
        r_min = 0.5
    else:
        pcnt = 700
        r_min = 1.5

    steps = 1000000
    dt = 0.01

    pset = ps.ParticlesSet(pcnt, dtype=np.float32)
    rand_c = rc.RandCluster()
    rand_c.insert3(
        X=pset.X,
        M=pset.M,
        start_indx=0,
        n=pset.size,
        radius=3.0,
        mass_rng=(0.5, 0.8),
        r_min=0.0,
    )

    if ocl_ok:
        occx = occ.OpenCLcontext(
            pset.size,
            pset.dim,
            occ.OCLC_X | occ.OCLC_V | occ.OCLC_A | occ.OCLC_M,
        )
        bubble_force = pb.PseudoBubbleOCL(
            pset.size,
            pset.dim,
            Consts=(r_min, 10),
            ocl_context=occx,
        )
        drag = dr.DragOCL(
            pset.size,
            pset.dim,
            Consts=0.01,
            ocl_context=occx,
        )
    else:
        bubble_force = pb.PseudoBubble(pset.size, pset.dim, Consts=(r_min, 10))
        drag = dr.Drag(pset.size, pset.dim, Consts=0.01)

    constf = cf.ConstForce(pset.size, dim=pset.dim, u_force=[0, 0, -10.0])

    multif = mf.MultipleForce(pset.size, pset.dim)
    multif.append_force(bubble_force)
    multif.append_force(constf)
    multif.append_force(drag)
    multif.set_masses(pset.M)

    solver = svs.StormerVerletSolver(multif, pset, dt)

    pset.set_boundary(rb.ReboundBoundary(bound=(-5.0, 5.0)))

    a = aogl.AnimatedGl()
    a.ode_solver = solver
    a.pset = pset
    a.steps = steps

    if ocl_ok:
        a.draw_particles.set_draw_model(a.draw_particles.DRAW_MODEL_VECTOR)

    a.init_rotation(-80, [0.7, 0.05, 0])
    a.build_animation()
    a.start()
