# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np


class MultipleForce(object):
    """Combine several host force models into one acceleration field."""

    def __init__(self, size, dim=3, m=None, Conts=None):
        self.__forces = []
        self.__M = np.zeros((size, 1))
        self.__A = np.zeros((size, dim))
        self.__F = np.zeros((size, dim))

        if m is not None:
            self.set_masses(m)

    def append_force(self, force):
        self.__forces.append(force)

    def set_masses(self, m):
        self.__M[:] = m
        for force in self.__forces:
            force.set_masses(m)

    def update_force(self, p_set):
        self.__A[:] = 0.0
        for force in self.__forces:
            self.__A[:] += force.update_force(p_set)

        self.__F[:] = self.__A * self.__M
        return self.__A

    def getA(self):
        return self.__A

    A = property(getA)

    def getF(self):
        return self.__F

    F = property(getF)


class MultipleForceOCL(object):
    """Compose OpenCL forces directly in the shared device acceleration buffer.

    Every appended force must expose ``update_force_device`` and share the
    exact same :class:`OpenCLcontext`.  The first force writes A and subsequent
    forces accumulate into it, so no intermediate acceleration array crosses
    PCIe.
    """

    def __init__(self, size, dim=3, m=None, ocl_context=None):
        if ocl_context is None:
            raise ValueError("MultipleForceOCL requires a shared OpenCL context")

        self.__size = int(size)
        self.__dim = int(dim)
        self.__occ = ocl_context
        self.__dtype = self.__occ.dtype
        self.__forces = []
        self.__M = np.zeros((size, 1), dtype=self.__dtype)
        self.__A = np.zeros((size, dim), dtype=self.__dtype)
        self.__F = np.zeros((size, dim), dtype=self.__dtype)

        if m is not None:
            self.set_masses(m)

    def append_force(self, force):
        if not hasattr(force, "update_force_device"):
            raise TypeError("MultipleForceOCL accepts only device-capable forces")
        if getattr(force, "ocl_context", None) is not self.__occ:
            raise ValueError("All MultipleForceOCL forces must share one OpenCL context")
        self.__forces.append(force)

    def set_masses(self, m):
        self.__M[:] = np.asarray(m, dtype=self.__dtype)
        self.__occ.set_from_host("M", self.__M)

        # Keep each component's host-side force property coherent. These are
        # one-time setup transfers and do not affect the integration hot path.
        for force in self.__forces:
            force.set_masses(self.__M)

    def update_force_device(self, p_set, accumulate=False, host_authoritative=False):
        if not self.__forces:
            self.__occ.A_cla.fill(self.__dtype(0.0), queue=self.__occ.CL_queue)
            self.__occ.mark_device_modified("A")
            return self.__occ.A_cla

        # A composed force is normally the complete force model.  If callers
        # request accumulation into an existing A, every component accumulates;
        # otherwise the first component initializes A and the rest add to it.
        first_accumulate = bool(accumulate)
        for index, force in enumerate(self.__forces):
            force.update_force_device(
                p_set,
                accumulate=(first_accumulate or index > 0),
                host_authoritative=host_authoritative,
            )
        self.__occ.mark_device_modified("A")
        return self.__occ.A_cla

    def update_force(self, p_set):
        # Preserve the Force-like host API for callers outside a device solver.
        self.update_force_device(p_set, host_authoritative=True)
        self.__occ.sync_to_host("A", self.__A)
        self.__F[:] = self.__A * self.__M
        return self.__A

    def getA(self):
        self.__occ.sync_to_host("A", self.__A)
        return self.__A

    A = property(getA)

    def getF(self):
        self.__occ.sync_to_host("A", self.__A)
        self.__F[:] = self.__A * self.__M
        return self.__F

    F = property(getF)

    def get_ocl_context(self):
        return self.__occ

    ocl_context = property(get_ocl_context)
