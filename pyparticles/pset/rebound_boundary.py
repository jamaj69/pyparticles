# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np
import pyparticles.pset.boundary as bd


class ReboundBoundary(bd.Boundary):
    def __init__(self, bound=(-1, 1), dim=3):
        self.set_boundary(bound, dim)
        self.set_normals()

    def set_normals(self):
        self.__N = np.zeros((2 * self.dim, self.dim))

        if self.dim >= 2:
            self.__N[0, :2] = np.array([1, 0])
            self.__N[1, :2] = np.array([-1, 0])
            self.__N[2, :2] = np.array([0, 1])
            self.__N[3, :2] = np.array([0, -1])

        if self.dim == 3:
            self.__N[4, :] = np.array([0, 0, 1])
            self.__N[5, :] = np.array([0, 0, -1])

    def needs_update(self, p_set):
        for i in range(self.dim):
            if np.any(p_set.X[:, i] < self.bound[i, 0]):
                return True
            if np.any(p_set.X[:, i] > self.bound[i, 1]):
                return True
        return False

    def boundary(self, p_set):
        v_mi = np.zeros(3)
        v_mx = np.zeros(3)
        changed = False

        for i in range(self.dim):
            j = 2 * i
            v_mi[:] = 0.0
            v_mx[:] = 0.0

            b_mi = p_set.X[:, i] < self.bound[i, 0]
            b_mx = p_set.X[:, i] > self.bound[i, 1]

            if np.any(b_mi) or np.any(b_mx):
                changed = True

            v_mi[i] = self.bound[i, 0]
            v_mx[i] = self.bound[i, 1]

            p_set.X[b_mi, :] = p_set.X[b_mi, :] + 2.0 * self.__N[j, :] * (
                v_mi - p_set.X[b_mi, :]
            )
            p_set.X[b_mx, :] = p_set.X[b_mx, :] + 2.0 * self.__N[j, :] * (
                v_mx - p_set.X[b_mx, :]
            )

            p_set.V[b_mi, i] = -p_set.V[b_mi, i]
            p_set.V[b_mx, i] = -p_set.V[b_mx, i]

        return changed
