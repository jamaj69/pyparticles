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
import scipy.spatial.distance as dist

import pyparticles.forces.force as fr


class LenardJones(fr.Force):
    r"""Compute the Lennard-Jones force between particles."""

    def __init__(self, size, dim=3, m=None, Consts=(1.0, 1.0)):
        self.__dim = dim
        self.__size = size
        self.__E = Consts[0]
        self.__O = Consts[1]

        self.__M = np.zeros((size, 1))
        pair_count = (size * (size - 1)) // 2
        self.__pF = np.zeros(pair_count)
        self.__V = np.zeros((size, size))
        self.__A = np.zeros((size, dim))

        if m is not None:
            self.set_masses(m)

    def set_masses(self, m):
        self.__M[:] = m

    def update_force(self, p_set):
        r = dist.pdist(p_set.X, "euclidean")

        self.__pF[:] = (
            4.0
            * self.__E
            * (
                12.0 * self.__O**12.0 / r**13.0
                - 6.0 * self.__O**6.0 / r**7.0
            )
            / r
        )

        F = dist.squareform(self.__pF)

        for i in range(p_set.dim):
            self.__V[:, :] = p_set.X[:, i]
            self.__V[:, :] = (self.__V[:, :].T - p_set.X[:, i]).T
            self.__A[:, i] = np.sum(F * self.__V[:, :] / self.__M.T, axis=0)

        return self.__A

    def getA(self):
        return self.__A

    A = property(getA, doc="Return the current accelerations")

    def getF(self):
        return self.__A * self.__M

    F = property(getF, doc="Return the current forces")
