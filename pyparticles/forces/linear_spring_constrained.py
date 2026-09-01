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
from scipy.sparse import dok_matrix

import pyparticles.forces.force_constrained as fcr


class LinearSpringConstrained(fcr.ForceConstrained):
    def __init__(self, size, dim, m=None, Consts=1.0, f_inter=None):
        super(LinearSpringConstrained, self).__init__(
            size, dim, m, Consts, f_inter=f_inter
        )

        self.__dim = dim
        self.__size = size
        self.__K = Consts

        self.__A = np.zeros((size, dim))
        self.__F = np.zeros((size, dim))
        self.__Fm = dok_matrix((size, size), dtype=float)
        self.__M = np.zeros((size, 1))

        if m is not None:
            self.set_masses(m)

    def set_masses(self, m):
        self.__M[:] = m

    def update_force(self, pset):
        connections = list(self.force_interactions.sparse.keys())

        for i in range(self.__dim):
            self.__Fm.clear()
            for k0, k1 in connections:
                self.__Fm[k0, k1] = pset.X[k1, i]
                self.__Fm[k1, k0] = pset.X[k0, i]

            force_matrix = -self.__K * (self.__Fm.T - self.__Fm).T
            self.__F[:, i] = np.asarray(force_matrix.sum(axis=0)).ravel()

        self.__A[:] = self.__F / self.__M
        return self.__A

    def getA(self):
        return self.__A

    A = property(getA)

    def getF(self):
        return self.__F

    F = property(getF)

    def get_const(self):
        return self.__K

    const = property(get_const)
