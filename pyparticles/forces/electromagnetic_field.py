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

import random

import numpy as np

import pyparticles.forces.force as fr


class ElectromagneticField(fr.Force):
    r"""
    Electromagnetic force for non-self-interacting charged particles.

    .. math::

        \mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})
    """

    def __init__(self, size, dim=3, m=None, q=None, Consts=1.0):
        self.__dim = dim
        self.__size = size

        self.__A = np.zeros((size, dim))
        self.__E = np.zeros((size, dim))
        self.__B = np.zeros((size, dim))
        self.__Fe = np.zeros((size, dim))
        self.__Fm = np.zeros((size, dim))
        self.__M = np.zeros((size, 1))
        self.__Q = np.zeros((size, 1))

        if m is not None:
            self.set_masses(m)
        if q is not None:
            self.set_charges(q)

        self.__el_fields = {}
        self.__ma_fields = {}

    def set_masses(self, m):
        self.__M[:] = m

    def set_charges(self, q):
        self.__Q[:] = q

    def append_electric_field(self, ef, key=None):
        if key is None:
            key = str(random.randint(0, 2**64))
        self.__el_fields[key] = ef
        return key

    def append_magnetic_field(self, bf, key=None):
        if key is None:
            key = str(random.randint(0, 2**64))
        self.__ma_fields[key] = bf
        return key

    def update_force(self, pset):
        self.__Fe[:] = 0.0
        self.__Fm[:] = 0.0

        for field in self.__el_fields.values():
            field(self.__E, pset.X)
            self.__Fe += self.__Q * self.__E

        for field in self.__ma_fields.values():
            field(self.__B, pset.X)
            self.__Fm += self.__Q * np.cross(pset.V, self.__B)

        self.__Fe += self.__Fm
        self.__A[:] = self.__Fe / self.__M
        return self.__A

    def getA(self):
        return self.__A

    A = property(getA)

    def getF(self):
        return self.__Fe

    F = property(getF)
