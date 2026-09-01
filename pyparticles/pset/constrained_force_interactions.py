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

import pyparticles.pset.constraint as ct


class ConstrainedForceInteractions(ct.Constraint):
    def __init__(self, pset=None):
        self.__S = dok_matrix((1, 1), dtype=np.byte)
        super(ConstrainedForceInteractions, self).__init__(pset=None)
        if pset is not None:
            self.pset = pset

    def get_pset(self):
        return super(ConstrainedForceInteractions, self).get_pset()

    def set_pset(self, pset):
        self.__S.resize((pset.size, pset.size))
        super(ConstrainedForceInteractions, self).set_pset(pset)

    pset = property(get_pset, set_pset)

    def add_connections(self, fc):
        for c in fc:
            i, j = int(c[0]), int(c[1])
            self.__S[i, j] = True

    def remove_connections(self, fc):
        for c in fc:
            i, j = int(c[0]), int(c[1])
            if (i, j) in self.__S:
                del self.__S[i, j]

    def get_dense(self):
        return np.asarray(self.__S.todense(), dtype=np.bool_)

    dense = property(get_dense)

    def get_sparse(self):
        return self.__S

    sparse = property(get_sparse)

    def get_items(self):
        return self.__S.items()

    items = property(get_items)
