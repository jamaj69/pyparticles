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

import pyparticles.pset.constraint as ct


class ConstrainedX(ct.Constraint):
    def __init__(self, pset=None):
        self.__X_cr = None
        self.__X_cr_i = None
        self.__X_free = None

        self.__use_slice_const = False
        self.__use_slice_free = False

        super(ConstrainedX, self).__init__(pset=pset)

    def add_x_constraint(self, indx, constr):
        """Add positional constraints and update the referenced particle set."""
        if isinstance(indx, slice):
            self.__X_cr = np.array(constr)
            self.__X_cr_i = indx
            self.__use_slice_const = True
            self.pset.X[indx, :] = constr
            self._optimize()
            return

        indx = np.asarray(indx, dtype=np.int64)
        constr = np.asarray(constr)

        if self.__X_cr is None:
            self.__X_cr = constr.copy()
            self.__X_cr_i = indx.copy()
        else:
            self.__X_cr_i = np.concatenate((self.__X_cr_i, indx))
            self.__X_cr = np.concatenate((self.__X_cr, constr))

        self._optimize()
        self.pset.X[indx, :] = constr

    def _optimize(self):
        """Use a slice for free indices when they form one contiguous range."""
        free = list(range(self.pset.size))

        if self.__X_cr_i is None:
            constrained = []
        elif isinstance(self.__X_cr_i, slice):
            constrained = range(*self.__X_cr_i.indices(self.pset.size))
        else:
            constrained = self.__X_cr_i

        for i in constrained:
            i = int(i)
            if i in free:
                free.remove(i)

        if not free:
            self.__X_free = np.array([], dtype=np.int64)
            self.__use_slice_free = False
            return

        sequential = all((free[i + 1] - free[i]) == 1 for i in range(len(free) - 1))
        if sequential:
            self.__X_free = slice(free[0], free[-1] + 1)
            self.__use_slice_free = True
        else:
            self.__X_free = np.asarray(free, dtype=np.int64)
            self.__use_slice_free = False

    def get_pset(self):
        return super(ConstrainedX, self).get_pset()

    def set_pset(self, pset):
        super(ConstrainedX, self).set_pset(pset)
        if self.__X_cr_i is not None and self.__X_cr is not None:
            pset.X[self.__X_cr_i, :] = self.__X_cr
            self._optimize()

    pset = property(get_pset, set_pset, doc="get and set the particles set (pset)")

    def remove_x_constraint(self, indxs):
        """Remove constraints whose particle indices are listed in *indxs*."""
        if self.__use_slice_const or self.__X_cr_i is None:
            return

        remove_positions = []
        for i in indxs:
            matches = np.flatnonzero(self.__X_cr_i == i)
            remove_positions.extend(matches.tolist())

        if not remove_positions:
            return

        remove_positions = np.asarray(sorted(set(remove_positions)), dtype=np.int64)
        self.__X_cr = np.delete(self.__X_cr, remove_positions, axis=0)
        self.__X_cr_i = np.delete(self.__X_cr_i, remove_positions, axis=0)

        if self.__X_cr_i.size == 0:
            self.__X_cr = None
            self.__X_cr_i = None

        self._optimize()

    def get_cx_indicies(self):
        """Return a copy of the constrained indices (legacy spelling kept)."""
        if isinstance(self.__X_cr_i, slice):
            return self.__X_cr_i
        if self.__X_cr_i is None:
            return None
        return np.copy(self.__X_cr_i)

    def set_free_indicies(self, indx):
        if isinstance(indx, slice):
            self.__X_free = indx
            self.__use_slice_free = True
            return

        indx = np.asarray(indx, dtype=np.int64)
        if self.__X_free is None:
            self.__X_free = indx
        elif isinstance(self.__X_free, slice):
            current = np.arange(self.pset.size, dtype=np.int64)[self.__X_free]
            self.__X_free = np.concatenate((current, indx))
        else:
            self.__X_free = np.concatenate((self.__X_free, indx))
        self.__use_slice_free = False

    def get_cx_free_indicies(self):
        return self.__X_free

    def clear_all_x_constraint(self):
        self.__X_cr = None
        self.__X_cr_i = None
        self.__X_free = slice(0, self.pset.size) if self.pset is not None else None
        self.__use_slice_const = False
        self.__use_slice_free = self.pset is not None

    def get_cX(self):
        if self.__X_cr_i is None:
            return None
        return self.pset.X[self.__X_cr_i, :]

    cX = property(get_cX, doc="return the constrained X elements")
