# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np
import pyparticles.pset.boundary as bd


class DefaultBoundary(bd.Boundary):
    r"""Move particles outside the domain through a user callback."""

    def __init__(self, bound=(-1, 1), dim=3, defualt_pos=None):
        self.set_boundary(bound, dim)
        self.__defualt_pos = defualt_pos

    def boundary(self, p_set):
        changed = False
        for i in range(self.dim):
            b_mi, = np.where(p_set.X[:, i] < self.bound[i, 0])
            b_mx, = np.where(p_set.X[:, i] > self.bound[i, 1])

            if len(b_mi) > 0:
                self.__defualt_pos(p_set, b_mi)
                changed = True

            if len(b_mx) > 0:
                self.__defualt_pos(p_set, b_mx)
                changed = True

        return changed
