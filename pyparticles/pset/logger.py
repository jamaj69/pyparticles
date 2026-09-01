# PyParticles : Particles simulation in python
# Copyright (C) 2012  Simone Riva
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import numpy as np


class Logger(object):
    """Circular in-memory logger for particle positions and velocities."""

    def __init__(self, pset, log_max_size, log_X=True, log_V=False, sim_time=None):
        self.__pset = pset
        self.__log_max_size = int(log_max_size)
        self.__size = 0
        self.__log_cnt = 0
        self.__La = 0
        self.__Lb = -1
        self.__state = 0

        if sim_time is not None:
            self.__sim_time = sim_time
            self.__log_time = np.zeros(self.__log_max_size)
        else:
            self.__sim_time = None
            self.__log_time = None

        self.__log_X = (
            np.zeros((self.__log_max_size, pset.size, pset.dim)) if log_X else None
        )
        self.__log_V = (
            np.zeros((self.__log_max_size, pset.size, pset.dim)) if log_V else None
        )

    def log(self):
        self.__log_cnt += 1

        if self.__state == 0:
            self.__size += 1
            self.__Lb += 1

        if self.__state in (1, 2):
            self.__La += 1
            self.__Lb += 1
            if self.__La >= self.__log_max_size:
                self.__La = 0
            if self.__Lb >= self.__log_max_size:
                self.__Lb = 0

        if self.__state != 0 and self.__La > self.__Lb:
            self.__state = 1
        if self.__state != 0 and self.__La < self.__Lb:
            self.__state = 2

        if self.__state == 0 and self.__Lb >= self.__log_max_size:
            self.__size = self.__log_max_size
            self.__Lb = 0
            self.__La = 1
            self.__state = 1

        if self.__sim_time is not None:
            self.__log_time[self.__Lb] = self.__sim_time.time
        if self.__log_X is not None:
            self.__log_X[self.__Lb, :, :] = self.__pset.X
        if self.__log_V is not None:
            self.__log_V[self.__Lb, :, :] = self.__pset.V

    def close_log(self):
        pass

    def __get_log_indices(self):
        if self.__size == 0:
            return np.empty(0, dtype=np.int32)

        if self.__state == 0:
            return np.arange(self.__La, self.__Lb + 1, dtype=np.int32)
        if self.__state == 1:
            return np.concatenate(
                (
                    np.arange(self.__La, self.log_max_size, dtype=np.int32),
                    np.arange(0, self.__Lb + 1, dtype=np.int32),
                )
            )
        return np.concatenate(
            (
                np.arange(self.__La, self.__Lb, dtype=np.int32),
                np.arange(self.__Lb, self.__log_max_size, dtype=np.int32),
            )
        )

    def get_log_array(self, i, log_X=True, log_V=False, log_time=False):
        ind = self.__get_log_indices()
        result = []
        if log_X:
            result.append(self.__log_X[ind, i, :])
        if log_V:
            result.append(self.__log_V[ind, i, :])
        if log_time:
            result.append(self.__log_time[ind])
        return tuple(result)

    def read_log_array(self, i, ta, log_X=True, log_V=False, log_time=False):
        ind = self.__get_log_indices()
        li = len(ind)

        if log_X:
            ta[0][0:li, :] = self.__log_X[ind, i, :]
        if log_V:
            ta[1][0:li, :] = self.__log_V[ind, i, :]
        if log_time:
            ta[2][0:li] = self.__log_time[ind]

        return (0, max(0, 2 * li - 2))

    def get_log_indices_segments(self, full=False):
        d = self.__log_max_size if full else self.__size
        if d < 2:
            return np.empty(0, dtype=np.uint32)

        i = np.arange(d, dtype=np.uint32)
        f = np.zeros(2 * d - 2, dtype=np.uint32)
        f[2:(2 * d - 2):2] = i[1:(d - 1)]
        f[1:(2 * d - 2):2] = i[1:]
        return f

    def resize(self, log_max_size):
        raise NotImplementedError("Logger.resize is not implemented")

    def jump(self):
        pass

    def get_log_max_size(self):
        return self.__log_max_size

    log_max_size = property(get_log_max_size, doc="get the max allowed size of the log")

    def get_log_size(self):
        return self.__size

    log_size = property(get_log_size)

    def get_log_X_enabled(self):
        return self.__log_X is not None

    def get_log_V_enabled(self):
        return self.__log_V is not None

    log_V_enabled = property(get_log_V_enabled)
    log_X_enabled = property(get_log_X_enabled)
