#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2021 Tiago de Paula Peixoto <tiago@skewed.de>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from .. import _get_rng, Vector_double, Vector_int64_t

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

from . util import *
from . base_states import *

from collections.abc import Iterable
import numpy as np

class HistState(object):
    def __init__(self, x, bins = None, alpha=1., bounded=None, discrete=None,
                 conditional=None):

        if discrete is not None and all(discrete):
            self.x = np.asarray(x, dtype="int64")
        else:
            self.x = np.asarray(x, dtype="double")
        self.D = self.x.shape[1]
        self.alpha = alpha

        if bounded is None:
            bounded = [(False, False)] * self.D
        if discrete is None:
            discrete = [False] * self.D
        self.bounded = self.obounded = [(bool(x), bool(y)) for x, y in bounded]
        self.discrete = self.odiscrete = [bool(x) for x in discrete]
        if conditional is None:
            conditional = self.D
        self.conditional = conditional

        if bins is None:
            bins = [1] * self.D

        self.bins = []
        for j in range(self.D):
            if isinstance(bins[j], int):
                b = np.linspace(self.x[:,j].min(), self.x[:,j].max(), bins[j] + 1)
                if discrete[j]:
                    b[-1] += 1
                else:
                    b[-1] += 1e-8
            else:
                b = bins[j]
            if all(self.discrete):
                b = Vector_int64_t(len(b), b)
            else:
                b = Vector_double(len(b), b)
            self.bins.append(b)

        self.obins = self.bins
        self._state = libinference.make_hist_state(self, self.D)

    def __copy__(self):
        return self.copy()

    def copy(self, **kwargs):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""

        return HistState(**dict(self.__getstate__(), **kwargs))

    def __getstate__(self):
        return dict(x=self.x, bins=[b.a.copy() for b in self.bins],
                    alpha=self.alpha, bounded=self.bounded,
                    discrete=self.discrete, conditional=self.conditional)

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        return "<HistState object with data shape %s, bin shape %s, discrete %s, bounded %s at 0x%x>" % \
            (self.x.shape, tuple(len(s) for s in self.bins), self.discrete,
             self.bounded, id(self))

    @copy_state_wrap
    def entropy(self, **kwargs):

        S = self._state.entropy()
        return S

    def _get_entropy_args(self, kwargs):
        return None

    def get_lpdf(self, x):
        if all(self.discrete):
            dtype = "int64"
        else:
            dtype = "double"
        return self._state.get_mle_lpdf(np.asarray(x, dtype=dtype))

    def predictive_sample(self, x=None, n=1):
        if all(self.discrete):
            dtype = "int64"
        else:
            dtype = "double"
        if self.conditional < self.D:
            x = np.asarray(x, dtype=dtype)
        else:
            x = np.zeros(1, dtype=dtype)
        return self._state.sample(n, x, _get_rng())

    @mcmc_sweep_wrap
    def mcmc_sweep(self, beta=1., niter=1, verbose=False,
                   **kwargs):
        mcmc_state = DictState(locals())
        mcmc_state.state = self._state

        dS, nattempts, nmoves = \
            libinference.hist_mcmc_sweep(mcmc_state, self._state, self.D,
                                         _get_rng())

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        return dS, nattempts, nmoves
