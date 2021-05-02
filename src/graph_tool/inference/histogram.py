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

from .. import _get_rng, Vector_double

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

from . util import *
from . blockmodel import _bm_test

from collections.abc import Iterable
import numpy as np

class HistState(object):
    def __init__(self, x, bins = None, alpha=1., bounded=None, discrete=None):

        self.x = np.asarray(x, dtype="double")
        self.D = self.x.shape[1]
        self.alpha = alpha

        if bounded is None:
            bounded = [(False, False)] * self.D
        if discrete is None:
            discrete = [False] * self.D
        self.bounded = self.obounded = [(bool(x), bool(y)) for x, y in bounded]
        self.discrete = self.odiscrete = [bool(x) for x in discrete]

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
            self.bins.append(Vector_double(len(b), b))


        self.obins = self.bins
        self._state = libinference.make_hist_state(self, self.D)

    def __copy__(self):
        return self.copy()

    def copy(self, b=None, n=None):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""

        return HistState(x=self.x, bins=self.bins, alpha=self.alpha,
                         bounded=self.bounded, discrete=self.discrete)

    def __getstate__(self):
        return dict(x=self.x, bins=[b.a for b in self.bins], alpha=self.alpha,
                    bounded=self.bounded, discrete=self.discrete)

    def __setstate__(self, state):
        self.__init__(**state)

    def entropy(self, **kwargs):

        S = self._state.entropy()

        if kwargs.pop("test", True) and _bm_test():
            assert not isnan(S) and not isinf(S), \
                "invalid entropy %g " % S

            state_copy = self.copy()
            Salt = state_copy.entropy(test=False)

            assert math.isclose(S, Salt, abs_tol=1e-8), \
                "entropy discrepancy after copying (%g %g %g)" % (S, Salt,
                                                                  S - Salt)

        return S

    def get_lpdf(self, x):
        return self._state.get_mle_lpdf(np.asarray(x, dtype="double"))

    def predictive_sample(self, n=1):
        return self._state.sample(n, _get_rng())

    def mcmc_sweep(self, beta=1., niter=1, verbose=False,
                   **kwargs):
        mcmc_state = DictState(locals())
        mcmc_state.state = self._state

        test = kwargs.pop("test", True)
        if _bm_test() and test:
            Si = self.entropy(test=True)

        dS, nattempts, nmoves = \
            libinference.hist_mcmc_sweep(mcmc_state, self._state, self.D,
                                         _get_rng())

        if _bm_test() and test:
            Sf = self.entropy(test=True)
            assert math.isclose(dS, (Sf - Si), abs_tol=1e-8), \
                "inconsistent entropy delta %g (%g)" % (dS, Sf - Si)

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        return dS, nattempts, nmoves
