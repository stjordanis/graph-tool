#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2020 Tiago de Paula Peixoto <tiago@skewed.de>
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

from __future__ import division, absolute_import, print_function
import sys
if sys.version_info < (3,):
    range = xrange

from .. import _degree, _prop, Graph, GraphView, _get_rng, Vector_size_t
from . blockmodel import DictState, get_entropy_args, _bm_test

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

import numpy as np
import math

class VICenterState(object):
    r"""Obtain the center of a set of partitions, according to the variation of
    information metric.

    Parameters
    ----------
    bs : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
        Initial block labels on the vertices. If not supplied, it will be
        randomly sampled.
    B : ``int`` (optional, default: ``None``)
        Number of blocks (or vertex groups). If not supplied it will be obtained
        from the parameter ``b``.
    """

    def __init__(self, bs, b=None):

        self.bs = bs = np.asarray(bs, dtype="int32")
        if b is None:
            b = np.zeros(bs.shape[1], dtype="int32")
        self.b = np.array(b, dtype="int32")

        self.g = Graph()
        self.g.add_vertex(bs.shape[1])
        self.bg = self.g
        self._abg = self.bg._get_any()
        self._state = libinference.make_vi_center_state(self)

        self._entropy_args = dict(adjacency=True, deg_entropy=True, dl=True,
                                  partition_dl=True, degree_dl=True,
                                  degree_dl_kind="distributed", edges_dl=True,
                                  dense=False, multigraph=True, exact=True,
                                  recs=True, recs_dl=True, beta_dl=1.,
                                  Bfield=True)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        b = copy.deepcopy(self.b, memo)
        bs = copy.deepcopy(self.bs, memo)
        return self.copy(bs=bs, b=b)

    def copy(self, bs=None, b=None):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""

        return VICenterState(bs=bs if bs is not None else self.bs,
                             b=b if b is not None else self.b)


    def __getstate__(self):
        return dict(bs=self.bs, b=self.b)

    def __setstate__(self, state):
        self.__init__(**state)

    def get_B(self):
        r"Returns the total number of blocks."
        return len(np.unique(self.b))

    def get_Be(self):
        r"""Returns the effective number of blocks, defined as :math:`e^{H}`, with
        :math:`H=-\sum_r\frac{n_r}{N}\ln \frac{n_r}{N}`, where :math:`n_r` is
        the number of nodes in group r.
        """
        w = np.array(np.bincount(self.b), dtype="double")
        w = w[w>0]
        w /= w.sum()
        return numpy.exp(-(w*log(w)).sum())

    def entropy(self):
        return self._state.entropy()

    def mcmc_sweep(self, beta=1.,d=.01, niter=1, entropy_args={},
                   allow_vacate=True, sequential=True, deterministic=False,
                   verbose=False, **kwargs):
        mcmc_state = DictState(locals())
        mcmc_state.entropy_args = get_entropy_args(self._entropy_args)
        mcmc_state.vlist = Vector_size_t()
        mcmc_state.vlist.resize(len(self.b))
        mcmc_state.vlist.a = np.arange(len(self.b))
        mcmc_state.state = self._state
        mcmc_state.c = 0
        mcmc_state.E = 0

        test = kwargs.pop("test", True)
        if _bm_test() and test:
            Si = self.entropy()

        dS, nattempts, nmoves = \
            libinference.vi_mcmc_sweep(mcmc_state, self._state,
                                       _get_rng())

        if _bm_test() and test:
            Sf = self.entropy(**entropy_args)
            assert math.isclose(dS, (Sf - Si), abs_tol=1e-8), \
                "inconsistent entropy delta %g (%g): %s" % (dS, Sf - Si,
                                                            str(entropy_args))

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        return dS, nattempts, nmoves


    def multiflip_mcmc_sweep(self, beta=1., psingle=100, psplit=1,
                             pmerge=1, pmergesplit=1, d=0.01, gibbs_sweeps=10,
                             niter=1, entropy_args={}, accept_stats=None,
                             verbose=False, **kwargs):

        gibbs_sweeps = max(gibbs_sweeps, 1)
        nproposal = Vector_size_t(4)
        nacceptance = Vector_size_t(4)
        force_move = kwargs.pop("force_move", False)
        mcmc_state = DictState(locals())
        mcmc_state.entropy_args = get_entropy_args(self._entropy_args)
        mcmc_state.state = self._state
        mcmc_state.c = 0
        mcmc_state.E = 0

        test = kwargs.pop("test", True)
        if _bm_test() and test:
            Si = self.entropy(**entropy_args)

        dS, nattempts, nmoves = \
            libinference.vi_multiflip_mcmc_sweep(mcmc_state, self._state,
                                                 _get_rng())

        if _bm_test() and test:
            Sf = self.entropy()
            assert math.isclose(dS, (Sf - Si), abs_tol=1e-8), \
                "inconsistent entropy delta %g (%g): %s" % (dS, Sf - Si,
                                                            str(entropy_args))

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))

        if accept_stats is not None:
            for key in ["proposal", "acceptance"]:
                if key not in accept_stats:
                    accept_stats[key] = numpy.zeros(len(nproposal),
                                                    dtype="uint64")
            accept_stats["proposal"] += nproposal.a
            accept_stats["acceptance"] += nacceptance.a

        return dS, nattempts, nmoves
