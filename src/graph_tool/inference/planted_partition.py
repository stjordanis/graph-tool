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

from .. import Graph, GraphView, _get_rng, Vector_size_t, PropertyMap, \
    group_vector_property
from . blockmodel import DictState, _bm_test, init_q_cache
from . util import *

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

import numpy as np
import math

def get_pp_entropy_args(kargs, ignore=None):
    kargs = kargs.copy()
    if ignore is not None:
        for a in ignore:
            kargs.pop(a, None)
    deg_dl_kind = kargs["degree_dl_kind"]
    if deg_dl_kind == "entropy":
        kind = libinference.deg_dl_kind.ent
    elif deg_dl_kind == "uniform":
        kind = libinference.deg_dl_kind.uniform
    elif deg_dl_kind == "distributed":
        kind = libinference.deg_dl_kind.dist
    ea = libinference.pp_entropy_args()
    ea.degree_dl_kind = kind
    ea.uniform = kargs["uniform"]
    del kargs["uniform"]
    del kargs["degree_dl_kind"]
    if len(kargs) > 0:
        raise ValueError("unrecognized entropy arguments: " +
                         str(list(kargs.keys())))
    return ea

class PPBlockState(object):
    r"""Obtain the center of a set of partitions, according to the variation of
    information metric or reduced mutual information.

    Parameters
    ----------
    bs : iterable of iterable of ``int``
        List of partitions.
    b : ``list`` or :class:`numpy.ndarray` (optional, default: ``None``)
        Initial partition. If not supplied, a partition into a single group will
        be used.
    RMI : ``bool`` (optional, default: ``False``)
         If ``True``, the reduced mutual information will be used, otherwise the
         variation of information metric will be used instead.
    """

    def __init__(self, g, b=None):

        self.g = GraphView(g, directed=False)
        if b is None:
            self.b = self.g.new_vp("int32_t")
        elif isinstance(g, PropertyMap):
            self.b = b.copy("int32_t")
        else:
            self.b = self.g.new_vp("int32_t", vals=b)

        self.wr = Vector_size_t()
        self.er = Vector_size_t()
        self.err = Vector_size_t()
        self.eio = Vector_size_t()

        init_q_cache(max(2 * max(self.g.num_edges(),
                                 self.g.num_vertices()), 100))

        self.bg = self.g
        self._abg = self.bg._get_any()
        self._state = libinference.make_pp_state(self)

        self._entropy_args = dict(uniform=False, degree_dl_kind="distributed")

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        g = copy.deepcopy(self.g, memo)
        b = copy.deepcopy(self.b, memo)
        return self.copy(g=g, b=b)

    def copy(self, g=None, b=None):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""
        return PPBlockState(g=g if g is not None else self.g,
                            b=b if b is not None else self.b)

    def __getstate__(self):
        return dict(g=self.g, b=self.b)

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        return "<PPBlockState object with %d blocks, for graph %s, at 0x%x>" % \
            (self.get_B(), str(self.g), id(self))

    def get_B(self):
        r"Returns the total number of blocks."
        return len(np.unique(self.b.fa))

    def get_Be(self):
        r"""Returns the effective number of blocks, defined as :math:`e^{H}`, with
        :math:`H=-\sum_r\frac{n_r}{N}\ln \frac{n_r}{N}`, where :math:`n_r` is
        the number of nodes in group r.
        """
        w = np.array(np.bincount(self.b.fa), dtype="double")
        w = w[w>0]
        w /= w.sum()
        return numpy.exp(-(w*log(w)).sum())

    def entropy(self, uniform=False, degree_dl_kind="distributed", **kwargs):
        entropy_args = dict(self._entropy_args, **locals())
        eargs = get_pp_entropy_args(entropy_args, ignore=["self", "kwargs"])

        if _bm_test() and kwargs.get("test", True):
            args = dict(uniform=uniform, degree_dl_kind=degree_dl_kind)

        S = self._state.entropy(eargs)

        if kwargs.pop("test", True) and _bm_test():
            assert not np.isnan(S) and not np.isinf(S), \
                "invalid entropy %g (%s) " % (S, str(args))

            args["test"] = False
            state_copy = self.copy()
            Salt = state_copy.entropy(**args)

            assert math.isclose(S, Salt, abs_tol=1e-8), \
                "entropy discrepancy after copying (%g %g %g)" % (S, Salt,
                                                                  S - Salt)

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        return S

    def draw(self, **kwargs):
        r"""Convenience wrapper to :func:`~graph_tool.draw.graph_draw` that
        draws the state of the graph as colors on the vertices and edges."""
        gradient = self.g.new_ep("double")
        gradient = group_vector_property([gradient])
        from graph_tool.draw import graph_draw
        return graph_draw(self.g,
                          vertex_fill_color=kwargs.get("vertex_fill_color",
                                                       self.b),
                          vertex_color=kwargs.get("vertex_color", self.b),
                          edge_gradient=kwargs.get("edge_gradient",
                                                   gradient),
                          **dmask(kwargs, ["vertex_fill_color",
                                           "vertex_color",
                                           "edge_gradient"]))

    def mcmc_sweep(self, beta=1., c=0.5, d=.01, niter=1, entropy_args={},
                   allow_vacate=True, sequential=True, deterministic=False,
                   verbose=False, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings rejection sampling MCMC to sample
        network partitions. See
        :meth:`graph_tool.inference.blockmodel.BlockState.mcmc_sweep` for the
        parameter documentation. """
        mcmc_state = DictState(locals())
        eargs = entropy_args
        entropy_args = dict(self._entropy_args, **entropy_args)
        mcmc_state.oentropy_args = get_pp_entropy_args(entropy_args)
        mcmc_state.vlist = Vector_size_t()
        mcmc_state.vlist.resize(self.g.num_vertices())
        mcmc_state.vlist.a = self.g.vertex_index.copy().fa
        mcmc_state.state = self._state
        mcmc_state.E = self.g.num_edges()

        test = kwargs.pop("test", True)
        if _bm_test() and test:
            Si = self.entropy(**eargs)

        dS, nattempts, nmoves = \
            libinference.pp_mcmc_sweep(mcmc_state, self._state,
                                       _get_rng())

        if _bm_test() and test:
            Sf = self.entropy(**eargs)
            assert math.isclose(dS, (Sf - Si), abs_tol=1e-8), \
                "inconsistent entropy delta %g (%g): %s" % (dS, Sf - Si,
                                                            str(entropy_args))

        if len(kwargs) > 0:
            raise ValueError("unrecognized keyword arguments: " +
                             str(list(kwargs.keys())))
        return dS, nattempts, nmoves


    def multiflip_mcmc_sweep(self, beta=1., c=0.5, psingle=100, psplit=1,
                             pmerge=1, pmergesplit=1, d=0.01, gibbs_sweeps=10,
                             niter=1, entropy_args={}, accept_stats=None,
                             verbose=False, **kwargs):
        r"""Perform sweeps of a merge-split Metropolis-Hastings rejection sampling MCMC
        to sample network partitions. See
        :meth:`graph_tool.inference.blockmodel.BlockState.mcmc_sweep` for the
        parameter documentation."""
        gibbs_sweeps = max(gibbs_sweeps, 1)
        nproposal = Vector_size_t(4)
        nacceptance = Vector_size_t(4)
        force_move = kwargs.pop("force_move", False)
        mcmc_state = DictState(locals())
        eargs = entropy_args
        entropy_args = dict(self._entropy_args, **entropy_args)
        mcmc_state.oentropy_args = get_pp_entropy_args(entropy_args)
        mcmc_state.state = self._state
        mcmc_state.E = self.g.num_edges()

        test = kwargs.pop("test", True)
        if _bm_test() and test:
            Si = self.entropy(**eargs)

        dS, nattempts, nmoves = \
            libinference.pp_multiflip_mcmc_sweep(mcmc_state, self._state,
                                                 _get_rng())

        if _bm_test() and test:
            Sf = self.entropy(**eargs)
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
