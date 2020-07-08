#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2020 Tiago de Paula Peixoto <tiago@skewed.de>
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

from .. import Graph, GraphView, _get_rng, _prop, PropertyMap, \
    perfect_prop_hash, Vector_size_t, group_vector_property
from . blockmodel import DictState, _bm_test
from . util import *
import numpy as np

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

def modularity(g, b, gamma=1., weight=None):
    r"""
    Calculate Newman's (generalized) modularity of a network partition.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be used.
    b : :class:`~graph_tool.VertexPropertyMap`
        Vertex property map with the community partition.
    gamma : ``float`` (optional, default: ``1.``)
        Resolution parameter.
    weight : :class:`~graph_tool.EdgePropertyMap` (optional, default: None)
        Edge property map with the optional edge weights.

    Returns
    -------
    Q : float
        Newman's modularity.

    Notes
    -----

    Given a specific graph partition specified by `prop`, Newman's modularity
    [newman-modularity-2006]_ is defined as:

    .. math::

          Q = \frac{1}{2E} \sum_r e_{rr}- \frac{e_r^2}{2E}

    where :math:`e_{rs}` is the number of edges which fall between
    vertices in communities s and r, or twice that number if :math:`r = s`, and
    :math:`e_r = \sum_s e_{rs}`.

    If weights are provided, the matrix :math:`e_{rs}` corresponds to the sum
    of edge weights instead of number of edges, and the value of :math:`E`
    becomes the total sum of edge weights.

    Examples
    --------
    >>> g = gt.collection.data["football"]
    >>> gt.modularity(g, g.vp.value_tsevans)
    0.5744393497...

    References
    ----------
    .. [newman-modularity-2006] M. E. J. Newman, "Modularity and community
       structure in networks", Proc. Natl. Acad. Sci. USA 103, 8577-8582 (2006),
       :doi:`10.1073/pnas.0601602103`, :arxiv:`physics/0602124`
    """

    if b.value_type() not in ["bool", "int16_t", "int32_t", "int64_t",
                              "unsigned long"]:
        b = perfect_prop_hash([b])[0]
    Q = libinference.modularity(g._Graph__graph, gamma,
                                _prop("e", g, weight),
                                _prop("v", g, b))
    return Q


def get_modularity_entropy_args(kargs, ignore=None):
    kargs = kargs.copy()
    if ignore is not None:
        for a in ignore:
            kargs.pop(a, None)
    ea = libinference.modularity_entropy_args()
    ea.gamma = kargs["gamma"]
    del kargs["gamma"]
    if len(kargs) > 0:
        raise ValueError("unrecognized entropy arguments: " +
                         str(list(kargs.keys())))
    return ea

class ModularityState(object):
    r"""Obtain the partition of a network according to Newman's modularity.

    .. warning::

       Do not use this approach in the analysis of networks without
       understanding the consequences. This algorithm is included only for
       comparison purposes. In general, the inference-based approaches based on
       :class:`~graph_tool.inference.blockmodel.BlockState`,
       :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState`, and
       :class:`~graph_tool.inference.planted_partition.PPBlockState` should be
       universally preferred.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Graph to be partitioned.
    b : :class:`~graph_tool.PropertyMap` (optional, default: ``None``)
        Initial partition. If not supplied, a partition into a single group will
        be used.

    """

    def __init__(self, g, b=None):

        self.g = GraphView(g, directed=False)
        if b is None:
            self.b = self.g.new_vp("int32_t")
        elif isinstance(g, PropertyMap):
            self.b = b.copy("int32_t")
        else:
            self.b = self.g.new_vp("int32_t", vals=b)

        self.er = Vector_size_t()
        self.err = Vector_size_t()

        self.bg = self.g
        self._abg = self.bg._get_any()
        self._state = libinference.make_modularity_state(self)

        self._entropy_args = dict(gamma=1.)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        g = copy.deepcopy(self.g, memo)
        b = copy.deepcopy(self.b, memo)
        return self.copy(g=g, b=b)

    def copy(self, g=None, b=None):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""
        return ModularityState(g=g if g is not None else self.g,
                                b=b if b is not None else self.b)

    def __getstate__(self):
        return dict(g=self.g, b=self.b)

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        return "<ModularityState object with %d blocks, for graph %s, at 0x%x>" % \
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

    def entropy(self, gamma=1., **kwargs):
        r"""Return the unnormalized negative generalized modularity.

        Notes
        -----

        The unnormalized negative generalized modularity is defined as

        .. math::

           -\sum_{ij}\left(A_{ij}-\gamma \frac{k_ik_j}{2E}\right)

        Where :math:`A_{ij}` is the adjacency matrix, :math:`k_i` is the degree
        of node :math:`i`, and :math:`E` is the total number of edges.

        """

        entropy_args = dict(self._entropy_args, **locals())
        eargs = get_modularity_entropy_args(entropy_args,
                                            ignore=["self", "kwargs"])

        if _bm_test() and kwargs.get("test", True):
            args = dict(gamma=gamma)

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


    def modularity(self, gamma=1):
        r"""Return the generalized modularity.

        Notes
        -----

        The generalized modularity is defined as

        .. math::

           \frac{1}{2E}\sum_{ij}\left(A_{ij}-\gamma \frac{k_ik_j}{2E}\right)

        Where :math:`A_{ij}` is the adjacency matrix, :math:`k_i` is the degree
        of node :math:`i`, and :math:`E` is the total number of edges.

        """

        Q = self.entropy(gamma=gamma)
        return -Q / (2 * self.g.num_edges())

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
        mcmc_state.oentropy_args = get_modularity_entropy_args(entropy_args)
        mcmc_state.vlist = Vector_size_t()
        mcmc_state.vlist.resize(self.g.num_vertices())
        mcmc_state.vlist.a = self.g.vertex_index.copy().fa
        mcmc_state.state = self._state
        mcmc_state.E = self.g.num_edges()

        test = kwargs.pop("test", True)
        if _bm_test() and test:
            Si = self.entropy(**eargs)

        dS, nattempts, nmoves = \
            libinference.modularity_mcmc_sweep(mcmc_state, self._state,
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


    def multiflip_mcmc_sweep(self, beta=1., c=0.5, psingle=None, psplit=1,
                             pmerge=1, pmergesplit=1, d=0.01, gibbs_sweeps=10,
                             niter=1, entropy_args={}, accept_stats=None,
                             verbose=False, **kwargs):
        r"""Perform sweeps of a merge-split Metropolis-Hastings rejection sampling MCMC
        to sample network partitions. See
        :meth:`graph_tool.inference.blockmodel.BlockState.mcmc_sweep` for the
        parameter documentation."""
        if psingle is None:
            psingle = self.g.num_vertices()
        gibbs_sweeps = max(gibbs_sweeps, 1)
        nproposal = Vector_size_t(4)
        nacceptance = Vector_size_t(4)
        force_move = kwargs.pop("force_move", False)
        mcmc_state = DictState(locals())
        eargs = entropy_args
        entropy_args = dict(self._entropy_args, **entropy_args)
        mcmc_state.oentropy_args = get_modularity_entropy_args(entropy_args)
        mcmc_state.state = self._state
        mcmc_state.E = self.g.num_edges()

        test = kwargs.pop("test", True)
        if _bm_test() and test:
            Si = self.entropy(**eargs)

        dS, nattempts, nmoves = \
            libinference.modularity_multiflip_mcmc_sweep(mcmc_state,
                                                         self._state,
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
