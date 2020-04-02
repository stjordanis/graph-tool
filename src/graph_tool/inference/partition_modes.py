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

from .. import _degree, _prop, Graph, GraphView, _get_rng, Vector_int32_t, \
    Vector_size_t
from . blockmodel import DictState, get_entropy_args, _bm_test

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

import numpy as np
import math

class PartitionModeState(object):
    def __init__(self, bs, relabel=True, nested=False, **kwargs):
        self.bs = {}
        self.nested = nested

        if len(kwargs) > 0:
            self._base = kwargs["base"]
        else:
            self._base = libinference.PartitionModeState()
            for b in bs:
                self.add_partition(b, relabel)

    def __copy__(self):
        return self.copy()

    def copy(self, bs=None):
        r"""Copies the state. The parameters override the state properties, and
         have the same meaning as in the constructor."""
        if bs is None:
            bs = list(self.get_nested_partitions().values())
        return PartitionModeState(bs=bs, relabel=False, nested=self.nested)

    def __getstate__(self):
        return dict(bs=list(self.get_nested_partitions().values()),
                    nested=self.nested)

    def __setstate__(self, state):
        self.__init__(**state, relabel=False)

    def add_partition(self, b, relabel=True):
        if self.nested:
            b = [Vector_int32_t(init=x) for x in b]
        else:
            b = [Vector_int32_t(init=b)]
        i = self._base.add_partition(b, relabel)
        self.bs[i] = b

    def remove_partition(self, i):
        self._base.remove_partition(i)
        if i in self.bs:
            del self.bs[i]

    def replace_partitions(self):
        return self._base.replace_partitions()

    def get_partition(self, i):
        return self._base.get_partition(i)

    def get_nested_partition(self, i):
        return self._base.get_nested_partition(i)

    def get_partitions(self):
        return self._base.get_partitions()

    def get_nested_partitions(self):
        return self._base.get_nested_partitions()

    def relabel(self):
        return self._base.relabel()

    def entropy(self):
        return self._base.entropy()

    def posterior_entropy(self, MLE=False):
        return self._base.posterior_entropy(MLE)

    def posterior_dev(self, MLE=False):
        return self._base.posterior_dev(MLE)

    def posterior_cerr(self, MLE=False):
        return self._base.posterior_cerror(MLE)

    def posterior_lprob(self, b, MLE=False):
        b = np.asarray(b, dtype="int32")
        return self._base.posterior_lprob(b, MLE)

    def get_coupled_state(self):
        base = self._base.get_coupled_state()
        if base is None:
            return None
        return PartitionModeState(bs=None, base=base, nested=self.nested)

    def get_marginal(self, g):
        bm = g.new_vp("vector<int>")
        self._base.get_marginal(g._Graph__graph, bm._get_any())
        return bm

    def get_map(self, g):
        b = g.new_vp("int")
        self._base.get_map(g._Graph__graph, b._get_any())
        return b

    def get_map_nested(self):
        return self._base.get_map_bs()

    def get_B(self):
        return self._base.get_B()

    def sample_partition(self, MLE=False):
        return self._base.sample_partition(MLE, _get_rng())

    def sample_nested_partition(self, MLE=False):
        return self._base.sample_nested_partition(MLE, _get_rng())

class ModeClusterState(object):
    def __init__(self, bs, b=None, B=1, nested=False, relabel=True):

        self.bs = []
        self.nested = nested

        for bv in bs:
            if self.nested:
                bv = [Vector_int32_t(init=x) for x in bv]
            else:
                bv = [Vector_int32_t(init=bv)]
            self.bs.append(bv)

        if b is None:
            self.b = np.random.randint(0, B, len(self.bs), dtype="int32")
        else:
            self.b = np.asarray(b, dtype="int32")

        self.relabel_init = relabel
        self.g = Graph()
        self.g.add_vertex(self.b.shape[0])
        self.bg = self.g
        self._abg = self.bg._get_any()
        self.obs = self.bs
        self._state = libinference.make_mode_cluster_state(self)

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
        if bs is None:
            if not self.nested:
                bs = [x[0] for x in self.bs]
            else:
                bs = self.bs
        return ModeClusterState(bs=bs, b=b if b is not None else self.b,
                                relabel=False, nested=self.nested)


    def __getstate__(self):
        return dict(bs=self.bs, b=self.b, nested=self.nested)

    def __setstate__(self, state):
        if not state["nested"]:
            state["bs"] = [x[0] for x in state["bs"]]
        self.__init__(**state, relabel=False)

    def get_mode(self, r):
        base = self._state.get_mode(r);
        return PartitionModeState(None, base=base, nested=self.nested)

    def get_modes(self):
        return [self.get_mode(r) for r in np.unique(self.b)]

    def get_wr(self):
        return np.array(np.bincount(self.b))

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
        return np.exp(-(w*np.log(w)).sum())

    def relabel(self, epsilon=1e-6, maxiter=100):
        return self._state.relabel_modes(epsilon, maxiter)

    def entropy(self):
        return self._state.entropy()

    def posterior_entropy(self, MLE=False):
        return self._state.posterior_entropy(MLE)

    def posterior_dev(self, MLE=False):
        return self._state.posterior_dev(MLE)

    def posterior_cerr(self, MLE=False):
        return self._state.posterior_cerror(MLE)

    def replace_partitions(self):
        return self._state.replace_partitions()

    def sample_partition(self, MLE=False):
        return self._state.sample_partition(MLE, _get_rng())

    def sample_nested_partition(self, MLE=False):
        return self._state.sample_nested_partition(MLE, _get_rng())

    def mcmc_sweep(self, beta=np.inf, d=.01, niter=1, entropy_args={},
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
            libinference.mode_clustering_mcmc_sweep(mcmc_state, self._state,
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


    def multiflip_mcmc_sweep(self, beta=np.inf, psingle=100, psplit=1,
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
            libinference.mode_clustering_multiflip_mcmc_sweep(mcmc_state,
                                                              self._state,
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
                    accept_stats[key] = np.zeros(len(nproposal),
                                                    dtype="uint64")
            accept_stats["proposal"] += nproposal.a
            accept_stats["acceptance"] += nacceptance.a

        return dS, nattempts, nmoves


def partition_overlap(x, y, norm=True):
    x = np.asarray(x, dtype="int32")
    y = np.asarray(y, dtype="int32")
    if len(x) != len(y):
        raise ValueError("x and y must be of the same length")
    m = libinference.partition_overlap(x, y)
    if norm:
        m /= len(x)
    return m

def contingency_graph(x, y):
    x = np.asarray(x, dtype="int32")
    y = np.asarray(y, dtype="int32")
    g = Graph(directed=False)
    g.ep.mrs = g.new_ep("int32_t")
    g.vp.label = g.new_vp("int32_t")
    g.vp.partition = g.new_vp("bool")
    libinference.get_contingency_graph(g._Graph__graph,
                                       g.vp.label._get_any(),
                                       g.ep.mrs._get_any(),
                                       g.vp.partition._get_any(),
                                       x, y)
    return g

def shuffle_partition_labels(x):
    x = np.asarray(x, dtype="int32").copy()
    libinference.partition_shuffle_labels(x, _get_rng())
    return x

def align_partition_labels(x, y):
    x = np.asarray(x, dtype="int32").copy()
    y = np.asarray(y, dtype="int32")
    libinference.align_partition_labels(x, y)
    return x


def overlap_center(bs, init=None):
    bs = np.asarray(bs, dtype="int")
    N = len(bs[0])
    if init is None:
        c = np.arange(0, N, dtype="int")
    else:
        c = init
    delta = 1
    while delta > 0:
        lc = c.copy()
        for b in bs:
            b[:] = align_partition_labels(b, c)
        for i in range(len(c)):
            nr = np.bincount(bs[:,i])
            c[i] = nr.argmax()
        delta = 1 - partition_overlap(c, lc)
    return c