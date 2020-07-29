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

from .. import _prop, Graph, GraphView, _get_rng, PropertyMap, \
    edge_endpoint_property

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_inference as libinference")

from . blockmodel import *
from . nested_blockmodel import *
from . blockmodel import _bm_test

import collections

def get_uentropy_args(kargs):
    ea = get_entropy_args(kargs, ignore=["latent_edges", "density"])
    uea = libinference.uentropy_args(ea)
    uea.latent_edges = kargs.get("latent_edges", True)
    uea.density = kargs.get("density", True)
    return uea

class UncertainBaseState(object):
    r"""Base state for uncertain network inference."""

    def __init__(self, g, nested=True, state_args={}, bstate=None,
                 self_loops=False, init_empty=False):

        self.g = g

        state_args = dict(state_args)
        if bstate is None:
            if init_empty:
                self.u = Graph(directed=g.is_directed())
                self.u.add_vertex(g.num_vertices())
                self.eweight = self.u.new_ep("int", val=1)
            elif "g" in state_args:
                self.u = state_args.pop("g")
                self.eweight = state_args.pop("eweight",
                                              self.u.new_ep("int", val=1))
            else:
                self.u = g.copy()
                self.eweight = self.u.new_ep("int", val=1)
        else:
            self.u = bstate.g.copy()
            if nested:
                self.eweight = bstate.levels[0].eweight
            else:
                self.eweight = bstate.eweight
            self.eweight = self.u.own_property(self.eweight.copy())
            if nested:
                bstate = bstate.copy(g=self.u,
                                     state_args=dict(bstate.state_args,
                                                     eweight=self.eweight))
            else:
                bstate = bstate.copy(g=self.u, eweight=self.eweight)

        self.u.set_fast_edge_removal()

        self.self_loops = self_loops
        N = self.u.num_vertices()
        if self.u.is_directed():
            if self_loops:
                M = N * N
            else:
                M = N * (N - 1)
        else:
            if self_loops:
                M = (N * (N + 1)) / 2
            else:
                M = (N * (N - 1)) / 2

        self.M = M

        if bstate is None:
            if nested:
                state_args["state_args"] = state_args.get("state_args", {})
                state_args["state_args"].update(dict(eweight=self.eweight))
                self.nbstate = NestedBlockState(self.u, **state_args)
                self.bstate = self.nbstate.levels[0]
            else:
                self.nbstate = None
                self.bstate = BlockState(self.u, eweight=self.eweight,
                                         **state_args)
        else:
            if nested:
                self.nbstate = bstate
                self.bstate = bstate.levels[0]
            else:
                self.nbstate = None
                self.bstate = bstate

        edges = self.g.get_edges()
        edges = numpy.concatenate((edges,
                                   numpy.ones(edges.shape,
                                              dtype=edges.dtype) * (N + 1)))
        self.slist = Vector_size_t(init=edges[:,0])
        self.tlist = Vector_size_t(init=edges[:,1])

        init_q_cache()

    def __getstate__(self):
        return dict(g=self.g, nested=self.nbstate is not None,
                    bstate=(self.nbstate if self.nbstate is not None else self.bstate),
                    self_loops=self.self_loops)

    def __setstate__(self, state):
        self.__init__(**state)

    def get_block_state(self):
        """Return the underlying block state, which can be either
        :class:`~graph_tool.inference.blockmodel.BlockState` or
        :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState`.
        """
        if self.nbstate is None:
            return self.bstate
        else:
            return self.nbstate

    def entropy(self, latent_edges=True, density=True, **kwargs):
        """Return the entropy, i.e. negative log-likelihood."""
        S = self._state.entropy(latent_edges, density)
        if self.nbstate is None:
            S += self.bstate.entropy(**kwargs)
        else:
            S += self.nbstate.entropy(**kwargs)

        if kwargs.get("test", True) and _bm_test():
            args = kwargs.copy()
            assert not isnan(S) and not isinf(S), \
                "invalid entropy %g (%s) " % (S, str(args))
            args["test"] = False
            state_copy = self.copy()
            Salt = state_copy.entropy(latent_edges, density, **args)

            assert math.isclose(S, Salt, abs_tol=1e-8), \
                "entropy discrepancy after copying (%g %g %g)" % (S, Salt,
                                                                  S - Salt)
        return S

    def virtual_remove_edge(self, u, v, entropy_args={}):
        dentropy_args = dict(self.bstate._entropy_args, **entropy_args)
        entropy_args = get_uentropy_args(dentropy_args)
        return self._state.remove_edge_dS(int(u), int(v), entropy_args)

    def virtual_add_edge(self, u, v, entropy_args={}):
        dentropy_args = dict(self.bstate._entropy_args, **entropy_args)
        entropy_args = get_uentropy_args(dentropy_args)
        return self._state.add_edge_dS(int(u), int(v), entropy_args)

    def set_state(self, g, w):
        if w.value_type() != "int32_t":
            w = w.copy("int32_t")
        self._state.set_state(g._Graph__graph, w._get_any())

    def _algo_sweep(self, algo, r=.5, **kwargs):
        kwargs = kwargs.copy()
        beta = kwargs.get("beta", 1.)
        niter = kwargs.get("niter", 1)
        edges_only = kwargs.pop("edges_only", False)
        verbose = kwargs.get("verbose", False)
        slist = self.slist
        tlist = self.tlist
        dentropy_args = dict(self.bstate._entropy_args,
                             **kwargs.get("entropy_args", {}))
        entropy_args = get_uentropy_args(dentropy_args)
        kwargs.get("entropy_args", {}).pop("latent_edges", None)
        kwargs.get("entropy_args", {}).pop("density", None)
        state = self._state

        mcmc_state = DictState(dict(kwargs, **locals()))

        kwargs.pop("xlog", None)
        kwargs.pop("xstep", None)
        kwargs.pop("xdefault", None)

        if _bm_test():
            Si = self.entropy(**dentropy_args)

        if self.nbstate is None:
            self.bstate._clear_egroups()
        else:
            self.nbstate._clear_egroups()
        if numpy.random.random() < r:
            edges = True
            dS, nattempts, nmoves = self._mcmc_sweep(mcmc_state)
        else:
            edges = False
            if self.nbstate is None:
                dS, nattempts, nmoves = algo(self.bstate, **kwargs)
            else:
                dS, nattempts, nmoves = algo(self.nbstate, **kwargs)

        if _bm_test():
            Sf = self.entropy(**dentropy_args)
            assert math.isclose(dS, (Sf - Si), abs_tol=1e-8), \
                "inconsistent entropy delta %g (%g): %s %s" % (dS, Sf - Si, edges,
                                                               str(dentropy_args))

        return dS, nattempts, nmoves

    def mcmc_sweep(self, r=.5, multiflip=True, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection sampling MCMC to
        sample network partitions and latent edges. The parameter ``r`` controls
        the probability with which edge move will be attempted, instead of
        partition moves. The remaining keyword parameters will be passed to
        :meth:`~graph_tool.inference.blockmodel.BlockState.mcmc_sweep` or
        :meth:`~graph_tool.inference.blockmodel.BlockState.multiflip_mcmc_sweep`,
        if ``multiflip=True``.
        """

        if multiflip:
            return self._algo_sweep(lambda s, **kw: s.multiflip_mcmc_sweep(**kw),
                                    r=r, **kwargs)
        else:
            return self._algo_sweep(lambda s, **kw: s.mcmc_sweep(**kw),
                                    r=r, **kwargs)

    def multiflip_mcmc_sweep(self, **kwargs):
        r"""Alias for :meth:`~UncertainBaseState.mcmc_sweep` with ``multiflip=True``."""
        return self.mcmc_sweep(multiflip=True, **kwargs)

    def get_edge_prob(self, u, v, entropy_args={}, epsilon=1e-8):
        r"""Return conditional posterior log-probability of edge :math:`(u,v)`."""
        entropy_args = dict(self.bstate._entropy_args, **entropy_args)
        ea = get_uentropy_args(entropy_args)
        return self._state.get_edge_prob(u, v, ea, epsilon)

    def get_edges_prob(self, elist, entropy_args={}, epsilon=1e-8):
        r"""Return conditional posterior log-probability of an edge list, with
        shape :math:`(E,2)`."""
        entropy_args = dict(self.bstate._entropy_args, **entropy_args)
        ea = get_uentropy_args(entropy_args)
        elist = numpy.asarray(elist, dtype="uint64")
        probs = numpy.zeros(elist.shape[0])
        self._state.get_edges_prob(elist, probs, ea, epsilon)
        return probs

    def get_graph(self):
        r"""Return the current inferred graph."""
        if self.self_loops:
            u = GraphView(self.u, efilt=self.eweight.fa > 0)
        else:
            es = edge_endpoint_property(self.u, self.u.vertex_index, "source")
            et = edge_endpoint_property(self.u, self.u.vertex_index, "target")
            u = GraphView(self.u, efilt=numpy.logical_and(self.eweight.fa > 0,
                                                          es.fa != et.fa))
        return u

    def collect_marginal(self, g=None):
        r"""Collect marginal inferred network during MCMC runs.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph` (optional, default: ``None``)
            Previous marginal graph.

        Returns
        -------
        g : :class:`~graph_tool.Graph`
            New marginal graph, with internal edge :class:`~graph_tool.EdgePropertyMap`
            ``"eprob"``, containing the marginal probabilities for each edge.

        Notes
        -----
        The posterior marginal probability of an edge :math:`(i,j)` is defined as

        .. math::

           \pi_{ij} = \sum_{\boldsymbol A}A_{ij}P(\boldsymbol A|\boldsymbol D)

        where :math:`P(\boldsymbol A|\boldsymbol D)` is the posterior
        probability given the data.

        """

        if g is None:
            g = Graph(directed=self.g.is_directed())
            g.add_vertex(self.g.num_vertices())
            g.gp.count = g.new_gp("int", 0)
            g.ep.count = g.new_ep("int")

        if "eprob" not in g.ep:
            g.ep.eprob = g.new_ep("double")

        u = self.get_graph()
        libinference.collect_marginal(g._Graph__graph,
                                      u._Graph__graph,
                                      _prop("e", g, g.ep.count))
        g.gp.count += 1
        g.ep.eprob.fa = g.ep.count.fa
        g.ep.eprob.fa /= g.gp.count
        return g

    def collect_marginal_multigraph(self, g=None):
        r"""Collect marginal latent multigraph during MCMC runs.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph` (optional, default: ``None``)
            Previous marginal multigraph.

        Returns
        -------
        g : :class:`~graph_tool.Graph`
            New marginal graph, with internal edge
            :class:`~graph_tool.EdgePropertyMap` ``"w"`` and ``"wcount"``,
            containing the edge multiplicities and their respective counts.

        Notes
        -----

        The mean posterior marginal multiplicity distribution of a multi-edge
        :math:`(i,j)` is defined as

        .. math::

           \pi_{ij}(w) = \sum_{\boldsymbol G}\delta_{w,G_{ij}}P(\boldsymbol G|\boldsymbol D)

        where :math:`P(\boldsymbol G|\boldsymbol D)` is the posterior
        probability of a multigraph :math:`\boldsymbol G` given the data.

        """

        if g is None:
            g = Graph(directed=self.g.is_directed())
            g.add_vertex(self.g.num_vertices())
            g.ep.w = g.new_ep("vector<int>")
            g.ep.wcount = g.new_ep("vector<int>")

        libinference.collect_marginal_count(g._Graph__graph,
                                            self.u._Graph__graph,
                                            _prop("e", self.u, self.eweight),
                                            _prop("e", g, g.ep.w),
                                            _prop("e", g, g.ep.wcount))
        return g

class UncertainBlockState(UncertainBaseState):
    r"""Inference state of an uncertain graph, using the stochastic block model as a
    prior.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Measured graph.
    q : :class:`~graph_tool.EdgePropertyMap`
        Edge probabilities in range :math:`[0,1]`.
    q_default : ``float`` (optional, default: ``0.``)
        Non-edge probability in range :math:`[0,1]`.
    aE : ``float`` (optional, default: ``NaN``)
        Expected total number of edges used in prior. If ``NaN``, a flat
        prior will be used instead.
    nested : ``boolean`` (optional, default: ``True``)
        If ``True``, a :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState`
        will be used, otherwise
        :class:`~graph_tool.inference.blockmodel.BlockState`.
    state_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to
        :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or
        :class:`~graph_tool.inference.blockmodel.BlockState`.
    bstate : :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or :class:`~graph_tool.inference.blockmodel.BlockState`  (optional, default: ``None``)
        If passed, this will be used to initialize the block state
        directly.
    self_loops : bool (optional, default: ``False``)
        If ``True``, it is assumed that the uncertain graph can contain
        self-loops.

    References
    ----------
    .. [peixoto-reconstructing-2018] Tiago P. Peixoto, "Reconstructing
       networks with unknown and heterogeneous errors", Phys. Rev. X 8
       041011 (2018). :doi:`10.1103/PhysRevX.8.041011`, :arxiv:`1806.07956`
    """

    def __init__(self, g, q, q_default=0., aE=numpy.nan, nested=True, state_args={},
                 bstate=None, self_loops=False, **kwargs):

        super().__init__(g, nested=nested, state_args=state_args, bstate=bstate,
                         self_loops=self_loops, **kwargs)
        self._q = q
        self._q_default = q_default

        self.p = (q.fa.sum() + (self.M - g.num_edges()) * q_default) / self.M

        self.q = self.g.new_ep("double", vals=log(q.fa) - log1p(-q.fa))
        self.q.fa -= log(self.p) - log1p(-self.p)
        if q_default > 0:
            self.q_default = log(q_default) - log1p(q_default)
            self.q_default -= log(self.p) - log1p(-self.p)
        else:
            self.q_default = -numpy.inf

        self.S_const = (log1p(-q.fa[q.fa<1]).sum() +
                        log1p(-q_default) * (self.M - self.g.num_edges())
                        - self.M * log1p(-self.p))

        self.aE = aE
        if numpy.isnan(aE):
            self.E_prior = False
        else:
            self.E_prior = True

        self._state = libinference.make_uncertain_state(self.bstate._state,
                                                        self)
    def __getstate__(self):
        state = super().__getstate__()
        return dict(state,  q=self._q, q_default=self._q_default,
                    aE=self.aE)

    def copy(self, **kwargs):
        """Return a copy of the state."""
        return UncertainBlockState(**dict(self.__getstate__(), **kwargs))

    def __copy__(self):
        return self.copy()

    def __repr__(self):
        return "<UncertainBlockState object with %s, at 0x%x>" % \
            (self.nbstate if self.nbstate is not None else self.bstate,
             id(self))

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_uncertain_sweep(mcmc_state,
                                                 self._state,
                                                 _get_rng())

class LatentMultigraphBlockState(UncertainBaseState):
    r"""Inference state of an erased Poisson multigraph, using the stochastic
    block model as a prior.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Measured graph.
    aE : ``float`` (optional, default: ``NaN``)
        Expected total number of edges used in prior. If ``NaN``, a flat
        prior will be used instead.
    nested : ``boolean`` (optional, default: ``True``)
        If ``True``, a :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState`
        will be used, otherwise
        :class:`~graph_tool.inference.blockmodel.BlockState`.
    state_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to
        :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or
        :class:`~graph_tool.inference.blockmodel.BlockState`.
    bstate : :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or :class:`~graph_tool.inference.blockmodel.BlockState`  (optional, default: ``None``)
        If passed, this will be used to initialize the block state
        directly.
    self_loops : bool (optional, default: ``False``)
        If ``True``, it is assumed that the uncertain graph can contain
        self-loops.

    References
    ----------
    .. [peixoto-latent-2020] Tiago P. Peixoto, "Latent Poisson models for
       networks with heterogeneous density", :arxiv:`2002.07803`
    """

    def __init__(self, g, aE=numpy.nan, nested=True, state_args={},
                 bstate=None, self_loops=False, **kwargs):

        super().__init__(g, nested=nested, state_args=state_args, bstate=bstate,
                         self_loops=self_loops, **kwargs)

        self.q = self.g.new_ep("double", val=numpy.inf)
        self.q_default = -numpy.inf
        self.S_const = 0

        self.aE = aE
        if numpy.isnan(aE):
            self.E_prior = False
        else:
            self.E_prior = True

        self._state = libinference.make_uncertain_state(self.bstate._state,
                                                        self)
    def __getstate__(self):
        state = super().__getstate__()
        return dict(state, aE=self.aE)

    def __setstate__(self, state):
        self.__init__(**state)

    def copy(self, **kwargs):
        """Return a copy of the state."""
        return LatentMultigraphBlockState(**dict(self.__getstate__(), **kwargs))

    def __copy__(self):
        return self.copy()

    def __repr__(self):
        return "<LatentMultigraphBlockState object with %s, at 0x%x>" % \
            (self.nbstate if self.nbstate is not None else self.bstate,
             id(self))

    def _mcmc_sweep(self, mcmc_state):
        mcmc_state.edges_only = True
        return libinference.mcmc_uncertain_sweep(mcmc_state,
                                                 self._state,
                                                 _get_rng())

class MeasuredBlockState(UncertainBaseState):
    r"""Inference state of a measured graph, using the stochastic block model as a
    prior.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Measured graph.
    n : :class:`~graph_tool.EdgePropertyMap`
        Edge property map of type ``int``, containing the total number of
        measurements for each edge.
    x : :class:`~graph_tool.EdgePropertyMap`
        Edge property map of type ``int``, containing the number of
        positive measurements for each edge.
    n_default : ``int`` (optional, default: ``1``)
        Total number of measurements for each non-edge.
    x_default : ``int`` (optional, default: ``0``)
        Total number of positive measurements for each non-edge.
    fn_params : ``dict`` (optional, default: ``dict(alpha=1, beta=1)``)
        Beta distribution hyperparameters for the probability of missing
        edges (false negatives).
    fp_params : ``dict`` (optional, default: ``dict(mu=1, nu=1)``)
        Beta distribution hyperparameters for the probability of spurious
        edges (false positives).
    aE : ``float`` (optional, default: ``NaN``)
        Expected total number of edges used in prior. If ``NaN``, a flat
        prior will be used instead.
    nested : ``boolean`` (optional, default: ``True``)
        If ``True``, a :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState`
        will be used, otherwise
        :class:`~graph_tool.inference.blockmodel.BlockState`.
    state_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to
        :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or
        :class:`~graph_tool.inference.blockmodel.BlockState`.
    bstate : :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or :class:`~graph_tool.inference.blockmodel.BlockState`  (optional, default: ``None``)
        If passed, this will be used to initialize the block state
        directly.
    self_loops : bool (optional, default: ``False``)
        If ``True``, it is assumed that the uncertain graph can contain
        self-loops.

    References
    ----------
    .. [peixoto-reconstructing-2018] Tiago P. Peixoto, "Reconstructing
       networks with unknown and heterogeneous errors", Phys. Rev. X 8
       041011 (2018). :doi:`10.1103/PhysRevX.8.041011`, :arxiv:`1806.07956`
    """

    def __init__(self, g, n, x, n_default=1, x_default=0,
                 fn_params=dict(alpha=1, beta=1), fp_params=dict(mu=1, nu=1),
                 aE=numpy.nan, nested=True, state_args={}, bstate=None,
                 self_loops=False, **kwargs):

        super().__init__(g, nested=nested, state_args=state_args, bstate=bstate,
                         **kwargs)

        self.aE = aE
        if numpy.isnan(aE):
            self.E_prior = False
        else:
            self.E_prior = True

        self.n = n
        self.x = x
        self.n_default = n_default
        self.x_default = x_default
        self.alpha = fn_params.get("alpha", 1)
        self.beta = fn_params.get("beta", 1)
        self.mu = fp_params.get("mu", 1)
        self.nu = fp_params.get("nu", 1)

        self._state = libinference.make_measured_state(self.bstate._state,
                                                       self)

    def __getstate__(self):
        state = super().__getstate__()
        return dict(state, n=self.n, x=self.x, n_default=self.n_default,
                    x_default=self.x_default,
                    fn_params=dict(alpha=self.alpha, beta=self.beta),
                    fp_params=dict(mu=self.mu, nu=self.nu), aE=self.aE)

    def copy(self, **kwargs):
        """Return a copy of the state."""
        return MeasuredBlockState(**dict(self.__getstate__(), **kwargs))

    def __repr__(self):
        return "<MeasuredBlockState object with %s, at 0x%x>" % \
            (self.nbstate if self.nbstate is not None else self.bstate,
             id(self))

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_measured_sweep(mcmc_state,
                                                self._state,
                                                _get_rng())

    def set_hparams(self, alpha, beta, mu, nu):
        """Set edge and non-edge hyperparameters."""
        self._state.set_hparams(alpha, beta, mu, nu)
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nu = nu

    def get_p_posterior(self):
        """Get beta distribution parameters for the posterior probability of missing edges."""
        T = self._state.get_T()
        M = self._state.get_M()
        return M - T + self.alpha, T + self.beta

    def get_q_posterior(self):
        """Get beta distribution parameters for the posterior probability of spurious edges."""
        N = self._state.get_N()
        X = self._state.get_X()
        T = self._state.get_T()
        M = self._state.get_M()
        return X - T + self.mu, N - X - (M - T) + self.nu

class MixedMeasuredBlockState(UncertainBaseState):
    r"""Inference state of a measured graph with heterogeneous errors, using the
    stochastic block model as a prior.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Measured graph.
    n : :class:`~graph_tool.EdgePropertyMap`
        Edge property map of type ``int``, containing the total number of
        measurements for each edge.
    x : :class:`~graph_tool.EdgePropertyMap`
        Edge property map of type ``int``, containing the number of
        positive measurements for each edge.
    n_default : ``int`` (optional, default: ``1``)
        Total number of measurements for each non-edge.
    x_default : ``int`` (optional, default: ``1``)
        Total number of positive measurements for each non-edge.
    fn_params : ``dict`` (optional, default: ``dict(alpha=1, beta=10)``)
        Beta distribution hyperparameters for the probability of missing
        edges (false negatives).
    fp_params : ``dict`` (optional, default: ``dict(mu=1, nu=10)``)
        Beta distribution hyperparameters for the probability of spurious
        edges (false positives).
    aE : ``float`` (optional, default: ``NaN``)
        Expected total number of edges used in prior. If ``NaN``, a flat
        prior will be used instead.
    nested : ``boolean`` (optional, default: ``True``)
        If ``True``, a :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState`
        will be used, otherwise
        :class:`~graph_tool.inference.blockmodel.BlockState`.
    state_args : ``dict`` (optional, default: ``{}``)
        Arguments to be passed to
        :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or
        :class:`~graph_tool.inference.blockmodel.BlockState`.
    bstate : :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or :class:`~graph_tool.inference.blockmodel.BlockState`  (optional, default: ``None``)
        If passed, this will be used to initialize the block state
        directly.
    self_loops : bool (optional, default: ``False``)
        If ``True``, it is assumed that the uncertain graph can contain
        self-loops.

    References
    ----------
    .. [peixoto-reconstructing-2018] Tiago P. Peixoto, "Reconstructing
       networks with unknown and heterogeneous errors", Phys. Rev. X 8
       041011 (2018). :doi:`10.1103/PhysRevX.8.041011`, :arxiv:`1806.07956`
    """

    def __init__(self, g, n, x, n_default=1, x_default=0,
                 fn_params=dict(alpha=1, beta=10), fp_params=dict(mu=1, nu=10),
                 aE=numpy.nan, nested=True, state_args={}, bstate=None,
                 self_loops=False, **kwargs):

        super().__init__(g, nested=nested, state_args=state_args, bstate=bstate,
                         **kwargs)
        self.aE = aE
        if numpy.isnan(aE):
            self.E_prior = False
        else:
            self.E_prior = True

        self.n = n
        self.x = x
        self.n_default = n_default
        self.x_default = x_default
        self.alpha = fn_params.get("alpha", 1)
        self.beta = fn_params.get("beta", 10)
        self.mu = fp_params.get("mu", 1)
        self.nu = fp_params.get("nu", 10)

        self._state = None

        self.q = self.g.new_ep("double")
        self.sync_q()

        self._state = libinference.make_uncertain_state(self.bstate._state,
                                                        self)

    def sync_q(self):
        ra, rb = self.transform(self.n.fa, self.x.fa)
        self.q.fa = ra - rb
        dra, drb = self.transform(self.n_default, self.x_default)
        self.q_default = dra - drb

        self.S_const = (self.M - self.g.num_edges()) * drb + rb.sum()

        if self._state is not None:
            self._state.set_q_default(self.q_default)
            self._state.set_S_const(self.S_const)

    def transform(self, na, xa):
        ra = (scipy.special.betaln(na - xa + self.alpha, xa + self.beta) -
              scipy.special.betaln(self.alpha, self.beta))
        rb = (scipy.special.betaln(xa + self.mu, na - xa + self.nu) -
              scipy.special.betaln(self.mu, self.nu))
        return ra, rb

    def set_hparams(self, alpha, beta, mu, nu):
        """Set edge and non-edge hyperparameters."""
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.nu = nu
        self.sync_q()

    def __getstate__(self):
        state = super().__getstate__()
        return dict(state, n=self.n, x=self.x, n_default=self.n_default,
                    x_default=self.x_default,
                    fn_params=dict(alpha=self.alpha, beta=self.beta),
                    fp_params=dict(mu=self.mu, nu=self.nu), aE=self.aE)

    def copy(self, **kwargs):
        """Return a copy of the state."""
        return MixedMeasuredBlockState(**dict(self.__getstate__(), **kwargs))

    def __copy__(self):
        return self.copy()

    def __setstate__(self, state):
        self.__init__(**state)

    def __repr__(self):
        return "<MixedMeasuredBlockState object with %s, at 0x%x>" % \
            (self.nbstate if self.nbstate is not None else self.bstate,
             id(self))

    def mcmc_sweep(self, r=.5, h=.1, hstep=1, multiflip=True, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection sampling MCMC to
        sample network partitions and latent edges. The parameter ``r`` controls
        the probability with which edge move will be attempted, instead of
        partition moves. The parameter ``h`` controls the relative probability
        with which hyperparamters moves will be attempted, and ``hstep`` is the
        size of the step.

        The remaining keyword parameters will be passed to
        :meth:`~graph_tool.inference.blockmodel.BlockState.mcmc_sweep` or
        :meth:`~graph_tool.inference.blockmodel.BlockState.multiflip_mcmc_sweep`,
        if ``multiflip=True``.
        """

        return super().mcmc_sweep(r=r, multiflip=multiflip, h=h, hstep=hstep,
                                  **kwargs)

    def _algo_sweep(self, algo, r=.5, h=.1, hstep=1, niter=1, **kwargs):
        if numpy.random.random() < h:
            dS = nt = nm = 0
            for i in range(niter):
                hs = [self.alpha, self.beta, self.mu, self.nu]
                j = numpy.random.randint(len(hs))

                f_dh = [max(0, hs[j] - hstep), hs[j] + hstep]
                pf = 1./(f_dh[1] - f_dh[0])

                old_hs = hs[j]
                hs[j] = f_dh[0] + numpy.random.random() * (f_dh[1] - f_dh[0])

                b_dh = [max(0, hs[j] - hstep), hs[j] + hstep]
                pb = 1./min(1, hs[j])

                latent_edges = kwargs.get("entropy_args", {}).get("latent_edges", True)
                density = False

                Sb = self._state.entropy(latent_edges, density)
                self.set_hparams(*hs)
                Sa = self._state.entropy(latent_edges, density)

                nt += 1
                if Sa < Sb or numpy.random.random() < exp(-(Sa-Sb) + log(pb) - log(pf)):
                    dS += Sa - Sb
                    nm +=1
                else:
                    hs[j] = old_hs
                    self.set_hparams(*hs)
            return (dS, nt, nm)
        else:
            return super()._algo_sweep(algo, r, niter=niter, **kwargs)

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_uncertain_sweep(mcmc_state,
                                                 self._state,
                                                 _get_rng())

class DynamicsBlockStateBase(UncertainBaseState):
    def __init__(self, g, s, t, x=None, aE=numpy.nan, nested=True,
                 state_args={}, bstate=None, self_loops=False,
                 **kwargs):
        r"""Base state for network reconstruction based on dynamical data, using
        the stochastic block model as a prior. This class is not meant to be
        instantiated directly, only indirectly via one of its subclasses."""

        super().__init__(g, nested=nested, state_args=state_args, bstate=bstate,
                         self_loops=self_loops)
        self.s = [g.own_property(x) for x in s]
        self.t = [g.own_property(x) for x in t]
        if x is None:
            x = self.u.new_ep("double")
        else:
            x = self.u.copy_property(x, g=x.get_graph())
        self.x = x

        self.aE = aE
        if numpy.isnan(aE):
            self.E_prior = False
        else:
            self.E_prior = True

        for k in kwargs.keys():
            v = kwargs[k]
            if isinstance(v, PropertyMap):
                kwargs[k] = g.own_property(v)
            elif (isinstance(v, collections.Iterable) and len(v) > 0 and
                  isinstance(v[0], PropertyMap)):
                kwargs[k] = [g.own_property(x) for x in v]
        self.params = kwargs
        self.os = [ns._get_any() for ns in s]
        self.ot = [nt._get_any() for nt in t]
        self._state = self._make_state()

    def set_params(self, params):
        r"""Sets the model parameters via the dictionary ``params``."""
        self.params = dict(self.params, **params)
        self._state.set_params(self.params)

    def __getstate__(self):
        state = super().__getstate__()
        return dict(state, s=self.s, t=self.t, x=self.x, aE=self.aE,
                    **self.params)

    def copy(self, **kwargs):
        """Return a copy of the state."""
        return type(self)(**dict(self.__getstate__(), **kwargs))

    def __copy__(self):
        return self.copy()

    def __repr__(self):
        return "<%s object with %s, at 0x%x>" % \
            (self.__class__.__name__,
             self.nbstate if self.nbstate is not None else self.bstate,
             id(self))

    def _move_proposal(self, name, beta, step, rg, transform, entropy_args):
        x = x_orig = self.params[name]

        if isinstance(x, collections.Iterable):
            idx = numpy.random.randint(len(x))
            x = x[idx]
        else:
            idx = None

        if transform is not None:
            x = transform[1](x)

        if rg is not None:
            mi = max(rg[0], x - step)
            ma = min(rg[1], x + step)
        else:
            mi = x - step
            ma = x + step
        nx = numpy.random.random() * (ma - mi) + mi

        a = 0
        if rg is not None:
            a -= -log(ma - mi)
            mi = max(rg[0], nx - step)
            ma = min(rg[1], nx + step)
            a += -log(ma - mi)

        if transform is not None:
            nx = transform[0](nx)

        latent_edges = entropy_args.get("latent_edges", True)
        density = False

        Sb = self._state.entropy(latent_edges, density)

        if idx is not None:
            y = self.params[name]
            y[idx] = nx
            nx = y

        self.set_params({name:nx})
        Sa = self._state.entropy(latent_edges, density)

        a += beta * (Sb - Sa)

        if a > 0 or numpy.random.random() < exp(a):
            self.set_params({name:nx})
            return Sa-Sb, 1, 1

        if idx is not None:
            y = self.params[name]
            y[idx] = x_orig
            x_orig = y
        self.set_params({name:x_orig})

        return 0, 0, 0

    def get_x(self):
        """Return latent edge covariates."""
        return self.x

    def get_edge_prob(self, u, v, x, entropy_args={}, epsilon=1e-8):
        r"""Return conditional posterior log-probability of edge :math:`(u,v)`."""
        entropy_args = dict(self.bstate._entropy_args, **entropy_args)
        ea = get_uentropy_args(entropy_args)
        return self._state.get_edge_prob(u, v, x, ea, epsilon)

    def collect_marginal(self, g=None):
        r"""Collect marginal inferred network during MCMC runs.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph` (optional, default: ``None``)
            Previous marginal graph.

        Returns
        -------
        g : :class:`~graph_tool.Graph`
            New marginal graph, with internal edge :class:`~graph_tool.EdgePropertyMap`
            ``"eprob"``, containing the marginal probabilities for each edge.

        Notes
        -----
        The posterior marginal probability of an edge :math:`(i,j)` is defined as

        .. math::

           \pi_{ij} = \sum_{\boldsymbol A}A_{ij}P(\boldsymbol A|\boldsymbol D)

        where :math:`P(\boldsymbol A|\boldsymbol D)` is the posterior
        probability given the data.

        """

        if g is None:
            g = Graph(directed=self.g.is_directed())
            g.add_vertex(self.g.num_vertices())
            g.gp.count = g.new_gp("int", 0)
            g.ep.count = g.new_ep("int")
            g.ep.eprob = g.new_ep("double")

        if "x" not in g.ep:
            g.ep.xsum = g.new_ep("double")
            g.ep.x2sum = g.new_ep("double")
            g.ep.x = g.new_ep("double")
            g.ep.xdev = g.new_ep("double")

        u = self.get_graph()
        x = self.get_x()
        libinference.collect_xmarginal(g._Graph__graph,
                                       u._Graph__graph,
                                       _prop("e", u, x),
                                       _prop("e", g, g.ep.count),
                                       _prop("e", g, g.ep.xsum),
                                       _prop("e", g, g.ep.x2sum))
        g.gp.count += 1
        g.ep.eprob.fa = g.ep.count.fa / g.gp.count
        g.ep.x.fa = g.ep.xsum.fa / g.gp.count
        g.ep.xdev.fa = sqrt(g.ep.x2sum.fa / g.gp.count - g.ep.x.fa ** 2)
        return g

class EpidemicsBlockState(DynamicsBlockStateBase):
    def __init__(self, g, s, beta, r, r_v=None, global_beta=None, active=None,
                 t=[], exposed=False, aE=numpy.nan, nested=True, state_args={},
                 bstate=None, self_loops=False, **kwargs):
        r"""Inference state for network reconstruction based on epidemic dynamics, using
        the stochastic block model as a prior.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
            Initial graph state.
        s : ``list`` of :class:`~graph_tool.VertexPropertyMap`
            Collection of time-series with node states over time. Each entry in
            this list must be a :class:`~graph_tool.VertexPropertyMap` with type
            ``vector<int>`` containing the states of each node in each time
            step. A value of ``1`` means infected and ``0`` susceptible. Other
            values are allowed (e.g. for recovered), but their actual value is
            unimportant for reconstruction.

            If the parameter ``t`` below is given, each property map value for a
            given node should contain only the states for the same points in
            time given by that parameter.
        beta : ``float`` or :class:`~graph_tool.EdgePropertyMap`
            Initial value of the global or local transmission probability for
            each edge.
        r : ``float``
            Spontaneous infection probability.
        r_v : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            If given, this will set the initial spontaneous infection
            probability for each node, and trigger the use of a model where this
            quantity is in principle different for each node.
        global_beta : ``float`` (optional, default: ``None``)
            If provided, and ``beta is None`` this will trigger the use of a
            model where all transmission probabilities on edges are the same,
            and given (initially) by this value.
        t : ``list`` of :class:`~graph_tool.VertexPropertyMap` (optional, default: ``[]``)
            If nonempty, this allows for a compressed representation of the
            time-series parameter ``s``, corresponding only to points in time
            where the state of each node changes. Each each entry in this list
            must be a :class:`~graph_tool.VertexPropertyMap` with type
            ``vector<int>`` containing the points in time where the state of
            each node change. The corresponding state of the nodes at these
            times are given by parameter ``s``.
        active : ``list`` of :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            If given, this specifies the points in time where each node is
            "active", and prepared to change its state according to the state of
            its neighbors. Each entry in this list must be a
            :class:`~graph_tool.VertexPropertyMap` with type ``vector<int>``
            containing the states of each node in each time step. A value of
            ``1`` means active and ``0`` inactive.
        exposed : ``boolean`` (optional, default: ``False``)
            If ``True``, the data is supposed to come from a SEI, SEIR,
            etc. model, where a susceptible node (valued ``0``) first transits
            to an exposed state (valued ``-1``) upon transmission, before
            transiting to the infective state (valued ``1``).
        aE : ``float`` (optional, default: ``NaN``)
            Expected total number of edges used in prior. If ``NaN``, a flat
            prior will be used instead.
        nested : ``boolean`` (optional, default: ``True``)
            If ``True``, a :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState`
            will be used, otherwise
            :class:`~graph_tool.inference.blockmodel.BlockState`.
        state_args : ``dict`` (optional, default: ``{}``)
            Arguments to be passed to
            :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or
            :class:`~graph_tool.inference.blockmodel.BlockState`.
        bstate : :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or :class:`~graph_tool.inference.blockmodel.BlockState`  (optional, default: ``None``)
            If passed, this will be used to initialize the block state
            directly.
        self_loops : bool (optional, default: ``False``)
            If ``True``, it is assumed that the inferred graph can contain
            self-loops.

        References
        ----------
        .. [peixoto-network-2019] Tiago P. Peixoto, "Network reconstruction and
           community detection from dynamics", Phys. Rev. Lett. 123 128301
           (2019), :doi:`10.1103/PhysRevLett.123.128301`, :arxiv:`1903.10833`
        """

        if beta is None:
            beta = global_beta
        x = kwargs.pop("x", None)
        if x is None:
            if bstate is not None:
                if nested:
                    u = bstate.levels[0].g
                else:
                    u = bstate.g
            else:
                u = g
            if isinstance(beta, PropertyMap):
                x = u.own_property(beta.copy())
            else:
                x = u.new_ep("double", val=beta)
            x.fa = log1p(-x.fa)
        super().__init__(g, s=s, t=t, global_beta=global_beta, active=active,
                         x=x, r=r, r_v=r_v, exposed=exposed, aE=aE,
                         nested=nested, state_args=state_args, bstate=bstate,
                         self_loops=self_loops, **kwargs)

    def _make_state(self):
        return libinference.make_epidemics_state(self.bstate._state, self)

    def __setstate__(self, state):
        beta = state.pop("beta", None)
        if "x" not in state:
            state["global_beta"] = beta
        self.__init__(**dict(state, beta=beta))

    def copy(self, **kwargs):
        """Return a copy of the state."""
        return type(self)(**dict(self.__getstate__(),
                                 **dict(kwargs, beta=kwargs.get("beta", None))))

    def set_params(self, params):
        r"""Sets the model parameters via the dictionary ``params``."""
        self.params = dict(self.params, **params)
        self._state.set_params(self.params)
        beta = self.params["global_beta"]
        if beta is not None:
            self.x.fa = log1p(-beta)
            self._state.reset_m()

    def get_x(self):
        """Return latent edge transmission probabilities."""
        x = self.x.copy()
        x.fa = 1-exp(x.fa)
        return x

    def mcmc_sweep(self, r=.5, p=.1, pstep=.1, h=.1, hstep=1, xstep=.1,
                   multiflip=True, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection sampling MCMC to
        sample network partitions and latent edges. The parameter ``r`` controls
        the probability with which edge move will be attempted, instead of
        partition moves. The parameter ``h`` controls the relative probability
        with which moves for the parameters ``r_v`` will be attempted, and
        ``hstep`` is the size of the step. The parameter ``p`` controls the
        relative probability with which moves for the parameters ``global_beta``
        and ``r`` will be attempted, and ``pstep`` is the size of the step.  The
        paramter ``xstep`` determines the size of the attempted steps for the
        edge transmission probabilities.

        The remaining keyword parameters will be passed to
        :meth:`~graph_tool.inference.blockmodel.BlockState.mcmc_sweep` or
        :meth:`~graph_tool.inference.blockmodel.BlockState.multiflip_mcmc_sweep`,
        if ``multiflip=True``.

        """

        return super().mcmc_sweep(r=r, p=p, pstep=p, h=h, hstep=hstep,
                                  xstep=xstep, multiflip=multiflip, **kwargs)

    def _algo_sweep(self, algo, r=.5, p=.1, pstep=.1, h=.1, hstep=1,
                    xstep=.1, niter=1, **kwargs):
        if numpy.random.random() < p:
            if self.params["r_v"] is None:
                h = 0
            if numpy.random.random() < h:
                mcmc_state = DictState(dict(beta=kwargs.get("beta", 1),
                                            hstep=hstep,
                                            verbose=kwargs.get("verbose", False),
                                            niter=niter,
                                            state=self._state))
                return libinference.mcmc_epidemics_sweep_r(mcmc_state,
                                                           self._state,
                                                           _get_rng())
            else:
                dS = nt = nm = 0
                for i in range(niter):
                    if self.params["global_beta"] is not None:
                        ret = self._move_proposal("global_beta",
                                                  kwargs.get("beta", 1),
                                                  pstep, (0, 1),
                                                  None,
                                                  kwargs.get("entropy_args", {}))
                        dS += ret[0]
                        nt += ret[1]
                        nm += ret[2]
                    ret = self._move_proposal("r",
                                              kwargs.get("beta", 1),
                                              pstep, (0, 1),
                                              None,
                                              kwargs.get("entropy_args", {}))
                    dS += ret[0]
                    nt += ret[1]
                    nm += ret[2]
                return (dS, nt, nm)
        else:
            beta = self.params["global_beta"]
            if beta is not None:
                xdefault = log1p(-beta)
            else:
                xdefault = 0
            return super()._algo_sweep(algo, r, xlog=True, xstep=xstep,
                                       xdefault=xdefault, niter=niter, **kwargs)

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_epidemics_sweep(mcmc_state, self._state,
                                                 _get_rng())

class IsingBaseBlockState(DynamicsBlockStateBase):
    def __init__(self, g, s, beta, x=None, h=None, t=None, aE=numpy.nan,
                 nested=True, state_args={}, bstate=None, self_loops=False,
                 has_zero=False, **kwargs):
        r"""Base state for network reconstruction based on the Ising model, using the
        stochastic block model as a prior. This class is not supposed to be
        instantiated directly. Instead one of its specialized subclasses must
        be used, which have the same signature: :class:`IsingGlauberBlockState`,
        :class:`PseudoIsingBlockState`, :class:`CIsingGlauberBlockState`,
        :class:`PseudoCIsingBlockState`.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
            Initial graph state.
        s : ``list`` of :class:`~graph_tool.VertexPropertyMap` or :class:`~graph_tool.VertexPropertyMap`
            Collection of time-series with node states over time, or a single
            time-series. Each time-series must be a
            :class:`~graph_tool.VertexPropertyMap` with type ``vector<int>``
            containing the Ising states (``-1`` or ``+1``) of each node in each
            time step.

            If the parameter ``t`` below is given, each property map value for a
            given node should contain only the states for the same points in
            time given by that parameter.
        beta : ``float``
            Initial value of the global inverse temperature.
        x : :class:`~graph_tool.EdgePropertyMap` (optional, default: ``None``)
            Initial value of the local coupling for each edge. If not given, a
            uniform value of ``1`` will be used.
        h : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
            If given, this will set the initial local fields of each
            node. Otherwise a value of ``0`` will be used.
        t : ``list`` of :class:`~graph_tool.VertexPropertyMap` (optional, default: ``[]``)
            If nonempty, this allows for a compressed representation of the
            time-series parameter ``s``, corresponding only to points in time
            where the state of each node changes. Each each entry in this list
            must be a :class:`~graph_tool.VertexPropertyMap` with type
            ``vector<int>`` containing the points in time where the state of
            each node change. The corresponding state of the nodes at these
            times are given by parameter ``s``.
        aE : ``float`` (optional, default: ``NaN``)
            Expected total number of edges used in prior. If ``NaN``, a flat
            prior will be used instead.
        nested : ``boolean`` (optional, default: ``True``)
            If ``True``, a :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState`
            will be used, otherwise
            :class:`~graph_tool.inference.blockmodel.BlockState`.
        state_args : ``dict`` (optional, default: ``{}``)
            Arguments to be passed to
            :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or
            :class:`~graph_tool.inference.blockmodel.BlockState`.
        bstate : :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` or :class:`~graph_tool.inference.blockmodel.BlockState`  (optional, default: ``None``)
            If passed, this will be used to initialize the block state
            directly.
        self_loops : bool (optional, default: ``False``)
            If ``True``, it is assumed that the inferred graph can contain
            self-loops.
        has_zero : bool (optional, default: ``False``)
            If ``True``, the three-state "Ising" model with values ``{-1,0,1}``
            is used.

        References
        ----------
        .. [peixoto-network-2019] Tiago P. Peixoto, "Network reconstruction and
           community detection from dynamics", Phys. Rev. Lett. 123 128301
           (2019), :doi:`10.1103/PhysRevLett.123.128301`, :arxiv:`1903.10833`

        """

        if isinstance(s, PropertyMap):
            s = [s]
        if isinstance(t, PropertyMap):
            t = [t]
        if t is None:
            t = []
        if h is None:
            h = g.new_vp("double")
        if bstate is not None:
            if nested:
                u = bstate.levels[0].g
            else:
                u = bstate.g
        else:
            u = g
        if x == 1:
            x = u.new_ep("double", 1)
        elif x is None:
            x = u.new_ep("double", numpy.random.random(g.num_edges()) - 1/2)
        super().__init__(g, s=s, t=t, beta=beta, x=x, aE=aE, nested=nested,
                         state_args=state_args, bstate=bstate,
                         self_loops=self_loops, h=h, has_zero=has_zero,
                         **kwargs)

    def __setstate__(self, state):
        beta = state.pop("beta", None)
        if "x" not in state:
            state["x"] = 1
        self.__init__(**dict(state, beta=beta))

    def get_x(self):
        """Return edge couplings."""
        x = self.x.copy()
        x.fa *= self.params["beta"]
        return x

    def mcmc_sweep(self, r=.5, p=.1, pstep=.1, h=.1, hstep=1, xstep=.1,
                   multiflip=True, **kwargs):
        r"""Perform sweeps of a Metropolis-Hastings acceptance-rejection sampling MCMC to
        sample network partitions and latent edges. The parameter ``r`` controls
        the probability with which edge move will be attempted, instead of
        partition moves. The parameter ``h`` controls the relative probability
        with which moves for the parameters ``r_v`` will be attempted, and
        ``hstep`` is the size of the step. The parameter ``p`` controls the
        relative probability with which moves for the parameters ``global_beta``
        and ``r`` will be attempted, and ``pstep`` is the size of the step. The
        paramter ``xstep`` determines the size of the attempted steps for the
        edge coupling parameters.

        The remaining keyword parameters will be passed to
        :meth:`~graph_tool.inference.blockmodel.BlockState.mcmc_sweep` or
        :meth:`~graph_tool.inference.blockmodel.BlockState.multiflip_mcmc_sweep`,
        if ``multiflip=True``.
        """

        return super().mcmc_sweep(r=r, p=p, pstep=p, h=h, hstep=hstep,
                                  xstep=xstep, multiflip=multiflip, **kwargs)

    def _algo_sweep(self, algo, r=.5, p=.1, pstep=1, h=.5, hstep=1, niter=1,
                    xstep=1, **kwargs):
        if numpy.random.random() < p:
            if numpy.random.random() < h:
                mcmc_state = DictState(dict(beta=kwargs.get("beta", 1),
                                            hstep=hstep,
                                            verbose=kwargs.get("verbose", False),
                                            niter=niter,
                                            n=numpy.random.randint(len(self.s)),
                                            state=self._state))
                return self._mcmc_sweep_h(mcmc_state)
            else:
                dS = nt = nm = 0
                for i in range(niter):
                    ret = self._move_proposal("beta",
                                              kwargs.get("beta", 1),
                                              pstep, None, None,
                                              kwargs.get("entropy_args", {}))
                    dS += ret[0]
                    nt += ret[1]
                    nm += ret[2]
                return (dS, nt, nm)
        else:
            return super()._algo_sweep(algo, r, xlog=False, xstep=xstep,
                                       xdefault=1, niter=niter, **kwargs)

class IsingGlauberBlockState(IsingBaseBlockState):
    def __init__(self, *args, **kwargs):
        r"""State for network reconstruction based on the Glauber dynamics of the Ising
        model, using the stochastic block model as a prior.

        See documentation for :class:`IsingBaseBlockState` for details.
        """

        super().__init__(*args, **kwargs)

    def _make_state(self):
        return libinference.make_ising_glauber_state(self.bstate._state, self)

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_ising_glauber_sweep(mcmc_state, self._state,
                                                     _get_rng())

    def _mcmc_sweep_h(self, mcmc_state):
        return libinference.mcmc_ising_glauber_sweep_h(mcmc_state, self._state,
                                                       _get_rng())

class CIsingGlauberBlockState(IsingBaseBlockState):
    def __init__(self, *args, **kwargs):
        r"""State for network reconstruction based on the Glauber dynamics of the
        continuous Ising model, using the stochastic block model as a prior.

        See documentation for :class:`IsingBaseBlockState` for details. Note
        that in this case the ``s`` parameter should contain property maps of
        type ``vector<double>``, with values in the range :math:`[-1,1]`.
        """
        super().__init__(*args, **kwargs)

    def _make_state(self):
        return libinference.make_cising_glauber_state(self.bstate._state, self)

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_cising_glauber_sweep(mcmc_state, self._state,
                                                      _get_rng())
    def _mcmc_sweep_h(self, mcmc_state):
        return libinference.mcmc_cising_glauber_sweep_h(mcmc_state, self._state,
                                                        _get_rng())

class PseudoIsingBlockState(IsingBaseBlockState):
    def __init__(self, *args, **kwargs):
        r"""State for network reconstruction based on the equilibrium configurations of
        the Ising model, using the Pseudolikelihood approximation and the
        stochastic block model as a prior.

        See documentation for :class:`IsingBaseBlockState` for details. Note
        that in this model "time-series" should be interpreted as a set of
        uncorrelated samples, not a temporal sequence.
        """
        super().__init__(*args, **kwargs)

    def _make_state(self):
        return libinference.make_pseudo_ising_state(self.bstate._state, self)

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_pseudo_ising_sweep(mcmc_state, self._state,
                                                    _get_rng())
    def _mcmc_sweep_h(self, mcmc_state):
        return libinference.mcmc_pseudo_ising_sweep_h(mcmc_state, self._state,
                                                      _get_rng())

class PseudoCIsingBlockState(IsingBaseBlockState):
    def __init__(self, *args, **kwargs):
        r"""State for network reconstruction based on the equilibrium configurations of
        the continuous Ising model, using the Pseudolikelihood approximation and
        the stochastic block model as a prior.

        See documentation for :class:`IsingBaseBlockState` for details. Note
        that in this model "time-series" should be interpreted as a set of
        uncorrelated samples, not a temporal sequence. Additionally, the ``s``
        parameter should contain property maps of type ``vector<double>``, with
        values in the range :math:`[-1,1]`.
        """
        super().__init__(*args, **kwargs)

    def _make_state(self):
        return libinference.make_pseudo_cising_state(self.bstate._state, self)

    def _mcmc_sweep(self, mcmc_state):
        return libinference.mcmc_pseudo_cising_sweep(mcmc_state, self._state,
                                                     _get_rng())
    def _mcmc_sweep_h(self, mcmc_state):
        return libinference.mcmc_pseudo_cising_sweep_h(mcmc_state, self._state,
                                                       _get_rng())


def marginal_multigraph_entropy(g, ecount):
    r"""Compute the entropy of the marginal latent multigraph distribution.

    Parameters
    ----------
    g : :class:`~graph_tool.Graph`
        Marginal multigraph.
    ecount : :class:`~graph_tool.EdgePropertyMap`
        Vector-valued edge property map containing the counts of edge
        multiplicities.

    Returns
    -------
    eh : :class:`~graph_tool.EdgePropertyMap`
        Marginal entropy of edge multiplicities.

    Notes
    -----

    The mean posterior marginal multiplicity distribution of a multi-edge
    :math:`(i,j)` is defined as

    .. math::

       \pi_{ij}(w) = \sum_{\boldsymbol G}\delta_{w,G_{ij}}P(\boldsymbol G|\boldsymbol D)

    where :math:`P(\boldsymbol G|\boldsymbol D)` is the posterior
    probability of a multigraph :math:`\boldsymbol G` given the data.

    The corresponding entropy is therefore given (in nats) by

    .. math::

       \mathcal{S}_{ij} = -\sum_w\pi_{ij}(w)\ln \pi_{ij}(w).

    """

    eh = g.new_ep("double")
    libinference.marginal_count_entropy(g._Graph__graph,
                                        _prop("e", g, ecount),
                                        _prop("e", g, eh))
    return eh

def marginal_multigraph_sample(g, ews, ecount):

    w = g.new_ep("int")
    libinference.marginal_multigraph_sample(g._Graph__graph,
                                                _prop("e", g, ews),
                                                _prop("e", g, ecount),
                                                _prop("e", g, w),
                                                _get_rng())
    return w

def marginal_multigraph_lprob(g, ews, ecount, ew):

    L = libinference.marginal_multigraph_lprob(g._Graph__graph,
                                               _prop("e", g, ews),
                                               _prop("e", g, ecount),
                                               _prop("e", g, ew))
    return L

def marginal_graph_sample(g, ep):
    w = g.new_ep("int")
    libinference.marginal_graph_sample(g._Graph__graph,
                                       _prop("e", g, ep),
                                       _prop("e", g, w),
                                       _get_rng())
    return w

def marginal_graph_lprob(g, ep, w):
    L = libinference.marginal_graph_lprob(g._Graph__graph,
                                          _prop("e", g, ep),
                                          _prop("e", g, w))
    return L
