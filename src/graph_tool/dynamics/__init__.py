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

"""
``graph_tool.dynamics`` - Dynamical processes
---------------------------------------------

This module contains implementations of some often-studied dynamical processes
that take place on networks.

Summary
+++++++

Discrete-time dynamics
======================

.. autosummary::
   :nosignatures:

   DiscreteStateBase
   EpidemicStateBase
   SIState
   SISState
   SIRState
   SIRSState
   VoterState
   MajorityVoterState
   BinaryThresholdState
   IsingGlauberState
   CIsingGlauberState
   IsingMetropolisState
   PottsGlauberState
   PottsMetropolisState
   AxelrodState
   BooleanState
   KirmanState

Continuous-time dynamics
========================

.. autosummary::
   :nosignatures:

   ContinuousStateBase
   KuramotoState


Contents
++++++++

"""

from .. import _degree, _prop, Graph, GraphView, _get_rng, PropertyMap, \
    EdgePropertyMap, VertexPropertyMap, _check_prop_scalar
from .. stats import label_self_loops
import numpy
import numpy.random
import collections

import scipy.integrate

from .. dl_import import dl_import
dl_import("from . import libgraph_tool_dynamics as lib_dynamics")

__all__ = ["DiscreteStateBase", "EpidemicStateBase", "SIState", "SISState",
           "SIRState", "SIRSState", "VoterState", "MajorityVoterState",
           "BinaryThresholdState", "IsingGlauberState", "CIsingGlauberState",
           "IsingMetropolisState", "PottsGlauberState", "PottsMetropolisState",
           "AxelrodState", "BooleanState", "KirmanState", "ContinuousStateBase",
           "KuramotoState"]

class DiscreteStateBase(object):
    def __init__(self, g, make_state, params, s=None, stype="int32_t"):
        r"""Base state for discrete-time dynamics. This class it not meant to be
        instantiated directly."""

        self.g = g
        if s is None:
            self.s = g.new_vp(stype)
        else:
            self.s = s.copy(stype)
        self.s_temp = self.s.copy()
        self.params = params
        self._state = make_state(g._Graph__graph, _prop("v", g, self.s),
                                 _prop("v", g, self.s_temp), params, _get_rng())
        self.reset_active()

    def copy(self):
        """Return a copy of the state."""
        return type(self)(self.g, s=self.s.copy(), **self.params)

    def __setstate__(self):
        return dict(g=self.g, s=self.s, params=self.params)

    def __getstate__(self, state):
        return type(self)(state["g"], s=state["s"], **state["params"])

    def get_state(self):
        """Returns the internal :class:`~graph_tool.VertexPropertyMap` with the current
        state."""
        return self.s

    def get_active(self):
        """Returns list of "active" nodes, for states where this concept is used."""
        return self._state.get_active()

    def reset_active(self):
        """Resets list of "active" nodes, for states where this concept is used."""
        self._state.reset_active(_get_rng())

    def iterate_sync(self, niter=1):
        """Updates nodes synchronously (i.e. a full "sweep" of all nodes in parallel),
        `niter` number of times. This function returns the number of nodes that
        changed state.

        If enabled during compilation, this algorithm runs in parallel
        (i.e. using more than one thread.)

        """
        return self._state.iterate_sync(niter, _get_rng())

    def iterate_async(self, niter=1):
        """Updates nodes asynchronously (i.e. single vertex chosen randomly), `niter`
        number of times. This function returns the number of nodes that changed
        state.
        """
        return self._state.iterate_async(niter, _get_rng())

class EpidemicStateBase(DiscreteStateBase):
    def __init__(self, g, constant_beta, make_state, params, v0=None, s=None):
        r"""Base state for epidemic dynamics. This class it not meant to be
        instantiated directly."""

        if s is not None:
            self.s = s
        else:
            self.s = g.new_vp("int32_t")
            if v0 is None:
                v0 = numpy.random.randint(0, g.num_vertices())
                v0 = g.vertex(v0, use_index=False)
            self.s[v0] = 1

        beta = params["beta"]
        weighted = isinstance(beta, EdgePropertyMap)
        if weighted:
            _check_prop_scalar(beta, "beta")
            if beta.value_type() != "double":
                if not constant_beta:
                    raise ValueError("if constant_beta == False, the type of beta must be double")
                beta = beta.copy("double")
        params["beta"] = beta

        for p in ["r", "epsilon", "gamma", "mu"]:
            if p not in params:
                continue
            if not isinstance(params[p], VertexPropertyMap):
                params[p] = g.new_vp("double", val=params[p])
            _check_prop_scalar(params[p], p)
            if params[p].value_type() != "double":
                params[p] = params[p].copy("double")
        self.params = params

        self.make_state = lambda *args: make_state(*args, weighted, constant_beta)


class SIState(EpidemicStateBase):
    def __init__(self, g, beta=1., r=0, exposed=False, epsilon=.1, v0=None,
                 s=None, constant_beta=True):
        r"""SI compartmental epidemic model.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        beta : ``float`` or :class:`~graph_tool.EdgePropertyMap` (optional, default: ``1.``)
           Transmission probability. If an :class:`~graph_tool.EdgePropertyMap`
           object is passed, it must contain the transmission probability for
           every edge.
        r : ``float`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``0.``)
           Spontaneous infection probability.
        exposed : ``boolean`` (optional, default: ``False``)
           If ``True``, an SEI model is simulated, with an additional "exposed"
           state.
        epsilon : ``float`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``.1``)
           Susceptible to exposed transition probability. This only has an
           effect if ``exposed=True``.
        v0 : ``int`` or :class:`~graph_tool.Vertex` (optional, default: ``None``)
           Initial infectious vertex. If not provided, and if the global state is
           also not provided via paramter ``s``, a random vertex will be chosen.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, all vertices will be
           initialized to the susceptible state.
        constant_beta : ``boolean`` (optional, default: ``True``)
           If ``True``, and ``beta`` is an edge property map, it will be assumed
           that the ``beta`` values do not change, such that the probability
           values can be pre-computed for efficiency. If ``beta`` is a
           ``float``, this option has no effect.

        Notes
        -----

        This implements an SI epidemic process [pastor-satorras-epidemic-2015]_,
        where nodes in the susceptible state (value 0) are infected by neighbours
        in the infectious state (value 1).

        If a node :math:`i` is updated at time :math:`t`, the transition
        probabilities from state :math:`s_i(t)` to state :math:`s_i(t+1)` are
        given as follows:

        1. If :math:`s_i(t) = 0`, we have :math:`s_i(t+1) = 1` with probability
            .. math::

               (1-r_i)\left[1-\prod_j(1-\beta_{ij})^{A_{ij}\delta_{s_j(t),1}}\right] + r_i,

           otherwise :math:`s_i(t+1) = 0`.

        2. If :math:`s_i(t) = 1`, we have :math:`s_i(t+1) = 1` with probability
           1.

        If the option ``exposed == True`` is given, then the states transit
        first from 0 to -1 (exposed) with probability given by 1. above, and
        then finally from -1 to 1 with probability :math:`\epsilon_i`.

        Examples
        --------

        .. testsetup:: SI

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: SI

           >>> g = gt.collection.data["pgp-strong-2009"]
           >>> state = gt.SIState(g, beta=0.01)
           >>> X = []
           >>> for t in range(1000):
           ...     ret = state.iterate_sync()
           ...     X.append(state.get_state().fa.sum())

           >>> figure(figsize=(6, 4))
           <...>
           >>> plot(X)
           [...]
           >>> xlabel(r"Time")
           Text(...)
           >>> ylabel(r"Infectious nodes")
           Text(...)
           >>> tight_layout()
           >>> savefig("SI.svg")

        .. figure:: SI.svg
           :align: center

           Number of infectious nodes vs. time for an SI dynamics.

        References
        ----------

        .. [pastor-satorras-epidemic-2015] Romualdo Pastor-Satorras, Claudio
           Castellano, Piet Van Mieghem, and Alessandro Vespignani, "Epidemic
           processes in complex networks", Rev. Mod. Phys. 87, 925 (2015)
           :doi:`10.1103/RevModPhys.87.925`, :arxiv:`1408.2701`

        """
        EpidemicStateBase.__init__(self, g, constant_beta,
                                   lib_dynamics.make_SEI_state if exposed else lib_dynamics.make_SI_state,
                                   dict(beta=beta, r=r, epsilon=epsilon),
                                   v0, s)
        DiscreteStateBase.__init__(self, g, self.make_state, self.params,
                                   self.s)

class SISState(DiscreteStateBase):
    def __init__(self, g, beta=1., gamma=.1, r=0, exposed=False, epsilon=.1,
                 v0=None, s=None, constant_beta=True):
        r"""SIS compartmental epidemic model.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        beta : ``float`` or :class:`~graph_tool.EdgePropertyMap` (optional, default: ``1.``)
           Transmission probability.
        gamma : ``float`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``.1``)
           Recovery probability.
        r : ``float`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``0.``)
           Spontaneous infection probability.
        exposed : ``boolean`` (optional, default: ``False``)
           If ``True``, an SEIS model is simulated, with an additional "exposed"
           state.
        epsilon : ``float`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``.1``)
           Susceptible to exposed transition probability. This only has an
           effect if ``exposed=True``.
        v0 : ``int`` or :class:`~graph_tool.Vertex` (optional, default: ``None``)
           Initial infectious vertex. If not provided, and if the global state is
           also not provided via paramter ``s``, a random vertex will be chosen.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, all vertices will be
           initialized to the susceptible state.
        constant_beta : ``boolean`` (optional, default: ``True``)
           If ``True``, and ``beta`` is an edge property map, it will be assumed
           that the ``beta`` values do not change, such that the probability
           values can be pre-computed for efficiency. If ``beta`` is a
           ``float``, this option has no effect.

        Notes
        -----

        This implements an SIS epidemic process
        [pastor-satorras-epidemic-2015]_, where nodes in the susceptible state
        (value 0) are infected by neighbours in the infectious state (value 1),
        which can then eventually recover to a susceptible state.

        If a node :math:`i` is updated at time :math:`t`, the transition
        probabilities from state :math:`s_i(t)` to state :math:`s_i(t+1)` are
        given as follows:

        1. If :math:`s_i(t) = 0`, we have :math:`s_i(t+1) = 1` with probability
            .. math::

               (1-r_i)\left[1-\prod_j(1-\beta_{ij})^{A_{ij}\delta_{s_j(t),1}}\right] + r_i,

           otherwise :math:`s_i(t+1) = 0`.

        2. If :math:`s_i(t) = 1`, we have :math:`s_i(t+1) = 0` with probability
           :math:`\gamma_i`, or :math:`s_i(t+1) = 1` with probability
           :math:`1-\gamma_i`.

        If the option ``exposed == True`` is given, then the states transit
        first from 0 to -1 (exposed) with probability given by 1. above, and
        then finally from -1 to 1 with probability :math:`\epsilon_i`.

        Examples
        --------

        .. testsetup:: SIS

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: SIS

           >>> g = gt.collection.data["pgp-strong-2009"]
           >>> state = gt.SISState(g, beta=0.01, gamma=0.007)
           >>> X = []
           >>> for t in range(1000):
           ...     ret = state.iterate_sync()
           ...     X.append(state.get_state().fa.sum())

           >>> figure(figsize=(6, 4))
           <...>
           >>> plot(X)
           [...]
           >>> xlabel(r"Time")
           Text(...)
           >>> ylabel(r"Infectious nodes")
           Text(...)
           >>> tight_layout()
           >>> savefig("SIS.svg")

        .. figure:: SIS.svg
           :align: center

           Number of infectious nodes vs. time for an SIS dynamics.

        References
        ----------

        .. [pastor-satorras-epidemic-2015] Romualdo Pastor-Satorras, Claudio
           Castellano, Piet Van Mieghem, and Alessandro Vespignani, "Epidemic
           processes in complex networks", Rev. Mod. Phys. 87, 925 (2015)
           :doi:`10.1103/RevModPhys.87.925`, :arxiv:`1408.2701`

        """
        EpidemicStateBase.__init__(self, g, constant_beta,
                                   lib_dynamics.make_SEIS_state if exposed else lib_dynamics.make_SIS_state,
                                   dict(beta=beta, gamma=gamma, r=r,
                                        epsilon=epsilon),
                                   v0, s)
        DiscreteStateBase.__init__(self, g, self.make_state, self.params, self.s)

class SIRState(DiscreteStateBase):
    def __init__(self, g, beta=1., gamma=.1, r=0, exposed=False, epsilon=.1,
                 v0=None, s=None, constant_beta=True):
        r"""SIR compartmental epidemic model.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        beta : ``float`` or :class:`~graph_tool.EdgePropertyMap` (optional, default: ``1.``)
           Transmission probability.
        gamma : ``float`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``.1``)
           Recovery probability.
        r : ``float`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``0.``)
           Spontaneous infection probability.
        exposed : ``boolean`` (optional, default: ``False``)
           If ``True``, an SEIR model is simulated, with an additional "exposed"
           state.
        epsilon : ``float`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``.1``)
           Susceptible to exposed transition probability. This only has an
           effect if ``exposed=True``.
        v0 : ``int`` or :class:`~graph_tool.Vertex` (optional, default: ``None``)
           Initial infectious vertex. If not provided, and if the global state is
           also not provided via paramter ``s``, a random vertex will be chosen.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, all vertices will be
           initialized to the susceptible state.
        constant_beta : ``boolean`` (optional, default: ``True``)
           If ``True``, and ``beta`` is an edge property map, it will be assumed
           that the ``beta`` values do not change, such that the probability
           values can be pre-computed for efficiency. If ``beta`` is a
           ``float``, this option has no effect.

        Notes
        -----

        This implements an SIR epidemic process
        [pastor-satorras-epidemic-2015]_, where nodes in the susceptible state
        (value 0) are infected by neighbours in the infectious state (value 1),
        which can then eventually recover to a recovered state (value 2).

        If a node :math:`i` is updated at time :math:`t`, the transition
        probabilities from state :math:`s_i(t)` to state :math:`s_i(t+1)` are
        given as follows:

        1. If :math:`s_i(t) = 0`, we have :math:`s_i(t+1) = 1` with probability
            .. math::

               (1-r_i)\left[1-\prod_j(1-\beta_{ij})^{A_{ij}\delta_{s_j(t),1}}\right] + r_i,

           otherwise :math:`s_i(t+1) = 0`.

        2. If :math:`s_i(t) = 1`, we have :math:`s_i(t+1) = 2` with probability
           :math:`\gamma_i`, or :math:`s_i(t+1) = 1` with probability
           :math:`1-\gamma_i`.

        If the option ``exposed == True`` is given, then the states transit
        first from 0 to -1 (exposed) with probability given by 1. above, and
        then finally from -1 to 1 with probability :math:`\epsilon_i`.

        Examples
        --------

        .. testsetup:: SIR

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: SIR

           >>> g = gt.collection.data["pgp-strong-2009"]
           >>> state = gt.SIRState(g, beta=0.01, gamma=0.0025)
           >>> S, X, R = [], [], []
           >>> for t in range(2000):
           ...     ret = state.iterate_sync()
           ...     s = state.get_state().fa
           ...     S.append((s == 0).sum())
           ...     X.append((s == 1).sum())
           ...     R.append((s == 2).sum())

           >>> figure(figsize=(6, 4))
           <...>
           >>> plot(S, label="Susceptible")
           [...]
           >>> plot(X, label="Infectious")
           [...]
           >>> plot(R, label="Recovered")
           [...]
           >>> xlabel(r"Time")
           Text(...)
           >>> ylabel(r"Number of nodes")
           Text(...)
           >>> legend(loc="best")
           <...>
           >>> tight_layout()
           >>> savefig("SIR.svg")

        .. figure:: SIR.svg
           :align: center

           Number of susceptible, infectious, and recovered nodes vs. time for an
           SIR dynamics.

        References
        ----------

        .. [pastor-satorras-epidemic-2015] Romualdo Pastor-Satorras, Claudio
           Castellano, Piet Van Mieghem, and Alessandro Vespignani, "Epidemic
           processes in complex networks", Rev. Mod. Phys. 87, 925 (2015)
           :doi:`10.1103/RevModPhys.87.925`, :arxiv:`1408.2701`

        """
        EpidemicStateBase.__init__(self, g, constant_beta,
                                   lib_dynamics.make_SEIR_state if exposed else lib_dynamics.make_SIR_state,
                                   dict(beta=beta, gamma=gamma, r=r,
                                        epsilon=epsilon),
                                   v0, s)
        DiscreteStateBase.__init__(self, g, self.make_state,
                                   self.params, self.s)

class SIRSState(DiscreteStateBase):
    def __init__(self, g, beta=1, gamma=.1, mu=.1, r=0, exposed=False,
                 epsilon=.1, v0=None, s=None, constant_beta=True):
        r"""SIRS compartmental epidemic model.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        beta : ``float`` or :class:`~graph_tool.EdgePropertyMap` (optional, default: ``1.``)
           Transmission probability.
        gamma : ``float`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``.1``)
           I to R recovery probability.
        mu : ``float`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``.1``)
           R to S recovery probability.
        r : ``float`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``0.``)
           Spontaneous infection probability.
        exposed : ``boolean`` (optional, default: ``False``)
           If ``True``, an SEIRS model is simulated, with an additional "exposed"
           state.
        epsilon : ``float`` or :class:`~graph_tool.VertexPropertyMap` (optional, default: ``.1``)
           Susceptible to exposed transition probability. This only has an
           effect if ``exposed=True``.
        v0 : ``int`` or :class:`~graph_tool.Vertex` (optional, default: ``None``)
           Initial infectious vertex. If not provided, and if the global state is
           also not provided via paramter ``s``, a random vertex will be chosen.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, all vertices will be
           initialized to the susceptible state.
        constant_beta : ``boolean`` (optional, default: ``True``)
           If ``True``, and ``beta`` is an edge property map, it will be assumed
           that the ``beta`` values do not change, such that the probability
           values can be pre-computed for efficiency. If ``beta`` is a
           ``float``, this option has no effect.

        Notes
        -----

        This implements an SIRS epidemic process
        [pastor-satorras-epidemic-2015]_, where nodes in the susceptible state
        (value 0) are infected by neighbours in the infectious state (value 1),
        which can then eventually recover to a recovered state (value 2), and
        finally back to the susceptible state.

        If a node :math:`i` is updated at time :math:`t`, the transition
        probabilities from state :math:`s_i(t)` to state :math:`s_i(t+1)` are
        given as follows:

        1. If :math:`s_i(t) = 0`, we have :math:`s_i(t+1) = 1` with probability
            .. math::

               (1-r_i)\left[1-\prod_j(1-\beta_{ij})^{A_{ij}\delta_{s_j(t),1}}\right] + r_i,

           otherwise :math:`s_i(t+1) = 0`.

        2. If :math:`s_i(t) = 1`, we have :math:`s_i(t+1) = 2` with probability
           :math:`\gamma_i`, or :math:`s_i(t+1) = 1` with probability
           :math:`1-\gamma_i`.

        3. If :math:`s_i(t) = 2`, we have :math:`s_i(t+1) = 1` with probability
           :math:`\mu_i`, or :math:`s_i(t+1) = 2` with probability
           :math:`1-\mu_i`.

        If the option ``exposed == True`` is given, then the states transit
        first from 0 to -1 (exposed) with probability given by 1. above, and
        then finally from -1 to 1 with probability :math:`\epsilon_i`.

        Examples
        --------

        .. testsetup:: SIRS

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: SIRS

           >>> g = gt.collection.data["pgp-strong-2009"]
           >>> state = gt.SIRSState(g, beta=0.2, gamma=0.025, mu=0.02)
           >>> S, X, R = [], [], []
           >>> for t in range(2000):
           ...     ret = state.iterate_sync()
           ...     s = state.get_state().fa
           ...     S.append((s == 0).sum())
           ...     X.append((s == 1).sum())
           ...     R.append((s == 2).sum())

           >>> figure(figsize=(6, 4))
           <...>
           >>> plot(S, label="Susceptible")
           [...]
           >>> plot(X, label="Infectious")
           [...]
           >>> plot(R, label="Recovered")
           [...]
           >>> xlabel(r"Time")
           Text(...)
           >>> ylabel(r"Number of nodes")
           Text(...)
           >>> legend(loc="best")
           <...>
           >>> tight_layout()
           >>> savefig("SIRS.svg")

        .. figure:: SIRS.svg
           :align: center

           Number of susceptible, infectious, and recovered nodes vs. time for an
           SIRS dynamics.

        References
        ----------

        .. [pastor-satorras-epidemic-2015] Romualdo Pastor-Satorras, Claudio
           Castellano, Piet Van Mieghem, and Alessandro Vespignani, "Epidemic
           processes in complex networks", Rev. Mod. Phys. 87, 925 (2015)
           :doi:`10.1103/RevModPhys.87.925`, :arxiv:`1408.2701`

        """
        EpidemicStateBase.__init__(self, g, constant_beta,
                                   lib_dynamics.make_SEIRS_state if exposed else lib_dynamics.make_SIRS_state,
                                   dict(beta=beta, gamma=gamma, mu=mu, r=r,
                                        epsilon=epsilon),
                                   v0, s)
        DiscreteStateBase.__init__(self, g, self.make_state,
                                   self.params, self.s)


class VoterState(DiscreteStateBase):
    def __init__(self, g, q=2, r=0., s=None):
        r"""Generalized q-state voter model dynamics.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        q : ``int`` (optional, default: ``2``)
           Number of opinions.
        r : ``float`` (optional, default: ``0.``)
           Random opinion probability.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, a random state will be chosen.

        Notes
        -----

        This implements the voter model dynamics [clifford-model-1973]_
        [holley-ergodic-1075]_ on a network.

        If a node :math:`i` is updated at time :math:`t`, the transition
        probabilities from state :math:`s_i(t)` to state :math:`s_i(t+1)` are
        given as follows:

        1. With a probability :math:`r` one of the :math:`q` opinions,
        :math:`x`, is chosen uniformly at random, and assigned to :math:`i`,
        i.e. :math:`s_i(t+1) = x`.

        2. Otherwise, a random (in-)neighbour :math:`j` is chosen. and its
        opinion is copied, i.e. :math:`s_i(t+1) = s_j(t)`.


        Examples
        --------

        .. testsetup:: voter

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: voter

           >>> g = gt.collection.data["pgp-strong-2009"]
           >>> state = gt.VoterState(g, q=4)
           >>> x = [[] for r in range(4)]
           >>> for t in range(2000):
           ...     ret = state.iterate_sync()
           ...     s = state.get_state().fa
           ...     for r in range(4):
           ...         x[r].append((s == r).sum())
           >>> figure(figsize=(6, 4))
           <...>
           >>> for r in range(4):
           ...     plot(x[r], label="Opinion %d" % r)
           [...]
           >>> xlabel(r"Time")
           Text(...)
           >>> ylabel(r"Number of nodes")
           Text(...)
           >>> legend(loc="best")
           <...>
           >>> tight_layout()
           >>> savefig("voter.svg")

        .. figure:: voter.svg
           :align: center

           Number of nodes with a given opinion vs. time for a voter model
           dynamics with :math:`q=4` opinions.

        References
        ----------
        .. [clifford-model-1973] Clifford, P., Sudbury, A., "A model for spatial
           conflict", Biometrika 60, 581–588 (1973). :doi:`10.1093/biomet/60.3.581`.
        .. [holley-ergodic-1075] Holley, R. A., Liggett, T. M., "Ergodic
           Theorems for Weakly Interacting Infinite Systems and the Voter Model",
           Ann. Probab. 3, 643–663 (1975). :doi:`10.1214/aop/1176996306`.

        """
        if s is None:
            s = g.new_vp("int", vals=numpy.random.randint(0, q, g.num_vertices()))
        DiscreteStateBase.__init__(self, g,
                                  lib_dynamics.make_voter_state,
                                  dict(q=q, r=r), s)

class MajorityVoterState(DiscreteStateBase):
    def __init__(self, g, q=2, r=0, s=None):
        r"""Generalized q-state majority voter model dynamics.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        q : ``int`` (optional, default: ``2``)
           Number of opinions.
        r : ``float`` (optional, default: ``0.``)
           Random opinion probability.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, a random state will be chosen.

        Notes
        -----

        This implements the majority voter model dynamics
        [oliveira-isotropic-1992]_ on a network.

        If a node :math:`i` is updated at time :math:`t`, the transition
        probabilities from state :math:`s_i(t)` to state :math:`s_i(t+1)` are
        given as follows:

        1. With a probability :math:`r` one of the :math:`q` opinions,
        :math:`x`, is chosen uniformly at random, and assigned to :math:`i`,
        i.e. :math:`s_i(t+1) = x`.

        2. Otherwise, the majority opinion :math:`x` held by all (in-)neighbours
        of :math:`i` is chosen. In case of a tie between two or more opinions, a
        random choice between them is made. The chosen opinion is then copied,
        i.e. :math:`s_i(t+1) = x`.


        Examples
        --------

        .. testsetup:: majority-voter

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: majority-voter

           >>> g = gt.collection.data["pgp-strong-2009"]
           >>> state = gt.MajorityVoterState(g, q=4)
           >>> x = [[] for r in range(4)]
           >>> for t in range(2000):
           ...     ret = state.iterate_async(niter=g.num_vertices())
           ...     s = state.get_state().fa
           ...     for r in range(4):
           ...         x[r].append((s == r).sum())
           >>> figure(figsize=(6, 4))
           <...>
           >>> for r in range(4):
           ...     plot(x[r], label="Opinion %d" % r)
           [...]
           >>> xlabel(r"Time")
           Text(...)
           >>> ylabel(r"Number of nodes")
           Text(...)
           >>> legend(loc="best")
           <...>
           >>> tight_layout()
           >>> savefig("majority-voter.svg")

        .. figure:: majority-voter.svg
           :align: center

           Number of nodes with a given opinion vs. time for a majority voter
           model dynamics with :math:`q=4` opinions.

        References
        ----------
        .. [oliveira-isotropic-1992] de Oliveira, M.J., "Isotropic majority-vote
           model on a square lattice", J Stat Phys 66: 273 (1992).
           :doi:`10.1007/BF01060069`.
        """
        if s is None:
            s = g.new_vp("int", vals=numpy.random.randint(0, q, g.num_vertices()))
        DiscreteStateBase.__init__(self, g,
                                   lib_dynamics.make_majority_voter_state,
                                   dict(q=q, r=r), s)

class BinaryThresholdState(DiscreteStateBase):
    def __init__(self, g, w=1., h=.5, r=0., s=None):
        r"""Generalized binary threshold dynamics.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        w : :class:`~graph_tool.EdgePropertyMap` or ``float`` (optional, default: ``1.``)
           Edge weights. If a scalar is provided, it's used for all edges.
        h : ``float`` (optional, default: ``.5``)
           Relative threshold value.
        r : ``float`` (optional, default: ``0.``)
           Input random flip probability.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, a random state will be chosen.

        Notes
        -----

        This implements a Boolean threshold model on a network.

        If a node :math:`i` is updated at time :math:`t`, the transition
        to state :math:`s_i(t+1)` is given by

        .. math::

           s_i(t+1) =
           \begin{cases}
               1, & \text{ if } \sum_jA_{ij}w_{ij}\hat s_j(t) > h k_i,\\
               0, & \text{ otherwise.}
           \end{cases}

        where :math:`k_i=\sum_jA_{ij}` and :math:`\hat s_i(t)` are the flipped
        inputs sampled with probability

        .. math::

           P(\hat s_i(t)|s_i(t)) = r^{1-\delta_{\hat s_i(t),s_i(t)}}(1-r)^{\delta_{\hat s_i(t),s_i(t)}}.

        Examples
        --------

        .. testsetup:: binary-threshold

           gt.seed_rng(44)
           np.random.seed(44)

        .. doctest:: binary-threshold

           >>> g = gt.GraphView(gt.collection.data["polblogs"].copy(), directed=False)
           >>> gt.remove_parallel_edges(g)
           >>> g = gt.extract_largest_component(g, prune=True)
           >>> state = gt.BinaryThresholdState(g, r=0.25)
           >>> ret = state.iterate_sync(niter=1000)
           >>> gt.graph_draw(g, g.vp.pos, vertex_fill_color=state.s, vcmap=cm.bone,
           ...               output="binary-threshold.pdf")
           <...>

        .. testcleanup:: binary-threshold

           conv_png("binary-threshold.pdf")

        .. figure:: binary-threshold.png
           :align: center
           :width: 80%

           State of a binary threshold dynamics on a political blog network.
        """

        if isinstance(w, PropertyMap):
            if w.value_type() != "double":
                w = w.copy("double")
        else:
            w = g.new_ep("double", val=w)

        if isinstance(h, PropertyMap):
            if h.value_type() != "double":
                h = w.copy("double")
        else:
            h = g.new_vp("double", val=h)

        if s is None:
            s = g.new_vp("int", vals=numpy.random.randint(0, 2, g.num_vertices()))

        DiscreteStateBase.__init__(self, g,
                                   lib_dynamics.make_binary_threshold_state,
                                   dict(w=w, h=h, r=r), s)

class IsingGlauberState(DiscreteStateBase):
    def __init__(self, g, beta=1., w=1., h=0., s=None):
        r"""Glauber dynamics of the Ising model.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        beta : ``float`` (optional, default: ``1.``)
           Inverse temperature.
        w : :class:`~graph_tool.EdgePropertyMap` or ``float`` (optional, default: ``1.``)
           Edge interaction strength. If a scalar is provided, it's used for all edges.
        h : :class:`~graph_tool.VertexPropertyMap` or ``float`` (optional, default: ``0.``)
           Vertex local field. If a scalar is provided, it's used for all vertices.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, a random state will be chosen.

        Notes
        -----

        This implements the Glauber dynamics of the Ising model [ising-model]_
        on a network.

        If a node :math:`i` is updated at time :math:`t`, the transition
        to state :math:`s_i(t+1) \in \{-1,+1\}` is done with probability

        .. math::

           P(s_i(t+1)|\boldsymbol s(t)) =
           \frac{\exp(\beta s_i(t+1)\sum_jA_{ij}w_{ij}s_j(t) + h_is_i(t+1))}
           {2\cosh(\beta\sum_jA_{ij}w_{ij}s_j(t) + h_i)}.

        Examples
        --------

        .. testsetup:: glauber-ising

           gt.seed_rng(47)
           np.random.seed(47)

        .. doctest:: glauber-ising

           >>> g = gt.GraphView(gt.collection.data["polblogs"].copy(), directed=False)
           >>> gt.remove_parallel_edges(g)
           >>> g = gt.extract_largest_component(g, prune=True)
           >>> state = gt.IsingGlauberState(g, beta=.05)
           >>> ret = state.iterate_async(niter=1000 * g.num_vertices())
           >>> gt.graph_draw(g, g.vp.pos, vertex_fill_color=state.s, vcmap=cm.bone,
           ...               output="glauber-ising.pdf")
           <...>

        .. testcleanup:: glauber-ising

           conv_png("glauber-ising.pdf")

        .. figure:: glauber-ising.png
           :align: center
           :width: 80%

           State of a Glauber Ising dynamics on a political blog network.

        References
        ----------
        .. [ising-model] https://en.wikipedia.org/wiki/Ising_model
        """

        if isinstance(w, PropertyMap):
            if w.value_type() != "double":
                w = w.copy("double")
        else:
            w = g.new_ep("double", val=w)
        if isinstance(h, PropertyMap):
            if h.value_type() != "double":
                h = h.copy("double")
        else:
            h = g.new_vp("double", val=h)
        if s is None:
            s = g.new_vp("int32_t",
                         vals=2 * numpy.random.randint(0, 2,
                                                       g.num_vertices()) - 1)
        DiscreteStateBase.__init__(self, g,
                                   lib_dynamics.make_ising_glauber_state,
                                   dict(w=w, h=h, beta=beta), s)

class CIsingGlauberState(DiscreteStateBase):
    def __init__(self, g, beta=1., w=1., h=0., s=None):
        r"""Glauber dynamics of the continuous Ising model.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        beta : ``float`` (optional, default: ``1.``)
           Inverse temperature.
        w : :class:`~graph_tool.EdgePropertyMap` or ``float`` (optional, default: ``1.``)
           Edge interaction strength. If a scalar is provided, it's used for all edges.
        h : :class:`~graph_tool.VertexPropertyMap` or ``float`` (optional, default: ``0.``)
           Vertex local field. If a scalar is provided, it's used for all vertices.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, a random state will be chosen.

        Notes
        -----

        This implements the Glauber dynamics of the continuous Ising model
        [ising-model]_ on a network.

        If a node :math:`i` is updated at time :math:`t`, the transition to
        state :math:`s_i(t+1) \in [-1,+1]` is done with probability density

        .. math::

           P(s_i(t+1)|\boldsymbol s(t)) =
           \frac{\exp(\beta s_i(t+1)\sum_jA_{ij}w_{ij}s_j(t) + h_is_i(t+1))}
           {Z(\beta\sum_jA_{ij}w_{ij}s_j(t) + h_i)},

        with :math:`Z(x) = 2\sinh(x)/x`.

        Examples
        --------

        .. testsetup:: glauber-cising

           gt.seed_rng(45)
           np.random.seed(45)

        .. doctest:: glauber-cising

           >>> g = gt.GraphView(gt.collection.data["polblogs"].copy(), directed=False)
           >>> gt.remove_parallel_edges(g)
           >>> g = gt.extract_largest_component(g, prune=True)
           >>> state = gt.CIsingGlauberState(g, beta=.2)
           >>> ret = state.iterate_async(niter=1000 * g.num_vertices())
           >>> gt.graph_draw(g, g.vp.pos, vertex_fill_color=state.s, vcmap=cm.magma,
           ...               output="glauber-cising.pdf")
           <...>

        .. testcleanup:: glauber-cising

           conv_png("glauber-cising.pdf")

        .. figure:: glauber-cising.png
           :align: center
           :width: 80%

           State of a continuous Glauber Ising dynamics on a political blog network.

        References
        ----------
        .. [ising-model] https://en.wikipedia.org/wiki/Ising_model

        """

        if isinstance(w, PropertyMap):
            if w.value_type() != "double":
                w = w.copy("double")
        else:
            w = g.new_ep("double", val=w)
        if isinstance(h, PropertyMap):
            if h.value_type() != "double":
                h = h.copy("double")
        else:
            h = g.new_vp("double", val=h)
        if s is None:
            s = g.new_vp("double", vals=2 * numpy.random.random(g.num_vertices()) - 1)
        DiscreteStateBase.__init__(self, g,
                                   lib_dynamics.make_cising_glauber_state,
                                   dict(w=w, h=h, beta=beta), s, stype="double")

class IsingMetropolisState(DiscreteStateBase):
    def __init__(self, g, beta=1, w=1, h=0, s=None):
        r"""Metropolis-Hastings dynamics of the Ising model.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        beta : ``float`` (optional, default: ``1.``)
           Inverse temperature.
        w : :class:`~graph_tool.EdgePropertyMap` or ``float`` (optional, default: ``1.``)
           Edge interaction strength. If a scalar is provided, it's used for all edges.
        h : :class:`~graph_tool.VertexPropertyMap` or ``float`` (optional, default: ``0.``)
           Vertex local field. If a scalar is provided, it's used for all vertices.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, a random state will be chosen.

        Notes
        -----

        This implements the Metropolis-Hastings dynamics
        [metropolis-equations-1953]_ [hastings-monte-carlo-1970]_ of the Ising
        model [ising-model]_ on a network.

        If a node :math:`i` is updated at time :math:`t`, the transition
        to state :math:`s_i(t+1) = -s_i(t)` is done with probability

        .. math::

           \min\left\{1, \exp\left[-2s_i(t)\left(h_i + \beta\sum_jA_{ij}w_{ij}s_j(t)\right)\right]\right\}

        otherwise we have :math:`s_i(t+1) = s_i(t)`.

        Examples
        --------

        .. testsetup:: metropolis-ising

           gt.seed_rng(42 + 1)
           np.random.seed(42 + 1)

        .. doctest:: metropolis-ising

           >>> g = gt.GraphView(gt.collection.data["polblogs"].copy(), directed=False)
           >>> gt.remove_parallel_edges(g)
           >>> g = gt.extract_largest_component(g, prune=True)
           >>> state = gt.IsingMetropolisState(g, beta=.1)
           >>> ret = state.iterate_async(niter=1000 * g.num_vertices())
           >>> gt.graph_draw(g, g.vp.pos, vertex_fill_color=state.s, vcmap=cm.bone,
           ...               output="metropolis-ising.pdf")
           <...>

        .. testcleanup:: metropolis-ising

           conv_png("metropolis-ising.pdf")

        .. figure:: metropolis-ising.png
           :align: center
           :width: 80%

           State of a Metropolis-Hastings Ising dynamics on a political blog network.

        References
        ----------
        .. [ising-model] https://en.wikipedia.org/wiki/Ising_model
        .. [metropolis-equations-1953] Metropolis, N., A.W. Rosenbluth,
           M.N. Rosenbluth, A.H. Teller, and E. Teller, "Equations of
           State Calculations by Fast Computing Machines," Journal of Chemical
           Physics, 21, 1087–1092 (1953). :doi:`10.1063/1.1699114`
        .. [hastings-monte-carlo-1970] Hastings, W.K., "Monte Carlo Sampling
           Methods Using Markov Chains and Their Applications," Biometrika, 57,
           97–109, (1970). :doi:`10.1093/biomet/57.1.97`

        """

        if isinstance(w, PropertyMap):
            if w.value_type() != "double":
                w = w.copy("double")
        else:
            w = g.new_ep("double", val=w)
        if isinstance(h, PropertyMap):
            if h.value_type() != "double":
                h = h.copy("double")
        else:
            h = g.new_vp("double", val=h)
        if s is None:
            s = g.new_vp("int32_t", vals=2 * numpy.random.randint(0, 2, g.num_vertices()) - 1)
        DiscreteStateBase.__init__(self, g,
                                   lib_dynamics.make_ising_metropolis_state,
                                   dict(w=w, h=h, beta=beta), s)

class PottsGlauberState(DiscreteStateBase):
    def __init__(self, g, f, w=1, h=0, s=None):
        r"""Glauber dynamics of the Potts model.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        f : list of lists or two-dimensional :class:`numpy.ndarray`
           Matrix of interactions between spin values, of dimension
           :math:`q\times q`, where :math:`q` is the number of spins.
        w : :class:`~graph_tool.EdgePropertyMap` or ``float`` (optional, default: ``1.``)
           Edge interaction strength. If a scalar is provided, it's used for all edges.
        h : :class:`~graph_tool.VertexPropertyMap` or iterable or ``float`` (optional, default: ``0.``)
           Vertex local field. If an iterable is provided, it will be used as
           the field for all vertices. If a scalar is provided, it will be used
           for all spins values and vertices.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, a random state will be chosen.

        Notes
        -----

        This implements the Glauber dynamics of the Potts model [potts-model]_
        on a network.

        If a node :math:`i` is updated at time :math:`t`, the transition
        to state :math:`s_i(t+1) \in \{0,\dots,q-1\}` is done with probability

        .. math::

           P(s_i(t+1)|\boldsymbol s(t)) \propto
           \exp\left(\sum_jA_{ij}w_{ij}f_{s_i(t+1), s_j(t)} + h^{(i)}_{s_i(t+1)}\right)

        Examples
        --------

        .. testsetup:: glauber-potts

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: glauber-potts

           >>> g = gt.GraphView(gt.collection.data["polblogs"].copy(), directed=False)
           >>> gt.remove_parallel_edges(g)
           >>> g = gt.extract_largest_component(g, prune=True)
           >>> f = np.eye(4) * 0.1
           >>> state = gt.PottsGlauberState(g, f)
           >>> ret = state.iterate_async(niter=1000 * g.num_vertices())
           >>> gt.graph_draw(g, g.vp.pos, vertex_fill_color=state.s,
           ...               output="glauber-potts.pdf")
           <...>

        .. testcleanup:: glauber-potts

           conv_png("glauber-potts.pdf")

        .. figure:: glauber-potts.png
           :align: center
           :width: 80%

           State of a Glauber Potts dynamics with :math:`q=4` on a political
           blog network.

        References
        ----------
        .. [potts-model] https://en.wikipedia.org/wiki/Potts_model

        """

        f = numpy.asarray(f, dtype="double")
        q = f.shape[0]
        if isinstance(w, PropertyMap):
            if w.value_type() != "double":
                w = w.copy("double")
        else:
            w = g.new_ep("double", val=w)
        if isinstance(h, PropertyMap):
            if h.value_type() != "vector<double>":
                h = h.copy("vector<double>")
        else:
            if not isinstance(h, collections.Iterable):
                h = [h] * q
            h = g.new_vp("vector<double>", val=h)
        if s is None:
            s = g.new_vp("int32_t", vals=numpy.random.randint(0, q, g.num_vertices()))
        DiscreteStateBase.__init__(self, g,
                                   lib_dynamics.make_potts_glauber_state,
                                   dict(f=f, w=w, h=h), s)

class PottsMetropolisState(DiscreteStateBase):
    def __init__(self, g, f, w=1, h=0, s=None):
        r"""Metropolis-Hastings dynamics of the Potts model.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        f : list of lists or two-dimensional :class:`numpy.ndarray`
           Matrix of interactions between spin values, of dimension
           :math:`q\times q`, where :math:`q` is the number of spins.
        w : :class:`~graph_tool.EdgePropertyMap` or ``float`` (optional, default: ``1.``)
           Edge interaction strength. If a scalar is provided, it's used for all edges.
        h : :class:`~graph_tool.VertexPropertyMap` or iterable or ``float`` (optional, default: ``0.``)
           Vertex local field. If an iterable is provided, it will be used as
           the field for all vertices. If a scalar is provided, it will be used
           for all spins values and vertices.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, a random state will be chosen.

        Notes
        -----

        This implements the Metropolis-Hastings dynamics
        [metropolis-equations-1953]_ [hastings-monte-carlo-1970]_ of the Potts
        model [potts-model]_ on a network.

        If a node :math:`i` is updated at time :math:`t`, the transition
        to state :math:`s_i(t+1) \in \{0,\dots,q-1\}` is done as follows:

        1. A spin value :math:`r` is sampled uniformly at random from the set
           :math:`\{0,\dots,q-1\}`.

        2. The transition :math:`s_i(t+1)=r` is made with probability

           .. math::
               \min\left[1, \exp\left(\sum_jA_{ij}w_{ij}(f_{r, s_j(t)}-f_{s_i(t), s_j(t)}) + h^{(i)}_{r} - h^{(i)}_{s_i(t)}\right)\right]

           otherwise we have :math:`s_i(t+1)=s_i(t)`.


        Examples
        --------

        .. testsetup:: metropolis-potts

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: metropolis-potts

           >>> g = gt.GraphView(gt.collection.data["polblogs"].copy(), directed=False)
           >>> gt.remove_parallel_edges(g)
           >>> g = gt.extract_largest_component(g, prune=True)
           >>> f = np.eye(4) * 0.1
           >>> state = gt.PottsGlauberState(g, f)
           >>> ret = state.iterate_async(niter=1000 * g.num_vertices())
           >>> gt.graph_draw(g, g.vp.pos, vertex_fill_color=state.s,
           ...               output="metropolis-potts.pdf")
           <...>

        .. testcleanup:: metropolis-potts

           conv_png("metropolis-potts.pdf")

        .. figure:: metropolis-potts.png
           :align: center
           :width: 80%

           State of a Metropolis-Hastings Potts dynamics with :math:`q=4` on a
           political blog network.

        References
        ----------
        .. [potts-model] https://en.wikipedia.org/wiki/Potts_model
        .. [metropolis-equations-1953] Metropolis, N., A.W. Rosenbluth,
           M.N. Rosenbluth, A.H. Teller, and E. Teller, "Equations of
           State Calculations by Fast Computing Machines," Journal of Chemical
           Physics, 21, 1087–1092 (1953). :doi:`10.1063/1.1699114`
        .. [hastings-monte-carlo-1970] Hastings, W.K., "Monte Carlo Sampling
           Methods Using Markov Chains and Their Applications," Biometrika, 57,
           97–109, (1970) :doi:`10.1093/biomet/57.1.97`

        """

        f = numpy.asarray(f, dtype="double")
        q = f.shape[0]
        if isinstance(w, PropertyMap):
            if w.value_type() != "double":
                w = w.copy("double")
        else:
            w = g.new_ep("double", val=w)
        if isinstance(h, PropertyMap):
            if h.value_type() != "vector<double>":
                h = h.copy("vector<double>")
        else:
            if not isinstance(h, collections.Iterable):
                h = [h] * q
            h = g.new_vp("vector<double>", val=h)
        if s is None:
            s = g.new_vp("int32_t", vals=numpy.random.randint(0, q, g.num_vertices()))
        DiscreteStateBase.__init__(self, g,
                                   lib_dynamics.make_potts_metropolis_state,
                                   dict(f=f, w=w, h=h), s)

class KirmanState(DiscreteStateBase):
    def __init__(self, g, d=.1, c1=.001, c2=.001, s=None):
        r"""Kirman's "ant colony" model.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        d : ``float`` (optional, default: ``.1``)
           Strategy infection probability.
        c1 : ``float`` (optional, default: ``.001``)
           Spontaneous transition probability to first strategy.
        c2 : ``float`` (optional, default: ``.001``)
           Spontaneous transition probability to second strategy.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, a random state will be chosen.

        Notes
        -----

        This implements Kirman's "ant colony" model [kirman_ants_1993]_ on a
        network.

        If a node :math:`i` is updated at time :math:`t`, the transition
        to state :math:`s_i(t+1) \in \{0,1\}` is done as follows:

        1. If :math:`s_i(t) = 0`, we have :math:`s_i(t) = 1` with probability
           :math:`c_1`.

        2. Otherwise if :math:`s_i(t) = 1`, we have :math:`s_i(t) = 0` with probability
           :math:`c_2`.

        3. Otherwise we have :math:`s_i(t+1) = 1 - s_i(t)` with probability

           .. math::
              1 - (1-d)^{\sum_jA_{ij}(1-\delta_{s_i(t), s_j(t)})}

        4. Otherwise we have :math:`s_i(t+1) = s_i(t)`.

        Examples
        --------

        .. testsetup:: kirman

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: kirman

           >>> g = gt.GraphView(gt.collection.data["polblogs"].copy(), directed=False)
           >>> gt.remove_parallel_edges(g)
           >>> g = gt.extract_largest_component(g, prune=True)
           >>> state = gt.KirmanState(g)
           >>> ret = state.iterate_sync(niter=1000)
           >>> gt.graph_draw(g, g.vp.pos, vertex_fill_color=state.s, vcmap=cm.bone,
           ...               output="kirman.pdf")
           <...>

        .. testcleanup:: kirman

           conv_png("kirman.pdf")

        .. figure:: kirman.png
           :align: center
           :width: 80%

           State of Kirman's model on a political blog network.

        References
        ----------
        .. [kirman_ants_1993] A. Kirman, "Ants, Rationality, and Recruitment",
           The Quarterly Journal of Economics 108, 137 (1993),
           :doi:`10.2307/2118498`.

        """
        if s is None:
            s = g.new_vp("int32_t", vals=numpy.random.randint(0, 2, g.num_vertices()))
        DiscreteStateBase.__init__(self, g,
                                   lib_dynamics.make_kirman_state,
                                   dict(d=d, c1=c1, c2=c2), s)

class AxelrodState(DiscreteStateBase):
    def __init__(self, g, f=10, q=2, r=0, s=None):
        r"""Axelrod's culture dissemination model.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        f : ``int`` (optional, default: ``10``)
           Number of features.
        q : ``int`` (optional, default: ``2``)
           Number of traits for each feature.
        r : ``float`` (optional, default: ``.0``)
           Spontaneous trait change probability.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, a random state will be chosen.

        Notes
        -----

        This implements Axelrod's model for culture dissemination
        [axelrod-dissemination-1997]_.

        Each node has a vector state :math:`\boldsymbol s^{(i)} \in
        \{0,\dots,q-1\}^f`.

        If a node :math:`i` is updated at time :math:`t`, the transition
        to state :math:`\boldsymbol s^{(i)}(t+1)` is done as follows:

        1. With probability :math:`r` a feature :math:`l` is chosen uniformly at
           random from the interval :math:`\{0,\dots,f-1\}`, and a trait
           :math:`r` is chosen uniformly at random from the interval
           :math:`\{0,\dots,q-1\}`, and the new state is set as
           :math:`s^{(i)}_l(t+1)=r`.

        2. Otherwise, a neighbour :math:`j` is chosen uniformly at random, and
           we let :math:`d` be the number of equal traits across features
           between :math:`i` and :math:`j`,

           .. math::
              d = \sum_{l=0}^{f-1} \delta_{s^{(i)}_l(t), s^{(j)}_l(t)}.

           Then with probability :math:`d/f` a trait :math:`l` is chosen
           uniformly at random from the set of differing features of size
           :math:`f-d`, i.e. :math:`\{l|s^{(i)}_l(t) \ne s^{(j)}_l(t)\}`, and
           the corresponding trait of :math:`j` is copied to :math:`i`:
           :math:`s^{(i)}_l(t+1) = s^{(j)}_l(t)`.

        3. Otherwise we have :math:`\boldsymbol s_i(t+1) = \boldsymbol s_i(t)`.

        Examples
        --------

        .. testsetup:: axelrod

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: axelrod

           >>> g = gt.GraphView(gt.collection.data["polblogs"].copy(), directed=False)
           >>> gt.remove_parallel_edges(g)
           >>> g = gt.extract_largest_component(g, prune=True)
           >>> state = gt.AxelrodState(g, f=10, q=30, r=0.005)
           >>> ret = state.iterate_async(niter=10000000)
           >>> gt.graph_draw(g, g.vp.pos,
           ...               vertex_fill_color=gt.perfect_prop_hash([state.s])[0],
           ...               vcmap=cm.magma, output="axelrod.pdf")
           <...>

        .. testcleanup:: axelrod

           conv_png("axelrod.pdf")

        .. figure:: axelrod.png
           :align: center
           :width: 80%

           State of Axelrod's model on a political blog network.

        References
        ----------
        .. [axelrod-dissemination-1997] Axelrod, R., "The Dissemination of
           Culture: A Model with Local Convergence and Global Polarization",
           Journal of Conflict Resolution, 41(2), 203–226
           (1997). :doi:`10.1177/0022002797041002001`

        """
        DiscreteStateBase.__init__(self, g,
                                   lib_dynamics.make_axelrod_state,
                                   dict(f=f, q=q, r=r), s,
                                   stype="vector<int32_t>")

class BooleanState(DiscreteStateBase):
    def __init__(self, g, f=None, p=.5, r=0, s=None):
        r"""Boolean network dynamics.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        f : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Vertex property map of type ``vector<bool>`` containing the Boolean
           functions. If not provided, the functions will be randomly chosen.
        p : ``float`` (optional, default: ``.5``)
           Output probability of random functions. This only has an effect if
           ``f is None``.
        r : ``float`` (optional, default: ``0.``)
           Input random flip probability.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, a random state will be chosen.

        Notes
        -----

        This implements a Boolean network model.

        If a node :math:`i` is updated at time :math:`t`, the transition
        to state :math:`s_i(t+1)` is given by

        .. math::

           s_i(t+1) = f^{(i)}_{\sum_{j\in \partial i}2^{\hat s_j(t)}}

        where :math:`\partial i` are the (in-)neighbors of :math:`i`, indexed
        from :math:`0` to :math:`k-1`, and :math:`\hat s_i(t)` are the flipped
        inputs sampled with probability

        .. math::

           P(\hat s_i(t)|s_i(t)) = r^{1-\delta_{\hat s_i(t),s_i(t)}}(1-r)^{\delta_{\hat s_i(t),s_i(t)}}.

        Examples
        --------

        .. testsetup:: boolean-network

           gt.seed_rng(42)
           np.random.seed(42)

        .. doctest:: boolean-network

           >>> g = gt.random_graph(50, lambda: (2,2))
           >>> state = gt.BooleanState(g)
           >>> ret = state.iterate_sync(niter=1000)
           >>> s0 = state.s.copy()
           >>> ret = state.iterate_sync(niter=1)
           >>> l = 1
           >>> while any(state.s.a != s0.a):
           ...     ret = state.iterate_sync(niter=1)
           ...     l += 1
           >>> print("Period length:", l)
           Period length: 3

        """

        if f is None:
            f = g.new_vp("vector<bool>")
        elif f.value_type() != "vector<bool>":
            f = f.copy("vector<bool>")
        DiscreteStateBase.__init__(self, g,
                                   lib_dynamics.make_boolean_state,
                                   dict(f=f, p=p, r=r), s,
                                   stype="bool")


class ContinuousStateBase(object):
    def __init__(self, g, make_state, params, t0=0, s=None, stype="double"):
        r"""Base state for continuous-time dynamics. This class it not meant to
        be instantiated directly.
        """
        self.g = g
        self.t = t0
        if s is None:
            self.s = g.new_vp(stype)
        else:
            self.s = s.copy(stype)
        self.s_diff = self.s.copy()
        self.params = params
        self._state = make_state(g._Graph__graph, _prop("v", g, self.s),
                                 _prop("v", g, self.s_diff), params, _get_rng())

    def copy(self):
        r"""Copy state."""
        return type(self)(g, s=self.s.copy(), **self.params)

    def __setstate__(self):
        return dict(g=self.g, s=self.s, params=self.params)

    def __getstate__(self, state):
        return type(self)(state["g"], s=state["s"], **state["params"])

    def get_state(self):
        r"""Returns the internal :class:`~graph_tool.VertexPropertyMap` with the current
        state."""
        return self.s

    def get_diff(self, dt):
        r"""Returns the current time derivative for all the nodes. The parameter ``dt``
        is the time interval in consideration, which is used only if the ODE has
        a stochastic component.

        If enabled during compilation, this algorithm runs in parallel.
        """
        self._state.get_diff_sync(self.t, dt, _get_rng())
        return self.s_diff.fa

    def solve(self, t, *args, **kwargs):
        r"""Integrate the system up to time ``t``. The remaining parameters are
        passed to :func:`scipy.integrate.solve_ivp`. This solver is not suitable
        for stochastic ODEs."""
        if t == self.t:
            return
        if t < self.t:
            raise ValueError("Can't integrate backwards in time")
        def f(t, y):
            self.s.fa = y.flatten()
            self.t = t
            return self.get_diff(0)
        ret = scipy.integrate.solve_ivp(f, (self.t, t), self.s.fa,
                                        **dict(kwargs, vectorized=True))
        self.t = ret.t[-1]
        self.s.fa = ret.y[:,-1]
        return ret

    def solve_euler(self, t, dt=0.001):
        r"""Integrate the system up o time ``t`` using a simple Euler's method
        with step size ``dt``. This solver is suitable for stochastic ODEs."""
        if t == self.t:
            return
        if t < self.t:
            raise ValueError("Can't integrate backwards in time")
        for t in numpy.arange(self.t, t + dt, dt):
            self.t = t
            self.s.fa += self.get_diff(dt) * dt


class KuramotoState(ContinuousStateBase):
    def __init__(self, g, omega=1, w=1, sigma=0, t0=0, s=None):
        r"""The Kuramoto model.

        Parameters
        ----------
        g : :class:`~graph_tool.Graph`
           Graph to be used for the dynamics
        omega : :class:`~graph_tool.VertexPropertyMap` or ``float`` (optional, default: ``1``)
           Intrinsic frequencies for each node. If a scalar is given, it will be
           used for all nodes.
        w : :class:`~graph_tool.EdgePropertyMap` or ``float`` (optional, default: ``1``)
           Coupling strength of each edge. If a scalar is given, it will be
           used for all edges.
        sigma : ``float`` (optional, default: ``.0``)
           Stochastic noise magnitude.
        s : :class:`~graph_tool.VertexPropertyMap` (optional, default: ``None``)
           Initial global state. If not provided, a random state will be chosen.

        Notes
        -----

        This implements Kuramoto's model for synchronization
        [kuramoto_self-entrainment_1975]_ [rodrigues_kuramoto_2016]_.

        Each node has an angle :math:`\theta_i`, which evolves in time obeying
        the differential equation:

        .. math::

           \frac{\mathrm{d}\theta_i}{\mathrm{d}t} = \omega_i + \sum_{j}A_{ij}w_{ij}\sin(\theta_j-\theta_i) + \sigma\xi_i(t),

        where :math:`\xi_i(t)` is a Gaussian noise with zero mean and unit
        variance.

        Examples
        --------

        .. testsetup:: kuramoto

           gt.seed_rng(49)
           np.random.seed(49)

        .. doctest:: kuramoto


           >>> g = gt.collection.data["karate"]
           >>> omega = g.new_vp("double", np.random.normal(0, 1, g.num_vertices())) 
           >>> state = gt.KuramotoState(g, omega=omega, w=1.5)
           >>> thetas = []
           >>> ts = linspace(0, 40, 1000)
           >>> for t in ts:
           ...     ret = state.solve(t, first_step=0.0001)
           ...     thetas.append(state.get_state().fa % (2 * pi))

           >>> figure(figsize=(6, 4))
           <...>
           >>> for v in g.vertices():
           ...    plot(ts, [t[int(v)] - t.mean() for t in thetas])
           [...]
           >>> xlabel(r"Time")
           Text(...)
           >>> ylabel(r"$\theta_i - \left<\theta\right>$")
           Text(...)
           >>> tight_layout()
           >>> savefig("karate-kuramoto.svg")

        .. figure:: karate-kuramoto.svg
           :align: center

           Kuramoto oscillator dynamics on the Karate Club network.

        References
        ----------
        .. [kuramoto_self-entrainment_1975] Y. Kuramoto, "Self-entrainment of a
           population of coupled non-linear oscillators", International
           Symposium on Mathematical Problems in Theoretical Physics. Lecture
           Notes in Physics, vol 39. Springer, Berlin, Heidelberg (1975),
           :doi:`10.1007/BFb0013365`
        .. [rodrigues_kuramoto_2016] Francisco A. Rodrigues, Thomas K. DM.Peron,
           Peng Ji, Jürgen Kurth, "The Kuramoto model in complex networks",
           Physics Reports 610 1-98 (2016) :doi:`10.1016/j.physrep.2015.10.008`,
           :arxiv:`1511.07139`

        """

        if not isinstance(omega, PropertyMap):
            omega = g.new_vp("double", val=omega)
        elif omega.value_type() != "double":
            omega = omega.copy("double")
        if not isinstance(w, PropertyMap):
            w = g.new_ep("double", val=w)
        elif w.value_type() != "double":
            w = w.copy("double")
        if s is None:
            s = g.new_vp("double",
                         vals=2 * numpy.pi * numpy.random.random(g.num_vertices()))

        ContinuousStateBase.__init__(self, g, lib_dynamics.make_kuramoto_state,
                                     dict(omega=omega, w=w, sigma=sigma), t0, s)

    def get_r_phi(self):
        r"""Returns the phase coherence :math:`r` and average phase :math:`\phi`,
        defined as

        .. math::
           re^{i\phi} = \frac{1}{N}\sum_j e^{i\theta_j}.

        """
        z = numpy.exp(self.s.fa * 1j).mean()
        return float(numpy.abs(z)), float(numpy.angle(z))