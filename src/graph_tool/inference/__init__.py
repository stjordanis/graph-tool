#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2019 Tiago de Paula Peixoto <tiago@skewed.de>
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


"""``graph_tool.inference`` - Statistical inference of generative network models
-----------------------------------------------------------------------------

This module contains algorithms for the identification of large-scale network
structure via the statistical inference of generative models.

.. note::

   An introduction to the concepts used here, as well as a basic HOWTO is
   included in the cookbook section: :ref:`inference-howto`.

Nonparametric stochastic block model inference
++++++++++++++++++++++++++++++++++++++++++++++

High-level functions
====================

.. autosummary::
   :nosignatures:

   ~graph_tool.inference.minimize.minimize_blockmodel_dl
   ~graph_tool.inference.minimize.minimize_nested_blockmodel_dl

State classes
=============

.. autosummary::
   :nosignatures:

   ~graph_tool.inference.blockmodel.BlockState
   ~graph_tool.inference.overlap_blockmodel.OverlapBlockState
   ~graph_tool.inference.layered_blockmodel.LayeredBlockState
   ~graph_tool.inference.nested_blockmodel.NestedBlockState
   ~graph_tool.inference.mcmc.TemperingState

Sampling and minimization
=========================

.. autosummary::
   :nosignatures:

   ~graph_tool.inference.mcmc.mcmc_equilibrate
   ~graph_tool.inference.mcmc.mcmc_anneal
   ~graph_tool.inference.mcmc.mcmc_multilevel
   ~graph_tool.inference.mcmc.multicanonical_equilibrate
   ~graph_tool.inference.mcmc.MulticanonicalState
   ~graph_tool.inference.bisection.bisection_minimize
   ~graph_tool.inference.nested_blockmodel.hierarchy_minimize

Auxiliary functions
===================

.. autosummary::
   :nosignatures:

   ~graph_tool.inference.blockmodel.model_entropy
   ~graph_tool.inference.blockmodel.mf_entropy
   ~graph_tool.inference.blockmodel.bethe_entropy
   ~graph_tool.inference.blockmodel.microstate_entropy
   ~graph_tool.inference.overlap_blockmodel.half_edge_graph
   ~graph_tool.inference.overlap_blockmodel.get_block_edge_gradient

Auxiliary classes
=================

.. autosummary::
   :nosignatures:

   ~graph_tool.inference.blockmodel.PartitionHist
   ~graph_tool.inference.blockmodel.BlockPairHist

Nonparametric network reconstruction
++++++++++++++++++++++++++++++++++++

State classes
=============

.. autosummary::
   :nosignatures:

   ~graph_tool.inference.uncertain_blockmodel.LatentMultigraphBlockState
   ~graph_tool.inference.uncertain_blockmodel.MeasuredBlockState
   ~graph_tool.inference.uncertain_blockmodel.MixedMeasuredBlockState
   ~graph_tool.inference.uncertain_blockmodel.UncertainBlockState
   ~graph_tool.inference.uncertain_blockmodel.UncertainBaseState
   ~graph_tool.inference.uncertain_blockmodel.DynamicsBlockStateBase
   ~graph_tool.inference.uncertain_blockmodel.EpidemicsBlockState
   ~graph_tool.inference.uncertain_blockmodel.IsingBaseBlockState
   ~graph_tool.inference.uncertain_blockmodel.IsingGlauberBlockState
   ~graph_tool.inference.uncertain_blockmodel.CIsingGlauberBlockState
   ~graph_tool.inference.uncertain_blockmodel.PseudoIsingBlockState
   ~graph_tool.inference.uncertain_blockmodel.PseudoCIsingBlockState

Semiparametric stochastic block model inference
+++++++++++++++++++++++++++++++++++++++++++++++

State classes
=============

.. autosummary::
   :nosignatures:

   ~graph_tool.inference.blockmodel_em.EMBlockState

Expectation-maximization Inference
==================================

.. autosummary::
   :nosignatures:

   ~graph_tool.inference.blockmodel_em.em_infer

Large-scale descriptors
+++++++++++++++++++++++

.. autosummary::
   :nosignatures:

   ~graph_tool.inference.modularity.modularity

Contents
++++++++

"""

from __future__ import division, absolute_import, print_function
import sys
if sys.version_info < (3,):
    range = xrange

__all__ = ["minimize_blockmodel_dl",
           "minimize_nested_blockmodel_dl",
           "BlockState",
           "OverlapBlockState",
           "LayeredBlockState",
           "NestedBlockState",
           "LatentMultigraphBlockState",
           "UncertainBlockState",
           "MeasuredBlockState",
           "UncertainBaseState",
           "MixedMeasuredBlockState",
           "EpidemicsBlockState",
           "IsingGlauberBlockState",
           "PseudoIsingBlockState",
           "CIsingGlauberBlockState",
           "PseudoCIsingBlockState",
           "mcmc_equilibrate",
           "mcmc_anneal",
           "mcmc_multilevel",
           "TemperingState",
           "multicanonical_equilibrate",
           "MulticanonicalState",
           "bisection_minimize",
           "hierarchy_minimize",
           "EMBlockState",
           "em_infer",
           "model_entropy",
           "mf_entropy",
           "bethe_entropy",
           "microstate_entropy",
           "PartitionHist",
           "BlockPairHist",
           "half_edge_graph",
           "get_block_edge_gradient",
           "get_hierarchy_tree",
           "modularity",
           "latent_multigraph"]

from . blockmodel import *
from . overlap_blockmodel import *
from . layered_blockmodel import *
from . nested_blockmodel import *
from . uncertain_blockmodel import *
from . mcmc import *
from . bisection import *
from . minimize import *
from . blockmodel_em import *
from . util import *
from . modularity import *
from . latent_multigraph import *
