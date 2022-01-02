#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2022 Tiago de Paula Peixoto <tiago@skewed.de>
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

.. currentmodule:: graph_tool.inference

.. autosummary::
   :nosignatures:

   minimize_blockmodel_dl
   minimize_nested_blockmodel_dl

State classes
=============

.. autosummary::
   :nosignatures:

   BlockState
   OverlapBlockState
   LayeredBlockState
   NestedBlockState
   PPBlockState
   ModularityState
   NormCutState
   TemperingState
   CliqueState

Abstract base classes
=====================
.. autosummary::
   :nosignatures:

   MCMCState
   MultiflipMCMCState
   MultilevelMCMCState
   GibbsMCMCState
   MulticanonicalMCMCState
   ExhaustiveSweepState

Sampling and minimization
=========================

.. autosummary::
   :nosignatures:

   mcmc_equilibrate
   mcmc_anneal
   multicanonical_equilibrate
   MulticanonicalState

Comparing and manipulating partitions
=====================================

.. autosummary::
   :nosignatures:

   PartitionModeState
   ModeClusterState
   PartitionCentroidState
   partition_overlap
   nested_partition_overlap
   variation_information
   mutual_information
   reduced_mutual_information
   contingency_graph
   shuffle_partition_labels
   order_partition_labels
   order_nested_partition_labels
   align_partition_labels
   align_nested_partition_labels
   partition_overlap_center
   nested_partition_overlap_center
   nested_partition_clear_null
   contiguous_map
   nested_contiguous_map

Auxiliary functions
===================

.. autosummary::
   :nosignatures:

   mf_entropy
   bethe_entropy
   microstate_entropy
   marginal_multigraph_entropy
   half_edge_graph
   get_block_edge_gradient

Auxiliary classes
=================

.. autosummary::
   :nosignatures:

   PartitionHist
   BlockPairHist

Nonparametric network reconstruction
++++++++++++++++++++++++++++++++++++

State classes
=============

.. autosummary::
   :nosignatures:

   LatentMultigraphBlockState
   LatentClosureBlockState
   MeasuredBlockState
   MeasuredClosureBlockState
   MixedMeasuredBlockState
   UncertainBlockState
   UncertainBaseState
   DynamicsBlockStateBase
   EpidemicsBlockState
   IsingBaseBlockState
   IsingGlauberBlockState
   CIsingGlauberBlockState
   PseudoIsingBlockState
   PseudoCIsingBlockState

Expectation-maximization Inference
==================================

.. autosummary::
   :nosignatures:

   latent_multigraph

Semiparametric stochastic block model inference
+++++++++++++++++++++++++++++++++++++++++++++++

State classes
=============

.. autosummary::
   :nosignatures:

   EMBlockState

Expectation-maximization Inference
==================================

.. autosummary::
   :nosignatures:

   em_infer

Large-scale descriptors
+++++++++++++++++++++++

.. autosummary::
   :nosignatures:

   modularity

Contents
++++++++

"""

__all__ = ["minimize_blockmodel_dl",
           "minimize_nested_blockmodel_dl",
           "BlockState",
           "OverlapBlockState",
           "LayeredBlockState",
           "NestedBlockState",
           "PPBlockState",
           "PartitionCentroidState",
           "PartitionModeState",
           "ModeClusterState",
           "ModularityState",
           "NormCutState",
           "LatentMultigraphBlockState",
           "UncertainBlockState",
           "MeasuredBlockState",
           "UncertainBaseState",
           "MixedMeasuredBlockState",
           "DynamicsBlockStateBase",
           "EpidemicsBlockState",
           "IsingBaseBlockState",
           "IsingGlauberBlockState",
           "PseudoIsingBlockState",
           "CIsingGlauberBlockState",
           "PseudoCIsingBlockState",
           "LatentLayerBaseState",
           "LatentClosureBlockState",
           "MeasuredClosureBlockState",
           "HistState",
           "CliqueState",
           "MCMCState",
           "MultiflipMCMCState",
           "MultilevelMCMCState",
           "GibbsMCMCState",
           "MulticanonicalMCMCState",
           "ExhaustiveSweepState",
           "DrawBlockState",
           "mcmc_equilibrate",
           "mcmc_anneal",
           "TemperingState",
           "multicanonical_equilibrate",
           "MulticanonicalState",
           "EMBlockState",
           "em_infer",
           "mf_entropy",
           "bethe_entropy",
           "microstate_entropy",
           "marginal_multigraph_entropy",
           "marginal_multigraph_sample",
           "marginal_graph_sample",
           "marginal_multigraph_lprob",
           "marginal_graph_lprob",
           "PartitionHist",
           "BlockPairHist",
           "half_edge_graph",
           "get_block_edge_gradient",
           "get_hierarchy_tree",
           "modularity",
           "latent_multigraph",
           "partition_overlap",
           "nested_partition_overlap",
           "contingency_graph",
           "contiguous_map",
           "nested_contiguous_map",
           "align_partition_labels",
           "align_nested_partition_labels",
           "shuffle_nested_partition_labels",
           "shuffle_partition_labels",
           "order_partition_labels",
           "order_nested_partition_labels",
           "partition_overlap_center",
           "nested_partition_overlap_center",
           "nested_partition_clear_null",
           "variation_information",
           "mutual_information",
           "reduced_mutual_information"]

from . base_states import *
from . blockmodel import *
from . overlap_blockmodel import *
from . layered_blockmodel import *
from . nested_blockmodel import *
from . uncertain_blockmodel import *
from . mcmc import *
from . minimize import *
from . blockmodel_em import *
from . util import *
from . modularity import *
from . norm_cut import *
from . latent_multigraph import *
from . partition_centroid import *
from . partition_modes import *
from . planted_partition import *
from . latent_layers import *
from . histogram import *
from . clique_decomposition import *
