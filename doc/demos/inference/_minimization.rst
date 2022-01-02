Inferring the best partition
----------------------------

The simplest and most efficient approach is to find the best
partition of the network by maximizing Eq. :eq:`model-posterior`
according to some version of the model. This is obtained via the
functions :func:`~graph_tool.inference.minimize_blockmodel_dl` or
:func:`~graph_tool.inference.minimize_nested_blockmodel_dl`, which
employs an agglomerative multilevel `Markov chain Monte Carlo (MCMC)
<https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ algorithm
[peixoto-efficient-2014]_.

We focus first on the non-nested model, and we illustrate its use with a
network of American football teams, which we load from the
:mod:`~graph_tool.collection` module:

.. testsetup:: football

   import os
   try:
      os.chdir("demos/inference")
   except FileNotFoundError:
       pass
   np.random.seed(42)
   gt.seed_rng(42)

.. testcode:: football

   g = gt.collection.data["football"]
   print(g)

which yields

.. testoutput:: football

   <Graph object, undirected, with 115 vertices and 613 edges, 4 internal vertex properties, 2 internal graph properties, at 0x...>

We then fit the degree-corrected model by calling:

.. testcode:: football

   state = gt.minimize_blockmodel_dl(g)

This returns a :class:`~graph_tool.inference.BlockState` object that
includes the inference results.

.. note::

   The inference algorithm used is stochastic by nature, and may return
   a different answer each time it is run. This may be due to the fact
   that there are alternative partitions with similar probabilities, or
   that the optimum is difficult to find. Note that the inference
   problem here is, in general, `NP-Hard
   <https://en.wikipedia.org/wiki/NP-hardness>`_, hence there is no
   efficient algorithm that is guaranteed to always find the best
   answer.

   Because of this, typically one would call the algorithm many times,
   and select the partition with the largest posterior probability of
   Eq. :eq:`model-posterior`, or equivalently, the minimum description
   length of Eq. :eq:`model-dl`. The description length of a fit can be
   obtained with the :meth:`~graph_tool.inference.BlockState.entropy`
   method. See also Sec. :ref:`sec_model_selection` below.


We may perform a drawing of the partition obtained via the
:mod:`~graph_tool.inference.BlockState.draw` method, that functions as a
convenience wrapper to the :func:`~graph_tool.draw.graph_draw` function

.. testcode:: football

   state.draw(pos=g.vp.pos, output="football-sbm-fit.svg")

which yields the following image.

.. figure:: football-sbm-fit.*
   :align: center
   :width: 400px

   Stochastic block model inference of a network of American college
   football teams. The colors correspond to inferred group membership of
   the nodes.

We can obtain the group memberships as a
:class:`~graph_tool.PropertyMap` on the vertices via the
:mod:`~graph_tool.inference.BlockState.get_blocks` method:

.. testcode:: football

   b = state.get_blocks()
   r = b[10]   # group membership of vertex 10
   print(r)

which yields:

.. testoutput:: football

   82

.. note::

   For reasons of algorithmic efficiency, the group labels returned are
   not necessarily contiguous, and they may lie in any subset of the
   range :math:`[0, N-1]`, where :math:`N` is the number of nodes in the
   network.

We may also access the matrix of edge counts between groups via
:mod:`~graph_tool.inference.BlockState.get_matrix`

.. testcode:: football

   # let us obtain a contiguous range first, which will facilitate
   # visualization

   b = gt.contiguous_map(state.get_blocks())
   state = state.copy(b=b)
              
   e = state.get_matrix()

   B = state.get_nonempty_B()
   matshow(e.todense()[:B, :B])
   savefig("football-edge-counts.svg")

.. figure:: football-edge-counts.*
   :align: center

   Matrix of edge counts between groups.

We may obtain the same matrix of edge counts as a graph, which has
internal edge and vertex property maps with the edge and vertex counts,
respectively:

.. testcode:: football

   bg = state.get_bg()
   ers = state.mrs    # edge counts
   nr = state.wr      # node counts

.. _sec_model_selection:

Hierarchical partitions
+++++++++++++++++++++++

The inference of the nested family of SBMs is done in a similar manner,
but we must use instead the
:func:`~graph_tool.inference.minimize_nested_blockmodel_dl` function. We
illustrate its use with the neural network of the `C. elegans
<https://en.wikipedia.org/wiki/Caenorhabditis_elegans>`_ worm:

.. testsetup:: celegans

   gt.seed_rng(44)

.. testcode:: celegans

   g = gt.collection.data["celegansneural"]
   print(g)

which has 297 vertices and 2359 edges.

.. testoutput:: celegans

   <Graph object, directed, with 297 vertices and 2359 edges, 2 internal vertex properties, 1 internal edge property, 2 internal graph properties, at 0x...>

A hierarchical fit of the degree-corrected model is performed as follows.

.. testcode:: celegans

   state = gt.minimize_nested_blockmodel_dl(g)

The object returned is an instance of a
:class:`~graph_tool.inference.NestedBlockState` class, which
encapsulates the results. We can again draw the resulting hierarchical
clustering using the
:meth:`~graph_tool.inference.NestedBlockState.draw` method:

.. testcode:: celegans

   state.draw(output="celegans-hsbm-fit.pdf")

.. testcleanup:: celegans

   conv_png("celegans-hsbm-fit.pdf")
                 

.. figure:: celegans-hsbm-fit.png
   :align: center
   :width: 80%

   Most likely hierarchical partition of the neural network of
   the *C. elegans* worm according to the nested degree-corrected SBM.

.. note::

   If the ``output`` parameter to
   :meth:`~graph_tool.inference.NestedBlockState.draw` is omitted, an
   interactive visualization is performed, where the user can re-order
   the hierarchy nodes using the mouse and pressing the ``r`` key.

A summary of the inferred hierarchy can be obtained with the
:meth:`~graph_tool.inference.NestedBlockState.print_summary` method,
which shows the number of nodes and groups in all levels:

.. testcode:: celegans

   state.print_summary()

.. testoutput:: celegans

   l: 0, N: 297, B: 15
   l: 1, N: 15, B: 6
   l: 2, N: 6, B: 2
   l: 3, N: 2, B: 1
   l: 4, N: 1, B: 1

The hierarchical levels themselves are represented by individual
:meth:`~graph_tool.inference.BlockState` instances obtained via the
:meth:`~graph_tool.inference.NestedBlockState.get_levels()` method:

.. testcode:: celegans

   levels = state.get_levels()
   for s in levels:
       print(s)
       if s.get_N() == 1:
           break

.. testoutput:: celegans

   <BlockState object with 297 blocks (15 nonempty), degree-corrected, for graph <Graph object, directed, with 297 vertices and 2359 edges, 2 internal vertex properties, 1 internal edge property, 2 internal graph properties, at 0x...>, at 0x...>
   <BlockState object with 15 blocks (6 nonempty), for graph <Graph object, directed, with 297 vertices and 125 edges, 2 internal vertex properties, 1 internal edge property, at 0x...>, at 0x...>
   <BlockState object with 8 blocks (2 nonempty), for graph <Graph object, directed, with 15 vertices and 32 edges, 2 internal vertex properties, 1 internal edge property, at 0x...>, at 0x...>
   <BlockState object with 2 blocks (1 nonempty), for graph <Graph object, directed, with 8 vertices and 4 edges, 2 internal vertex properties, 1 internal edge property, at 0x...>, at 0x...>
   <BlockState object with 1 blocks (1 nonempty), for graph <Graph object, directed, with 2 vertices and 1 edge, 2 internal vertex properties, 1 internal edge property, at 0x...>, at 0x...>

This means that we can inspect the hierarchical partition just as before:

.. testcode:: celegans

   r = levels[0].get_blocks()[46]    # group membership of node 46 in level 0
   print(r)
   r = levels[1].get_blocks()[r]     # group membership of node 46 in level 1
   print(r)
   r = levels[2].get_blocks()[r]     # group membership of node 46 in level 2
   print(r)

.. testoutput:: celegans

   278
   4
   4

Refinements using merge-split MCMC
++++++++++++++++++++++++++++++++++

The agglomerative algorithm behind
:func:`~graph_tool.inference.minimize_blockmodel_dl` and
:func:`~graph_tool.inference.minimize_nested_blockmodel_dl` has
a log-linear complexity on the size of the network, and it usually works
very well in finding a good estimate of the optimum
partition. Nevertheless, it's often still possible to find refinements
without starting the whole algorithm from scratch using a greedy
algorithm based on a merge-split MCMC with zero temperature
[peixoto-merge-split-2020]_. This is achieved by following the
instructions in Sec. :ref:`sampling`, while setting the inverse
temperature parameter ``beta`` to infinity. For example, an equivalent
to the above minimization for the `C. elegans` network is the following:

.. testcode:: celegans-mcmc

   g = gt.collection.data["celegansneural"]

   state = gt.minimize_nested_blockmodel_dl(g)

   S1 = state.entropy()
   
   for i in range(1000): # this should be sufficiently large
       state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

   S2 = state.entropy()

   print("Improvement:", S2 - S1)

.. testoutput:: celegans-mcmc

   Improvement: -36.887009...
      
Whenever possible, this procedure should be repeated several times, and
the result with the smallest description length (obtained via the
:meth:`~graph_tool.inference.BlockState.entropy` method)
should be chosen. In more demanding situations, better results still can
be obtained, at the expense of a longer computation time, by using the
:meth:`~graph_tool.inference.mcmc_anneal` function, which
implements `simulated annealing
<https://en.wikipedia.org/wiki/Simulated_annealing>`_:

.. testcode:: celegans-mcmc-anneal

   g = gt.collection.data["celegansneural"]

   state = gt.minimize_nested_blockmodel_dl(g)

   gt.mcmc_anneal(state, beta_range=(1, 10), niter=1000, mcmc_equilibrate_args=dict(force_niter=10))
