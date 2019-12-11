Inferring the best partition
----------------------------

The simplest and most efficient approach is to find the best
partition of the network by maximizing Eq. :eq:`model-posterior`
according to some version of the model. This is obtained via the
functions :func:`~graph_tool.inference.minimize.minimize_blockmodel_dl` or
:func:`~graph_tool.inference.minimize.minimize_nested_blockmodel_dl`, which
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
   gt.seed_rng(8)

.. testcode:: football

   g = gt.collection.data["football"]
   print(g)

which yields

.. testoutput:: football

   <Graph object, undirected, with 115 vertices and 613 edges at 0x...>

we then fit the degree-corrected model by calling

.. testcode:: football

   state = gt.minimize_blockmodel_dl(g)

This returns a :class:`~graph_tool.inference.blockmodel.BlockState` object that
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
   obtained with the :meth:`~graph_tool.inference.blockmodel.BlockState.entropy`
   method. See also Sec. :ref:`sec_model_selection` below.


We may perform a drawing of the partition obtained via the
:mod:`~graph_tool.inference.blockmodel.BlockState.draw` method, that functions as a
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
:mod:`~graph_tool.inference.blockmodel.BlockState.get_blocks` method:

.. testcode:: football

   b = state.get_blocks()
   r = b[10]   # group membership of vertex 10
   print(r)

which yields:

.. testoutput:: football

   3

We may also access the matrix of edge counts between groups via
:mod:`~graph_tool.inference.blockmodel.BlockState.get_matrix`

.. testcode:: football

   e = state.get_matrix()

   matshow(e.todense())
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
:func:`~graph_tool.inference.minimize.minimize_nested_blockmodel_dl` function. We
illustrate its use with the neural network of the `C. elegans
<https://en.wikipedia.org/wiki/Caenorhabditis_elegans>`_ worm:

.. testsetup:: celegans

   gt.seed_rng(51)

.. testcode:: celegans

   g = gt.collection.data["celegansneural"]
   print(g)

which has 297 vertices and 2359 edges.

.. testoutput:: celegans

   <Graph object, directed, with 297 vertices and 2359 edges at 0x...>

A hierarchical fit of the degree-corrected model is performed as follows.

.. testcode:: celegans

   state = gt.minimize_nested_blockmodel_dl(g)

The object returned is an instance of a
:class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` class, which
encapsulates the results. We can again draw the resulting hierarchical
clustering using the
:meth:`~graph_tool.inference.nested_blockmodel.NestedBlockState.draw` method:

.. testcode:: celegans

   state.draw(output="celegans-hsbm-fit.svg")

.. figure:: celegans-hsbm-fit.*
   :align: center

   Most likely hierarchical partition of the neural network of
   the *C. elegans* worm according to the nested degree-corrected SBM.

.. note::

   If the ``output`` parameter to
   :meth:`~graph_tool.inference.nested_blockmodel.NestedBlockState.draw` is omitted, an
   interactive visualization is performed, where the user can re-order
   the hierarchy nodes using the mouse and pressing the ``r`` key.

A summary of the inferred hierarchy can be obtained with the
:meth:`~graph_tool.inference.nested_blockmodel.NestedBlockState.print_summary` method,
which shows the number of nodes and groups in all levels:

.. testcode:: celegans

   state.print_summary()

.. testoutput:: celegans

   l: 0, N: 297, B: 16
   l: 1, N: 16, B: 8
   l: 2, N: 8, B: 3
   l: 3, N: 3, B: 1

The hierarchical levels themselves are represented by individual
:meth:`~graph_tool.inference.blockmodel.BlockState` instances obtained via the
:meth:`~graph_tool.inference.nested_blockmodel.NestedBlockState.get_levels()` method:

.. testcode:: celegans

   levels = state.get_levels()
   for s in levels:
       print(s)

.. testoutput:: celegans

    <BlockState object with 16 blocks (16 nonempty), degree-corrected, for graph <Graph object, directed, with 297 vertices and 2359 edges at 0x...>, at 0x...>
    <BlockState object with 8 blocks (8 nonempty), for graph <Graph object, directed, with 16 vertices and 134 edges at 0x...>, at 0x...>
    <BlockState object with 3 blocks (3 nonempty), for graph <Graph object, directed, with 8 vertices and 50 edges at 0x...>, at 0x...>
    <BlockState object with 1 blocks (1 nonempty), for graph <Graph object, directed, with 3 vertices and 8 edges at 0x...>, at 0x...>

This means that we can inspect the hierarchical partition just as before:

.. testcode:: celegans

   r = levels[0].get_blocks()[46]    # group membership of node 46 in level 0
   print(r)
   r = levels[1].get_blocks()[r]     # group membership of node 46 in level 1
   print(r)
   r = levels[2].get_blocks()[r]     # group membership of node 46 in level 2
   print(r)

.. testoutput:: celegans

   2
   1
   0
