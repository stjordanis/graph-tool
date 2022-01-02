.. _planted_partition:

Assortative community structure
-------------------------------

Traditionally, `"community structure"
<https://en.wikipedia.org/wiki/Community_structure>`_ in the proper
sense refers to groups of nodes that are more connected to each other
than to nodes of other communities. The SBM is capable of representing
this kind of structure without any problems, but in some circumstances
it might make sense to search exclusively for assortative communities
[lizhi-statistical-2020]_. A version of the SBM that is constrained in
this way is called the "planted partition model", which can be inferred
with graph-tool using
:class:`~graph_tool.inference.PPBlockState`. This
class behaves just like
:class:`~graph_tool.inference.BlockState`, therefore all
algorithms described in this documentation work in the same way. Below
we show how this model can be inferred for the football network
considered previously

.. testsetup:: assortative

   import os
   try:
       os.chdir("demos/inference")
   except FileNotFoundError:
       pass
   gt.seed_rng(42)

.. testcode:: assortative

   g = gt.collection.data["football"]

   # We can use the same agglomerative heuristic as before, but we need
   # to specify PPBlockState as the internal state.

   state = gt.minimize_blockmodel_dl(g, state=gt.PPBlockState)

   # Now we run 100 sweeps of the MCMC with zero temperature, as a
   # refinement. This is often not necessary.

   state.multiflip_mcmc_sweep(beta=np.inf, niter=100)

   state.draw(pos=g.vp.pos, output="football-pp.svg")

.. figure:: football-pp.*
   :align: center
   :width: 350px

   Best fit of the degree-corrected planted partition model to a
   network of American college football teams.

It is possible to perform model comparison with other model variations
in the same manner as described in :ref:`sec_model_selection` below.
