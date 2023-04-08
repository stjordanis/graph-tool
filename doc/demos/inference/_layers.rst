Layered networks
----------------

The edges of the network may be distributed in discrete "layers",
representing distinct types if interactions
[peixoto-inferring-2015]_. Extensions to the SBM may be defined for such
data, and they can be inferred using the exact same interface shown
above, except one should use the
:class:`~graph_tool.inference.LayeredBlockState`
class, instead of
:class:`~graph_tool.inference.BlockState`. This class takes
two additional parameters: the ``ec`` parameter, that must correspond to
an edge :class:`~graph_tool.PropertyMap` with the layer/covariate values
on the edges, and the Boolean ``layers`` parameter, which if ``True``
specifies a layered model, otherwise one with categorical edge
covariates (not to be confused with the weighted models in
Sec. :ref:`weights`).

If we use :func:`~graph_tool.inference.minimize_blockmodel_dl`, this can
be achieved simply by passing the option ``layers=True`` as well as the
appropriate value of ``state_args``, which will be propagated to
:class:`~graph_tool.inference.LayeredBlockState`'s constructor.

As an example, let us consider a social network of tribes, where two
types of interactions were recorded, amounting to either friendship or
enmity [read-cultures-1954]_. We may apply the layered model by
separating these two types of interactions in two layers:

.. testsetup:: layered-model

   import os
   try:
       os.chdir("demos/inference")
   except FileNotFoundError:
       pass
   gt.seed_rng(42)
         
.. testcode:: layered-model

   g = gt.collection.ns["new_guinea_tribes"]

   # The edge types are stored in the edge property map "weights".

   # Note the different meanings of the two 'layers' parameters below: The
   # first enables the use of LayeredBlockState, and the second selects
   # the 'edge layers' version (instead of 'edge covariates').

   state = gt.minimize_nested_blockmodel_dl(g,                                            
                                            state_args=dict(base_type=gt.LayeredBlockState,
                                                            state_args=dict(ec=g.ep.weight, layers=True)))

   state.draw(edge_color=g.ep.weight.copy("double"), edge_gradient=[],
              ecmap=(matplotlib.cm.coolwarm_r, .6), edge_pen_width=5,  eorder=g.ep.weight,
              output="tribes-sbm-edge-layers.svg")

.. figure:: tribes-sbm-edge-layers.*
   :align: center
   :width: 350px

   Best fit of the degree-corrected SBM with edge layers for a network
   of tribes, with edge layers shown as colors. The groups show two
   enemy tribes.

It is possible to perform model averaging of all layered variants
exactly like for the regular SBMs as was shown above.
