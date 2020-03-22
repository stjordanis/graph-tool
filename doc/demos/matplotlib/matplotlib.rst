Integration with matplotlib
===========================

The drawing capabilities of ``graph-tool`` (see :mod:`~graph_tool.draw`
module) can be integrated with `matplotlib <https://matplotlib.org>`_,
as we demonstrate in the following.

.. note::

   Since ``graph-tool`` uses `cairo <https://cairographics.org>`_ for
   drawing, it is necessary to use matplotlib's cairo backend, otherwise
   integration is not possible. The cairo backend can be used by calling:

   .. code::

      switch_backend("cairo")


Drawing with matplotlib is done by calling
:func:`~graph_tool.draw.graph_draw` and passing a container (e.g.
:class:`matplotlib.axes.Axes`) as the ``mplfig`` parameter. For example:

.. testsetup::

   gt.seed_rng(44)
   
.. testcode::

   switch_backend("cairo")

   fig, ax = subplots(2, 2, figsize=(12, 11.5)) 

   g = gt.collection.data["polbooks"]

   gt.graph_draw(g, g.vp.pos, vertex_size=1.5, mplfig=ax[0,0])

   ax[0,0].set_xlabel("$x$ coordinate")
   ax[0,0].set_ylabel("$y$ coordinate")

   state = gt.minimize_nested_blockmodel_dl(g)

   state.draw(mplfig=ax[0,1])

   ax[0,1].set_xlabel("$x$ coordinate")
   ax[0,1].set_ylabel("$y$ coordinate")

   g = gt.collection.data["lesmis"]
   gt.graph_draw(g, g.vp.pos, vertex_size=1.5, mplfig=ax[1,0])

   ax[1,0].set_xlabel("$x$ coordinate")
   ax[1,0].set_ylabel("$y$ coordinate")

   state = gt.minimize_nested_blockmodel_dl(g)

   state.draw(mplfig=ax[1,1])

   ax[1,1].set_xlabel("$x$ coordinate")
   ax[1,1].set_ylabel("$y$ coordinate")

   subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.06)
   fig.savefig("gt-mpl.pdf")


.. testcleanup::

   conv_png("gt-mpl.pdf")


.. figure:: gt-mpl.png
   :width: 90%

   Four subplots showing networks drawn using graph-tool.