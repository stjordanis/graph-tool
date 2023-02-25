.. _sec_faq:

FAQ
===

How do I retrieve a vertex by its property map value?
-----------------------------------------------------

There are two ways to achieve this. The first is using the
:func:`~graph_tool.util.find_vertex` function that searches for all
vertices with a given property map value. However, this function has a
:math:`O(N)` complexity, where :math:`N` is the number of nodes. This
means that it can be inefficient if the search is performed frequently.

The second approach is to perform this lookup quickly, in :math:`O(1)`
time, but it requires the user to keep her own dictionary that maps
vertices to property values, e.g.:

   >>> from collections import defaultdict
   >>> g = gt.Graph()
   >>> g.gp.vmap = g.new_gp("object", val=defaultdict(list))
   >>> def add_vertex(g, prop, value):
   ...    v = g.add_vertex()
   ...    prop[v] = value
   ...    g.gp.vmap[value].append(v)
   ...    return g
   >>> name = g.new_vp("string")
   >>> add_vertex(g, name, "bob")
   <...>
   >>> add_vertex(g, name, "eve")
   <...>
   >>> add_vertex(g, name, "steve")
   <...>
   >>> g.gp.vmap["eve"]
   [<Vertex object with index '1' at 0x...>]

.. admonition:: Why not include this functionality automatically?

   Property map values are not guaranteed to be unique, therefore
   there's no inherent bijection between property map values and their
   corresponding vertices.

   Note that, in the example above, the user still needs to manually
   modify ``g.gp.vmap`` whenever a value of the property map ``name``
   has changed. To enforce this kind of bookkeeping throughout the
   library would incur a prohibitive performance cost — not to mention
   a significant increase in complexity.

How do I cite graph-tool?
-------------------------

You can cite graph-tool in your publications as follows:

    Tiago P. Peixoto, "The graph-tool python library", figshare. (2014)
    :doi:`10.6084/m9.figshare.1164194`

Here is a more convenient `BibTeX <http://www.bibtex.org>`_ entry:

.. code-block:: none

    @article{peixoto_graph-tool_2014,
             title = {The graph-tool python library},
             url = {http://figshare.com/articles/graph_tool/1164194},
             doi = {10.6084/m9.figshare.1164194},
             urldate = {2014-09-10},
             journal = {figshare},
             author = {Peixoto, Tiago P.},
             year = {2014},
             keywords = {all, complex networks, graph, network, other}}

More information can be found at the `figshare site
<http://figshare.com/articles/graph_tool/1164194>`_.




