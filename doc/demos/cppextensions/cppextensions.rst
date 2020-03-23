Writing extensions in C++
=========================

It's possible to extend graph-tool with code written in C++, as we
explain in this text. This is useful for project-specific code that is
not available in the library, but nevertheless needs to run faster than
what is possible with the pure Python API.

.. note::

   In order to write C++ extensions, a general background on `C++ with
   templates <https://en.wikipedia.org/wiki/Template_(C%2B%2B)>`_, the
   `Boost Graph Library (BGL)
   <https://www.boost.org/doc/libs/release/libs/graph/doc/>`_ and
   `boost.Python
   <https://www.boost.org/doc/libs/release/libs/python/doc/>`_ is
   strongly recommended.

A C++ extension to graph-tool consists of a compiled object file that is
imported as a Python module, and it contains one or more functions that
operate on :class:`~graph_tool.Graph` and
:class:`~graph_tool.PropertyMap` objects. We demonstrate with an example
that computes `k-cores
<https://en.wikipedia.org/wiki/Degeneracy_(graph_theory)>`_ using the
algorithm of [batagelj]_ (this is already implemented in
:func:`~graph_tool.topology.kcore_decomposition`, but serves as a good
example).

The first part is a file :download:`kcore.hh <kcore.hh>` that defines a
function template that implements the desired algorithm:

.. literalinclude:: kcore.hh
   :language: c++
   :linenos:

The second part is a compilation unit :download:`kcore.cc <kcore.cc>`
that sets up the Python-C++ interface:

.. literalinclude:: kcore.cc
   :language: c++
   :linenos:

When using the `gt_dispatch<>()` function, we can make use of the
following type lists for the graph views:

.. tabularcolumns:: |l|l|

.. table::

    ==================================  ======================
    List of graph views                 Description
    ==================================  ======================
    ``all_graph_views``                 All possible graph views
    ``always_directed``                 All directed graph views 
    ``never_directed``                  All undirected graph views
    ``always_reversed``                 All directed but reversed graph views
    ``never_reversed``                  All undirected, and directed but not reversed graph views
    ``always_directed_never_reversed``  All directed but not reversed graph views
    ``never_filtered``                  All unfiltered graph views
    ``never_filtered_never_reversed``   All unfiltered and not reversed graph views
    ==================================  ======================

Likewise, for the types of property maps we have:

.. tabularcolumns:: |l|l|

.. table::

    =====================================  ======================
    List of property maps                  Description
    =====================================  ======================
    ``vertex_properties``                  All vertex property maps
    ``edge_properties``                    All edge property maps
    ``writable_vertex_properties``         All writable vertex property maps
    ``writable_edge_properties``           All writable edge property maps
    ``vertex_scalar_properties``           All scalar-valued vertex property maps
    ``edge_scalar_properties``             All scalar-valued edge property maps
    ``writable_vertex_scalar_properties``  All writable scalar-valued vertex property maps
    ``writable_edge_scalar_properties``    All writable scalar-valued edge property maps
    ``vertex_integer_properties``          All integer-valued vertex property maps
    ``edge_integer_properties``            All integer-valued edge property maps
    ``vertex_floating_properties``         All floating-point-valued vertex property maps
    ``edge_floating_properties``           All floating-point-valued edge property maps
    ``vertex_scalar_vector_properties``    All vertex property maps with vector of scalar types
    ``edge_scalar_vector_properties``      All edge property maps with vector of scalar types
    ``vertex_integer_vector_properties``   All vertex property maps with vector of integer types
    ``edge_integer_vector_properties``     All edge property maps with vector of integer types
    ``vertex_floating_vector_properties``  All vertex property maps with vector of floating-point types
    ``edge_floating_vector_properties``    All edge property maps with vector of floating-point types
    =====================================  ======================

Finally, we need to bind things from the Python side with the file
:download:`kcore.py <kcore.py>`:

.. literalinclude:: kcore.py
   :linenos:

Naturally, before we can use the extension, we need to compile it. To do
this, we need to inform the compiler where to find the graph-tool
include files. For that we use the `pkg-config
<https://www.freedesktop.org/wiki/Software/pkg-config/>`_
infrastructure, by calling ``pkg-config --cflags --libs graph-tool-py3.7``
(in case graph-tool was compiled for Python 3.7). To keep things
organized, we create a :download:`Makefile <Makefile>`:

.. literalinclude:: Makefile
   :language: make
   :linenos:

(If you use MacOS, you may want to use a different compiler such as
``clang++``.) After compiling by typing ``make``, we can import and use
the ``kcore.py`` module:

.. testsetup:: kcore

   import os, sys
   try:
       os.chdir("demos/cppextensions")
   except FileNotFoundError:
       pass
   import subprocess
   subprocess.call("make")
   sys.path.append(".")

.. doctest:: kcore

   >>> from kcore import kcore_decomposition
   >>> g = gt.collection.data["football"]
   >>> c = kcore_decomposition(g)
   >>> print(c.a)
   [8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8
    8 8 8 8 8 7 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8
    8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8
    8 8 8 8]

Range-based iteration over vertices and edges
+++++++++++++++++++++++++++++++++++++++++++++

Users familiar with the `BGL
<https://www.boost.org/doc/libs/release/libs/graph/doc/>`_ may have
noticed that iteration over vertices and edges was done in a peculiar
way in the example above. The standard BGL way of iterating is something
like:

.. code-block:: c++

   auto rg = vertices(g);
   for(auto vi = rg.first; vi != rg.second; ++vi)
   {
       auto v = *vi;  // get the vertex descriptor
   }

In graph-tool iteration can also be done in a more convenient way using
`range-based iteration
<https://en.cppreference.com/w/cpp/language/range-for>`_:

.. code-block:: c++

   for(auto v : vertices_range(g))
   {
       // v is the vertex descriptor
   }

Both approaches above are equivalent. Graph-tool also provides
``edges_range()``, ``out_edges_range()``, ``in_edges_range()``, etc., in
an analogous fashion.
   
Extracting specific property maps
+++++++++++++++++++++++++++++++++

Sometimes it's more convenient to extract property map of known types
directly instead of relying on ``gt_dispatch<>()``. This can be done
simply by calling ``boost::any_cast`` as follows:

.. code-block:: c++

   void kcore_bind(GraphInterface& gi, boost::any core_map)
   {
       // Vertex property map of type int32_t
       typedef typename vprop_map_t<int32_t>::type vprop_t;
       vprop_t c = boost::any_cast<vprop_t>(core_map);

       gt_dispatch<>()
           ([&](auto& g){ kcore_decomposition(g, c.get_unchecked()); },
            all_graph_views()) (gi.get_graph_view());
   };
              
Checked and unchecked property maps
+++++++++++++++++++++++++++++++++++

Property maps in graph-tool can either be 'checked' or 'unchecked', when
they are seen from the C++ side. A checked property map automatically
resizes itself if the underlying graph changes (via the addition of
nodes or edges). However this involves bounds checking for every lookup,
which can result in performance hit. For static graphs, it's better to
use unchecked property maps, which do not perform bounds checking (but
do not adapt to a changing graph).

The default property map types are by default `checked`:

.. code-block:: c++

    // vprop_t and eprop_t below are checked vertex and edge
    // property maps of type int32_t
                
    typedef typename vprop_map_t<int32_t>::type vprop_t;
    typedef typename eprop_map_t<int32_t>::type eprop_t;

The type hidden in a ``boost::any`` that comes from Python is a checked
property map. **However** the types propagated by ``gt_dispatch<>()``
are by default 'unchecked'. It's easy to obtain checked and unchecked
versions of each kind of property map:

.. code-block:: c++

    // core_map is of type boost::any

    typedef typename vprop_map_t<int32_t>::type vprop_t;
    vprop_t c = boost::any_cast<vprop_t>(core_map);

    // c is a checked property map. We can obtain an unchecked version
    // as follows:

    auto uc = c.get_unchecked();   // The type of uc is vprop_map_t<int32_t>::type::unchecked_t

    // If the underlying graph has changed size in the mean-time, we
    // need to specify the new size when creating the unchecked map:

    auto uc2 = c.get_unchecked(10000);

    // From a checked property map, we can get the checked type in a similar fashion:

    auto c2 = uc.get_checked();

    // c and c2 have the same type.
    
References
++++++++++

.. [batagelj] Vladimir Batagelj, Matjaž Zaveršnik, "Fast
   algorithms for determining (generalized) core groups in social
   networks", Advances in Data Analysis and Classification Volume 5,
   Issue 2, pp 129-145 (2011), :DOI:`10.1007/s11634-010-0079-y`,
   :arxiv:`cs/0310049`
