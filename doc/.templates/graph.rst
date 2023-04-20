{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if objtype  == "class" %}

{% if objname != "Graph" %}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:

   {% block methods %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}

   {% if methods %}
   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}
   {% endif %}

   {% if attributes %}
   {% for item in attributes %}
   .. autoattribute:: {{ item }}
   {%- endfor %}
   {% endif %}

   {% endblock %}

{% else %}

.. autoclass:: Graph
   :no-members:
   :no-undoc-members:

   .. automethod:: copy

   .. rubric:: Iterating over vertices and edges

   See :ref:`sec_iteration` for more documentation and examples.

   Iterator-based interface with descriptors:

   .. automethod:: vertices
   .. automethod:: edges

   Iterator-based interface without descriptors:

   .. automethod:: iter_vertices
   .. automethod:: iter_edges

   .. automethod:: iter_out_edges
   .. automethod:: iter_in_edges
   .. automethod:: iter_all_edges

   .. automethod:: iter_out_neighbors
   .. automethod:: iter_in_neighbors
   .. automethod:: iter_all_neighbors

   Array-based interface:

   .. automethod:: get_vertices
   .. automethod:: get_edges
   .. automethod:: get_out_edges
   .. automethod:: get_in_edges
   .. automethod:: get_all_edges
   .. automethod:: get_out_neighbors
   .. automethod:: get_in_neighbors
   .. automethod:: get_all_neighbors
   .. automethod:: get_out_degrees
   .. automethod:: get_in_degrees
   .. automethod:: get_total_degrees

   .. rubric:: Obtaining vertex and edge descriptors

   .. automethod:: vertex
   .. automethod:: edge

   .. rubric:: Number of vertices and edges

   .. automethod:: num_vertices
   .. automethod:: num_edges

   .. rubric:: Modifying vertices and edges

   The following functions allow for addition and removal of
   vertices in the graph.

   .. automethod:: add_vertex
   .. automethod:: remove_vertex

   The following functions allow for addition and removal of
   edges in the graph.

   .. automethod:: add_edge
   .. automethod:: remove_edge
   .. automethod:: add_edge_list

   .. automethod:: set_fast_edge_removal
   .. automethod:: get_fast_edge_removal

   The following functions allow for easy removal of vertices and
   edges from the graph.

   .. automethod:: clear
   .. automethod:: clear_vertex
   .. automethod:: clear_edges

   After the removal of many edges and/or vertices, the underlying
   containers may have a capacity that significantly exceeds the size
   of the graph. The function below corrects this.

   .. automethod:: shrink_to_fit

   .. rubric:: Directedness and reversal of edges

   .. note::

      These functions do not actually modify the graph, and are fully
      reversible. They are also very cheap, with an :math:`O(1)`
      complexity.

   .. automethod:: set_directed
   .. automethod:: is_directed

   .. automethod:: set_reversed
   .. automethod:: is_reversed


   .. rubric:: Creation of new property maps

   .. automethod:: new_property
   .. automethod:: new_vertex_property
   .. automethod:: new_vp
   .. automethod:: new_edge_property
   .. automethod:: new_ep
   .. automethod:: new_graph_property
   .. automethod:: new_gp

   New property maps can be created by copying already existing
   ones.

   .. automethod:: copy_property

   .. automethod:: degree_property_map

   .. rubric:: Index property maps

   .. autoattribute:: vertex_index
   .. autoattribute:: edge_index
   .. autoattribute:: edge_index_range
   .. automethod:: reindex_edges

   .. rubric:: Internal property maps

   Internal property maps are just like regular property maps, with
   the only exception that they are saved and loaded to/from files
   together with the graph itself. See :ref:`internal property maps <sec_internal_props>`
   for more details.

   .. note::

      All dictionaries below are mutable. However, any dictionary
      returned below is only an one-way proxy to the internally-kept
      properties. If you modify this object, the change will be
      propagated to the internal dictionary, but not
      vice-versa. Keep this in mind if you intend to keep a copy of
      the returned object.

   .. autoattribute:: properties
   .. autoattribute:: vertex_properties
   .. autoattribute:: vp
   .. autoattribute:: edge_properties
   .. autoattribute:: ep
   .. autoattribute:: graph_properties
   .. autoattribute:: gp
   .. automethod:: own_property
   .. automethod:: list_properties


   .. rubric:: Filtering of vertices and edges.

   See :ref:`sec_graph_filtering` for more details.

   .. note::

      These functions do not actually modify the graph, and are fully
      reversible. They are also very cheap, and have an :math:`O(1)`
      complexity.

   .. automethod:: set_filters
   .. automethod:: set_vertex_filter
   .. automethod:: get_vertex_filter
   .. automethod:: set_edge_filter
   .. automethod:: get_edge_filter
   .. automethod:: clear_filters

   .. warning::

     The purge functions below irreversibly remove the filtered
     vertices or edges from the graph. Note that, contrary to the
     functions above, these are :math:`O(V)` and :math:`O(E)`
     operations, respectively.

   .. automethod:: purge_vertices
   .. automethod:: purge_edges

   .. rubric:: I/O operations

   See :ref:`sec_graph_io` for more details.

   .. automethod:: load
   .. automethod:: save

{% endif %}

{% else %}

.. auto{{ objtype }}:: {{ objname }}

{% endif %}
