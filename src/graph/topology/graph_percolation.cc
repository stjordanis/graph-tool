// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2020 Tiago de Paula Peixoto <tiago@skewed.de>
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 3 of the License, or (at your option) any
// later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "graph_tool.hh"
#include "numpy_bind.hh"

#include "graph_percolation.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void percolate_edge(GraphInterface& gi, boost::any tree, boost::any size,
                    python::object edges, python::object max_size, bool second)
{
    typedef property_map_type::apply<int64_t,
                                     GraphInterface::vertex_index_map_t>::type
        tree_t;
    tree_t tree_map;
    try
    {
        tree_map = any_cast<tree_t>(tree);
    }
    catch (bad_any_cast&)
    {
        throw GraphException("tree map must be a vertex property map of value type int64_t");
    }

    tree_t size_map;
    try
    {
        size_map = any_cast<tree_t>(size);
    }
    catch (bad_any_cast&)
    {
        throw GraphException("size map must be a vertex property map of value type int64_t");
    }

    multi_array_ref<uint64_t, 2> es = get_array<uint64_t, 2>(edges);
    multi_array_ref<uint64_t, 1> ms = get_array<uint64_t, 1>(max_size);

    run_action<graph_tool::detail::never_directed>()
        (gi, [&](auto& g){ edge_percolate(g, tree_map, size_map, ms, es, second); })();
}


void percolate_vertex(GraphInterface& gi, boost::any tree, boost::any size,
                      boost::any visited, python::object vertices,
                      python::object max_size, bool second)
{
    typedef property_map_type::apply<int64_t,
                                     GraphInterface::vertex_index_map_t>::type
        tree_t;
    tree_t tree_map;
    try
    {
        tree_map = any_cast<tree_t>(tree);
    }
    catch (bad_any_cast&)
    {
        throw GraphException("tree map must be a vertex property map of value type int64_t");
    }

    tree_t size_map;
    try
    {
        size_map = any_cast<tree_t>(size);
    }
    catch (bad_any_cast&)
    {
        throw GraphException("size map must be a vertex property map of value type int64_t");
    }

    typedef property_map_type::apply<uint8_t,
                                     GraphInterface::vertex_index_map_t>::type
        visited_t;

    visited_t visited_map;
    try
    {
        visited_map = any_cast<visited_t>(visited);
    }
    catch (bad_any_cast&)
    {
        throw GraphException("visited map must be a vertex property map of value type uint8_t");
    }

    multi_array_ref<uint64_t, 1> vs = get_array<uint64_t, 1>(vertices);
    multi_array_ref<uint64_t, 1> ms = get_array<uint64_t, 1>(max_size);

    run_action<graph_tool::detail::never_directed>()
        (gi, [&](auto& g){ vertex_percolate(g, tree_map, size_map, visited_map,
                                            ms, vs, second); })();
}

#include <boost/python.hpp>

void export_percolation()
{
    using namespace boost::python;

    def("percolate_edge", percolate_edge);
    def("percolate_vertex", percolate_vertex);
};
