// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2023 Tiago de Paula Peixoto <tiago@skewed.de>
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

#include "graph_filtering.hh"
#include "graph_random_edges.hh"
#include "numpy_bind.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void _add_random_edges(GraphInterface& gi, size_t E, bool parallel,
                       bool self_loops, bool filtered, boost::any w, rng_t& rng)
{
    if (!w.empty() && !belongs<writable_edge_scalar_properties>()(w))
        throw ValueException("edge weight property must be scalar and writable");

    typedef UnityPropertyMap<int,GraphInterface::edge_t> weight_map_t;
    typedef boost::mpl::push_back<writable_edge_scalar_properties, weight_map_t>::type
        weight_props_t;

    if(w.empty())
        w = weight_map_t();

    run_action<>()
        (gi,
         [&](auto& g, auto eweight) { add_random_edges(g, E, parallel, self_loops,
                                                       filtered, get_checked(eweight), rng); },
         weight_props_t())(w);
}

void _remove_random_edges(GraphInterface& gi, size_t E, boost::any w,
                          bool counts, rng_t& rng)
{
    if (!w.empty() && !belongs<writable_edge_scalar_properties>()(w))
        throw ValueException("edge weight property must be scalar and writeable");

    typedef UnityPropertyMap<int,GraphInterface::edge_t> weight_map_t;
    typedef boost::mpl::push_back<writable_edge_scalar_properties, weight_map_t>::type
        weight_props_t;

    if(w.empty())
        w = weight_map_t();

    run_action<>()
        (gi,
         [&](auto& g, auto eweight) { remove_random_edges(g, E, eweight, counts,
                                                          rng); },
         weight_props_t())(w);
}

using namespace boost::python;

void export_random_edges()
{
    def("add_random_edges", &_add_random_edges);
    def("remove_random_edges", &_remove_random_edges);
}
