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
#include "graph_contract_edges.hh"
#include "numpy_bind.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void _contract_parallel_edges(GraphInterface& gi, boost::any w)
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
         [&](auto& g, auto eweight) { contract_parallel_edges(g, eweight); },
         weight_props_t())(w);
}

void _expand_parallel_edges(GraphInterface& gi, boost::any w)
{
    if (!w.empty() && !belongs<edge_scalar_properties>()(w))
        throw ValueException("edge weight property must be scalar");

    typedef UnityPropertyMap<int,GraphInterface::edge_t> weight_map_t;
    typedef boost::mpl::push_back<edge_scalar_properties, weight_map_t>::type
        weight_props_t;

    if(w.empty())
        w = weight_map_t();

    run_action<>()
        (gi,
         [&](auto& g, auto eweight) { expand_parallel_edges(g, eweight); },
         weight_props_t())(w);
}

using namespace boost::python;

void export_contract_edges()
{
    def("contract_parallel_edges", &_contract_parallel_edges);
    def("expand_parallel_edges", &_expand_parallel_edges);
}
