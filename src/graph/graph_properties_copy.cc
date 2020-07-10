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

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_properties.hh"

#include "graph_properties_copy.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void GraphInterface::copy_vertex_property(const GraphInterface& src,
                                          boost::any prop_src,
                                          boost::any prop_tgt)
{
    gt_dispatch<>()
        (
            [&](auto&& graph, auto&& a2, auto&& a3)
            {
                return copy_property<vertex_selector, vertex_properties>()(
                    std::forward<decltype(graph)>(graph),
                    std::forward<decltype(a2)>(a2),
                    std::forward<decltype(a3)>(a3), prop_src);
            },
            all_graph_views(), all_graph_views(), writable_vertex_properties())(
            this->get_graph_view(), src.get_graph_view(), prop_tgt);
}

bool compare_vertex_properties(const GraphInterface& gi,
                               boost::any prop1,
                               boost::any prop2)
{
    bool ret = false;
    gt_dispatch<>()
        ([&](auto& g, auto p1, auto p2)
         {
             ret = compare_props<vertex_selector>(g, p1, p2);
         },
         all_graph_views(), vertex_properties(), vertex_properties())
        (gi.get_graph_view(), prop1, prop2);
    return ret;
}

bool compare_edge_properties(const GraphInterface& gi,
                             boost::any prop1,
                             boost::any prop2)
{
    bool ret = false;
    gt_dispatch<>()
        ([&](auto& g, auto p1, auto p2)
         {
             ret = compare_props<edge_selector>(g, p1, p2);
         },
         all_graph_views(), edge_properties(), edge_properties())
        (gi.get_graph_view(), prop1, prop2);
    return ret;
}
