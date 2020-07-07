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

#include <boost/mpl/contains.hpp>
#include <boost/python/extract.hpp>

#include "graph_properties_copy.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void GraphInterface::copy_edge_property(const GraphInterface& src,
                                        boost::any prop_src,
                                        boost::any prop_tgt)
{
    gt_dispatch<>()
        (
            [&](auto&& graph, auto&& a2, auto&& a3)
            {
                return copy_property<edge_selector, edge_properties>()(
                    std::forward<decltype(graph)>(graph),
                    std::forward<decltype(a2)>(a2),
                    std::forward<decltype(a3)>(a3), prop_src);
            },
            all_graph_views(), all_graph_views(), writable_edge_properties())(
            this->get_graph_view(), src.get_graph_view(), prop_tgt);
}

void copy_external_edge_property(const GraphInterface& src,
                                 const GraphInterface& tgt,
                                 boost::any prop_src,
                                 boost::any prop_tgt)
{
    gt_dispatch<>()
        (
            [&](auto&& graph, auto&& a2, auto&& a3)
            {
                return copy_external_edge_property_dispatch<edge_properties>()(
                    std::forward<decltype(graph)>(graph),
                    std::forward<decltype(a2)>(a2),
                    std::forward<decltype(a3)>(a3), prop_src);
            },
            all_graph_views(), all_graph_views(), writable_edge_properties())(
            tgt.get_graph_view(), src.get_graph_view(), prop_tgt);
}
