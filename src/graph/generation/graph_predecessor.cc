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

#include "graph_predecessor.hh"

using namespace graph_tool;
using namespace boost;

void predecessor_graph(GraphInterface& gi, GraphInterface& gpi,
                       boost::any pred_map)
{
    run_action<>()
        (gi,
         [&](auto&& graph, auto&& a2)
         {
             return get_predecessor_graph()
                 (std::forward<decltype(graph)>(graph), gpi.get_graph(),
                  std::forward<decltype(a2)>(a2));
         },
         vertex_scalar_properties())(pred_map);
}
