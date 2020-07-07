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
#include "graph_properties.hh"
#include "graph_filtering.hh"
#include "graph_selectors.hh"
#include "graph_util.hh"
#include "graph_python_interface.hh"

#include "graph_properties_map_values.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void edge_property_map_values(GraphInterface& g, boost::any src_prop,
                              boost::any tgt_prop, boost::python::object mapper);

void property_map_values(GraphInterface& g, boost::any src_prop,
                         boost::any tgt_prop, boost::python::object mapper,
                         bool edge)
{
    if (!edge)
    {
        run_action<graph_tool::detail::always_directed_never_reversed>()
            (g,
             [&](auto&& graph, auto&& a2, auto&& a3)
             {
                 return do_map_values()
                     (std::forward<decltype(graph)>(graph),
                      std::forward<decltype(a2)>(a2),
                      std::forward<decltype(a3)>(a3), mapper);
             },
             vertex_properties(),
             writable_vertex_properties())(src_prop, tgt_prop);
    }
    else
    {
        edge_property_map_values(g, src_prop, tgt_prop, mapper);
    }
}
