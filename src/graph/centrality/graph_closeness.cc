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

#include "graph_filtering.hh"
#include "graph.hh"
#include "graph_selectors.hh"
#include "graph_properties.hh"

#include "graph_closeness.hh"

#include <boost/python.hpp>

using namespace std;
using namespace graph_tool;

void do_get_closeness(GraphInterface& gi, boost::any weight,
                      boost::any closeness, bool harmonic, bool norm)
{
    if (weight.empty())
    {
        run_action<>()
            (gi,
             [&](auto&& graph, auto&& a2)
             {
                 return get_closeness()
                     (std::forward<decltype(graph)>(graph),
                      gi.get_vertex_index(), no_weightS(),
                      std::forward<decltype(a2)>(a2), harmonic, norm);
             },
             writable_vertex_scalar_properties())(closeness);
    }
    else
    {
        run_action<>()
            (gi,
             [&](auto&& graph, auto&& a2, auto&& a3)
             {
                 return get_closeness()
                     (std::forward<decltype(graph)>(graph),
                      gi.get_vertex_index(), std::forward<decltype(a2)>(a2),
                      std::forward<decltype(a3)>(a3), harmonic, norm);
             },
             edge_scalar_properties(),
             writable_vertex_scalar_properties())(weight, closeness);
    }
}

void export_closeness()
{
    boost::python::def("closeness", &do_get_closeness);
}
