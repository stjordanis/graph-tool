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

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_properties.hh"

#include <boost/graph/transitive_closure.hpp>

using namespace graph_tool;
using namespace boost;

struct get_transitive_closure
{
    template <class Graph,  class TCGraph>
    void operator()(Graph& g, TCGraph& tcg) const
    {
        boost::transitive_closure(g, tcg);
    }
};


void transitive_closure_dispatch(GraphInterface& gi, GraphInterface& tcgi)
{
    run_action<graph_tool::detail::always_directed>()
        (gi,
         [&](auto&& graph)
         {
             return get_transitive_closure()
                 (std::forward<decltype(graph)>(graph), tcgi.get_graph());
         })();
}

#include <boost/python.hpp>

using namespace boost::python;

#define __MOD__ topology
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("transitive_closure", &transitive_closure_dispatch);
 });
