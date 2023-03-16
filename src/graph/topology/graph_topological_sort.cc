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
#include "graph.hh"
#include "graph_properties.hh"

#include <boost/graph/topological_sort.hpp>

using namespace std;
using namespace boost;
using namespace graph_tool;

struct get_topological_sort
{
    template <class Graph>
    void operator()(const Graph& g, vector<int32_t>& sort) const
    {
        sort.clear();
        topological_sort(g, std::back_inserter(sort));
    }
};

bool topological_sort_dispatch(GraphInterface& gi, vector<int32_t>& sort)
{
    try
    {
        run_action<>()
            (gi,
             [&](auto&& graph)
             {
                 return get_topological_sort()
                     (std::forward<decltype(graph)>(graph), sort);
             })();
        return true;
    }
    catch (not_a_dag& e)
    {
        return false;
    }
}

#include <boost/python.hpp>

using namespace boost::python;

#define __MOD__ topology
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("topological_sort", &topological_sort_dispatch);
 });
