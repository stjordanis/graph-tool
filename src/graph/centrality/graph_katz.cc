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

#include <boost/python.hpp>

#include "graph.hh"
#include "graph_selectors.hh"
#include "graph_katz.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void katz(GraphInterface& g, boost::any w, boost::any c, boost::any beta,
          long double alpha, double epsilon, size_t max_iter)
{
    if (!w.empty() && !belongs<writable_edge_scalar_properties>()(w))
        throw ValueException("edge property must be writable");
    if (!belongs<vertex_floating_properties>()(c))
        throw ValueException("centrality vertex property must be of floating point"
                             " value type");
    if (!beta.empty() && !belongs<vertex_floating_properties>()(beta))
        throw ValueException("personalization vertex property must be of floating point"
                             " value type");

    typedef UnityPropertyMap<int,GraphInterface::edge_t> weight_map_t;
    typedef boost::mpl::push_back<writable_edge_scalar_properties, weight_map_t>::type
        weight_props_t;

    if(w.empty())
        w = weight_map_t();

    typedef UnityPropertyMap<int,GraphInterface::vertex_t> beta_map_t;
    typedef boost::mpl::push_back<vertex_floating_properties, beta_map_t>::type
        beta_props_t;

    if(beta.empty())
        beta = beta_map_t();

    run_action<>()
        (g,
         [&](auto&& graph, auto&& a2, auto&& a3, auto&& a4)
         {
             return get_katz()
                 (std::forward<decltype(graph)>(graph), g.get_vertex_index(),
                  std::forward<decltype(a2)>(a2),
                  std::forward<decltype(a3)>(a3),
                  std::forward<decltype(a4)>(a4), alpha, epsilon, max_iter);
         },
         weight_props_t(), vertex_floating_properties(),
         beta_props_t())(w, c, beta);
}

void export_katz()
{
    using namespace boost::python;
    def("get_katz", &katz);
}
