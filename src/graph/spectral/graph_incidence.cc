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

#include <boost/python.hpp>
#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "numpy_bind.hh"

#include "graph_selectors.hh"
#include "graph_properties.hh"

#include "graph_incidence.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void incidence(GraphInterface& g, boost::any vindex, boost::any eindex,
               python::object odata, python::object oi,
               python::object oj)
{
    if (!belongs<vertex_scalar_properties>()(vindex))
        throw ValueException("index vertex property must have a scalar value type");
    if (!belongs<edge_scalar_properties>()(eindex))
        throw ValueException("index edge property must have a scalar value type");

    multi_array_ref<double,1> data = get_array<double,1>(odata);
    multi_array_ref<int32_t,1> i = get_array<int32_t,1>(oi);
    multi_array_ref<int32_t,1> j = get_array<int32_t,1>(oj);
    run_action<>()
        (g,
         [&](auto&& graph, auto&& a2, auto&& a3)
         {
             return get_incidence()
                 (std::forward<decltype(graph)>(graph),
                  std::forward<decltype(a2)>(a2),
                  std::forward<decltype(a3)>(a3), data, i, j);
         },
         vertex_scalar_properties(), edge_scalar_properties())(vindex, eindex);
}

void incidence_matvec(GraphInterface& g, boost::any vindex, boost::any eindex,
                      python::object ov, python::object oret, bool transpose)
{
    if (!belongs<vertex_scalar_properties>()(vindex))
        throw ValueException("index vertex property must have a scalar value type");

    multi_array_ref<double,1> v = get_array<double,1>(ov);
    multi_array_ref<double,1> ret = get_array<double,1>(oret);

    run_action<>()
        (g,
         [&](auto&& graph, auto&& vi, auto&& ei)
         {
             return inc_matvec(graph, vi, ei, v, ret, transpose);
         },
         vertex_scalar_properties(), edge_scalar_properties())(vindex, eindex);
}

void incidence_matmat(GraphInterface& g, boost::any vindex, boost::any eindex,
                      python::object ov, python::object oret, bool transpose)
{
    if (!belongs<vertex_scalar_properties>()(vindex))
        throw ValueException("index vertex property must have a scalar value type");

    multi_array_ref<double,2> v = get_array<double,2>(ov);
    multi_array_ref<double,2> ret = get_array<double,2>(oret);

    run_action<>()
        (g,
         [&](auto&& graph, auto&& vi, auto&& ei)
         {
             return inc_matmat(graph, vi, ei, v, ret, transpose);
         },
         vertex_scalar_properties(), edge_scalar_properties())(vindex, eindex);
}
