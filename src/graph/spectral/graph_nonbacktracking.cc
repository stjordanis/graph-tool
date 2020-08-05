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

#include "graph_nonbacktracking.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void nonbacktracking(GraphInterface& gi, boost::any index,
                     std::vector<int64_t>& i, std::vector<int64_t>& j)
{
    if (!belongs<edge_scalar_properties>()(index))
        throw ValueException("index vertex property must have a scalar value type");

    run_action<>()
        (gi, [&](auto& g, auto idx){ get_nonbacktracking(g, idx, i, j);},
         edge_scalar_properties())(index);

}

void nonbacktracking_matvec(GraphInterface& g, boost::any index,
                            python::object ov, python::object oret,
                            bool transpose)
{
    if (!belongs<edge_scalar_properties>()(index))
        throw ValueException("index vertex property must have a scalar value type");

    multi_array_ref<double,1> v = get_array<double,1>(ov);
    multi_array_ref<double,1> ret = get_array<double,1>(oret);

    run_action<>()
        (g,
         [&](auto&& graph, auto&& ei)
         {
             if (!transpose)
                 nbt_matvec<false>(graph, ei, v, ret);
             else
                 nbt_matvec<true>(graph, ei, v, ret);
         },
         edge_scalar_properties())(index);
}

void nonbacktracking_matmat(GraphInterface& g, boost::any index,
                            python::object ov, python::object oret,
                            bool transpose)
{
    if (!belongs<edge_scalar_properties>()(index))
        throw ValueException("index vertex property must have a scalar value type");

    multi_array_ref<double,2> v = get_array<double,2>(ov);
    multi_array_ref<double,2> ret = get_array<double,2>(oret);

    run_action<>()
        (g,
         [&](auto&& graph, auto&& ei)
         {
             if (!transpose)
                 nbt_matmat<false>(graph, ei, v, ret);
             else
                 nbt_matmat<true>(graph, ei, v, ret);
         },
         edge_scalar_properties())(index);
}

void compact_nonbacktracking(GraphInterface& gi, std::vector<int64_t>& i,
                             std::vector<int64_t>& j, std::vector<double>& x)
{
    run_action<>()
        (gi, [&](auto& g){ get_compact_nonbacktracking(g, i, j, x);})();

}

void compact_nonbacktracking_matvec(GraphInterface& g, boost::any index,
                                    python::object ov, python::object oret,
                                    bool transpose)
{
    if (!belongs<vertex_scalar_properties>()(index))
        throw ValueException("index vertex property must have a scalar value type");

    multi_array_ref<double,1> v = get_array<double,1>(ov);
    multi_array_ref<double,1> ret = get_array<double,1>(oret);

    run_action<>()
        (g,
         [&](auto&& graph, auto&& ei)
         {
             if (!transpose)
                 cnbt_matvec<false>(graph, ei, v, ret);
             else
                 cnbt_matvec<true>(graph, ei, v, ret);
         },
         vertex_scalar_properties())(index);
}

void compact_nonbacktracking_matmat(GraphInterface& g, boost::any index,
                                    python::object ov, python::object oret,
                                    bool transpose)
{
    if (!belongs<vertex_scalar_properties>()(index))
        throw ValueException("index vertex property must have a scalar value type");

    multi_array_ref<double,2> v = get_array<double,2>(ov);
    multi_array_ref<double,2> ret = get_array<double,2>(oret);

    run_action<>()
        (g,
         [&](auto&& graph, auto&& ei)
         {
             if (!transpose)
                 cnbt_matmat<false>(graph, ei, v, ret);
             else
                 cnbt_matmat<true>(graph, ei, v, ret);
         },
         vertex_scalar_properties())(index);
}
