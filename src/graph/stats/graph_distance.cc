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

#include "graph_distance.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

typedef Histogram<size_t, size_t, 1> hist_t;

python::object distance_histogram(GraphInterface& gi, boost::any weight,
                                  const vector<long double>& bins)
{
    python::object ret;

    if (weight.empty())
    {
        run_action<>()
            (gi,
             [&](auto&& graph)
             {
                 return get_distance_histogram()
                     (std::forward<decltype(graph)>(graph),
                      gi.get_vertex_index(), no_weightS(), bins, ret);
             })();
    }
    else
    {
        run_action<>()
            (gi,
             [&](auto&& graph, auto&& a2)
             {
                 return get_distance_histogram()
                     (std::forward<decltype(graph)>(graph),
                      gi.get_vertex_index(), std::forward<decltype(a2)>(a2),
                      bins, ret);
             },
             edge_scalar_properties())(weight);
    }
    return ret;
}

void export_distance()
{
    python::def("distance_histogram", &distance_histogram);
}
