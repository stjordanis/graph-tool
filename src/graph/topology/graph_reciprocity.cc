// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2022 Tiago de Paula Peixoto <tiago@skewed.de>
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
#include "graph_selectors.hh"
#include "graph_properties.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

struct get_reciprocity
{
    template <class Graph, class Weight>
    void operator()(const Graph& g, Weight w, double& reciprocity) const
    {
        typedef typename property_traits<Weight>::value_type val_t;
        val_t L = 0, Lbd = 0;

        #pragma omp parallel if (num_vertices(g) > OPENMP_MIN_THRESH) \
            reduction(+: L, Lbd)
        parallel_edge_loop_no_spawn
            (g,
             [&](auto e)
             {
                 auto we = w[e];
                 auto v = source(e, g);
                 auto t = target(e, g);
                 for (auto er : out_edges_range(t, g))
                 {
                     auto u = target(er, g);
                     if (u == v)
                     {
                         Lbd += std::min(w[er], we);
                         break;
                     }
                 }
                 L += we;
             });
        reciprocity = Lbd / double(L);
    }
};

double reciprocity(GraphInterface& gi, boost::any aw)
{
    typedef UnityPropertyMap<int,GraphInterface::edge_t> weight_map_t;
    typedef boost::mpl::push_back<edge_scalar_properties, weight_map_t>::type
        weight_props_t;

    if (aw.empty())
        aw = weight_map_t();

    double reciprocity;
    run_action<>()
        (gi,
         [&](auto&& graph, auto w)
         {
             return get_reciprocity()
                 (std::forward<decltype(graph)>(graph), w, reciprocity);
         }, weight_props_t())(aw);
    return reciprocity;
}
