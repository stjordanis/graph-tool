// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2020 Tiago de Paula Peixoto <tiago@skewed.de>
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// you should have received a copy of the GNU Lesser General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#ifndef GRAPH_CLUSTERING_HH
#define GRAPH_CLUSTERING_HH

#include "config.h"

#include "hash_map_wrap.hh"
#include <boost/mpl/if.hpp>

#ifdef _OPENMP
#include "omp.h"
#endif

#ifndef __clang__
#include <ext/numeric>
using __gnu_cxx::power;
#else
template <class Value>
Value power(Value value, int n)
{
    return pow(value, n);
}
#endif

namespace graph_tool
{
using namespace boost;
using namespace std;

// calculates the number of triangles to which v belongs
template <class Graph, class EWeight, class VProp>
auto get_triangles(typename graph_traits<Graph>::vertex_descriptor v,
                   EWeight& eweight, VProp& mark, const Graph& g)
{
    typedef typename property_traits<EWeight>::value_type val_t;
    val_t triangles = 0, w1 = 0, w2 = 0, k = 0;

    for (auto e : out_edges_range(v, g))
    {
        auto n = target(e, g);
        if (n == v)
            continue;
        auto w = eweight[e];
        mark[n] = w;
        k += w;
    }

    for (auto e : out_edges_range(v, g))
    {
        auto n = target(e, g);
        if (n == v)
            continue;
        w1 = eweight[e];
        for (auto e2 : out_edges_range(n, g))
        {
            auto n2 = target(e2, g);
            if (n2 == n)
                continue;
            w2 = eweight[e2];
            triangles += mark[n2] * w1 * w2;
        }
    }

    for (auto n : adjacent_vertices_range(v, g))
        mark[n] = 0;

    if (graph_tool::is_directed(g))
        return make_pair(val_t(triangles), val_t((k * (k - 1))));
    else
        return make_pair(val_t(triangles / 2), val_t((k * (k - 1)) / 2));
}


// retrieves the global clustering coefficient
struct get_global_clustering
{
    template <class Graph, class EWeight>
    void operator()(const Graph& g, EWeight eweight, double& c, double& c_err) const
    {
        typedef typename property_traits<EWeight>::value_type val_t;
        val_t triangles = 0, n = 0;
        vector<val_t> mask(num_vertices(g), 0);

        #pragma omp parallel if (num_vertices(g) > OPENMP_MIN_THRESH) \
            firstprivate(mask) reduction(+:triangles, n)
        parallel_vertex_loop_no_spawn
                (g,
                 [&](auto v)
                 {
                     auto temp = get_triangles(v, eweight, mask, g);
                     triangles += temp.first;
                     n += temp.second;
                 });
        c = double(triangles) / n;

        // "jackknife" variance
        c_err = 0.0;
        double cerr = 0.0;
        #pragma omp parallel if (num_vertices(g) > OPENMP_MIN_THRESH) \
            firstprivate(mask) reduction(+:cerr)
        parallel_vertex_loop_no_spawn
                (g,
                 [&](auto v)
                 {
                     auto temp = get_triangles(v, eweight, mask, g);
                     double cl = double(triangles - temp.first) /
                         (n - temp.second);
                     cerr += power(c - cl, 2);
                 });
        c_err = sqrt(cerr);
    }
};

// sets the local clustering coefficient to a property
struct set_clustering_to_property
{
    template <class Graph, class EWeight, class ClustMap>
    void operator()(const Graph& g, EWeight eweight, ClustMap clust_map) const
    {
        typedef typename property_traits<EWeight>::value_type val_t;
        vector<val_t> mask(num_vertices(g), false);

        #pragma omp parallel if (num_vertices(g) > OPENMP_MIN_THRESH) \
            firstprivate(mask)
        parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 auto triangles = get_triangles(v, eweight, mask, g);
                 double clustering = (triangles.second > 0) ?
                     double(triangles.first)/triangles.second :
                     0.0;
                 clust_map[v] = clustering;
             });
    }

    template <class Graph>
    struct get_undirected_graph
    {
        typedef typename mpl::if_
           <std::is_convertible<typename graph_traits<Graph>::directed_category,
                                directed_tag>,
            const undirected_adaptor<Graph>,
            const Graph& >::type type;
    };
};

} //graph-tool namespace

#endif // GRAPH_CLUSTERING_HH
