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

#ifndef GRAPH_KNN_HH
#define GRAPH_KNN_HH

#include <tuple>
#include <iostream>
#include <random>
#include <boost/functional/hash.hpp>

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "parallel_rng.hh"

#include "random.hh"

#include "hash_map_wrap.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;


template <bool parallel, class Graph, class Dist, class Weight, class RNG>
void gen_knn(Graph& g, Dist&& d, size_t k, double r, double epsilon,
             Weight eweight, RNG& rng_)
{
    parallel_rng<rng_t>::init(rng_);

    auto cmp =
        [] (auto& x, auto& y)
        {
            return get<1>(x) < get<1>(y);
        };

    typedef std::set<std::tuple<size_t, double>, decltype(cmp)> set_t;
    std::vector<set_t> B(num_vertices(g), set_t(cmp));

    std::vector<size_t> vs;
    for (auto v : vertices_range(g))
        vs.push_back(v);

    #pragma omp parallel if (num_vertices(g) > OPENMP_MIN_THRESH && parallel) \
        firstprivate(vs)
    parallel_vertex_loop_no_spawn
        (g,
         [&](auto v)
         {
             auto& rng = parallel_rng<rng_t>::get(rng_);
             for (auto u : random_permutation_range(vs, rng))
             {
                 if (u == v)
                     continue;
                 double l = d(v, u);
                 B[v].insert({u, l});
                 if (B[v].size() == k)
                     break;
             }
         });

    std::bernoulli_distribution rsample(r);

    double delta = epsilon + 1;
    while (delta > epsilon)
    {
        for (auto v : vertices_range(g))
            clear_vertex(v, g);
        for (auto v : vertices_range(g))
        {
            for (auto& u : B[v])
                add_edge(v, get<0>(u), g);
        }

        size_t c = 0;
        #pragma omp parallel if (num_vertices(g) > OPENMP_MIN_THRESH && parallel) \
            reduction(+:c)
        parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 auto& rng = parallel_rng<rng_t>::get(rng_);

                 auto& Bv = B[v];
                 for (auto u : all_neighbors_range(v, g))
                 {
                     if (!rsample(rng))
                         continue;

                     for (auto w : all_neighbors_range(u, g))
                     {
                         if (w == u || w == v || !rsample(rng))
                             continue;

                         double l = d(v, w);
                         auto iter = Bv.lower_bound({w, l});
                         if (iter != Bv.end() && get<0>(*iter) != w)
                         {
                             Bv.insert(iter, {w, l});
                             iter = Bv.end();
                             --iter;
                             Bv.erase(iter);
                             ++c;
                         }
                     }
                 }
             });

        delta = c / double(vs.size() * k);
    }

    for (auto v : vertices_range(g))
        clear_vertex(v, g);

    for (auto v : vertices_range(g))
    {
        for (auto& u : B[v])
        {
            auto e = add_edge(v, get<0>(u), g);
            eweight[e.first] = get<1>(u);
        }
    }
}

template <bool parallel, class Graph, class Dist, class Weight>
void gen_knn_exact(Graph& g, Dist&& d, size_t k, Weight eweight)
{
    std::vector<std::vector<std::tuple<size_t, double>>> vs(num_vertices(g));
    #pragma omp parallel if (num_vertices(g) > OPENMP_MIN_THRESH && parallel)
    parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 auto& ns = vs[v];
                 for (auto u : vertices_range(g))
                 {
                     if (u == v)
                         continue;
                     ns.emplace_back(u, d(v, u));
                 }
                 nth_element(ns.begin(),
                             ns.begin() + k,
                             ns.end(),
                             [] (auto& x, auto& y)
                             {
                                 return get<1>(x) < get<1>(y);
                             });
                 ns.resize(k);
                 ns.shrink_to_fit();
             });

    for (auto v : vertices_range(g))
    {
        for (auto& u : vs[v])
        {
            auto e = add_edge(v, get<0>(u), g);
            eweight[e.first] = get<1>(u);
        }
    }
}


} // graph_tool namespace

#endif // GRAPH_KNN_HH
