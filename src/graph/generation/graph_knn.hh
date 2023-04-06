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

#ifndef GRAPH_KNN_HH
#define GRAPH_KNN_HH

#include <tuple>
#include <random>
#include <boost/functional/hash.hpp>

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "parallel_rng.hh"
#include "idx_map.hh"

#include "random.hh"

#include "hash_map_wrap.hh"

#include "graph_contract_edges.hh"
#include "shared_heap.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class D>
class CachedDist
{
public:
    template <class Graph>
    CachedDist(Graph& g, D& d)
        : _d(d)
    {
        _dist_cache.resize(num_vertices(g));
    }

    double operator()(size_t v, size_t u)
    {
        auto& cache = _dist_cache[u];
        auto iter = cache.find(v);
        if (iter == cache.end())
        {
            double d = _d(v, u);
            cache[v] = d;
            return d;
        }
        return iter->second;
    }

private:
    std::vector<gt_hash_map<size_t, double>> _dist_cache;
    D& _d;
};

template <class Graph, class D>
auto make_cached_dist(Graph& g, D& d)
{
    return CachedDist<D>(g, d);
}

template <bool parallel, class Graph>
auto get_forbidden(Graph& g)
{
    std::vector<gt_hash_set<size_t>> forbid(num_vertices(g));
    #pragma omp parallel if (parallel)
    parallel_vertex_loop_no_spawn
        (g,
         [&](auto v)
         {
             auto& fv = forbid[v];
             for (auto w : in_or_out_neighbors_range(v, g))
                 fv.insert(w);
         });
    return forbid;
}

template <bool parallel, class Graph, class Dist, class Weight, class F,
          class RNG>
void gen_knn(Graph& g, Dist&& d, size_t k, double r, double epsilon,
             Weight eweight, F& forbid, RNG& rng_)
{
    parallel_rng<rng_t>::init(rng_);

    auto cmp =
        [] (auto& x, auto& y)
        {
            return get<1>(x) < get<1>(y);
        };

    std::vector<std::vector<std::tuple<size_t, double>>>
        B(num_vertices(g));

    std::vector<size_t> vs;
    for (auto v : vertices_range(g))
        vs.push_back(v);

    #pragma omp parallel if (parallel) firstprivate(vs)
    parallel_vertex_loop_no_spawn
        (g,
         [&](auto v)
         {
             auto& rng = parallel_rng<rng_t>::get(rng_);
             auto& fv = forbid[v];
             for (auto u : random_permutation_range(vs, rng))
             {
                 if (u == v || (!fv.empty() && fv.find(u) != fv.end()))
                     continue;
                 double l = d(u, v);
                 auto& Bv = B[v];
                 Bv.emplace_back(u, l);
                 std::push_heap(Bv.begin(), Bv.end(), cmp);
                 if (Bv.size() == k)
                     break;
             }
         });

    auto build_graph = [&]()
        {
            for (auto v : vertices_range(g))
                clear_vertex(v, g);

            for (auto v : vertices_range(g))
            {
                for (auto u : forbid[v])
                    add_edge(u, v, g);

                for (auto& [u, l] : B[v])
                {
                    auto e = add_edge(u, v, g).first;
                    eweight[e] = l;
                }
            }
        };

    idx_set<size_t> visited;
    std::bernoulli_distribution rsample(r);

    double delta = epsilon + 1;
    while (delta > epsilon)
    {
        build_graph();

        size_t c = 0;
        #pragma omp parallel if (parallel) reduction(+:c) firstprivate(visited)
        parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 auto& rng = parallel_rng<rng_t>::get(rng_);

                 visited.clear();
                 for (auto u : in_neighbors_range(v, g))
                     visited.insert(u);

                 auto& fv = forbid[v];
                 auto& Bv = B[v];
                 for (auto u : all_neighbors_range(v, g))
                 {
                     if (!rsample(rng))
                         continue;

                     for (auto w : all_neighbors_range(u, g))
                     {
                         if (w == u || w == v || !rsample(rng) ||
                             visited.find(w) != visited.end() ||
                             (!fv.empty() && fv.find(w) != fv.end()))
                             continue;

                         double l = d(w, v);
                         if (l < get<1>(Bv.front()))
                         {
                             std::pop_heap(Bv.begin(), Bv.end(), cmp);
                             Bv.back() = {w, l};
                             std::push_heap(Bv.begin(), Bv.end(), cmp);
                             c++;
                         }
                         visited.insert(w);
                     }
                 }
             });

        delta = c / double(vs.size() * k);
    }
    build_graph();
}

template <bool parallel, class Graph, class Dist, class Weight, class RNG>
void gen_knn(Graph& g, Dist&& d, size_t k, double r, double epsilon,
             Weight eweight, RNG& rng_)
{
    auto forbid = get_forbidden<parallel>(g);
    gen_knn<parallel>(g, d, k, r, epsilon, eweight, forbid, rng_);
}

template <bool parallel, class Graph, class Dist, class Weight, class F>
void gen_knn_exact(Graph& g, Dist&& d, size_t k, Weight eweight, F& forbid)
{
    std::vector<std::vector<std::tuple<size_t, double>>> vs(num_vertices(g));
    #pragma omp parallel if (parallel)
    parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 auto& fv = forbid[v];
                 auto& ns = vs[v];
                 for (auto u : vertices_range(g))
                 {
                     if (u == v || (!fv.empty() && fv.find(u) != fv.end()))
                         continue;
                     ns.emplace_back(u, d(u, v));
                 }
                 if (ns.size() <= k)
                     return;
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
        for (auto u : forbid[v])
            add_edge(u, v, g);
        for (auto& [u, w] : vs[v])
        {
            auto e = add_edge(u, v, g).first;
            eweight[e] = w;
        }
    }
}

template <bool parallel, class Graph, class Dist, class Weight>
void gen_knn_exact(Graph& g, Dist&& d, size_t k, Weight eweight)
{
    auto forbid = get_forbidden<parallel>(g);
    gen_knn_exact<parallel>(g, d, k, eweight, forbid);
}

template <bool parallel, class Graph, class Dist, class Weight, class F>
void gen_k_nearest_exact(Graph& g, Dist&& d, size_t k, bool directed,
                         Weight eweight, F& forbid)
{
    std::vector<std::tuple<std::tuple<size_t, size_t>, double>> pairs;

    auto heap = make_shared_heap(pairs, k,
                                 [](auto& x, auto& y)
                                 {
                                     return get<1>(x) < get<1>(y);
                                 });

    std::vector<size_t> vs;
    for (auto v : vertices_range(g))
        vs.push_back(v);

    size_t N = num_vertices(g);
    #pragma omp parallel for if (parallel) firstprivate(heap)
    for (size_t i = 0; i < N; ++i)
    {
        auto v = vertex(i, g);
        if (!is_valid_vertex(v, g))
            continue;
        auto& fv = forbid[v];
        for (auto u : vs)
        {
            if (u == v || (!fv.empty() && fv.find(u) != fv.end()))
                continue;
            if (!directed && u > v)
                continue;
            auto l = d(u, v);
            heap.push({{u, v}, l});
        }
    }

    heap.merge();

    for (auto& [uv, l] : pairs)
    {
        auto& [u, v] = uv;
        auto e = add_edge(u, v, g).first;
        eweight[e] = l;
    }
}

template <bool parallel, class Graph, class Dist, class Weight>
void gen_k_nearest_exact(Graph& g, Dist&& d, size_t k, bool directed,
                         Weight eweight)
{
    auto forbid = get_forbidden<parallel>(g);
    gen_k_nearest_exact<parallel>(g, d, k, directed, eweight, forbid);
}

template <bool parallel, class Graph, class Dist, class Weight, class RNG>
void gen_k_nearest(Graph& g, Dist&& d, size_t k, double r, double epsilon,
                   Weight eweight, bool directed, RNG& rng)
{
    auto forbid = get_forbidden<parallel>(g);

    size_t N = num_vertices(g);
    if (N * N <= 4 * k)
    {
        gen_k_nearest_exact<parallel>(g, d, k, directed, eweight, forbid);
        return;
    }

    size_t nk = ceil((4. * k)/num_vertices(g)) + 2;
    gen_knn<parallel>(g, d, nk, r, epsilon, eweight, forbid, rng);

    typedef typename graph_traits<Graph>::edge_descriptor edge_t;
    std::vector<std::tuple<edge_t, double>> redges;

    size_t E = num_edges(g);
    auto heap = make_shared_heap(redges, E - 2 * k,
                                 [](auto& x, auto& y)
                                 {
                                     return get<1>(x) > get<1>(y);
                                 });

    #pragma omp parallel for firstprivate(heap)
    for (size_t i = 0; i < N; ++i)
    {
        auto v = vertex(i, g);
        if (!is_valid_vertex(v, g))
            continue;
        for (auto e : out_edges_range(v, g))
            heap.push({e, eweight[e]});
    };

    heap.merge();

    typename eprop_map_t<uint8_t>::type eremove(g.get_edge_index_range(),
                                                get(edge_index_t(), g));
    parallel_loop(redges, [&](auto, auto& el){ eremove[get<0>(el)] = true; });

    N = 0;
    std::vector<uint8_t> select(num_vertices(g));
    #pragma omp parallel reduction(+:N)
    parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 select[v] = true;
                 for (auto e : in_edges_range(v, g))
                 {
                     if (eremove[e])
                     {
                         select[v] = false;
                         break;
                     }
                 }
                 if (select[v])
                     N++;
             });

    adj_list<> g_(num_vertices(g));
    g_.set_keep_epos(true);
    typename eprop_map_t<double>::type ew(get(edge_index_t(), g_));

    auto u = make_filt_graph(g_, boost::keep_all(),
                             [&](auto v) { return select[v]; });

    if (N * N <= 4 * k)
    {
        gen_k_nearest_exact<parallel>(u, d, k, directed, ew, forbid);
    }
    else
    {
        nk = ceil((4. * k)/N) + 2;
        gen_knn<parallel>(u, d, nk, r, epsilon, ew, forbid, rng);
        //gen_knn_exact<parallel>(u, d, nk, ew, forbid);
    }

    if (!directed)
    {
        undirected_adaptor g_u(g);
        contract_parallel_edges(g_u, dummy_property_map());
        undirected_adaptor u_u(g_);
        contract_parallel_edges(u_u, dummy_property_map());
    }

    gt_hash_set<std::tuple<size_t, size_t>> visited;
    std::vector<std::tuple<std::tuple<size_t, size_t>, double>> pairs;
    auto collect_heap =
        [&](auto& g, auto& ew)
        {
            auto heap = make_shared_heap(pairs, k,
                                         [](auto& x, auto& y)
                                         {
                                             return get<1>(x) < get<1>(y);
                                         });
            for (auto e : edges_range(g))
            {
                size_t u = source(e, g);
                size_t v = target(e, g);
                if (!directed && u > v)
                    std::swap(u, v);
                if (visited.find({u,v}) != visited.end())
                    continue;
                visited.insert({u, v});
                auto l = ew[e];
                heap.push({{u,v}, l});
            }

            heap.merge();
        };

    collect_heap(g, eweight);
    collect_heap(u, ew);

    for (auto v : vertices_range(g))
    {
        clear_vertex(v, g);
        for (auto u : forbid[v])
            add_edge(u, v, g);
    }

    for (auto& [uv, l] : pairs)
    {
        auto& [u, v] = uv;
        auto e = add_edge(u, v, g).first;
        eweight[e] = l;
    }
}

} // graph_tool namespace

#endif // GRAPH_KNN_HH
