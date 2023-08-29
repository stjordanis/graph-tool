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
#include <list>
#include <shared_mutex>
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

template <bool parallel, class Graph, class Dist, class Weight, class Hint, class RNG>
void gen_knn(Graph& g, Dist&& d, size_t k, double r, size_t max_rk,
             double epsilon, Weight eweight, Hint& hint, bool verbose, RNG& rng_)
{
    parallel_rng<rng_t> prng(rng_);

    auto cmp =
        [] (auto& x, auto& y)
        {
            return get<1>(x) < get<1>(y);
        };

    std::vector<std::vector<std::tuple<size_t, double>>>
        B(num_vertices(g));

    std::vector<size_t> vs, vs_;
    for (auto v : vertices_range(g))
        vs.push_back(v);
    vs_ = vs;

    vector<gt_hash_map<size_t,size_t>> all_seen(num_vertices(g));

    if (verbose)
        cout << "random init" << endl;

    #pragma omp parallel if (parallel) firstprivate(vs_)
    parallel_loop_no_spawn
        (vs,
         [&](auto, auto v)
         {
             auto& rng = prng.get(rng_);
             auto& seen = all_seen[v];
             auto& Bv = B[v];
             for (auto u : random_permutation_range(vs_, rng))
             {
                 if (u == v)
                     continue;
                 double l = d(u, v);
                 Bv.emplace_back(u, l);
                 std::push_heap(Bv.begin(), Bv.end(), cmp);
                 seen[u] = 0;
                 if (Bv.size() == k)
                     break;
             }
             for (auto u : all_neighbors_range(v, hint))
             {
                 if (seen.find(u) != seen.end() || u == v)
                     continue;
                 double l = d(u, v);
                 if (l < get<1>(Bv.front()))
                 {
                     std::pop_heap(Bv.begin(), Bv.end(), cmp);
                     Bv.back() = {u, l};
                     std::push_heap(Bv.begin(), Bv.end(), cmp);
                 }
                 seen[u] = 0;
             }
         });

    auto build_vertex = [&](auto v)
        {
            for (auto& [u, l] : B[v])
            {
                auto e = add_edge(u, v, g).first;
                eweight[e] = l;
            }
        };

    std::vector<std::vector<size_t>> out_neighbors(num_vertices(g));
    std::vector<std::vector<size_t>> hint_neighbors(num_vertices(hint));

    std::bernoulli_distribution rsample(r);

    size_t iter = 1;
    double delta = epsilon + 1;
    while (delta > epsilon)
    {
        if (verbose)
            cout << "build graph" << endl;

        for (size_t v : vs)
            clear_vertex(v, g);
        for (auto v : vs)
            build_vertex(v);

        #pragma omp parallel if (parallel)
        parallel_loop_no_spawn
            (vs,
             [&](auto, auto v)
             {
                 auto& rng = prng.get(rng_);
                 auto& us = out_neighbors[v];
                 auto& hus = hint_neighbors[v];
                 us.clear();
                 for (auto u : out_neighbors_range(v, g))
                     us.push_back(u);
                 for (auto u : all_neighbors_range(v, hint))
                     hus.push_back(u);
                 if (max_rk < us.size())
                 {
                     size_t i = 0;
                     for ([[maybe_unused]] auto u : random_permutation_range(us, rng))
                     {
                         if (++i == max_rk)
                             break;
                     }
                     us.erase(us.begin() + max_rk, us.end());
                 }

                 if (max_rk < hus.size())
                 {
                     size_t i = 0;
                     for ([[maybe_unused]] auto u : random_permutation_range(hus, rng))
                     {
                         if (++i == max_rk)
                             break;
                     }
                     hus.erase(hus.begin() + max_rk, hus.end());
                 }
             });

        if (verbose)
            cout << "update neighbors" << endl;

        size_t c = 0;
        size_t n = 0;

        #pragma omp parallel if (parallel) reduction(+:c,n)
        parallel_loop_no_spawn
            (vs,
             [&](auto, auto v)
             {
                 auto& rng = prng.get(rng_);

                 auto& seen = all_seen[v];

                 auto& Bv = B[v];

                 for (auto& [v, l] : Bv)
                     seen[v] = iter;

                 auto update =
                     [&](size_t u, size_t w)
                     {
                         if (u == w || w == v)
                             return;

                         auto it = seen.find(w);
                         if (it != seen.end())
                         {
                             it->second = iter;
                             return;
                         }

                         if (!rsample(rng))
                             return;

                         double l = d(w, v);

                         if (l < get<1>(Bv.front()))
                         {
                             std::pop_heap(Bv.begin(), Bv.end(), cmp);
                             Bv.back() = {w, l};
                             std::push_heap(Bv.begin(), Bv.end(), cmp);
                             c++;
                         }

                         seen[w] = iter;
                         n++;
                     };

                 for (auto u : all_neighbors_range(v, hint))
                 {
                     update(v, u);
                     for (auto w : hint_neighbors[u])
                         update(u, w);
                 }

                 for (auto u : in_neighbors_range(v, g))
                 {
                     for (auto w : in_neighbors_range(u, g))
                         update(u, w);
                     for (auto w : out_neighbors[u])
                         update(u, w);
                 }

                 for (auto u : out_neighbors[v])
                 {
                     update(v, u);
                     for (auto w : in_neighbors_range(u, g))
                         update(u, w);
                     for (auto w : out_neighbors[u])
                         update(u, w);
                 }

                 for (auto it = seen.begin(); it != seen.end(); ++it)
                 {
                     if (it->second < iter)
                         seen.erase(it);
                 }

                 seen.resize(0);
             });

        delta = c / double(vs.size() * k);

        if (verbose)
            cout << iter++ << " " << delta << " " << c << " " << n << endl;
    }

    for (size_t v : vs)
        clear_vertex(v, g);
    for (auto v : vs)
        build_vertex(v);
}


template <bool parallel, class Graph, class Dist, class Weight, class RNG>
void gen_knn_local(Graph& g, Dist&& d, size_t k, double r, double epsilon,
                   Weight eweight, bool verbose, RNG& rng_)
{
    parallel_rng<rng_t> prng(rng_);

    auto cmp =
        [] (auto& x, auto& y)
        {
            return get<1>(x) < get<1>(y);
        };

    std::vector<std::vector<std::tuple<size_t, double, bool>>>
        B(num_vertices(g));
    std::vector<gt_hash_set<size_t>> Bset(num_vertices(g));
    vector<gt_hash_set<size_t>> all_visited(num_vertices(g));

    std::vector<size_t> vs, vs_;
    for (auto v : vertices_range(g))
        vs.push_back(v);
    vs_ = vs;

    if (verbose)
        cout << "random init" << endl;

    #pragma omp parallel if (parallel) firstprivate(vs_)
    parallel_loop_no_spawn
        (vs,
         [&](auto, auto v)
         {
             auto& rng = prng.get(rng_);
             auto& visited = all_visited[v];
             for (auto u : random_permutation_range(vs_, rng))
             {
                 if (u == v)
                     continue;
                 double l = d(u, v);
                 auto& Bv = B[v];
                 Bv.emplace_back(u, l, true);
                 std::push_heap(Bv.begin(), Bv.end(), cmp);
                 Bset[v].insert(u);
                 visited.insert(u);
                 if (Bv.size() == k)
                     break;
             }
         });

    std::vector<std::shared_mutex> mutex(num_vertices(g));

    std::vector<std::vector<size_t>> vnew(num_vertices(g));
    std::vector<std::vector<size_t>> rvnew(num_vertices(g));
    std::vector<std::vector<size_t>> vold(num_vertices(g));
    std::vector<std::vector<size_t>> rvold(num_vertices(g));

    idx_set<size_t, false, false> visited(num_vertices(g));
    std::bernoulli_distribution rsample(std::min(r, 1.));

    size_t iter = 0;
    double delta = epsilon + 1;
    while (delta > epsilon)
    {
        if (verbose)
            cout << "build graph" << endl;

        parallel_loop(vs,
                      [&](auto, auto v)
                      {
                          auto& rng = prng.get(rng_);

                          vnew[v].clear();
                          vold[v].clear();
                          rvnew[v].clear();
                          rvold[v].clear();
                          for (auto& [u, l, m] : B[v])
                          {
                              if (m)
                              {
                                  if (rsample(rng))
                                  {
                                      vnew[v].push_back(u);
                                      m = false;
                                  }
                              }
                              else
                              {
                                  vold[v].push_back(u);
                              }
                          }
                      }, vs.size());

        for (auto v : vs)
        {
            for (auto u : vnew[v])
                rvnew[u].push_back(v);
            for (auto u : vold[v])
                rvold[u].push_back(v);
        }

        if (verbose)
            cout << "update neighbors (local)" << endl;

        size_t rK = std::max(size_t(ceil(r * k)), size_t(1));
        size_t c = 0;
        size_t n = 0;

        #pragma omp parallel if (parallel) firstprivate(visited) reduction(+:c,n)
        parallel_loop_no_spawn
            (vs,
             [&](auto, auto v)
             {
                 auto& rng = prng.get(rng_);

                 visited.clear();
                 for (auto u : vnew[v])
                     visited.insert(u);
                 size_t i = 0;
                 for (auto u : random_permutation_range(rvnew[v], rng))
                 {
                     if (visited.find(u) != visited.end())
                         continue;
                     visited.insert(u);
                     vnew[v].push_back(u);
                     if (++i == rK)
                         break;
                 }

                 visited.clear();
                 for (auto u : vold[v])
                     visited.insert(u);
                 i = 0;
                 for (auto u : random_permutation_range(rvold[v], rng))
                 {
                     if (visited.find(u) != visited.end())
                         continue;
                     visited.insert(u);
                     vold[v].push_back(u);
                     if (++i == rK)
                         break;
                 }

                 auto update_heap =
                     [&](auto s, auto l, auto& Bt, auto& Bst)
                     {
                         if (l < get<1>(Bt.front()))
                         {
                             if (Bst.find(s) != Bst.end())
                                 return;
                             Bst.erase(get<0>(Bt.front()));
                             Bst.insert(s);
                             std::pop_heap(Bt.begin(), Bt.end(), cmp);
                             Bt.back() = {s, l, true};
                             std::push_heap(Bt.begin(), Bt.end(), cmp);
                             c++;
                         }
                     };

                 auto update_pair =
                     [&](auto s, auto t)
                     {
                         auto& Bt = B[t];
                         auto& Bst = Bset[t];
                         auto& vis = all_visited[t];
                         n++;

                         auto do_ulocked =
                             [&](auto&& f)
                             {
                                 if constexpr (parallel)
                                 {
                                     std::unique_lock lock(mutex[t]);
                                     f();
                                 }
                                 else
                                 {
                                     f();
                                 }
                             };

                         auto do_slocked =
                             [&](auto&& f)
                             {
                                 if constexpr (parallel)
                                 {
                                     std::shared_lock lock(mutex[t]);
                                     return f();
                                 }
                                 else
                                 {
                                     return f();
                                 }
                             };

                         if (do_slocked([&]()
                                        {
                                            if (vis.find(s) != vis.end())
                                                return true;
                                            return false;
                                        }))
                             return;

                         double l = d(s, t);

                         do_ulocked([&]()
                                    {
                                        update_heap(s, l, Bt, Bst);
                                        vis.insert(s);
                                    });
                     };

                 for (auto s : vnew[v])
                 {
                     for (auto t : vnew[v])
                     {
                        if (t >= s)
                            continue;
                         update_pair(s, t);
                         update_pair(t, s);
                     }
                 }

                 for (auto s : vnew[v])
                 {
                     for (auto t : vold[v])
                     {
                         if (t == s)
                             continue;
                         update_pair(s, t);
                         update_pair(t, s);
                     }
                 }
             });

        delta = c / double(vs.size() * k);

        if (verbose)
            cout << iter++ << " " << delta << " " << c << " " << n << endl;
    }

    for (size_t v : vs)
        clear_vertex(v, g);
    for (auto v : vs)
    {
        for (auto& [u, l, m] : B[v])
        {
            auto e = add_edge(u, v, g).first;
            eweight[e] = l;
        }
    }
}

template <bool parallel, class Graph, class Dist, class Weight>
void gen_knn_exact(Graph& g, Dist&& d, size_t k, Weight eweight)
{
    std::vector<size_t> vs;
    for (auto v : vertices_range(g))
        vs.push_back(v);
    std::vector<std::vector<std::tuple<size_t, double>>> us(num_vertices(g));

    #pragma omp parallel if (parallel)
    parallel_loop_no_spawn
        (vs,
         [&](auto, auto v)
         {
             auto& ns = us[v];
             for (size_t u : vertices_range(g))
             {
                 if (u == v)
                     continue;
                 ns.emplace_back(u, d(u, v));
             }
             if (ns.size() <= k)
                 return;
             nth_element(ns.begin(),
                         ns.begin() + k,
                         ns.end(),
                         [](auto& x, auto& y)
                         {
                             return get<1>(x) < get<1>(y);
                         });
             ns.resize(k);
             ns.shrink_to_fit();
         });

    for (auto v : vs)
    {
        for (auto& [u, w] : us[v])
        {
            auto e = add_edge(u, v, g).first;
            eweight[e] = w;
        }
    }
}

template <bool parallel, class Graph, class Dist, class Weight>
void gen_k_nearest_exact(Graph& g, Dist&& d, size_t k, bool directed,
                         Weight eweight)
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

    #pragma omp parallel if (parallel) firstprivate(heap)
    parallel_loop_no_spawn
        (vs,
         [&](auto, auto v)
         {
             for (auto u : vs)
             {
                 if (u == v)
                     continue;
                 if (!directed && u > v)
                     continue;
                 auto l = d(u, v);
                 heap.push({{u, v}, l});
             }
         });

    heap.merge();

    for (auto& [uv, l] : pairs)
    {
        auto& [u, v] = uv;
        auto e = add_edge(u, v, g).first;
        eweight[e] = l;
    }
}

template <class DescriptorProperty>
class MaskFilter
{
public:
    typedef typename boost::property_traits<DescriptorProperty>::value_type value_t;
    MaskFilter(){}
    MaskFilter(DescriptorProperty& filtered_property)
        : _filtered_property(&filtered_property) {}

    template <class Descriptor>
    inline bool operator() (Descriptor&& d) const
    {
        return get(*_filtered_property, d);
    }

    DescriptorProperty& get_filter() { return *_filtered_property; }
    constexpr bool is_inverted() { return false; }

private:
    DescriptorProperty* _filtered_property;
};


template <bool parallel, class Graph, class Dist, class Weight, class Hint, class RNG>
void gen_k_nearest(Graph& g, Dist&& d, size_t m, double r, size_t max_rk,
                   double epsilon, Weight eweight, Hint& hint, bool local,
                   bool directed, bool verbose, RNG& rng)
{
    size_t N = num_vertices(g);
    std::vector<bool> select(N, true);
    typename eprop_map_t<bool>::type::unchecked_t eselect(get(edge_index_t(), g));

    auto u = make_filt_graph(g, MaskFilter<decltype(eselect)>(eselect),
                             [&](auto v) { return select[v]; });

    auto uhint = make_filt_graph(hint, keep_all(),
                                 [&](auto v) { return select[v]; });

    size_t iter = 0;
    while (N > 1)
    {
        if (verbose)
            cout << "m = " << m <<  " nearest iteration: " << iter++ << endl;

        if (N * N <= 4 * m)
        {
            if (verbose)
                cout << "Running exact m nearest with N = " << N << endl;
            gen_k_nearest_exact<parallel>(u, d, m, directed, eweight);
            break;
        }

        size_t nk = ceil((4. * m)/N);

        eselect.reserve(N * nk);

        if (verbose)
            cout << "Running KNN with N = " << N << " and k = " << nk << endl;
        if (local)
            gen_knn_local<parallel>(u, d, nk, r, epsilon, eweight, verbose, rng);
        else
            gen_knn<parallel>(u, d, nk, r, max_rk, epsilon, eweight, uhint, verbose, rng);

        typedef typename graph_traits<Graph>::edge_descriptor edge_t;
        std::vector<std::tuple<edge_t, double>> medges;

        // 2m shortest directed pairs
        auto heap = make_shared_heap(medges, 2 * m,
                                     [](auto& x, auto& y)
                                     {
                                         return get<1>(x) < get<1>(y);
                                     });
        if (verbose)
            cout << "Keeping 2m = " << 2 * m << " of "
                 << nk * N << " closest directed pairs..." << endl;

        #pragma omp parallel if (parallel) firstprivate(heap)
        parallel_edge_loop_no_spawn(u,
                                    [&](auto& e)
                                    { heap.push({e, eweight[e]}); });
        heap.merge();

        if (verbose)
            cout << "heap size: " << medges.size()
                 << ", top: " << get<1>(medges.front()) << endl;

        if (verbose)
            cout << "Selecting nodes..." << endl;

        typename eprop_map_t<bool>::type ekeep(get(edge_index_t(), g));
        for (auto& el : medges)
            ekeep[get<0>(el)] = true;

        N = 0;
        std::vector<bool> nselect(select);
        #pragma omp parallel if (parallel) reduction(+:N)
        parallel_vertex_loop_no_spawn
            (u,
             [&](auto v)
             {
                 nselect[v] = true;
                 for (auto e : in_edges_range(v, u))
                 {
                     if (!ekeep[e])
                     {
                         nselect[v] = false;
                         break;
                     }
                 }
                 if (nselect[v])
                     N++;
             });

        select = nselect;
        if (N > 1)
        {
            #pragma omp parallel if (parallel)
            parallel_edge_loop_no_spawn(u, [&](auto& e){ eselect[e] = false; });
        }
    };

    if (verbose)
        cout << "Removing parallel edges..." << endl;

    if (!directed)
    {
        undirected_adaptor g_u(g);
        contract_parallel_edges(g_u, dummy_property_map());
    }
    else
    {
        contract_parallel_edges(g, dummy_property_map());
    }

    if (verbose)
        cout << "Selecting best m = " << m << " out of "
             << num_edges(g) << " edges..." << endl;

    std::vector<std::tuple<std::tuple<size_t, size_t>, double>> pairs;

    auto heap = make_shared_heap(pairs, m,
                                 [](auto& x, auto& y)
                                 {
                                     return get<1>(x) < get<1>(y);
                                 });

    #pragma omp parallel if (parallel) firstprivate(heap)
    parallel_edge_loop_no_spawn
        (g,
         [&](auto& e)
         {
             size_t u = source(e, g);
             size_t v = target(e, g);
             if (!directed && u > v)
                 std::swap(u, v);
             auto l = eweight[e];
             heap.push({{u, v}, l});
         });
    heap.merge();

    if (verbose)
        cout << "E = " << pairs.size()
             << ", top: " << get<1>(pairs.front()) << endl;

    if (verbose)
        cout << "Building graph..." << endl;

    for (auto v : vertices_range(g))
        clear_vertex(v, g);

    for (auto& [uv, l] : pairs)
    {
        auto& [u, v] = uv;
        auto e = add_edge(u, v, g).first;
        eweight[e] = l;
    }
}

} // graph_tool namespace

#endif // GRAPH_KNN_HH
