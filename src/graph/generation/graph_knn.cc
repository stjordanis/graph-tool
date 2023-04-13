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

#include "graph_knn.hh"
#include "numpy_bind.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void generate_knn(GraphInterface& gi, boost::python::object om, size_t k,
                  double r, double epsilon, bool cache, boost::any aw,
                  bool verbose, rng_t& rng)

{
    typedef eprop_map_t<double>::type emap_t;
    auto w = any_cast<emap_t>(aw);

    try
    {
        auto m = get_array<double, 2>(om);

        auto d_e =
            [&](auto u, auto v)
            {
                double d = 0;
                auto x = m[u];
                auto y = m[v];
                for (size_t i = 0; i < m.shape()[1]; ++i)
                    d += pow(x[i] - y[i], 2);
                return sqrt(d);
            };

        if (!cache)
        {
            run_action<always_directed_never_filtered_never_reversed>()
                (gi, [&](auto& g)
                     {
                         gen_knn<true>(g, d_e, k, r, epsilon, w, verbose, rng);
                     })();
        }
        else
        {
            run_action<always_directed_never_filtered_never_reversed>()
                (gi, [&](auto& g)
                     {
                         auto d = make_cached_dist(g, d_e);
                         gen_knn<true>(g, d, k, r, epsilon, w, verbose, rng);
                     })();
        }
    }
    catch (InvalidNumpyConversion&)
    {
        if (!cache)
        {
            auto d_e =
                [&](auto v, auto u)
                {
                    double d = python::extract<double>(om(v, u));
                    return d;
                };

            run_action<always_directed_never_filtered_never_reversed>()
                (gi, [&](auto& g)
                     {
                         gen_knn<false>(g, d_e, k, r, epsilon, w, verbose, rng);
                     })();
        }
        else
        {
            auto d_e =
                [&](auto v, auto u)
                {
                    double d;
                    #pragma omp critical
                    d = python::extract<double>(om(v, u));
                    return d;
                };

            run_action<always_directed_never_filtered_never_reversed>()
                (gi, [&](auto& g)
                     {
                         auto d = make_cached_dist(g, d_e);
                         gen_knn<true>(g, d, k, r, epsilon, w, verbose, rng);
                     })();
        }
    }
}

template <class T, class M>
double euclidean(T u, T v, const M& m)
{
    double d = 0;
    auto mu = m[u];
    auto mv = m[v];
    for (size_t i = 0; i < m.shape()[1]; ++i)
    {
        auto x = mu[i] - mv[i];
        d += x * x;
    }
    return sqrt(d);
}

void generate_knn_exact(GraphInterface& gi, boost::python::object om, size_t k,
                        boost::any aw)
{
    typedef eprop_map_t<double>::type emap_t;
    auto w = any_cast<emap_t>(aw);

    try
    {
        auto m = get_array<double, 2>(om);
        run_action<always_directed_never_filtered_never_reversed>()
            (gi, [&](auto& g) { gen_knn_exact<true>(g,
                                                    [&](auto u, auto v)
                                                    { return euclidean(u, v, m); },
                                                    k, w); })();
    }
    catch (InvalidNumpyConversion&)
    {
        run_action<always_directed_never_filtered_never_reversed>()
            (gi, [&](auto& g) { gen_knn_exact<false>(g,
                                              [&](auto u, auto v)
                                              {
                                                  double d;
                                                  d = python::extract<double>(om(u, v));
                                                  return d;
                                              },
                                              k, w); })();
    }
}

void generate_k_nearest(GraphInterface& gi, boost::python::object om, size_t k,
                        double r, double epsilon, bool cache, boost::any aw,
                        bool directed, bool verbose, rng_t& rng)

{
    typedef eprop_map_t<double>::type emap_t;
    auto w = any_cast<emap_t>(aw);

    try
    {
        auto m = get_array<double, 2>(om);

        auto d_e =
            [&](auto u, auto v)
            {
                return euclidean(u, v, m);
            };

        if (!cache)
        {
            run_action<always_directed_never_filtered_never_reversed>()
                (gi, [&](auto& g)
                     {
                         gen_k_nearest<true>(g, d_e, k, r, epsilon, w, directed,
                                             verbose, rng);
                     })();
        }
        else
        {
            run_action<always_directed_never_filtered_never_reversed>()
                (gi, [&](auto& g)
                     {
                         auto d = make_cached_dist(g, d_e);
                         gen_k_nearest<true>(g, d, k, r, epsilon, w, directed,
                                             verbose, rng);
                     })();
        }
    }
    catch (InvalidNumpyConversion&)
    {
        if (!cache)
        {
            auto d_e =
                [&](auto v, auto u)
                {
                    double d = python::extract<double>(om(v, u));
                    return d;
                };

            run_action<always_directed_never_filtered_never_reversed>()
                (gi, [&](auto& g)
                     {
                         gen_k_nearest<false>(g, d_e, k, r, epsilon, w,
                                              directed, verbose, rng);
                     })();
        }
        else
        {
            auto d_e =
                [&](auto v, auto u)
                {
                    double d;
                    #pragma omp critical
                    d = python::extract<double>(om(v, u));
                    return d;
                };

            run_action<always_directed_never_filtered_never_reversed>()
                (gi, [&](auto& g)
                     {
                         auto d = make_cached_dist(g, d_e);
                         gen_k_nearest<true>(g, d, k, r, epsilon, w, directed,
                                             verbose, rng);
                     })();
        }
    }
}

void generate_k_nearest_exact(GraphInterface& gi, boost::python::object om, size_t k,
                              boost::any aw, bool directed)
{
    typedef eprop_map_t<double>::type emap_t;
    auto w = any_cast<emap_t>(aw);

    try
    {
        auto m = get_array<double, 2>(om);
        run_action<always_directed_never_filtered_never_reversed>()
            (gi, [&](auto& g) { gen_k_nearest_exact<true>(g,
                                                          [&](auto u, auto v)
                                                          { return euclidean(u, v, m); },
                                                          k, directed, w); })();
    }
    catch (InvalidNumpyConversion&)
    {
        run_action<always_directed_never_filtered_never_reversed>()
            (gi, [&](auto& g) { gen_k_nearest_exact<false>(g,
                                                           [&](auto u, auto v)
                                                           {
                                                               double d;
                                                               d = python::extract<double>(om(u, v));
                                                               return d;
                                                           },
                                                           k, directed, w); })();
    }
}


using namespace boost::python;

#define __MOD__ generation
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     def("gen_knn", &generate_knn);
     def("gen_knn_exact", &generate_knn_exact);
     def("gen_k_nearest", &generate_k_nearest);
     def("gen_k_nearest_exact", &generate_k_nearest_exact);
 });
