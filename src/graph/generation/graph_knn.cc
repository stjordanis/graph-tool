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

#include "graph_knn.hh"
#include "numpy_bind.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

template <class D>
class CachedDist
{
public:
    CachedDist(GraphInterface& gi, D& d)
        : _d(d)
    {
        run_action<>()
            (gi, [&](auto& g) { _dist_cache.resize(num_vertices(g)); })();
    }

    double operator()(size_t v, size_t u)
    {
        auto& cache = _dist_cache[v];
        auto iter = cache.find(u);
        if (iter == cache.end())
        {
            double d = _d(v, u);
            cache[u] = d;
            return d;
        }
        return iter->second;
    }

private:
    std::vector<gt_hash_map<size_t, double>> _dist_cache;
    D& _d;
};

template <class D>
auto make_cached_dist(GraphInterface& gi, D& d)
{
    return CachedDist<D>(gi, d);
}

void generate_knn(GraphInterface& gi, boost::python::object om, size_t k,
                  double r, double epsilon, bool cache, boost::any aw,
                  rng_t& rng)

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
            run_action<>()
                (gi, [&](auto& g) { gen_knn<true>(g, d_e, k, r, epsilon, w, rng); })();
        }
        else
        {
            auto d = make_cached_dist(gi, d_e);
            run_action<>()
                (gi, [&](auto& g) { gen_knn<true>(g, d, k, r, epsilon, w, rng); })();
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

            run_action<>()
                (gi, [&](auto& g) { gen_knn<false>(g, d_e, k, r, epsilon, w, rng); })();
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

            auto d = make_cached_dist(gi, d_e);
            run_action<>()
                (gi, [&](auto& g) { gen_knn<true>(g, d, k, r, epsilon, w, rng); })();
        }
    }
}

void generate_knn_exact(GraphInterface& gi, boost::python::object om, size_t k,
                        boost::any aw)
{
    typedef eprop_map_t<double>::type emap_t;
    auto w = any_cast<emap_t>(aw);

    try
    {
        auto m = get_array<double, 2>(om);
        run_action<>()
            (gi, [&](auto& g) { gen_knn_exact<true>(g,
                                                    [&](auto u, auto v)
                                                    {
                                                        double d = 0;
                                                        for (size_t i = 0; i < m.shape()[1]; ++i)
                                                            d += pow(m[u][i] - m[v][i], 2);
                                                        return sqrt(d);
                                                    },
                                                    k, w); })();
    }
    catch (InvalidNumpyConversion&)
    {
        run_action<>()
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
