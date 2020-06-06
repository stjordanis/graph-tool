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

#ifndef GRAPH_NEIGHBOR_SAMPLER_HH
#define GRAPH_NEIGHBOR_SAMPLER_HH

#include "config.h"

#include "graph_tool.hh"

// Sample neighbors efficiently
// =============================

namespace graph_tool
{

template <class Graph, class Weighted, class Dynamic>
class NeighborSampler
{
public:
    typedef typename boost::graph_traits<Graph>::vertex_descriptor vertex_t;
    typedef typename boost::graph_traits<Graph>::edge_descriptor edge_t;

    template <class Eprop>
    NeighborSampler(Graph& g, Eprop& eweight, bool self_loops=false)
        : _g(g),
          _sampler(get(vertex_index_t(), g), num_vertices(g)),
          _self_loops(self_loops)
    {
        init(eweight);
    }

    template <class Eprop>
    void init_vertex(size_t v, Eprop& eweight)
    {
        init_vertex(v, eweight, Weighted());
    }

    template <class Eprop>
    void init_vertex(size_t v, Eprop& eweight, std::true_type)
    {
        std::vector<item_t> us;
        std::vector<double> ps;

        for (auto e : out_edges_range(v, _g))
        {
            auto u = target(e, _g);

            double w = eweight[e];

            if (w == 0)
                continue;

            if (u == v)
            {
                if (!_self_loops)
                    continue;
                if constexpr (!is_directed_::apply<Graph>::type::value)
                    w /= 2;
            }

            us.push_back(u);
            ps.push_back(w);
        }

        if constexpr (is_directed_::apply<Graph>::type::value)
        {
            for (auto e : in_edges_range(v, _g))
            {
                auto u = source(e, _g);

                if (u == v)
                    continue;

                auto w = eweight[e];

                if (w == 0)
                    continue;

                us.push_back(u);
                ps.push_back(w);
            }
        }

        _sampler[v] = sampler_t(us, ps);
    }

    template <class Eprop>
    void init_vertex(size_t v, Eprop&, std::false_type)
    {
        auto& sampler = _sampler[v];
        sampler.clear();

        [[maybe_unused]] gt_hash_set<size_t> sl_set;
        [[maybe_unused]] auto eindex = get(edge_index_t(), _g);

        for (auto e : out_edges_range(v, _g))
        {
            auto u = target(e, _g);
            if (u == v)
            {
                if (!_self_loops)
                    continue;
                if constexpr (!is_directed_::apply<Graph>::type::value)
                {
                    if (sl_set.find(eindex[e]) != sl_set.end())
                        continue;
                    sl_set.insert(eindex[e]);
                }
            }
            sampler.push_back(u);
        }

        if constexpr (is_directed_::apply<Graph>::type::value)
        {
            for (auto e : in_edges_range(v, _g))
            {
                auto u = source(e, _g);

                if (u == v)
                    continue;

                sampler.push_back(u);
            }
        }
    }

    template <class Eprop>
    void init(Eprop& eweight)
    {
        for (auto v : vertices_range(_g))
            init_vertex(v, eweight);
    }

    template <class RNG>
    vertex_t sample(vertex_t v, RNG& rng)
    {
        auto& sampler = _sampler[v];
        return sample_item(sampler, rng);
    }

    bool empty(vertex_t v)
    {
        return _sampler[v].empty();
    }

    void resize(size_t n)
    {
        _sampler.resize(n);
    }

private:
    typedef vertex_t item_t;

    template <class RNG>
    const item_t& sample_item(std::vector<item_t>& sampler, RNG& rng)
    {
        return uniform_sample(sampler, rng);
    }

    template <class Sampler, class RNG>
    const item_t& sample_item(Sampler& sampler, RNG& rng)
    {
        return sampler.sample(rng);
    }

    Graph& _g;

    typedef typename std::conditional<Weighted::value,
                                      typename std::conditional<Dynamic::value,
                                                                DynamicSampler<item_t>,
                                                                Sampler<item_t,
                                                                        boost::mpl::false_>>::type,
                                      std::vector<item_t>>::type
        sampler_t;

    typedef typename vprop_map_t<sampler_t>::type::unchecked_t vsampler_t;
    vsampler_t _sampler;

    bool _self_loops;
};

}

#endif // GRAPH_NEIGHBOR_SAMPLER_HH
