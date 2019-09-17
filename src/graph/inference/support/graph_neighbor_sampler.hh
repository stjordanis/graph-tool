// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2019 Tiago de Paula Peixoto <tiago@skewed.de>
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
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
          _sampler_pos(get(edge_index_t(), g)),
          _self_loops(self_loops)
    {
        init(eweight);
    }

    template <class Eprop>
    void init_vertex(size_t v, Eprop& eweight)
    {
        _sampler[v].clear();

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

            insert(v, u, w, e);
        }

        if constexpr (is_directed_::apply<Graph>::type::value)
        {

            for (auto e : in_edges_range(v, _g))
            {
                auto u = source(e, _g);

                if (!_self_loops && u == v)
                    continue;

                auto w = eweight[e];

                if (w == 0)
                    continue;

                insert(v, u, w, e);
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
        auto& item = sample_item(sampler, rng);
        return get_u(item);
    }

    bool empty(vertex_t v)
    {
        return _sampler[v].empty();
    }

    void resize(size_t n)
    {
        _sampler.resize(n);
    }

    template <class Edge>
    void remove(vertex_t v, vertex_t u, Edge&& e)
    {
        if (v == u && !_self_loops)
            return;
        auto& sampler = _sampler[v];
        auto& pos = _sampler_pos[e];
        bool is_src = (get_src(e) == u);
        remove_item({is_src, e}, sampler, pos);
    }

    template <class Weight, class Edge>
    void insert(vertex_t v, vertex_t u, Weight w, Edge&& e)
    {
        if (v == u && !_self_loops)
            return;
        auto& sampler = _sampler[v];
        auto& pos = _sampler_pos[e];
        bool is_src = (get_src(e) == u);
        insert_item({is_src, e}, w, sampler, pos);
    }

private:
    typedef std::tuple<bool, edge_t> item_t;

    vertex_t get_src(const edge_t& e)
    {
        if constexpr (is_directed_::apply<Graph>::type::value)
            return source(e, _g);
        else
            return std::min(source(e, _g), target(e, _g));
    }

    vertex_t get_tgt(const edge_t& e)
    {
        if constexpr (is_directed_::apply<Graph>::type::value)
            return target(e, _g);
        else
            return std::max(source(e, _g), target(e, _g));
    }

    vertex_t get_u(const item_t& item)
    {
        if (get<0>(item))
            return get_src(get<1>(item));
        else
            return get_tgt(get<1>(item));
    }

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

    size_t& get_pos(const item_t& u, std::tuple<size_t, size_t>& pos)
    {
        if (get<0>(u))
            return get<0>(pos);
        else
            return get<1>(pos);
    }

    void remove_item(const item_t& u, std::vector<item_t>& sampler,
                     std::tuple<size_t, size_t>& pos)
    {
        auto u_pos = get_pos(u, pos);
        if (u_pos >= sampler.size() || sampler[u_pos] != u)
            return;
        auto& back = sampler.back();
        auto& e = get<1>(back);
        auto& bpos = _sampler_pos[e];
        get_pos(back, bpos) = u_pos;
        sampler[u_pos] = back;
        sampler.pop_back();

    }

    template <class Sampler>
    void remove_item(const item_t& u, Sampler& sampler,
                     std::tuple<size_t, size_t>& pos)
    {
        auto i = get_pos(u, pos);
        if (!sampler.is_valid(i) || sampler[i] != u)
            return;
        sampler.remove(i);
    }


    template <class Weight>
    void insert_item(const item_t& u, Weight, std::vector<item_t>& sampler,
                     std::tuple<size_t, size_t>& pos)
    {
        get_pos(u, pos) = sampler.size();
        sampler.push_back(u);
    }

    template <class Weight>
    void insert_item(const item_t& u, Weight w, DynamicSampler<item_t>& sampler,
                     std::tuple<size_t, size_t>& pos)
    {
        get_pos(u, pos) = sampler.insert(u, w);
    }

    Graph& _g;

    typedef typename std::conditional<Weighted::value,
                                      typename std::conditional<Dynamic::value,
                                                                DynamicSampler<item_t>,
                                                                Sampler<item_t,
                                                                        boost::mpl::false_>>::type,
                                      vector<item_t>>::type
        sampler_t;

    typedef typename vprop_map_t<sampler_t>::type::unchecked_t vsampler_t;
    vsampler_t _sampler;

    typedef typename eprop_map_t<std::tuple<size_t, size_t>>::type pos_map_t;
    pos_map_t _sampler_pos;

    bool _self_loops;
};

}

#endif // GRAPH_NEIGHBOR_SAMPLER_HH
