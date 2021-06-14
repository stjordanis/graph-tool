// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2021 Tiago de Paula Peixoto <tiago@skewed.de>
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

#ifndef GRAPH_BLOCKMODEL_ELIST_HH
#define GRAPH_BLOCKMODEL_ELIST_HH

#include "../generation/sampler.hh"
#include "../generation/dynamic_sampler.hh"

namespace graph_tool
{

// ====================================
// Construct and manage half-edge lists
// ====================================

class EGroups
{
public:
    template <class Eprop, class BGraph>
    void init(BGraph& bg, Eprop& mrs)
    {
        clear();
        _egroups.resize(num_vertices(bg));
        _pos.resize(num_vertices(bg));

        for (auto e : edges_range(bg))
        {
            insert_edge(source(e, bg), target(e, bg), mrs[e]);
            insert_edge(target(e, bg), source(e, bg), mrs[e]);
        }
    }

    void add_block()
    {
        _egroups.emplace_back();
        _pos.emplace_back();
    }

    void clear()
    {
        _egroups.clear();
        _egroups.shrink_to_fit();
        _pos.clear();
        _pos.shrink_to_fit();
    }

    bool empty()
    {
        return _egroups.empty();
    }

    void insert_edge(size_t s, size_t t, int weight)
    {
        auto& pos = _pos[s];
        auto iter = pos.find(t);
        if (iter == pos.end())
            iter = pos.insert({t, _null_pos}).first;
        insert_edge(t, weight, _egroups[s], iter->second);
        if (iter->second == _null_pos)
            pos.erase(iter);
    }

    template <class EV>
    void insert_edge(size_t t, int weight, EV& elist, size_t& pos)
    {
        if (pos != _null_pos)
        {
            assert(elist.is_valid(pos) && elist[pos] == t);
            elist.update(pos, weight, true);
            if (elist.get_prob(pos) == 0)
            {
                elist.remove(pos);
                pos = _null_pos;
            }
        }
        else
        {
            if (weight > 0)
                pos = elist.insert(t, weight);
            else
                pos = _null_pos;
        }
    }

    template <bool Add, class Vertex, class Eprop, class Vprop, class Graph>
    void modify_vertex(Vertex v, Vprop& b, Eprop& eweight, Graph& g)
    {
        auto iter_edges = [&](auto&& range)
        {
            for (auto e : range)
            {
                auto ew = (Add) ?  eweight[e] : -eweight[e];
                auto s = b[source(e, g)];
                auto t = b[target(e, g)];
                insert_edge(s, t, ew);
                if (source(e, g) != target(e, g))
                    insert_edge(t, s, ew);
            }
        };

        iter_edges(out_edges_range(v, g));
        if constexpr (is_directed_::apply<Graph>::type::value)
            iter_edges(in_edges_range(v, g));
    }

    template <class Vertex, class Vprop, class Eprop, class Graph>
    void add_vertex(Vertex v, Vprop& b, Eprop& eweight, Graph& g)
    {
        modify_vertex<true>(v, b, eweight, g);
    }

    template <class Vertex, class Vprop, class Eprop, class Graph>
    void remove_vertex(Vertex v, Vprop& b, Eprop& eweight, Graph& g)
    {
        modify_vertex<false>(v, b, eweight, g);
    }

    template <class RNG>
    size_t sample_edge(size_t r, RNG& rng)
    {
        auto s = _egroups[r].sample(rng);
        assert(s != numeric_limits<size_t>::max());
        return s;
    }

private:
    vector<DynamicSampler<size_t>> _egroups;
    vector<gt_hash_map<size_t, size_t>> _pos;
    static constexpr size_t _null_pos = numeric_limits<size_t>::max();
};

} // namespace graph_tool

#endif //GRAPH_BLOCKMODEL_ELIST_HH
