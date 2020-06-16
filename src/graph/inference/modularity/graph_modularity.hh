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

#ifndef GRAPH_MODULARITY_HH
#define GRAPH_MODULARITY_HH

#include "config.h"

#include <vector>

#include "../blockmodel/graph_blockmodel_util.hh"
#include "../support/graph_state.hh"

#include "openmp_lock.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

struct modularity_entropy_args_t
{
    double gamma;
};


typedef vprop_map_t<int32_t>::type vmap_t;

#define BLOCK_STATE_params                                                     \
    ((g, &, never_directed, 1))                                                \
    ((_abg, &, boost::any&, 0))                                                \
    ((b,, vmap_t, 0))                                                          \
    ((er, &, vector<size_t>&, 0))                                              \
    ((err, &, vector<size_t>&, 0))

GEN_STATE_BASE(ModularityStateBase, BLOCK_STATE_params)

template <class... Ts>
class ModularityState
    : public ModularityStateBase<Ts...>
{
public:
    GET_PARAMS_USING(ModularityStateBase<Ts...>, BLOCK_STATE_params)
    GET_PARAMS_TYPEDEF(Ts, BLOCK_STATE_params)

    typedef partition_stats<false> partition_stats_t;

    template <class... ATs,
              typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
    ModularityState(ATs&&... args)
        : ModularityStateBase<Ts...>(std::forward<ATs>(args)...),
        _bg(boost::any_cast<std::reference_wrapper<bg_t>>(__abg)),
        _N(HardNumVertices()(_g)),
        _E(HardNumEdges()(_g)),
        _empty_pos(_N),
        _candidate_pos(_N),
        _bclabel(_N),
        _pclabel(_N),
        _wr(_N)
    {
        _wr.resize(num_vertices(_g), 0);
        _er.resize(num_vertices(_g), 0);
        _err.resize(num_vertices(_g), 0);

        for (auto v : vertices_range(_g))
        {
            auto r = _b[v];
            _er[r] += out_degree(v, _g);
            _wr[r]++;
        }

        for (size_t r = 0; r < _N; ++r)
        {
            if (_wr[r] == 0)
                add_element(_empty_blocks, _empty_pos, r);
            else
                add_element(_candidate_blocks, _candidate_pos, r);
        }

        for (auto e : edges_range(_g))
        {
            auto r = _b[source(e, _g)];
            auto s = _b[target(e, _g)];
            if (r == s)
                _err[r] += 2;
        }
    }

    typedef typename
        std::conditional<is_directed_::apply<g_t>::type::value,
                         GraphInterface::multigraph_t,
                         undirected_adaptor<GraphInterface::multigraph_t>>::type
        bg_t;
    bg_t& _bg;

    size_t _N;
    size_t _E;

    std::vector<size_t> _empty_blocks;
    std::vector<size_t> _empty_pos;
    std::vector<size_t> _candidate_blocks;
    std::vector<size_t> _candidate_pos;

    std::vector<size_t> _bclabel;
    std::vector<size_t> _pclabel;
    std::vector<size_t> _wr;

    constexpr static BlockStateVirtualBase* _coupled_state = nullptr;

    typedef int m_entries_t;

    UnityPropertyMap<int,GraphInterface::vertex_t> _vweight;
    UnityPropertyMap<int,GraphInterface::edge_t> _eweight;
    simple_degs_t _degs;
    std::vector<size_t> _bmap;

    typedef modularity_entropy_args_t _entropy_args_t;

    bool _egroups_update = true;

    // =========================================================================
    // State modification
    // =========================================================================

    void move_vertex(size_t v, size_t nr)
    {
        size_t r = _b[v];
        if (nr == r)
            return;

        size_t k = 0;
        size_t m = 0;
        for (auto e : out_edges_range(v, _g))
        {
            ++k;
            auto u = target(e, _g);
            if (u == v)
            {
                ++m;
                continue;
            }
            size_t s = _b[u];
            if (s == r)
                _err[r] -= 2;
            else if (s == nr)
                _err[nr] += 2;
        }

        _err[r] -= m;
        _err[nr] += m;

        _er[r] -= k;
        _er[nr] += k;

        _wr[r]--;
        _wr[nr]++;

        if (_wr[r] == 0)
        {
            add_element(_empty_blocks, _empty_pos, r);
            remove_element(_candidate_blocks, _candidate_pos, r);
        }

        if (_wr[nr] == 1)
        {
            remove_element(_empty_blocks, _empty_pos, nr);
            add_element(_candidate_blocks, _candidate_pos, nr);
        }

        _b[v] = nr;
    }

    size_t virtual_remove_size(size_t v)
    {
        return _wr[_b[v]] - 1;
    }

    constexpr void add_block(size_t)
    {
    }

    double virtual_move(size_t v, size_t r, size_t nr,
                        const modularity_entropy_args_t& ea)
    {
        if (r == nr)
            return 0;

        std::array<int, 2> derr({0,0});
        size_t k = 0;
        size_t m = 0;
        for (auto e : out_edges_range(v, _g))
        {
            ++k;
            auto u = target(e, _g);
            if (u == v)
            {
                ++m;
                continue;
            }
            size_t s = _b[u];
            if (s == r)
                derr[0] -= 2;
            else if (s == nr)
                derr[1] += 2;
        }
        derr[0] -= m;
        derr[1] += m;

        double Qb = 0;
        double Qa = 0;

        double M = 2 * _E;

        Qb += _err[r] - ea.gamma * _er[r] * (_er[r] / M);
        Qb += _err[nr] - ea.gamma * _er[nr] * (_er[nr] / M);

        Qa += (_err[r] + derr[0]) - ea.gamma * (_er[r] - k) * ((_er[r] - k) / M);
        Qa += (_err[nr] + derr[1]) - ea.gamma * (_er[nr] + k) * ((_er[nr] + k) / M);

        double dS = -(Qa - Qb);
        return dS;
    }

    size_t get_empty_block(size_t, bool)
    {
        return _empty_blocks.back();
    }

    size_t sample_block(size_t v, double c, double d, rng_t& rng)
    {
        std::bernoulli_distribution new_r(d);
        if (d > 0 && !_empty_blocks.empty() && new_r(rng))
            return uniform_sample(_empty_blocks, rng);
        std::bernoulli_distribution adj(c);
        auto iter = out_neighbors(v, _g);
        if (c > 0 && iter.first != iter.second && adj(rng))
        {
            auto w = uniform_sample(iter.first, iter.second, rng);
            return _b[w];
        }
        return uniform_sample(_candidate_blocks, rng);
    }

    // Computes the move proposal probability
    double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                         bool reverse)
    {
        size_t B = _candidate_blocks.size();
        if (reverse)
        {
            if (_wr[s] == 1)
                return d;
            if (_wr[r] == 0)
                B++;
        }
        else
        {
            if (_wr[s] == 0)
                return d;
        }

        size_t k_r = 0;
        size_t k_s = 0;
        size_t k = 0;
        for (auto w : out_neighbors_range(v, _g))
        {
            if (size_t(_b[w]) == r)
                k_r++;
            if (size_t(_b[w]) == s)
                k_s++;
            k++;
        }

        double p = ((reverse) ? k_r : k_s) / double(k);

        if (B == _N)
            d = 0;

        return (1. - d) * (c * p + (1 - c)/B);
    }

    template <class MEntries>
    double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                         bool reverse, MEntries&&)
    {
        return get_move_prob(v, r, s, c, d, reverse);
    }

    template <class EArgs, class MEntries>
    double virtual_move(size_t v, size_t r, size_t nr, EArgs&& ea, MEntries&&)
    {
        return virtual_move(v, r, nr, ea);
    }

    double entropy(const modularity_entropy_args_t& ea)
    {
        double Q = 0;
        double M = 2 * _E;
        for (auto r : _candidate_blocks)
            Q += _err[r] - ea.gamma * _er[r] * (_er[r] / M);
        return -Q;
    }

    template <class MCMCState>
    void init_mcmc(MCMCState&)
    {
    }

    constexpr size_t node_weight(size_t)
    {
        return 1;
    }

    bool is_last(size_t v)
    {
        return _wr[_b[v]] == 1;
    }

    constexpr bool allow_move(size_t, size_t)
    {
        return true;
    }

    template <class V>
    void push_state(V&) {}
    void pop_state() {}
    void store_next_state(size_t) {}
    void clear_next_state() {}

};

} // graph_tool namespace

#endif //GRAPH_MODULARITY_HH
