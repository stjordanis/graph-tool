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

#ifndef GRAPH_PARTITION_MODE_CLUSTERING_HH
#define GRAPH_PARTITION_MODE_CLUSTERING_HH

#include "config.h"

#include <vector>

#include "../blockmodel/graph_blockmodel_util.hh"
#include "../blockmodel/graph_blockmodel_partition.hh"
#include "../support/graph_state.hh"
#include "graph_partition_mode.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef multi_array_ref<int32_t,2> bs_t;
typedef multi_array_ref<int32_t,1> b_t;

#define BLOCK_STATE_params                                                     \
    ((g, &, always_directed_never_reversed, 1))                                \
    ((_abg, &, boost::any&, 0))                                                \
    ((bs,, bs_t, 0))                                                           \
    ((relabel_init,, bool, 0))                                                 \
    ((b,, b_t, 0))

GEN_STATE_BASE(ModeClusterStateBase, BLOCK_STATE_params)

template <class T>
std::vector<T> vrange(size_t N)
{
    std::vector<T> vs(N);
    std::iota(vs.begin(), vs.end(), 0);
    return vs;
}

template <class... Ts>
class ModeClusterState
    : public ModeClusterStateBase<Ts...>
{
public:
    GET_PARAMS_USING(ModeClusterStateBase<Ts...>, BLOCK_STATE_params)
    GET_PARAMS_TYPEDEF(Ts, BLOCK_STATE_params)

    template <class... ATs,
              typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
    ModeClusterState(ATs&&... args)
        : ModeClusterStateBase<Ts...>(std::forward<ATs>(args)...),
        _bg(boost::any_cast<std::reference_wrapper<bg_t>>(__abg)),
        _M(_bs.shape()[0]),
        _N(_bs.shape()[1]),
        _pos(_M),
        _modes(_M, PartitionModeState(_N)),
        _wr(_M),
        _empty_pos(_M),
        _candidate_pos(_M),
        _bclabel(_M),
        _pclabel(_M),
        _vs(vrange<size_t>(_M)),
        _partition_stats(_g, _b, _vs, 0, _M, _vweight, _eweight, _degs, _bmap),
        _next_state(_M)
    {
        for (size_t r : _b)
            _wr[r]++;

        for (size_t r = 0; r < _M; ++r)
        {
            if (_wr[r] == 0)
                add_element(_empty_blocks, _empty_pos, r);
            else
                add_element(_candidate_blocks, _candidate_pos, r);
        }

        for (size_t i = 0; i < _M; ++i)
        {
            auto r = _b[i];
            auto x = get_partition(i);
            _pos[i] = _modes[r].add_partition(x, _relabel_init);
        }
    }

    typedef typename
        std::conditional<is_directed_::apply<g_t>::type::value,
                         GraphInterface::multigraph_t,
                         undirected_adaptor<GraphInterface::multigraph_t>>::type
        bg_t;
    bg_t& _bg;

    size_t _M;
    size_t _N;
    std::vector<size_t> _pos;
    std::vector<PartitionModeState> _modes;

    std::vector<size_t> _wr;

    std::vector<size_t> _empty_blocks;
    std::vector<size_t> _empty_pos;
    std::vector<size_t> _candidate_blocks;
    std::vector<size_t> _candidate_pos;

    std::vector<size_t> _bclabel;
    std::vector<size_t> _pclabel;

    UnityPropertyMap<int,GraphInterface::vertex_t> _vweight;
    UnityPropertyMap<int,GraphInterface::edge_t> _eweight;
    simple_degs_t _degs;
    std::vector<size_t> _bmap;
    std::vector<size_t> _vs;

    partition_stats<false> _partition_stats;

    std::vector<std::vector<std::tuple<size_t,multi_array<int32_t,1>>>> _lstack;
    std::vector<multi_array<int32_t,1>> _next_state;
    std::vector<size_t> _next_list;

    constexpr static BlockStateVirtualBase* _coupled_state = nullptr;

    typedef int m_entries_t;

    bool _egroups_update = true;

    b_t get_partition(size_t i)
    {
        auto* data = &_bs[i][0];
        return b_t(data, extents[_N]);
    }

    PartitionModeState& get_mode(size_t r)
    {
        return _modes[r];
    }

    // =========================================================================
    // State modification
    // =========================================================================

    void move_vertex(size_t v, size_t nr)
    {
        size_t r = _b[v];
        if (nr == r && _next_state[v].size() == 0)
            return;

        _modes[r].remove_partition(_pos[v]);

        auto x = get_partition(v);
        if (_next_state[v].size() == 0)
        {
            _pos[v] = _modes[nr].add_partition(x, true);
        }
        else
        {
            x = _next_state[v];
            _pos[v] = _modes[nr].add_partition(x, false);
        }

        if (nr == r)
            return;

        _wr[r]--;
        _wr[nr]++;

        _partition_stats.remove_vertex(v, r, false, _g,
                                       _vweight, _eweight,
                                       _degs);
        _partition_stats.add_vertex(v, nr, false, _g,
                                    _vweight, _eweight,
                                    _degs);

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

    template <class VS>
    void push_state(VS& vs)
    {
        _lstack.emplace_back();
        auto& back = _lstack.back();
        for (auto v : vs)
        {
            auto b = get_partition(v);
            back.emplace_back(v, b);
        }
    }

    void pop_state()
    {
        auto& back = _lstack.back();
        for (auto vx : back)
        {
            size_t v = get<0>(vx);
            auto& x = get<1>(vx);
            auto b = get_partition(v);
            auto r = _b[v];
            _modes[r].remove_partition(_pos[v]);
            b = x;
            _pos[v] = _modes[r].add_partition(b, false);
        }
        _lstack.pop_back();
    }

    void store_next_state(size_t i)
    {
        _next_state[i].resize(extents[_N]);
        _next_state[i] = get_partition(i);
        _next_list.push_back(i);
    }

    void clear_next_state()
    {
        for (auto v : _next_list)
            _next_state[v].resize(extents[0]);
        _next_list.clear();
    }

    size_t virtual_remove_size(size_t v)
    {
        return _wr[_b[v]] - 1;
    }

    constexpr size_t add_block()
    {
        return 0;
    }

    double virtual_move(size_t v, size_t r, size_t nr)
    {
        if (r == nr)
            return 0;

        double dS = 0;

        auto x = get_partition(v);

        dS += _modes[r].virtual_remove_partition(x);
        dS += _modes[nr].virtual_add_partition(x);

        dS += _partition_stats.get_delta_partition_dl(v, r, nr, _vweight);

        return dS;
    }

    size_t get_empty_block(size_t)
    {
        return _empty_blocks.back();
    }

    size_t sample_block(size_t, double, double d, rng_t& rng)
    {
        std::bernoulli_distribution new_r(d);
        if (d > 0 && !_empty_blocks.empty() && new_r(rng))
            return uniform_sample(_empty_blocks, rng);
        return uniform_sample(_candidate_blocks, rng);
    }

    // Computes the move proposal probability
    double get_move_prob(size_t, size_t r, size_t s, double, double d, bool reverse)
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

        if (B == _M)
            d = 0;

        return (1. - d) / B;
    }

    template <class MEntries>
    double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                         bool reverse, MEntries&&)
    {
        return get_move_prob(v, r, s, c, d, reverse);
    }

    template <class EArgs>
    double virtual_move(size_t v, size_t r, size_t nr, EArgs&&)
    {
        return virtual_move(v, r, nr);
    }

    template <class EArgs, class MEntries>
    double virtual_move(size_t v, size_t r, size_t nr, EArgs&&, MEntries&&)
    {
        return virtual_move(v, r, nr);
    }

    void replace_partitions()
    {
        for (size_t j = 0; j < _M; ++j)
        {
            auto& mode = _modes[_b[j]];
            auto b = get_partition(j);
            double dS = mode.virtual_remove_partition(b);
            mode.remove_partition(_pos[j]);
            dS += mode.virtual_add_partition(b);
            _pos[j] = mode.add_partition(b, dS < 0);
        }
    }

    double entropy()
    {
        double S = 0;
        for (auto r: _candidate_blocks)
            S += _modes[r].entropy();
        S += _partition_stats.get_partition_dl();
        return S;
    }

    double posterior_entropy()
    {
        double S = 0;
        for (size_t r = 0; r < _wr.size(); ++r)
        {
            S += (_modes[r].posterior_entropy() * _wr[r]) / _M;
            S += -xlogx(_wr[r] / double(_M));
        }
        return S;
    }

    void relabel_mode(PartitionModeState& x, PartitionModeState& base)
    {
        adj_list<> g;
        typename vprop_map_t<int32_t>::type label(get(vertex_index_t(), g));
        typename eprop_map_t<long double>::type mrs(get(edge_index_t(), g));

        get_contingency_graph(g, label, mrs, x._nr, base._nr);

        typedef typename graph_traits<adj_list<>>::vertex_descriptor vertex_t;
        typename vprop_map_t<vertex_t>::type match(get(vertex_index_t(), g));

        auto u = undirected_adaptor<adj_list<>>(g);
        maximum_weighted_matching(u, mrs, match);

        idx_map<int32_t, size_t> x_vertices;
        for (auto v : vertices_range(g))
        {
            if (v < x._B)
                x_vertices[label[v]] = v;
            else
                break;
        }

        std::vector<int32_t> unmatched;
        size_t max_s = 0;
        for (size_t r = 0; r < x._count.size(); ++r)
        {
            if (x._count[r] == 0)
                continue;
            auto s = match[x_vertices[r]];
            if (s == graph_traits<adj_list<>>::null_vertex())
                unmatched.push_back(r);
            else
                max_s = std::max(max_s, s);
        }
        std::sort(unmatched.begin(), unmatched.end(),
                  [&](auto r, auto s) { return x._count[r] > x._count[s]; });
        idx_map<int32_t, int32_t> umatch;
        for (auto& r : unmatched)
            umatch[r] = ++max_s;


        idx_map<int, int> rpos;
        for (auto& jb : x._bs)
        {
            auto b = x.get_partition(jb.first);
            for (auto& r : b)
            {
                auto v = match[x_vertices[r]];
                if (v != graph_traits<adj_list<>>::null_vertex())
                    r = label[v];
                else
                    r = umatch[r];
            }
        }
        x.rebuild_nr();
    }

    void relabel_modes()
    {
        auto modes = vrange<size_t>(_M);
        std::sort(modes.begin(), modes.end(),
                  [&](auto r, auto s) { return _wr[r] > _wr[s]; });
        PartitionModeState base(_N);

        for (size_t r : modes)
        {
            auto& mode = _modes[r];
            if (!base._bs.empty())
                relabel_mode(mode, base);
            else
                mode.relabel();
            for (auto& jb : mode._bs)
            {
                auto b = mode.get_partition(jb.first);
                base.add_partition(b, false);
            }
        }
    }

    void init_mcmc(double, double)
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
};

} // graph_tool namespace

#endif //GRAPH_PARTITION_MODE_CLUSTERING_HH
