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

#ifndef GRAPH_PARTITION_CENTROID_HH
#define GRAPH_PARTITION_CENTROID_HH

#include "config.h"

#include <vector>

#include "../blockmodel/graph_blockmodel_util.hh"
#include "../support/graph_state.hh"

#include "openmp_lock.hh"

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
    ((b,, b_t, 0))

GEN_STATE_BASE(VICenterStateBase, BLOCK_STATE_params)

template <class... Ts>
class VICenterState
    : public VICenterStateBase<Ts...>
{
public:
    GET_PARAMS_USING(VICenterStateBase<Ts...>, BLOCK_STATE_params)
    GET_PARAMS_TYPEDEF(Ts, BLOCK_STATE_params)

    template <class... ATs,
              typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
    VICenterState(ATs&&... args)
        : VICenterStateBase<Ts...>(std::forward<ATs>(args)...),
        _bg(boost::any_cast<std::reference_wrapper<bg_t>>(__abg)),
        _mrs(_bs.shape()[0]),
        _nr(_bs.shape()[0]),
        _N(_bs.shape()[1]),
        _wr(_N),
        _empty_pos(_N),
        _candidate_pos(_N),
        _bclabel(_N),
        _pclabel(_N)
    {
        for (size_t r : _b)
            _wr[r]++;

        for (size_t r = 0; r < _N; ++r)
        {
            if (_wr[r] == 0)
                add_element(_empty_blocks, _empty_pos, r);
            else
                add_element(_candidate_blocks, _candidate_pos, r);
        }

        for (size_t i = 0; i < _mrs.size(); ++i)
        {
            for (size_t v = 0; v < _N; ++v)
            {
                auto r = _b[v];
                auto s = _bs[i][v];
                _mrs[i][{r,s}]++;
                _nr[i][s]++;
            }
        }
    }

    typedef typename
        std::conditional<is_directed_::apply<g_t>::type::value,
                         GraphInterface::multigraph_t,
                         undirected_adaptor<GraphInterface::multigraph_t>>::type
        bg_t;
    bg_t& _bg;

    std::vector<gt_hash_map<std::tuple<size_t, size_t>, size_t>> _mrs;
    std::vector<gt_hash_map<size_t, size_t>> _nr;

    size_t _N;

    std::vector<size_t> _wr;

    std::vector<size_t> _empty_blocks;
    std::vector<size_t> _empty_pos;
    std::vector<size_t> _candidate_blocks;
    std::vector<size_t> _candidate_pos;

    std::vector<size_t> _bclabel;
    std::vector<size_t> _pclabel;

    typedef char _entropy_args_t;

    constexpr static BlockStateVirtualBase* _coupled_state = nullptr;

    typedef int m_entries_t;

    bool _egroups_update = true;

    // =========================================================================
    // State modification
    // =========================================================================

    void move_vertex(size_t v, size_t nr)
    {
        size_t r = _b[v];
        if (nr == r)
            return;

        _wr[r]--;
        _wr[nr]++;

        //#pragma omp parallel for schedule(runtime)
        for (size_t i = 0; i < _mrs.size(); ++i)
        {
            auto& mrsi = _mrs[i];
            size_t s = _bs[i][v];
            auto iter = mrsi.find({r, s});
            assert(iter != mrsi.end());
            iter->second--;
            if (iter->second == 0)
                mrsi.erase(iter);
            mrsi[{nr,s}]++;
        }

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

    double virtual_move(size_t v, size_t r, size_t nr)
    {
        if (r == nr)
            return 0;

        double Sb = 0;
        double Sa = 0;

        Sb += (xlogx_fast(_wr[r]) + xlogx_fast(_wr[nr])) * _mrs.size();
        Sa += (xlogx_fast(_wr[r]-1) + xlogx_fast(_wr[nr]+1)) * _mrs.size();

        #pragma omp parallel for schedule(runtime) reduction(+:Sa, Sb) \
            if (_mrs.size() > OPENMP_MIN_THRESH)
        for (size_t i = 0; i < _mrs.size(); ++i)
        {
            auto& mrsi = _mrs[i];
            size_t s = _bs[i][v];

            size_t mrs = mrsi[{r,s}];
            assert(mrs > 0);
            auto iter = mrsi.find({nr, s});
            size_t mnrs = (iter != mrsi.end()) ? iter->second : 0;

            Sb += -2 * (xlogx_fast(mrs) + xlogx_fast(mnrs));
            Sa += -2 * (xlogx_fast(mrs-1) + xlogx_fast(mnrs+1));
        }
        return (Sa - Sb);
    }

    size_t get_empty_block(size_t, bool)
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
    double get_move_prob(size_t, size_t r, size_t s, double, double d,
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

        if (B == _N)
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

    double entropy()
    {
        double S = 0, S_n = 0;
        for (auto nr : _wr)
            S_n += xlogx_fast(nr);

        #pragma omp parallel for schedule(runtime) reduction(+:S)
        for (size_t i = 0; i < _mrs.size(); ++i)
        {
            for (auto& c : _mrs[i])
                S -= 2 * xlogx_fast(c.second);
            for (auto& rn : _nr[i])
                S += xlogx_fast(rn.second);
            S += S_n;
        }
        return S;
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

#endif //GRAPH_PARTITION_CENTROID_HH
