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

#ifndef GRAPH_BLOCKMODEL_MULTIFLIP_MCMC_HH
#define GRAPH_BLOCKMODEL_MULTIFLIP_MCMC_HH

#include "config.h"

#include <vector>
#include <algorithm>
#include <queue>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include "graph_blockmodel_util.hh"
#include <boost/mpl/vector.hpp>

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_BLOCK_STATE_params(State)                                         \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((c,, double, 0))                                                          \
    ((a1,, double, 0))                                                         \
    ((d,, double, 0))                                                          \
    ((psplit,, double, 0))                                                     \
    ((gibbs_sweeps,, size_t, 0))                                               \
    ((entropy_args,, entropy_args_t, 0))                                       \
    ((verbose,, bool, 0))                                                      \
    ((niter,, size_t, 0))

enum class move_t { single_node = 0, split, merge, null };

template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCBlockStateBase, MCMC_BLOCK_STATE_params(State))

    template <class... Ts>
    class MCMCBlockState
        : public MCMCBlockStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MCMCBlockStateBase<Ts...>,
                         MCMC_BLOCK_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_BLOCK_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCBlockState(ATs&&... as)
           : MCMCBlockStateBase<Ts...>(as...),
            _g(_state._g),
            _groups(num_vertices(_state._bg)),
            _vpos(get(vertex_index_t(), _state._g),
                  num_vertices(_state._g)),
            _rpos(get(vertex_index_t(), _state._bg),
                  num_vertices(_state._bg)),
            _btemp(get(vertex_index_t(), _state._g),
                   num_vertices(_state._g)),
            _btemp2(get(vertex_index_t(), _state._g),
                    num_vertices(_state._g)),
            _bprev(get(vertex_index_t(), _state._g),
                   num_vertices(_state._g)),
            _bnext(get(vertex_index_t(), _state._g),
                   num_vertices(_state._g))
        {
            _state.init_mcmc(_c,
                             (_entropy_args.partition_dl ||
                              _entropy_args.degree_dl ||
                              _entropy_args.edges_dl));
            for (auto v : vertices_range(_state._g))
            {
                if (_state._vweight[v] == 0)
                    continue;
                add_element(_groups[_state._b[v]], _vpos, v);
                _N += _state._vweight[v];
            }

            for (auto r : vertices_range(_state._bg))
            {
                if (_state._wr[r] == 0)
                    continue;
                add_element(_rlist, _rpos, r);
            }
        }

        typename state_t::g_t& _g;

        std::vector<std::vector<size_t>> _groups;
        typename vprop_map_t<size_t>::type::unchecked_t _vpos;
        typename vprop_map_t<size_t>::type::unchecked_t _rpos;

        typename vprop_map_t<int>::type::unchecked_t _btemp;
        typename vprop_map_t<int>::type::unchecked_t _btemp2;
        typename vprop_map_t<int>::type::unchecked_t _bprev;
        typename vprop_map_t<int>::type::unchecked_t _bnext;

        std::vector<size_t> _rlist;
        std::vector<size_t> _vs;

        constexpr static move_t _null_move = move_t::null;

        size_t _N = 0;

        double _dS;
        double _a;
        size_t _s = null_group;
        size_t _t = null_group;
        size_t _u = null_group;
        size_t _v = null_group;

        size_t node_state(size_t r)
        {
            return r;
        }

        constexpr bool skip_node(size_t)
        {
            return false;
        }

        size_t sample_new_group(size_t v, rng_t& rng)
        {
            _state.get_empty_block(v);
            auto t = uniform_sample(_state._empty_blocks, rng);

            if (_state._coupled_state != nullptr)
            {
                auto r = _state._b[v];
                _state._coupled_state->sample_branch(t, r, rng);
                _state._bclabel[t] = _state._bclabel[r];
            }
            if (t >= _groups.size())
            {
                _groups.resize(t + 1);
                _rpos.resize(t + 1);
            }
            assert(_state._wr[t] == 0);
            return t;
        }

        void move_vertex(size_t v, size_t r)
        {
            size_t s = _state._b[v];
            if (s == r)
                return;
            remove_element(_groups[s], _vpos, v);
            _state.move_vertex(v, r);
            add_element(_groups[r], _vpos, v);
        }

        template <class RNG, bool forward=true>
        std::tuple<size_t, size_t, double, double> split(size_t t, size_t r,
                                                         size_t s, RNG& rng)
        {
            if (forward)
                _vs = _groups[t];
            std::shuffle(_vs.begin(), _vs.end(), rng);

            std::array<size_t, 2> rt = {null_group, null_group};
            std::array<double, 2> ps;
            double lp = -log(2);
            double dS = 0;
            for (auto v : _vs)
            {
                if (forward)
                    _bprev[v] = t;
                else
                    _btemp[v] = _state._b[v];

                if (rt[0] == null_group)
                {
                    if (forward)
                        rt[0] = sample_new_group(v, rng);
                    else
                        rt[0] = r;
                    dS += _state.virtual_move(v, _state._b[v], rt[0],
                                              _entropy_args);
                    if (forward)
                        move_vertex(v, rt[0]);
                    else
                        _state.move_vertex(v, rt[0]);
                    continue;
                }

                if (rt[1] == null_group)
                {
                    if (forward)
                        // rt[1] = (_state.virtual_remove_size(v) == 0) ?
                        //     t : sample_new_group(v, rng);
                        rt[1] = sample_new_group(v, rng);
                    else
                        rt[1] = s;
                    dS += _state.virtual_move(v, _state._b[v], rt[1],
                                              _entropy_args);
                    if (forward)
                        move_vertex(v, rt[1]);
                    else
                        _state.move_vertex(v, rt[1]);
                    continue;
                }

                ps[0] = -_state.virtual_move(v, _state._b[v], rt[0],
                                             _entropy_args);
                ps[1] = -_state.virtual_move(v, _state._b[v], rt[1],
                                             _entropy_args);

                double Z = 0, p0 = 0;
                if (!std::isinf(_beta))
                {
                    Z = log_sum(_beta * ps[0], _beta * ps[1]);
                    p0 = _beta * ps[0] - Z;
                }
                else
                {
                    p0 =  (ps[0] < ps[1]) ? 0 : -numeric_limits<double>::infinity();
                }

                std::bernoulli_distribution sample(exp(p0));
                if (sample(rng))
                {
                    if (forward)
                        move_vertex(v, rt[0]);
                    else
                        _state.move_vertex(v, rt[0]);
                    lp += p0;
                    dS -= ps[0];
                }
                else
                {
                    if (forward)
                        move_vertex(v, rt[1]);
                    else
                        _state.move_vertex(v, rt[1]);
                    if (!std::isinf(_beta))
                        lp += _beta * ps[1] - Z;
                    dS -= ps[1];
                }
            }

            // gibbs sweep
            for (size_t i = 0; i < (forward ? _gibbs_sweeps : _gibbs_sweeps - 1); ++i)
            {
                lp = 0;
                std::array<double,2> p = {0,0};
                for (auto v : _vs)
                {
                    size_t bv = _state._b[v];
                    size_t nbv = (bv == rt[0]) ? rt[1] : rt[0];
                    double ddS;
                    if (_state.virtual_remove_size(v) > 0)
                        ddS = _state.virtual_move(v, bv, nbv, _entropy_args);
                    else
                        ddS = std::numeric_limits<double>::infinity();

                    if (!std::isinf(_beta) && !std::isinf(ddS))
                    {
                        double Z = log_sum(0., -ddS * _beta);
                        p[0] = -ddS * _beta - Z;
                        p[1] = -Z;
                    }
                    else
                    {
                        if (ddS < 0)
                        {
                            p[0] = 0;
                            p[1] = -std::numeric_limits<double>::infinity();
                        }
                        else
                        {
                            p[0] = -std::numeric_limits<double>::infinity();;
                            p[1] = 0;
                        }
                    }

                    std::bernoulli_distribution sample(exp(p[0]));
                    if (sample(rng))
                    {
                        if (forward)
                            move_vertex(v, nbv);
                        else
                            _state.move_vertex(v, nbv);
                        lp += p[0];
                        dS += ddS;
                    }
                    else
                    {
                        lp += p[1];
                    }
                }
            }

            return {rt[0], rt[1], dS, lp};
        }

        template <class RNG>
        double split_prob(size_t t, size_t r, size_t s, RNG& rng)
        {
            split<RNG, false>(t, r, s, rng);

            for (auto v : _vs)
                _btemp2[v] = _state._b[v];

            double lp1 = 0, lp2 = 0;
            for (bool swap : std::array<bool,2>({false, true}))
            {
                if (swap)
                    std::swap(r, s);
                for (auto v : _vs)
                {
                    size_t bv = _state._b[v];
                    size_t nbv = (bv == r) ? s : r;
                    double ddS;
                    if (_state.virtual_remove_size(v) > 0)
                        ddS = _state.virtual_move(v, bv, nbv, _entropy_args);
                    else
                        ddS = std::numeric_limits<double>::infinity();

                    if (!std::isinf(ddS))
                        ddS *= _beta;

                    double Z = log_sum(0., -ddS);

                    double p;
                    if ((size_t(_bprev[v]) == r) == (nbv == r))
                    {
                        _state.move_vertex(v, nbv);
                        p = -ddS - Z;
                    }
                    else
                    {
                        p = -Z;
                    }

                    if (swap)
                        lp2 += p;
                    else
                        lp1 += p;
                }

                if (!swap)
                {
                    for (auto v : _vs)
                        _state.move_vertex(v, _btemp2[v]);
                }
            }

            for (auto v : _vs)
                _state.move_vertex(v, _btemp[v]);

            return log_sum(lp1, lp2) - log(2);;
        }

        bool allow_merge(size_t r, size_t s)
        {
            // if (_state._coupled_state != nullptr)
            // {
            //     auto& bh = _state._coupled_state->get_b();
            //     if (bh[r] != bh[s])
            //         return false;
            // }
            //return _state._bclabel[r] == _state._bclabel[s];
            return _state.allow_move(r, s);
        }


        template <class RNG>
        std::tuple<size_t, double> merge(size_t r, size_t s, RNG& rng)
        {
            double dS = 0;

            _vs = _groups[r];
            _vs.insert(_vs.end(), _groups[s].begin(), _groups[s].end());

            size_t t = sample_new_group(_vs.front(), rng);

            for (auto v : _vs)
            {
                size_t bv = _bprev[v] = _state._b[v];
                dS +=_state.virtual_move(v, bv, t, _entropy_args);
                move_vertex(v, t);
            }

            return {t, dS};
        }

        double get_move_prob(size_t r, size_t s)
        {
            double prs = 0, prr = 0;
            for (auto v : _groups[r])
            {
                prs += _state.get_move_prob(v, r, s, _c, 0, false);
                prr += _state.get_move_prob(v, r, r, _c, 0, false);
            }
            prs /= _groups[r].size();
            prr /= _groups[r].size();
            return prs/(1-prr);
        }

        double merge_prob(size_t r, size_t s)
        {
            double pr = get_move_prob(r, s);
            double ps = get_move_prob(s, r);
            return log(pr + ps) - log(2);
            return pr;
        }

        template <class RNG>
        std::tuple<move_t,size_t> move_proposal(size_t r, RNG& rng)
        {
            move_t move;
            double pf = 0, pb = 0;
            _dS = _a = 0;

            std::bernoulli_distribution single(_a1);
            if (single(rng))
            {
                move = move_t::single_node;
                auto v = uniform_sample(_groups[r], rng);
                _s = _state.sample_block(v, _c, _d, rng);
                if (_s >= _groups.size())
                {
                    _groups.resize(_s + 1);
                    _rpos.resize(_s + 1);
                }
                if (r == _s || !_state.allow_move(r, _s))
                    return {_null_move, 1};
                if (_d == 0 && _groups[r].size() == 1 && !std::isinf(_beta))
                    return {_null_move, 1};
                _dS = _state.virtual_move(v, r, _s, _entropy_args);
                if (!std::isinf(_beta))
                {
                    pf = log(_state.get_move_prob(v, r, _s, _c, _d, false));
                    pb = log(_state.get_move_prob(v, _s, r, _c, _d, true));

                    pf += -safelog_fast(_rlist.size());
                    pf += -safelog_fast(_groups[r].size());
                    int dB = 0;
                    if (_groups[_s].empty())
                        dB++;
                    if (_groups[r].size() == 1)
                        dB--;
                    pb += -safelog_fast(_rlist.size() + dB);
                    pb += -safelog_fast(_groups[_s].size() + 1);
                }
                _vs.clear();
                _vs.push_back(v);
                _bprev[v] = r;
                _bnext[v] = _s;
            }
            else
            {
                std::bernoulli_distribution do_split(_psplit);
                if (do_split(rng))
                {
                    if (_groups[r].size() < 2 || _rlist.size() > _N - 2)
                        return {_null_move, 1};
                    move = move_t::split;
                    if (!std::isinf(_beta))
                        pf = log(_psplit);
                    std::tie(_s, _t, _dS, pf) = split(r, null_group, null_group, rng);
                    if (!std::isinf(_beta))
                        pb = merge_prob(_s, _t) + log(1 - _psplit);
                }
                else
                {
                    if (_rlist.size() == 1 || _rlist.size() == _N)
                        return {_null_move, 1};
                    move = move_t::merge;
                    _s = r;
                    while (_s == r)
                    {
                        size_t v = uniform_sample(_groups[r], rng);
                        _s = _state.sample_block(v, _c, 0, rng);
                    }
                    if (!allow_merge(r, _s))
                        return {_null_move, _groups[r].size() + _groups[_s].size()};
                     if (!std::isinf(_beta))
                        pf = merge_prob(r, _s) + log(1 - _psplit);
                    std::tie(_t, _dS) = merge(r, _s, rng);
                    if (!std::isinf(_beta))
                        pb = split_prob(_t, r, _s, rng) + log(_psplit);
                }
                for (auto v : _vs)
                {
                    _bnext[v] = _state._b[v];
                    move_vertex(v, _bprev[v]);
                }
            }

            _a = pb - pf;

            size_t nproposals = (move == move_t::single_node || std::isinf(_beta)) ?
                _vs.size() : _vs.size() * _gibbs_sweeps;

            return {move, nproposals};
        }

        std::tuple<double, double>
        virtual_move_dS(size_t, move_t)
        {
            return {_dS, _a};
        }

        void perform_move(size_t r, move_t move)
        {
            for (auto v : _vs)
                move_vertex(v, _bnext[v]);

            switch (move)
            {
            case move_t::single_node:
                if (_groups[r].empty())
                    remove_element(_rlist, _rpos, r);
                if (!has_element(_rlist, _rpos, _s))
                    add_element(_rlist, _rpos, _s);
                break;
            case move_t::split:
                remove_element(_rlist, _rpos, r);
                add_element(_rlist, _rpos, _s);
                add_element(_rlist, _rpos, _t);
                break;
            case move_t::merge:
                remove_element(_rlist, _rpos, r);
                remove_element(_rlist, _rpos, _s);
                add_element(_rlist, _rpos, _t);
                break;
            default:
                break;
            }
        }

        constexpr bool is_deterministic()
        {
            return false;
        }

        constexpr bool is_sequential()
        {
            return false;
        }

        auto& get_vlist()
        {
            return _rlist;
        }

        size_t get_N()
        {
            return _N;
        }

        double get_beta()
        {
            return _beta;
        }

        size_t get_niter()
        {
            return _niter;
        }

        constexpr void step(size_t, move_t)
        {
        }
    };
};

std::ostream& operator<<(std::ostream& os, move_t move);

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_MCMC_HH
