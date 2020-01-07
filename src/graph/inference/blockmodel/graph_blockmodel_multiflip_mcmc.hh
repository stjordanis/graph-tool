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
    ((d,, double, 0))                                                          \
    ((psingle,, double, 0))                                                    \
    ((psplit,, double, 0))                                                     \
    ((pmerge,, double, 0))                                                     \
    ((pmergesplit,, double, 0))                                                \
    ((nproposal, &, vector<size_t>&, 0))                                       \
    ((nacceptance, &, vector<size_t>&, 0))                                     \
    ((gibbs_sweeps,, size_t, 0))                                               \
    ((entropy_args,, entropy_args_t, 0))                                       \
    ((verbose,, int, 0))                                                       \
    ((force_move,, bool, 0))                                                   \
    ((niter,, size_t, 0))

enum class move_t { single = 0, split, merge, mergesplit, null };

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
            _bnext(get(vertex_index_t(), _state._g),
                   num_vertices(_state._g)),
            _btemp(get(vertex_index_t(), _state._g),
                   num_vertices(_state._g))
        {
            _state.init_mcmc(_c,
                             (_entropy_args.partition_dl ||
                              _entropy_args.degree_dl ||
                              _entropy_args.edges_dl));
            for (auto v : vertices_range(_state._g))
            {
                if (_state.node_weight(v) == 0)
                    continue;
                add_element(_groups[_state._b[v]], _vpos, v);
                _N += _state.node_weight(v);
            }

            for (auto r : vertices_range(_state._bg))
            {
                if (_state._wr[r] == 0)
                    continue;
                add_element(_rlist, _rpos, r);
            }

            std::vector<move_t> moves
                = {move_t::single, move_t::split, move_t::merge,
                   move_t::mergesplit};
            std::vector<double> probs
                = {_psingle, _psplit, _pmerge, _pmergesplit};
            _move_sampler = Sampler<move_t, mpl::false_>(moves, probs);
        }

        typename state_t::g_t& _g;

        std::vector<std::vector<size_t>> _groups;
        typename vprop_map_t<size_t>::type::unchecked_t _vpos;
        typename vprop_map_t<size_t>::type::unchecked_t _rpos;
        size_t _nmoves = 0;

        std::vector<std::vector<std::tuple<size_t, size_t>>> _bstack;

        Sampler<move_t, mpl::false_> _move_sampler;

        void _push_b_dispatch() {}

        template <class... Vs>
        void _push_b_dispatch(const std::vector<size_t>& vs, Vs&&... vvs)
        {
            auto& back = _bstack.back();
            for (auto v : vs)
                back.emplace_back(v, _state._b[v]);
            _push_b_dispatch(std::forward<Vs>(vvs)...);
        }

        template <class... Vs>
        void push_b(Vs&&... vvs)
        {
            _bstack.emplace_back();
            _push_b_dispatch(std::forward<Vs>(vvs)...);
        }

        void pop_b()
        {
            auto& back = _bstack.back();
            for (auto& vb : back)
            {
                size_t v = get<0>(vb);
                size_t s = get<1>(vb);
                move_vertex(v, s);
            }
            _bstack.pop_back();
        }

        std::vector<size_t> _rlist;
        std::vector<size_t> _vs;

        typename vprop_map_t<int>::type::unchecked_t _bnext;
        typename vprop_map_t<int>::type::unchecked_t _btemp;

        constexpr static move_t _null_move = move_t::null;

        size_t _N = 0;

        double _dS;
        double _a;

        size_t node_state(size_t r)
        {
            return r;
        }

        constexpr bool skip_node(size_t)
        {
            return false;
        }

        template <bool sample_branch=true>
        size_t sample_new_group(size_t v, rng_t& rng)
        {
            _state.get_empty_block(v);
            auto t = uniform_sample(_state._empty_blocks, rng);

            auto r = _state._b[v];
            _state._bclabel[t] = _state._bclabel[r];
            if (_state._coupled_state != nullptr)
            {
                if constexpr (sample_branch)
                {
                    do
                    {
                        _state._coupled_state->sample_branch(t, r, rng);
                    }
                    while(!_state.allow_move(r, t));
                }
                else
                {
                    auto& bh = _state._coupled_state->get_b();
                    bh[t] = bh[r];
                }
                auto& hpclabel = _state._coupled_state->get_pclabel();
                hpclabel[t] = _state._pclabel[v];
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
            _nmoves++;
        }


        template <class RNG>
        std::tuple<double, double>
        gibbs_sweep(std::vector<size_t>& vs, size_t r, size_t s,
                    double beta, RNG& rng)
        {
            double lp = 0, dS = 0;
            std::array<double,2> p = {0,0};
            std::shuffle(vs.begin(), vs.end(), rng);
            for (auto v : vs)
            {
                size_t bv = _state._b[v];
                size_t nbv = (bv == r) ? s : r;
                double ddS;
                if (_state.virtual_remove_size(v) > 0)
                    ddS = _state.virtual_move(v, bv, nbv, _entropy_args);
                else
                    ddS = std::numeric_limits<double>::infinity();

                if (!std::isinf(beta) && !std::isinf(ddS))
                {
                    double Z = log_sum(0., -ddS * beta);
                    p[0] = -ddS * beta - Z;
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
                    move_vertex(v, nbv);
                    lp += p[0];
                    dS += ddS;
                }
                else
                {
                    lp += p[1];
                }
            }
            return {dS, lp};
        }

        template <bool forward=true, class RNG>
        std::tuple<double, size_t, size_t>
        stage_split_random(std::vector<size_t>& vs, size_t r, size_t s, RNG& rng)
        {
            std::array<size_t, 2> rt = {null_group, null_group};
            std::array<double, 2> ps;
            double dS = 0;

            std::uniform_real_distribution<> unit(0, 1);
            double p = unit(rng);

            std::shuffle(vs.begin(), vs.end(), rng);
            for (auto v : vs)
            {
                if (rt[0] == null_group)
                {
                    rt[0] = r;
                    dS += _state.virtual_move(v, _state._b[v], rt[0],
                                              _entropy_args);
                    move_vertex(v, rt[0]);
                    continue;
                }

                if (rt[1] == null_group)
                {
                    if constexpr (forward)
                        rt[1] = (s == null_group) ? sample_new_group(v, rng) : s;
                    else
                        rt[1] = s;
                    dS += _state.virtual_move(v, _state._b[v], rt[1],
                                              _entropy_args);
                    move_vertex(v, rt[1]);
                    continue;
                }

                ps[0] = log(p);
                ps[1] = log1p(-p);

                double Z = log_sum(ps[0], ps[1]);
                double p0 = ps[0] - Z;
                std::bernoulli_distribution sample(exp(p0));
                if (sample(rng))
                {
                    dS += _state.virtual_move(v, _state._b[v], rt[0],
                                              _entropy_args);
                    move_vertex(v, rt[0]);
                }
                else
                {
                    dS += _state.virtual_move(v, _state._b[v], rt[1],
                                              _entropy_args);
                    move_vertex(v, rt[1]);
                }
            }
            return {dS, rt[0], rt[1]};
        }

        template <bool forward=true, class RNG>
        std::tuple<double, size_t, size_t>
        stage_split_scatter(std::vector<size_t>& vs, size_t r, size_t s, RNG& rng)
        {
            std::array<size_t, 2> rt = {null_group, null_group};
            std::array<double, 2> ps;
            double dS = 0;

            if (s != null_group && _groups[s].empty())
                _state.move_vertex(_groups[r].front(), s);

            size_t t;
            if (_rlist.size() < (forward ? _N - 1 : _N))
                t = sample_new_group<false>(_groups[r].front(), rng);
            else
                t = r;

            if (s != null_group && _groups[s].empty())
                _state.move_vertex(_groups[r].front(), r);

            for (auto v : _groups[r])
            {
                dS += _state.virtual_move(v, _state._b[v], t,
                                          _entropy_args);
                move_vertex(v, t);
            }

            if constexpr (!forward)
            {
                for (auto v : _groups[s])
                {
                    dS += _state.virtual_move(v, _state._b[v], t,
                                              _entropy_args);
                    move_vertex(v, t);
                }
            }

            std::shuffle(vs.begin(), vs.end(), rng);
            for (auto v : vs)
            {
                if (rt[0] == null_group)
                {
                    rt[0] = r;
                    dS += _state.virtual_move(v, _state._b[v], rt[0],
                                              _entropy_args);
                    move_vertex(v, rt[0]);
                    continue;
                }

                if (rt[1] == null_group)
                {
                    if constexpr (forward)
                        rt[1] = (s == null_group) ? sample_new_group(v, rng) : s;
                    else
                        rt[1] = s;
                    dS += _state.virtual_move(v, _state._b[v], rt[1],
                                              _entropy_args);
                    move_vertex(v, rt[1]);
                    continue;
                }

                ps[0] = _state.virtual_move(v, _state._b[v], rt[0],
                                            _entropy_args);
                ps[1] = _state.virtual_move(v, _state._b[v], rt[1],
                                            _entropy_args);;

                double Z = log_sum(ps[0], ps[1]);
                double p0 = ps[0] - Z;
                std::bernoulli_distribution sample(exp(p0));
                if (sample(rng))
                {
                    dS += ps[0];
                    move_vertex(v, rt[0]);
                }
                else
                {
                    dS += ps[1];
                    move_vertex(v, rt[1]);
                }
            }
            return {dS, rt[0], rt[1]};
        }

        template <class RNG, bool forward=true>
        std::tuple<size_t, double, double> split(size_t r, size_t s,
                                                 RNG& rng)
        {
            auto vs = _groups[r];

            if constexpr (!forward)
                vs.insert(vs.end(), _groups[s].begin(), _groups[s].end());

            double dS;
            std::array<size_t, 2> rt;

            std::bernoulli_distribution stage_sample(.5);
            if (stage_sample(rng))
                std::tie(dS, rt[0], rt[1]) = stage_split_random<forward>(vs, r, s, rng);
            else
                std::tie(dS, rt[0], rt[1]) = stage_split_scatter<forward>(vs, r, s, rng);

            for (size_t i = 0; i < _gibbs_sweeps - 1; ++i)
            {
                auto ret = gibbs_sweep(vs, rt[0], rt[1],
                                       (i < _gibbs_sweeps / 2) ? 1 : _beta,
                                       rng);
                dS += get<0>(ret);
            }

            double lp = 0;
            if constexpr (forward)
            {
                auto ret = gibbs_sweep(vs, rt[0], rt[1], _beta, rng);
                dS += get<0>(ret);
                lp = get<1>(ret);
            }

            return {rt[1], dS, lp};
        }

        template <class RNG>
        double split_prob(size_t r, size_t s, RNG& rng)
        {
            auto vs = _groups[r];
            vs.insert(vs.end(), _groups[s].begin(), _groups[s].end());

            push_b(vs);

            for (auto v : vs)
                _btemp[v] = _state._b[v];

            split<RNG, false>(r, s, rng);

            std::shuffle(vs.begin(), vs.end(), rng);

            double lp = 0;
            for (auto v : vs)
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

                size_t tbv = _btemp[v];
                if (tbv == nbv)
                {
                    move_vertex(v, nbv);
                    lp += -ddS - Z;
                }
                else
                {
                    lp += -Z;
                }
            }

            pop_b();

            return lp;
        }

        bool allow_merge(size_t r, size_t s)
        {
            return _state.allow_move(r, s);
        }

        double merge(size_t r, size_t s)
        {
            double dS = 0;

            auto vs = _groups[r];

            for (auto v : vs)
            {
                size_t bv = _state._b[v];
                dS +=_state.virtual_move(v, bv, s, _entropy_args);
                move_vertex(v, s);
            }

            return dS;
        }

        template <class RNG>
        size_t sample_move(size_t r, RNG& rng)
        {
            auto s = r;
            while (s == r)
            {
                size_t v = uniform_sample(_groups[r], rng);
                s  = _state.sample_block(v, _c, 0, rng);
            }
            return s;
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
            return log(get_move_prob(r, s));
        }

        template <class RNG>
        std::tuple<size_t, double, double, double>
        sample_merge(size_t r, RNG& rng)
        {
            size_t s = sample_move(r, rng);

            if (!allow_merge(r, s))
                return {null_group, 0., 0., 0.};

            double pf = 0, pb = 0;
            if (!std::isinf(_beta))
            {
                pf = merge_prob(r, s);
                pb = split_prob(s, r, rng);
            }

            if (_verbose)
                cout << "merge " << _groups[r].size() << " " << _groups[s].size();

            double dS = merge(r, s);

            if (_verbose)
                cout << " " << dS << " " << pf << "  " << pb << endl;

            return {s, dS, pf, pb};
        }

        template <class RNG>
        std::tuple<size_t, double, double, double>
        sample_split(size_t r, size_t s, RNG& rng)
        {
            double dS, pf, pb=0;
            std::tie(s, dS, pf) = split(r, s, rng);
            if (!std::isinf(_beta))
                pb = merge_prob(s, r);

            if (_verbose)
                cout << "split " << _groups[r].size() << " " << _groups[s].size()
                     << " " << dS << " " << pf << " " << pb << endl;

            return {s, dS, pf, pb};
        }

        template <class RNG>
        std::tuple<move_t,size_t> move_proposal(size_t r, RNG& rng)
        {
            double pf = 0, pb = 0;
            _dS = _a = 0;
            _vs.clear();
            _nmoves = 0;

            move_t move = _move_sampler.sample(rng);

            switch (move)
            {
            case move_t::single:
                {
                    auto v = uniform_sample(_groups[r], rng);
                    auto s = _state.sample_block(v, _c, _d, rng);
                    if (s >= _groups.size())
                    {
                        _groups.resize(s+ 1);
                        _rpos.resize(s+ 1);
                    }
                    if (r == s || !_state.allow_move(r, s))
                        return {_null_move, 1};
                    if (_d == 0 && _groups[r].size() == 1 && !std::isinf(_beta))
                        return {_null_move, 1};
                    _dS = _state.virtual_move(v, r, s, _entropy_args);
                    if (!std::isinf(_beta))
                    {
                        pf = log(_state.get_move_prob(v, r, s, _c, _d, false));
                        pb = log(_state.get_move_prob(v, s, r, _c, _d, true));

                        pf += -safelog_fast(_rlist.size());
                        pf += -safelog_fast(_groups[r].size());
                        int dB = 0;
                        if (_groups[s].empty())
                            dB++;
                        if (_groups[r].size() == 1)
                            dB--;
                        pb += -safelog_fast(_rlist.size() + dB);
                        pb += -safelog_fast(_groups[s].size() + 1);
                    }
                    _vs.clear();
                    _vs.push_back(v);
                    _bnext[v] = s;
                    _nmoves++;
                }
                break;

            case move_t::split:
                {
                    if (_groups[r].size() < 2)
                        return {_null_move, 1};

                    _state._egroups_update = false;

                    _vs = _groups[r];
                    push_b(_vs);

                    size_t s;
                    std::tie(s, _dS, pf) = split(r, null_group, rng);

                    if (!std::isinf(_beta))
                    {
                        pf += log(_psplit);
                        pb = merge_prob(s, r) + log(_pmerge);
                    }

                    if (_verbose)
                        cout << "split proposal: " << _groups[r].size() << " "
                             << _groups[s].size() << " " << _dS << " " << pb - pf
                             << " " << -_dS + pb - pf << endl;

                    for (auto v : _vs)
                        _bnext[v] = _state._b[v];
                    pop_b();

                    _state._egroups_update = true;
                }
                break;

            case move_t::merge:
                {
                    if (_rlist.size() == 1)
                        return {_null_move, 1};

                    auto s = sample_move(r, rng);

                    if (!allow_merge(r, s))
                        return {_null_move, 1};

                    _state._egroups_update = false;

                    if (!std::isinf(_beta))
                    {
                        pf = merge_prob(r, s) + log(_pmerge);
                        pb = split_prob(s, r, rng) + log(_psplit);
                    }

                    _vs = _groups[r];
                    push_b(_vs);

                    _dS = merge(r, s);

                    for (auto v : _vs)
                        _bnext[v] = _state._b[v];
                    pop_b();

                    _state._egroups_update = true;

                    if (_verbose)
                        cout << "merge proposal: " <<  _groups[r].size() << " "
                             << _groups[s].size() << " " << _dS << " " << pb - pf
                             << " " << -_dS + pb - pf << endl;
                }
                break;

            case move_t::mergesplit:
                {
                    if (_rlist.size() == 1)
                        return {_null_move, 1};

                    _state._egroups_update = false;

                    push_b(_groups[r]);

                    auto ret = sample_merge(r, rng);
                    size_t s = get<0>(ret);

                    if (s == null_group)
                    {
                        while (!_bstack.empty())
                            pop_b();
                        _state._egroups_update = true;
                        return {_null_move, 1};
                    }

                    _dS += get<1>(ret);
                    pf += get<2>(ret);
                    pb += get<3>(ret);

                    push_b(_groups[s]);

                    ret = sample_split(s, r, rng);
                    _dS += get<1>(ret);
                    pf += get<2>(ret);
                    pb += get<3>(ret);

                    for (auto& vs : _bstack)
                        for (auto& vb : vs)
                        {
                            auto v = get<0>(vb);
                            _vs.push_back(v);
                            _bnext[v] = _state._b[v];
                        }

                    while (!_bstack.empty())
                        pop_b();

                    _state._egroups_update = true;

                    if (_verbose)
                        cout << "mergesplit proposal: " << _dS << " " << pb - pf
                             << " " << -_dS + pb - pf << endl;
                }
                break;

            default:
                return {_null_move, 0};
                break;
            }

            _a = pb - pf;

            if (size_t(move) >= _nproposal.size())
            {
                _nproposal.resize(size_t(move) + 1);
                _nacceptance.resize(size_t(move) + 1);
            }
            _nproposal[size_t(move)]++;

            if (_force_move)
            {
                _nmoves = std::numeric_limits<size_t>::max();
                _a = _dS * _beta + 1;
            }

            return {move, _nmoves};
        }

        std::tuple<double, double>
        virtual_move_dS(size_t, move_t)
        {
            return {_dS, _a};
        }

        void perform_move(size_t, move_t move)
        {
            for (auto v : _vs)
            {
                size_t r = _state._b[v];
                size_t s = _bnext[v];
                if (_groups[s].empty())
                    add_element(_rlist, _rpos, s);

                move_vertex(v, s);

                if (_groups[r].empty() && has_element(_rlist, _rpos, r))
                    remove_element(_rlist, _rpos, r);
            }

            _nacceptance[size_t(move)]++;
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
