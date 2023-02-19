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

#ifndef GRAPH_HISTOGRAM_MCMC_HH
#define GRAPH_HISTOGRAM_MCMC_HH

#include "config.h"

#include <vector>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include <boost/mpl/vector.hpp>

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_HIST_STATE_params(State)                                          \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((verbose,, int, 0))                                                       \
    ((niter,, size_t, 0))

enum class hmove_t { move = 0, add, remove, null };

ostream& operator<<(ostream& s, hmove_t v);

template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCHistStateBase, MCMC_HIST_STATE_params(State))

    template <class... Ts>
    class MCMCHistState
        : public MCMCHistStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MCMCHistStateBase<Ts...>,
                         MCMC_HIST_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_HIST_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCHistState(ATs&&... as)
           : MCMCHistStateBase<Ts...>(as...)
        {
            _state.update_bounds();
        }

        constexpr static hmove_t _null_move = hmove_t::null;

        size_t node_state(size_t)
        {
            return 0;
        }

        constexpr bool skip_node(size_t)
        {
            return false;
        }

        size_t _i;
        size_t _j;
        double _x;

        constexpr static double _epsilon = 1e-8;

        template <class RNG>
        hmove_t move_proposal(size_t, RNG& rng)
        {
            std::uniform_int_distribution<size_t>
                random_j(0, _state._D-1);
            _j = random_j(rng);

            std::uniform_int_distribution<size_t>
                random_i(0, _state._bins[_j]->size()-1);
            _i = random_i(rng);


            size_t m;
            if (_i == _state._bins[_j]->size()-1)
            {
                m = 0;
            }
            else if (_i == 0)
            {
                std::uniform_int_distribution<size_t> random(0, 1);
                m = random(rng);
            }
            else
            {
                std::uniform_int_distribution<size_t> random(0, 2);
                m = random(rng);
            }

            hmove_t move = hmove_t::null;
            switch (m)
            {
            case 0:
                move = hmove_t::move;
                {
                    if (_i == 0)
                    {
                        if (_state._bounded[_j].first)
                        {
                            move = hmove_t::null;
                        }
                        else
                        {
                            auto w = _state._bounds[_j].first - *_state._bins[_j]->begin();
                            if (_state._discrete[_j])
                            {
                                w++;
                                std::geometric_distribution<int64_t> d(1./(2 * w));
                                _x = _state._bounds[_j].first - d(rng) - 1;
                            }
                            else
                            {
                                w = std::max(double(w), _epsilon);
                                std::exponential_distribution<double> d(1./(2 * w));
                                _x = _state._bounds[_j].first - d(rng);
                                // if (_x > _state._bounds[_j].first ||
                                //     _x >= *(_state._bins[_j]->begin() + 1))
                                //     move = hmove_t::null;
                            }

                            assert(_x <= _state._bounds[_j].first);
                        }
                    }
                    else if (_i == _state._bins[_j]->size()-1)
                    {
                        if (_state._bounded[_j].second)
                        {
                            move = hmove_t::null;
                        }
                        else
                        {
                            auto w = *_state._bins[_j]->rbegin() - _state._bounds[_j].second;
                            if (_state._discrete[_j])
                            {
                                w++;
                                std::geometric_distribution<int64_t> d(1./(2 * w));
                                _x = _state._bounds[_j].second + d(rng) + 1;
                            }
                            else
                            {
                                w = std::max(double(w), _epsilon);
                                std::exponential_distribution<double> d(1./(2 * w));
                                _x = _state._bounds[_j].second + d(rng);
                                if (_x == _state._bounds[_j].second)
                                    move = hmove_t::null;
                                // if (_x <= _state._bounds[_j].second ||
                                //     _x <= *(_state._bins[_j]->end() - 2))
                                //     move = hmove_t::null;
                            }
                            assert(_x > _state._bounds[_j].second);
                        }
                    }
                    else
                    {
                        if (_state._discrete[_j])
                        {
                            std::uniform_int_distribution<int64_t>
                                random_x((*_state._bins[_j])[_i-1]+1,
                                         (*_state._bins[_j])[_i+1]-1);
                            _x = random_x(rng);
                        }
                        else
                        {
                            std::uniform_real_distribution<double>
                                random_x((*_state._bins[_j])[_i-1],
                                         (*_state._bins[_j])[_i+1]);
                            _x = random_x(rng);
                            if (_x <= (*_state._bins[_j])[_i-1] ||
                                _x >= (*_state._bins[_j])[_i+1])
                                move = hmove_t::null;
                        }
                    }
                }
                break;
            case 1:
                move = hmove_t::add;
                {
                    if (_state._discrete[_j])
                    {
                        auto a = (*_state._bins[_j])[_i] + 1;
                        auto b = (*_state._bins[_j])[_i+1] - 1;
                        if (b < a)
                        {
                            move = hmove_t::null;
                        }
                        else
                        {
                            std::uniform_int_distribution<int64_t> random_x(a, b);
                            _x = random_x(rng);
                        }
                    }
                    else
                    {
                        std::uniform_real_distribution<double>
                            random_x((*_state._bins[_j])[_i],
                                     (*_state._bins[_j])[_i+1]);
                        _x = random_x(rng);
                        if (_x <= (*_state._bins[_j])[_i] ||
                            _x >= (*_state._bins[_j])[_i+1])
                            move = hmove_t::null;
                    }
                }
                break;
            case 2:
                move = hmove_t::remove;
                break;
            default:
                break;
            }

            return move;
        }

        std::tuple<double, double>
        virtual_move_dS(size_t, hmove_t move)
        {
            double dS = 0;
            double pf = 0;
            double pb = 0;

            switch (move)
            {
            case hmove_t::move:
                dS = _state.virtual_move_edge(_j, _i, _x);

                if (_i == 0)
                {
                    auto w = _state._bounds[_j].first - *_state._bins[_j]->begin();
                    auto nw = _state._bounds[_j].first - _x;
                    if (_state._discrete[_j])
                    {
                        auto delta_f = nw - 1;
                        auto delta_b = w - 1;
                        w++;
                        nw++;
                        double p = 1./(2 * w);
                        pf = log1p(-p) * delta_f + log(p);
                        p = 1./(2 * nw);
                        pb = log1p(-p) * delta_b + log(p);
                    }
                    else
                    {
                        w = std::max(double(w), 1e-8);
                        nw = std::max(double(nw), 1e-8);
                        double lf = 1./(2 * w);
                        double lb = 1./(2 * nw);
                        double delta_f = nw;
                        double delta_b = w;
                        pf = -lf * delta_f - log(lf);
                        pb = -lb * delta_b - log(lb);
                    }
                }
                else if (_i == _state._bins[_j]->size()-1)
                {
                    auto w = *_state._bins[_j]->rbegin() - _state._bounds[_j].second;
                    auto nw = _x - _state._bounds[_j].second;
                    if (_state._discrete[_j])
                    {
                        auto delta_f = nw - 1;
                        auto delta_b = w - 1;
                        w++;
                        nw++;
                        double p = 1./(2 * w);
                        pf = log1p(-p) * delta_f + log(p);
                        p = 1./(2 * nw);
                        pb = log1p(-p) * delta_b + log(p);
                    }
                    else
                    {
                        w = std::max(double(w), 1e-8);
                        nw = std::max(double(nw), 1e-8);
                        double lf = 1./(2 * w);
                        double lb = 1./(2 * nw);
                        double delta_f = nw;
                        double delta_b = w;
                        pf = -lf * delta_f - log(lf);
                        pb = -lb * delta_b - log(lb);
                    }
                }

                break;
            case hmove_t::add:
                dS = _state.template virtual_change_edge<true>(_j, _i, _x);
                pf = -safelog_fast(_state._bins[_j]->size() - 2);
                pb = -safelog_fast(_state._bins[_j]->size() - 1);
                break;
            case hmove_t::remove:
                dS = _state.template virtual_change_edge<false>(_j, _i, 0.);
                pf = -safelog_fast(_state._bins[_j]->size() - 2);
                pb = -safelog_fast(_state._bins[_j]->size() - 3);
                break;
            default:
                break;
            }

            return std::make_tuple(dS, pb - pf);
        }

        void perform_move(size_t, hmove_t move)
        {
            switch (move)
            {
            case hmove_t::move:
                _state.move_edge(_j, _i, _x);
                break;
            case hmove_t::add:
                _state.add_edge(_j, _i, _x);
                break;
            case hmove_t::remove:
                _state.remove_edge(_j, _i);
                break;
            default:
                break;
            }
        }

        bool is_deterministic()
        {
            return false;
        }

        bool is_sequential()
        {
            return false;
        }

        std::array<size_t,1> _vlist = {0};
        auto& get_vlist()
        {
            return _vlist;
        }

        double get_beta()
        {
            return _beta;
        }

        size_t get_niter()
        {
            return _niter;
        }

        void step(size_t, hmove_t)
        {
        }
    };
};


} // graph_tool namespace

#endif //GRAPH_HISTOGRAM_MCMC_HH
