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

enum class move_t { move = 0, add, remove, null };

ostream& operator<<(ostream& s, move_t v);

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
        }

        constexpr static move_t _null_move = move_t::null;

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

        template <class RNG>
        move_t move_proposal(size_t, RNG& rng)
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

            move_t move = move_t::null;
            switch (m)
            {
            case 0:
                move = move_t::move;
                {
                    if (_i == 0)
                    {
                        if (_state._bounded[_j].first)
                        {
                            move = move_t::null;
                        }
                        else
                        {
                            double w = _state._bounds[_j].first - *_state._bins[_j]->begin();
                            if (_state._discrete[_j])
                            {
                                std::geometric_distribution<int> d(1./(2 * w + 1));
                                _x = _state._bounds[_j].first - d(rng) - 1;
                            }
                            else
                            {
                                w = std::max(w, 1e-8);
                                std::exponential_distribution<double> d(1./(2 * w));
                                _x = _state._bounds[_j].first - d(rng);
                                if (_x > _state._bounds[_j].first ||
                                    _x >= *(_state._bins[_j]->begin() + 1))
                                    move = move_t::null;
                            }

                            assert(_x <= _state._bounds[_j].first);
                        }
                    }
                    else if (_i == _state._bins[_j]->size()-1)
                    {
                        if (_state._bounded[_j].second)
                        {
                            move = move_t::null;
                        }
                        else
                        {
                            double w = *_state._bins[_j]->rbegin() - _state._bounds[_j].second;
                            if (_state._discrete[_j])
                            {
                                std::geometric_distribution<int> d(1./(2 * w + 1));
                                _x = _state._bounds[_j].second + d(rng) + 1;
                            }
                            else
                            {
                                w = std::max(w, 1e-8);
                                std::exponential_distribution<double> d(1./(2 * w));
                                _x = _state._bounds[_j].second + d(rng);
                                if (_x == _state._bounds[_j].second)
                                    move = move_t::null;
                                if (_x <= _state._bounds[_j].second ||
                                    _x <= *(_state._bins[_j]->end() - 2))
                                    move = move_t::null;
                            }
                            assert(_x > _state._bounds[_j].second);
                        }
                    }
                    else
                    {
                        if (_state._discrete[_j])
                        {
                            std::uniform_int_distribution<int>
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
                                move = move_t::null;
                        }
                    }
                }
                break;
            case 1:
                move = move_t::add;
                {
                    if (_state._discrete[_j])
                    {
                        int a = (*_state._bins[_j])[_i] + 1;
                        int b = (*_state._bins[_j])[_i+1] - 1;
                        if (b < a)
                        {
                            move = move_t::null;
                        }
                        else
                        {
                            std::uniform_int_distribution<int> random_x(a, b);
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
                            move = move_t::null;
                    }
                }
                break;
            case 2:
                move = move_t::remove;
                break;
            default:
                break;
            }

            return move;
        }

        std::tuple<double, double>
        virtual_move_dS(size_t, move_t move)
        {
            double dS = 0;
            double pf = 0;
            double pb = 0;

            switch (move)
            {
            case move_t::move:
                dS = _state.virtual_move_edge(_j, _i, _x);

                if (_i == 0)
                {
                    double w = _state._bounds[_j].first - *_state._bins[_j]->begin();
                    double nw = _state._bounds[_j].first - _x;
                    if (_state._discrete[_j])
                    {
                        w -= 1;
                        nw -= 1;
                        double p = 1./(2 * w + 1);
                        pf = log1p(-p) * nw + log(p);
                        std::swap(w, nw);
                        p = 1./(2 * w + 1);
                        pb = log1p(-p) * nw + log(p);
                    }
                    else
                    {
                        pf = -nw/(2 * std::max(w, 1e-8)) - log(2 * std::max(w, 1e-8));
                        std::swap(w, nw);
                        pb = -nw/(2 * std::max(w, 1e-8)) - log(2 * std::max(w, 1e-8));
                    }
                }
                else if (_i == _state._bins[_j]->size()-1)
                {
                    double w = *_state._bins[_j]->rbegin() - _state._bounds[_j].second;
                    double nw = _x - _state._bounds[_j].second;
                    if (_state._discrete[_j])
                    {
                        w -= 1;
                        nw -= 1;
                        double p = 1./(2 * w + 1);
                        pf = log1p(-p) * nw + log(p);
                        std::swap(w, nw);
                        p = 1./(2 * w + 1);
                        pb = log1p(-p) * nw + log(p);
                    }
                    else
                    {
                        pf = -nw/(2 * std::max(w, 1e-8)) - log(2 * std::max(w, 1e-8));
                        std::swap(w, nw);
                        pb = -nw/(2 * std::max(w, 1e-8)) - log(2 * std::max(w, 1e-8));
                    }
                }

                break;
            case move_t::add:
                dS = _state.virtual_add_edge(_j, _i, _x);
                pf = -safelog_fast(_state._bins[_j]->size() - 2);
                pb = -safelog_fast(_state._bins[_j]->size() - 1);
                break;
            case move_t::remove:
                dS = _state.virtual_remove_edge(_j, _i);
                pf = -safelog_fast(_state._bins[_j]->size() - 2);
                pb = -safelog_fast(_state._bins[_j]->size() - 3);
                break;
            default:
                break;
            }

            return std::make_tuple(dS, pb - pf);
        }

        void perform_move(size_t, move_t move)
        {
            switch (move)
            {
            case move_t::move:
                _state.move_edge(_j, _i, _x);
                break;
            case move_t::add:
                _state.add_edge(_j, _i, _x);
                break;
            case move_t::remove:
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

        void step(size_t, move_t)
        {
        }
    };
};


} // graph_tool namespace

#endif //GRAPH_HISTOGRAM_MCMC_HH
