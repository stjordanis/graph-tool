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

#ifndef GRAPH_BLOCKMODEL_DYNAMICS_MCMC_HH
#define GRAPH_BLOCKMODEL_DYNAMICS_MCMC_HH

#include "config.h"

#include <vector>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include "graph_blockmodel_dynamics.hh"
#include "graph_blockmodel_sample_edge.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

template <size_t n, typename... T>
typename std::enable_if<(n >= sizeof...(T))>::type
    print_tuple(std::ostream&, const std::tuple<T...>&)
{}

template <size_t n, typename... T>
typename std::enable_if<(n < sizeof...(T))>::type
    print_tuple(std::ostream& os, const std::tuple<T...>& tup)
{
    if (n != 0)
        os << ", ";
    os << std::get<n>(tup);
    print_tuple<n+1>(os, tup);
}

template <typename... T>
std::ostream& operator<<(std::ostream& os, const std::tuple<T...>& tup)
{
    os << "[";
    print_tuple<0>(os, tup);
    return os << "]";
}


typedef std::vector<size_t> vlist_t;

#define MCMC_DYNAMICS_STATE_params(State)                                      \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((xstep,, double, 0))                                                      \
    ((xlog,, bool, 0))                                                         \
    ((xdefault,, double, 0))                                                   \
    ((entropy_args,, uentropy_args_t, 0))                                      \
    ((verbose,, int, 0))                                                       \
    ((niter,, size_t, 0))


template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCDynamicsStateBase, MCMC_DYNAMICS_STATE_params(State))

    template <class... Ts>
    class MCMCDynamicsState
        : public MCMCDynamicsStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MCMCDynamicsStateBase<Ts...>,
                         MCMC_DYNAMICS_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_DYNAMICS_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCDynamicsState(ATs&&... as)
            : MCMCDynamicsStateBase<Ts...>(as...),
              _edge_sampler(_state._block_state),
              _vlist(num_vertices(_state._u))
        {
        }

        SBMEdgeSampler<typename State::block_state_t> _edge_sampler;

        std::tuple<size_t, size_t> _e;
        std::vector<size_t> _vlist;
        constexpr static std::tuple<int, double> _null_move = {0, .0};

        std::tuple<size_t, size_t> get_edge()
        {
            return _e;
        }

        std::tuple<size_t, double> node_state(size_t u, size_t v)
        {
            auto&& e = _state.get_u_edge(u, v);
            if (e == _state._null_edge)
                return {0, .0};
            return {_state._eweight[e], _state._xc[e]};
        }

        std::tuple<size_t, double> node_state(size_t)
        {
            size_t u, v;
            std::tie(u, v) = get_edge();
            return node_state(u, v);
        }

        template <class T>
        bool skip_node(T&)
        {
            return false;
        }

        template <class RNG>
        std::tuple<int, double> move_proposal(size_t, RNG& rng)
        {
            _e = _edge_sampler.sample(rng);

            std::bernoulli_distribution coin(.5);
            size_t m;
            double x;
            std::tie(m, x) = node_state(get<0>(_e), get<1>(_e));
            if (m > 0)
            {
                if (_xstep == 0 || coin(rng))
                {
                    if (coin(rng))
                        return {-1, 0};
                    return {1, 0};
                }
                if (_xlog)
                {
                    double p = 1 - exp(x);
                    std::uniform_real_distribution<> u(std::max(0., p - _xstep),
                                                       std::min(1., p + _xstep));
                    p = u(rng);
                    double nx = log1p(-p);
                    return {0, nx - x};
                }
                else
                {
                    std::uniform_real_distribution<> u(std::max(-.5, x - _xstep),
                                                       std::min(.5, x + _xstep));
                    return {0, u(rng) - x};
                }
            }
            else
            {
                if (_xstep == 0)
                    return {1, _xdefault};
                if (_xlog)
                {
                    std::uniform_real_distribution<> u(0, 1);
                    return {1, log1p(-u(rng))};
                }
                else
                {
                    std::uniform_real_distribution<> u(-.5, .5);
                    return {1, u(rng)};
                }
            }
        }

        double get_move_prob(int dm, double, int m, double x)
        {
            if (dm == -1)
            {
                if (_xstep == 0)
                    return (m > 0) ? 1/2. : 0;
                return (m > 0) ? 1/4. : 0;
            }
            if (dm == 0)
            {
                if (_xstep == 0)
                    return 0.;
                if (m > 0)
                {
                    if (_xlog)
                    {
                        double p = 1 - exp(x);
                        return (1/2.) / (std::min(1., p + _xstep) -
                                         std::max(0., p - _xstep));
                    }
                    else
                    {
                        return (1/2.) / (std::min(.5, x + _xstep) -
                                         std::max(-.5, x - _xstep));
                    }
                }
                return 0;
            }
            if (dm == 1)
            {
                if (m > 0)
                    return 1/4.;
                return 1.;
            }
            return 0.;
        }

        std::tuple<double, double>
        virtual_move_dS(size_t, std::tuple<int, double>& delta)
        {
            int dm = get<0>(delta);
            double dx = get<1>(delta);

            size_t u, v;
            std::tie(u, v) = get_edge();

            double dS = 0;
            if (dm == 0)
                dS = _state.update_edge_dS(u, v, dx, _entropy_args);
            else if (dm < 0)
                dS = _state.remove_edge_dS(u, v, _entropy_args);
            else
                dS = _state.add_edge_dS(u, v, dx, _entropy_args);

            size_t m;
            double x;
            std::tie(m, x) = node_state(u, v);

            double a = 0;
            if (dm != 0)
                a += (_edge_sampler.log_prob(u, v, m + dm, dm) -
                      _edge_sampler.log_prob(u, v, m, 0));

            a -= log(get_move_prob(dm, dx, m, x));
            a += log(get_move_prob(-dm, -dx, m + dm, x + dx));

            return std::make_tuple(dS, a);
        }

        void perform_move(size_t,  std::tuple<int, double>& delta)
        {
            int dm = get<0>(delta);
            double dx = get<1>(delta);

            size_t u, v;
            std::tie(u, v) = get_edge();
            if (dm == 0)
            {
                _state.update_edge(u, v, dx);
            }
            else if (dm < 0)
            {
                size_t m = get<0>(node_state(u, v));
                _edge_sampler.template update_edge<false>(u, v, m);
                _state.remove_edge(u, v);
            }
            else
            {
                size_t m = get<0>(node_state(u, v));
                _state.add_edge(u, v, dx);
                _edge_sampler.template update_edge<true>(u, v, m);
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

        template <class T>
        void step(T&, std::tuple<int, double>&)
        {
        }
    };
};


} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_DYNAMICS_MCMC_HH
