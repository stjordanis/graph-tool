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

#ifndef GRAPH_BLOCKMODEL_DYNAMICS_EPIDEMICS_MCMC_H_HH
#define GRAPH_BLOCKMODEL_DYNAMICS_EPIDEMICS_MCMC_H_HH

#include "config.h"

#include <vector>

#include "graph_tool.hh"
#include "../support/graph_state.hh"
#include "graph_blockmodel_dynamics.hh"
#include "graph_blockmodel_dynamics_discrete.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef std::vector<size_t> vlist_t;

#define MCMC_EPIDEMICS_STATE_params(State)                                     \
    ((__class__,&, mpl::vector<python::object>, 1))                            \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((hstep,, double, 0))                                                      \
    ((verbose,, int, 0) )                                                      \
    ((niter,, size_t, 0))


template <class State, class GetH>
struct MCMC
{
    GEN_STATE_BASE(MCMCEpidemicsStateBase, MCMC_EPIDEMICS_STATE_params(State))

    template <class... Ts>
    class MCMCEpidemicsState
        : public MCMCEpidemicsStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MCMCEpidemicsStateBase<Ts...>,
                         MCMC_EPIDEMICS_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_EPIDEMICS_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCEpidemicsState(ATs&&... as)
            : MCMCEpidemicsStateBase<Ts...>(as...)
        {
            for (auto v : vertices_range(_state._u))
                _vlist.push_back(v);
        }

        std::vector<size_t> _vlist;
        double _null_move = std::numeric_limits<double>::quiet_NaN();

        double node_state(size_t v)
        {
            return GetH()(v, _state._dstate);
        }

        template <class T>
        bool skip_node(T&)
        {
            return false;
        }

        template <class RNG>
        double move_proposal(size_t v, RNG& rng)
        {
            double h = GetH()(v, _state._dstate);
            std::uniform_real_distribution<> step(-_hstep, _hstep);
            double x = 2 * atanh(2 * h - 1) + step(rng);
            return (tanh(x/2) + 1)/2;
        }

        std::tuple<double, double>
        virtual_move_dS(size_t v, double h)
        {
            auto& dstate = _state._dstate;

            double old_h = GetH()(v, _state._dstate);

            GetH()(v, _state._dstate) = h;
            double Sa = -dstate.get_node_prob(v);

            GetH()(v, _state._dstate) = old_h;
            double Sb = -dstate.get_node_prob(v);

            double a;
            if (h == 0 || h == 1)
                a = -numeric_limits<double>::infinity();
            else
                a = (log1p(-pow(2 * h - 1, 2)) -
                     log1p(-pow(2 * old_h - 1, 2)));

            return {Sa - Sb, a};
        }

        void perform_move(size_t v, double h)
        {
            GetH()(v, _state._dstate) = h;
        }

        bool is_deterministic()
        {
            return true;
        }

        bool is_sequential()
        {
            return true;
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
        void step(T&, int)
        {
        }
    };
};


} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_DYNAMICS_EPIDEMICS_MCMC_H_HH
