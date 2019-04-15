// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2018 Tiago de Paula Peixoto <tiago@skewed.de>
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

#include "graph_tool.hh"
#include "random.hh"

#include <boost/python.hpp>

#include "../blockmodel/graph_blockmodel.hh"
#define BASE_STATE_params BLOCK_STATE_params
#include "graph_blockmodel_dynamics.hh"
#include "graph_blockmodel_dynamics_discrete.hh"
#include "graph_blockmodel_dynamics_epidemics_mcmc_h.hh"
#include "../support/graph_state.hh"
#include "../loops/mcmc_loop.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, BlockState, BLOCK_STATE_params)

template <class BaseState>
struct Dyn : Dynamics<BaseState, SIState> {};

template <class BaseState>
GEN_DISPATCH(dynamics_state, Dyn<BaseState>::template DynamicsState,
             DYNAMICS_STATE_params)

struct GetR
{
    template <class State>
    auto& operator()(size_t v, State& state)
    {
        return state._r_v[v];
    }
};

template <class State>
struct BMCMC: public MCMC<State, GetR> {};

template <class State>
GEN_DISPATCH(mcmc_epidemics_state, BMCMC<State>::template MCMCEpidemicsState,
             MCMC_EPIDEMICS_STATE_params(State))

python::object mcmc_epidemics_sweep_r(python::object omcmc_state,
                                      python::object odynamics_state,
                                      rng_t& rng)
{
    python::object ret;
    auto dispatch = [&](auto* block_state)
    {
        typedef typename std::remove_pointer<decltype(block_state)>::type
            state_t;

        dynamics_state<state_t>::dispatch
            (odynamics_state,
             [&](auto& ls)
             {
                 typedef typename std::remove_reference<decltype(ls)>::type
                     dynamics_state_t;

                 mcmc_epidemics_state<dynamics_state_t>::make_dispatch
                     (omcmc_state,
                      [&](auto& s)
                      {
                          auto ret_ = mcmc_sweep(s, rng);
                          ret = tuple_apply([&](auto&... args){ return python::make_tuple(args...); }, ret_);
                      });
             },
             false);
    };
    block_state::dispatch(dispatch);
    return ret;
}

void export_epidemics_mcmc_r()
{
    using namespace boost::python;
    def("mcmc_epidemics_sweep_r", &mcmc_epidemics_sweep_r);
}
