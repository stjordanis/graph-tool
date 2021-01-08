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

#include "graph_tool.hh"
#include "random.hh"

#include <boost/python.hpp>

#include "../blockmodel/graph_blockmodel.hh"
#define BASE_STATE_params BLOCK_STATE_params
#include "graph_blockmodel_latent_layers.hh"
#include "graph_blockmodel_latent_layers_mcmc.hh"
#include "graph_blockmodel_latent_closure.hh"
#include "../support/graph_state.hh"
#include "../loops/mcmc_loop.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, BlockState, BLOCK_STATE_params)

template <class BaseState>
GEN_DISPATCH(latent_layers_state, LatentLayers<BaseState>::template LatentLayersState,
             LATENT_LAYERS_STATE_params)

template <class BaseState>
GEN_DISPATCH(latent_closure_state, LatentClosure<BaseState>::template LatentClosureState,
             LATENT_CLOSURE_STATE_params)

template <class State>
GEN_DISPATCH(mcmc_latent_layers_state, MCMC<State>::template MCMCLatentLayersState,
             MCMC_LATENT_LAYERS_STATE_params(State))

python::object mcmc_latent_closure_sweep(python::object omcmc_state,
                                         python::object olatent_layers_state,
                                         rng_t& rng)
{
    python::object ret;
    auto dispatch = [&](auto* block_state)
    {
        typedef typename std::remove_pointer<decltype(block_state)>::type
            block_state_t;

        latent_closure_state<block_state_t>::dispatch
            ([&](auto* cs)
             {
                 typedef typename std::remove_reference<decltype(*cs)>::type c_state_t;

                 latent_layers_state<c_state_t>::dispatch
                     (olatent_layers_state,
                      [&](auto& ls)
                      {
                          typedef typename std::remove_reference<decltype(ls)>::type
                              latent_layers_state_t;

                          mcmc_latent_layers_state<latent_layers_state_t>::make_dispatch
                              (omcmc_state,
                               [&](auto& s)
                               {
                                   auto ret_ = mcmc_sweep(s, rng);
                                   ret = tuple_apply([&](auto&... args){ return python::make_tuple(args...); }, ret_);
                               });
                      },
                      false);
             });
    };
    block_state::dispatch(dispatch);
    return ret;
}

void export_latent_closure_mcmc()
{
    using namespace boost::python;
    def("mcmc_latent_closure_sweep", &mcmc_latent_closure_sweep);
}
