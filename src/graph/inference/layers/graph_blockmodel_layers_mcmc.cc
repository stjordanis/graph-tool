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

#define GRAPH_BLOCKMODEL_RMAP_ENABLE
#include "../blockmodel/graph_blockmodel_util.hh"
#include "../blockmodel/graph_blockmodel.hh"
#define BASE_STATE_params BLOCK_STATE_params
#include "graph_blockmodel_layers.hh"
#include "../blockmodel/graph_blockmodel_mcmc.hh"
#include "../loops/mcmc_loop.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, BlockState, BLOCK_STATE_params)

template <class BaseState>
GEN_DISPATCH(layered_block_state, Layers<BaseState>::template LayeredBlockState,
             LAYERED_BLOCK_STATE_params)

template <class State>
GEN_DISPATCH(mcmc_block_state, MCMC<State>::template MCMCBlockState,
             MCMC_BLOCK_STATE_params(State))

python::object mcmc_layered_sweep(python::object omcmc_state,
                                  python::object olayered_state,
                                  rng_t& rng)
{
#ifdef GRAPH_BLOCKMODEL_LAYERS_ENABLE
    python::object ret;
    auto dispatch = [&](auto* block_state)
    {
        typedef typename std::remove_pointer<decltype(block_state)>::type
            state_t;

        layered_block_state<state_t>::dispatch
            (olayered_state,
             [&](auto& ls)
             {
                 typedef typename std::remove_reference<decltype(ls)>::type
                     layered_state_t;

                 mcmc_block_state<layered_state_t>::make_dispatch
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
#endif
}

class MCMC_sweep_base
{
public:
    virtual std::tuple<double, size_t, size_t> run(rng_t&) = 0;
};

template <class State>
class MCMC_sweep : public MCMC_sweep_base
{
public:
    MCMC_sweep(State& s) : _s(s) {}
    virtual ~MCMC_sweep() {}

    virtual std::tuple<double, size_t, size_t> run(rng_t& rng)
    {
        return mcmc_sweep(_s, rng);
    }
private:
    State _s;
};

python::object mcmc_layered_sweep_parallel(python::object omcmc_states,
                                           python::object olayered_states,
                                           rng_t& rng)
{
#ifdef GRAPH_BLOCKMODEL_LAYERS_ENABLE
    std::vector<std::shared_ptr<MCMC_sweep_base>> sweeps;

    size_t N = python::len(omcmc_states);
    for (size_t i = 0; i < N; ++ i)
    {
        auto dispatch = [&](auto* block_state)
            {
                typedef typename std::remove_pointer<decltype(block_state)>::type
                    state_t;

                layered_block_state<state_t>::dispatch
                    (olayered_states[i],
                     [&](auto& ls)
                     {
                         typedef typename std::remove_reference<decltype(ls)>::type
                             layered_state_t;

                         mcmc_block_state<layered_state_t>::make_dispatch
                             (omcmc_states[i],
                              [&](auto& s)
                              {
                                  typedef typename std::remove_reference<decltype(s)>::type
                                      s_t;
                                  sweeps.push_back(std::make_shared<MCMC_sweep<s_t>>(s));
                              });
                     },
                     false);
            };
        block_state::dispatch(dispatch);
    }

    parallel_rng<rng_t>::init(rng);

    std::vector<std::tuple<double, size_t, size_t>> rets(N);

    #pragma omp parallel for schedule(runtime)
    for (size_t i = 0; i < N; ++i)
    {
        auto& rng_ = parallel_rng<rng_t>::get(rng);
        rets[i] = sweeps[i]->run(rng_);
    }

    python::list orets;
    for (auto& ret : rets)
        orets.append(tuple_apply([&](auto&... args){ return python::make_tuple(args...); }, ret));
    return std::move(orets);
#endif
}

void export_layered_blockmodel_mcmc()
{
    using namespace boost::python;
    def("mcmc_layered_sweep", &mcmc_layered_sweep);
    def("mcmc_layered_sweep_parallel", &mcmc_layered_sweep_parallel);
}
