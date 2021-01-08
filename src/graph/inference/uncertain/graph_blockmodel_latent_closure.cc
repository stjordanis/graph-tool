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

#define BOOST_PYTHON_MAX_ARITY 40
#include <boost/python.hpp>

#include "graph_tool.hh"
#include "random.hh"

#include "../blockmodel/graph_blockmodel.hh"
#define BASE_STATE_params BLOCK_STATE_params
#include "graph_blockmodel_latent_closure.hh"
#include "graph_blockmodel_latent_layers.hh"
#include "../support/graph_state.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, BlockState, BLOCK_STATE_params)

template <class BaseState>
GEN_DISPATCH(latent_layers_state, LatentLayers<BaseState>::template LatentLayersState,
             LATENT_LAYERS_STATE_params)

template <class BaseState>
GEN_DISPATCH(latent_closure_state, LatentClosure<BaseState>::template LatentClosureState,
             LATENT_CLOSURE_STATE_params)

python::object make_latent_closure_state(boost::python::object oeblock_state,
                                         boost::python::object opblock_state,
                                         boost::python::object olatent_layers_state,
                                         size_t L)
{
    python::list state;
    auto dispatch = [&](auto& eblock_state)
        {
            typedef typename std::remove_reference<decltype(eblock_state)>::type
                state_t;

            state_t& pblock_state = python::extract<state_t&>(opblock_state);

            for (size_t l = 0; l < L; ++l)
            {
                latent_closure_state<state_t>::make_dispatch
                    (olatent_layers_state,
                     [&](auto& s)
                     {
                         state.append(python::object(s));
                     },
                     eblock_state, pblock_state, l);
            }

            latent_closure_state<state_t>::dispatch
                ([&](auto* cs)
                 {
                     typedef typename std::remove_reference<decltype(*cs)>::type c_state_t;

                     std::vector<c_state_t*> cstates;
                     for (size_t l = 0; l < L; ++l)
                         cstates.push_back(&python::extract<c_state_t&>(state[l])());
                     for (size_t l = 0; l < L; ++l)
                         cstates[l]->set_cstates(cstates);

                     std::vector<std::reference_wrapper<c_state_t>> layers;
                     for (size_t l = 0; l < L; ++l)
                         layers.push_back(*cstates[l]);

                     latent_layers_state<c_state_t>::make_dispatch
                         (olatent_layers_state,
                          [&](auto& s)
                          {
                              state.append(python::object(s));
                          },
                          layers);
                 });
        };
    block_state::dispatch(oeblock_state, dispatch);
    return state;
}


void export_latent_closure_state()
{
    using namespace boost::python;

    def("make_latent_closure_state", &make_latent_closure_state);

    block_state::dispatch
        ([&](auto* bs)
         {
             typedef typename std::remove_reference<decltype(*bs)>::type block_state_t;

             latent_closure_state<block_state_t>::dispatch
                 ([&](auto* cs)
                  {
                      typedef typename std::remove_reference<decltype(*cs)>::type c_state_t;

                      class_<c_state_t>
                          c(name_demangle(typeid(c_state_t).name()).c_str(),
                            no_init);
                      c.def("entropy", &c_state_t::entropy);

                      latent_layers_state<c_state_t>::dispatch
                          ([&](auto* s)
                           {
                               typedef typename std::remove_reference<decltype(*s)>::type state_t;
                               class_<state_t>
                                   c(name_demangle(typeid(state_t).name()).c_str(),
                                     no_init);
                               c.def("remove_edge", &state_t::remove_edge)
                                   .def("add_edge", &state_t::add_edge)
                                   .def("remove_edge_dS", &state_t::remove_edge_dS)
                                   .def("add_edge_dS", &state_t::add_edge_dS)
                                   .def("entropy", &state_t::entropy)
                                   .def("set_hparams", &state_t::set_hparams)
                                   .def("get_N", &state_t::get_N)
                                   .def("get_X", &state_t::get_X)
                                   .def("get_T", &state_t::get_T)
                                   .def("get_M", &state_t::get_M);
                           });
                  });
         });
}
