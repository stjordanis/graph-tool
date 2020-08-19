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

#include <boost/python.hpp>

#include "graph_tool.hh"
#include "random.hh"

#include "../blockmodel/graph_blockmodel.hh"
#define BASE_STATE_params BLOCK_STATE_params
#include "graph_blockmodel_dynamics.hh"
#include "graph_blockmodel_dynamics_discrete.hh"
#include "../support/graph_state.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, BlockState, BLOCK_STATE_params)

template <class BaseState>
struct Dyn : Dynamics<BaseState, PseudoIsingState> {};

template <class BaseState>
GEN_DISPATCH(dynamics_state, Dyn<BaseState>::template DynamicsState,
             DYNAMICS_STATE_params)

python::object make_pseudo_ising_state(boost::python::object oblock_state,
                                       boost::python::object opseudo_ising_state)
{
    python::object state;
    auto dispatch = [&](auto& block_state)
        {
            typedef typename std::remove_reference<decltype(block_state)>::type
            state_t;

            dynamics_state<state_t>::make_dispatch
                (opseudo_ising_state,
                 [&](auto& s)
                 {
                     state = python::object(s);
                 },
                 block_state);
        };
    block_state::dispatch(oblock_state, dispatch);
    return state;
}

void export_pseudo_ising_state()
{
    using namespace boost::python;

    def("make_pseudo_ising_state", &make_pseudo_ising_state);

    block_state::dispatch
        ([&](auto* bs)
         {
             typedef typename std::remove_reference<decltype(*bs)>::type block_state_t;

             dynamics_state<block_state_t>::dispatch
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
                          .def("get_node_prob", &state_t::get_node_prob)
                          .def("get_edge_prob",
                               +[](state_t& state, size_t u, size_t v, double x,
                                   const uentropy_args_t& ea, double epsilon)
                                {
                                    return get_edge_prob(state, u, v, ea,
                                                         epsilon, x);
                                })
                          .def("get_edges_prob",
                               +[](state_t& state, python::object edges,
                                   python::object probs, const uentropy_args_t& ea,
                                   double epsilon)
                                {
                                    get_xedges_prob(state, edges, probs, ea,
                                                    epsilon);
                                })
                          .def("set_params", &state_t::set_params);
                  });
         });

}
