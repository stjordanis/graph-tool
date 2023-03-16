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

#include "graph_tool.hh"
#include "random.hh"

#include <boost/python.hpp>

#include "../blockmodel/graph_blockmodel.hh"
#define BASE_STATE_params BLOCK_STATE_params
#include "graph_blockmodel_uncertain.hh"
#include "../support/graph_state.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, BlockState, BLOCK_STATE_params)

template <class BaseState>
GEN_DISPATCH(uncertain_state, Uncertain<BaseState>::template UncertainState,
             UNCERTAIN_STATE_params)

python::object make_uncertain_state(boost::python::object oblock_state,
                                    boost::python::object ouncertain_state)
{
    python::object state;
    auto dispatch = [&](auto& block_state)
        {
            typedef typename std::remove_reference<decltype(block_state)>::type
            state_t;

            uncertain_state<state_t>::make_dispatch
                (ouncertain_state,
                 [&](auto& s)
                 {
                     state = python::object(s);
                 },
                 block_state);
        };
    block_state::dispatch(oblock_state, dispatch);
    return state;
}


#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;

    class_<uentropy_args_t, bases<entropy_args_t>>("uentropy_args",
                                                   init<entropy_args_t>())
        .def_readwrite("latent_edges", &uentropy_args_t::latent_edges)
        .def_readwrite("density", &uentropy_args_t::density);

    def("make_uncertain_state", &make_uncertain_state);

    block_state::dispatch
        ([&](auto* bs)
         {
             typedef typename std::remove_reference<decltype(*bs)>::type block_state_t;

             uncertain_state<block_state_t>::dispatch
                 ([&](auto* s)
                  {
                      typedef typename std::remove_reference<decltype(*s)>::type state_t;

                      class_<state_t, bases<>, std::shared_ptr<state_t>>
                          c(name_demangle(typeid(state_t).name()).c_str(),
                            no_init);
                      c.def("remove_edge", &state_t::remove_edge)
                          .def("add_edge", &state_t::add_edge)
                          .def("set_state",
                               +[](state_t& state, GraphInterface& gi,
                                   boost::any aw)
                                {
                                    typedef eprop_map_t<int32_t>::type emap_t;
                                    auto w = any_cast<emap_t>(aw).get_unchecked();
                                    gt_dispatch<>()
                                        ([&](auto& g)
                                         { set_state(state, g, w); },
                                         all_graph_views())
                                        (gi.get_graph_view());
                                })
                          .def("remove_edge_dS", &state_t::remove_edge_dS)
                          .def("add_edge_dS", &state_t::add_edge_dS)
                          .def("entropy", &state_t::entropy)
                          .def("set_q_default", &state_t::set_q_default)
                          .def("set_S_const", &state_t::set_S_const)
                          .def("get_edge_prob",
                               +[](state_t& state, size_t u, size_t v,
                                   const uentropy_args_t& ea, double epsilon)
                                {
                                    return get_edge_prob(state, u, v, ea,
                                                         epsilon);
                                })
                          .def("get_edges_prob",
                               +[](state_t& state, python::object edges,
                                   python::object probs, const uentropy_args_t& ea,
                                   double epsilon)
                                {
                                    get_edges_prob(state, edges, probs, ea,
                                                   epsilon);
                                });

                  });
         });

});
