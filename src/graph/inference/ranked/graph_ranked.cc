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

#include "../support/graph_state.hh"

#define GRAPH_VIEWS never_filtered_never_reversed
#include "../blockmodel/graph_blockmodel.hh"

#include "graph_ranked.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, BlockState, BLOCK_STATE_params)

template <class BState>
GEN_DISPATCH(ranked_state, OState<BState>::template RankedState, RANKED_STATE_params)

python::object make_ranked_state(boost::python::object oustate,
                                 boost::python::object ostate)
{

    python::object state;
    auto dispatch_d = [&](auto& ustate)
    {
        typedef typename std::remove_reference<decltype(ustate)>::type
            ustate_t;

        ranked_state<ustate_t>
                ::make_dispatch(ostate,
                                [&](auto& s){state = python::object(s);},
                                ustate);
    };
    block_state::dispatch(oustate, dispatch_d);
    return state;
}

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;
    def("make_ranked_state", &make_ranked_state);

    block_state::dispatch
        ([&](auto* ds)
         {
             typedef typename std::remove_reference<decltype(*ds)>::type dstate_t;
             ranked_state<dstate_t>::dispatch
                 ([&](auto* s)
                 {
                     typedef typename std::remove_reference<decltype(*s)>::type state_t;
                     void (state_t::*move_vertex)(size_t, size_t) =
                         &state_t::move_vertex;
                     double (state_t::*virtual_move)(size_t, size_t, size_t,
                                                     entropy_args_t& ea) =
                         &state_t::virtual_move;

                     class_<state_t, bases<>, std::shared_ptr<state_t>>
                         c(name_demangle(typeid(state_t).name()).c_str(),
                           no_init);
                     c.def("move_vertex", move_vertex)
                         .def("virtual_move", virtual_move)
                         .def("entropy", &state_t::entropy)
                         .def("couple_state", &state_t::couple_state)
                         .def("decouple_state", &state_t::decouple_state)
                         .def("get_Es",
                              +[](state_t& state)
                              {
                                  return python::make_tuple(state._E[0],
                                                            state._E[1],
                                                            state._E[2]);
                              });
                 });
         });
});
