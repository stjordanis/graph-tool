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

#include "graph_planted_partition.hh"
#include "../support/graph_state.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, PPState, BLOCK_STATE_params)

python::object make_pp_state(boost::python::object ostate)
{
    python::object state;
    block_state::make_dispatch(ostate,
                               [&](auto& s){state = python::object(s);});
    return state;
}

void export_pp_state()
{
    using namespace boost::python;
    def("make_pp_state", &make_pp_state);

    block_state::dispatch
        ([&](auto* s)
         {
             typedef typename std::remove_reference<decltype(*s)>::type state_t;
             void (state_t::*move_vertex)(size_t, size_t) =
                 &state_t::move_vertex;
             double (state_t::*virtual_move)(size_t, size_t, size_t,
                                             const pp_entropy_args_t& ea) =
                 &state_t::virtual_move;

             class_<state_t>
                 c(name_demangle(typeid(state_t).name()).c_str(),
                   no_init);
             c.def("move_vertex", move_vertex)
                 .def("virtual_move", virtual_move)
                 .def("entropy", &state_t::entropy);
         });

    class_<pp_entropy_args_t>("pp_entropy_args")
        .def_readwrite("uniform", &pp_entropy_args_t::uniform)
        .def_readwrite("degree_dl_kind", &pp_entropy_args_t::degree_dl_kind);
}
