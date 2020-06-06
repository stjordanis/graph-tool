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

#include "graph_partition_mode_clustering.hh"
#include "../support/graph_state.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, ModeClusterState, BLOCK_STATE_params)

python::object make_mode_cluster_state(boost::python::object ostate)
{
    python::object state;
    block_state::make_dispatch(ostate,
                               [&](auto& s){state = python::object(s);});
    return state;
}

PartitionModeState::bv_t get_bv(python::object ob);

void export_mode_cluster_state()
{
    using namespace boost::python;
    def("make_mode_cluster_state", &make_mode_cluster_state);

    block_state::dispatch
        ([&](auto* s)
         {
             typedef typename std::remove_reference<decltype(*s)>::type state_t;
             void (state_t::*move_vertex)(size_t, size_t) =
                 &state_t::move_vertex;
             double (state_t::*virtual_move)(size_t, size_t, size_t) =
                 &state_t::virtual_move;

             class_<state_t>
                 c(name_demangle(typeid(state_t).name()).c_str(),
                   no_init);
             c.def("move_vertex", move_vertex)
                 .def("virtual_move", virtual_move)
                 .def("virtual_add_partition",
                      +[](state_t& state, object ob, size_t r, bool relabel)
                       {
                           auto bv = get_bv(ob);
                           return state.virtual_add_partition(bv, r, relabel);
                       })
                 .def("add_partition",
                      +[](state_t& state, object ob, size_t r, bool relabel)
                       {
                           auto bv = get_bv(ob);
                           state.add_partition(bv, r, relabel);
                       })
                 .def("entropy", &state_t::entropy)
                 .def("posterior_entropy", &state_t::posterior_entropy)
                 .def("posterior_lprob",
                      +[](state_t& state, size_t r, object obv, bool MLE)
                       {
                           PartitionModeState::bv_t bv;
                           for (int i = 0; i < python::len(obv); ++i)
                           {
                               PartitionModeState::b_t& b =
                                   python::extract<PartitionModeState::b_t&>(obv[i]);
                               bv.emplace_back(b);
                           }
                           return state.posterior_lprob(r, bv, MLE);
                       })
                 .def("relabel_modes", &state_t::relabel_modes)
                 .def("replace_partitions",
                      +[](state_t& state, rng_t& rng)
                       {
                           return state.replace_partitions(rng);
                       })
                 .def("get_mode",
                      +[](state_t& state, size_t r) -> PartitionModeState&
                       {
                           return state.get_mode(r);
                       },
                      return_internal_reference<>())
                 .def("sample_partition",
                      +[](state_t& state, bool MLE, rng_t& rng)
                       {
                           auto rb = state.sample_partition(MLE, rng);
                           return python::make_tuple(rb.first,
                                                     wrap_vector_owned(rb.second));
                       })
                 .def("sample_nested_partition",
                      +[](state_t& state, bool MLE, bool fix_empty, rng_t& rng)
                       {
                           python::list obv;
                           auto rbv = state.sample_nested_partition(MLE,
                                                                    fix_empty,
                                                                    rng);
                           for (auto& b : rbv.second)
                               obv.append(wrap_vector_owned(b));
                           return python::make_tuple(rbv.first, obv);
                       });
         });
}
