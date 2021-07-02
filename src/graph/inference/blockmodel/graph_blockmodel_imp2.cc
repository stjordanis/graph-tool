// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2021 Tiago de Paula Peixoto <tiago@skewed.de>
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

#include "graph_blockmodel_util.hh"
#include "graph_blockmodel.hh"
#include "../uncertain/graph_blockmodel_sample_edge.hh"
#include "../support/graph_state.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, BlockState, BLOCK_STATE_params)

using namespace boost::python;

void export_sbm_state()
{
    class_<BlockStateVirtualBase, boost::noncopyable>
        ("BlockStateVirtualBase", no_init);

    block_state::dispatch
        ([&](auto* s)
         {
             typedef typename std::remove_reference<decltype(*s)>::type state_t;
             void (state_t::*move_vertex)(size_t, size_t) =
                 &state_t::move_vertex;
             double (state_t::*virtual_move)(size_t, size_t, size_t,
                                             const entropy_args_t&) =
                 &state_t::virtual_move;
             size_t (state_t::*sample_block)(size_t, double, double, rng_t&) =
                 &state_t::sample_block;
             size_t (state_t::*random_neighbor)(size_t, rng_t&) =
                 &state_t::random_neighbor;
             double (state_t::*get_move_prob)(size_t, size_t, size_t, double,
                                              double, bool) =
                 &state_t::get_move_prob;
             void (state_t::*set_partition)(boost::any&) =
                 &state_t::set_partition;

             class_<state_t, bases<BlockStateVirtualBase>>
                 c(name_demangle(typeid(state_t).name()).c_str(),
                   no_init);
             c.def("move_vertex", move_vertex)
                 .def("set_partition", set_partition)
                 .def("virtual_move", virtual_move)
                 .def("sample_block", sample_block)
                 .def("sample_neighbor", random_neighbor)
                 .def("entropy", &state_t::entropy)
                 .def("get_partition_dl", &state_t::get_partition_dl)
                 .def("get_deg_dl", &state_t::get_deg_dl)
                 .def("get_move_prob", get_move_prob)
                 .def("reset_partition_stats",
                      &state_t::reset_partition_stats)
                 .def("init_partition_stats",
                      &state_t::init_partition_stats)
                 .def("couple_state",
                      &state_t::couple_state)
                 .def("decouple_state",
                      &state_t::decouple_state)
                 .def("get_B_E",
                      &state_t::get_B_E)
                 .def("get_B_E_D",
                      &state_t::get_B_E_D)
                 .def("clear_egroups",
                      &state_t::clear_egroups)
                 .def("sync_emat",
                      &state_t::sync_emat)
                 .def("get_edge_sampler",
                      +[](state_t& state, bool edges_only)
                      {
                          return SBMEdgeSampler(state, edges_only);
                      });

             typedef SBMEdgeSampler<state_t> edge_sampler_t;
             class_<edge_sampler_t>
                 es(name_demangle(typeid(edge_sampler_t).name()).c_str(),
                    no_init);
             es.def("sample",
                    +[](edge_sampler_t& es, rng_t& rng)
                    {
                        auto e = es.sample(rng);
                        return python::make_tuple(get<0>(e), get<1>(e));
                    })
                 .def("log_prob",
                      +[](edge_sampler_t& es, size_t u, size_t v, size_t m)
                      {
                          return es.log_prob(u, v, m, 0);
                      });
         });
}
