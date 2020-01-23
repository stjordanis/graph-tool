// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2019 Tiago de Paula Peixoto <tiago@skewed.de>
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
#include "numpy_bind.hh"

#include <boost/python.hpp>

#include "graph_partition_mode.hh"

using namespace boost;
using namespace graph_tool;

void export_partition_mode()
{
    using namespace boost::python;

    class_<PartitionModeState>
        ("PartitionModeState", init<size_t>())
        .def("add_partition",
             +[](PartitionModeState& state, object ob, bool relabel)
              {
                  auto b = get_array<int32_t, 1>(ob);
                  state.add_partition(b, relabel);
              })
        .def("remove_partition",
             +[](PartitionModeState& state, size_t i)
              {
                  state.remove_partition(i);
              })
        .def("replace_partitions", &PartitionModeState::replace_partitions)
        .def("get_marginal",
             +[](PartitionModeState& state,
                 GraphInterface& gi, boost::any obm)
              {
                  run_action<>()
                      (gi, [&](auto& g, auto bm)
                      {
                          state.get_marginal(g, bm);
                      }, vertex_scalar_vector_properties())(obm);
              })
        .def("get_partition",
             +[](PartitionModeState& state, size_t i)
              {
                  auto b = state.get_partition(i);
                  return wrap_multi_array_not_owned(b);
              })
        .def("get_partitions",
             +[](PartitionModeState& state)
              {
                  python::dict obs;
                  auto& bs = state.get_partitions();
                  for (auto& kb : bs)
                  {
                      auto b = state.get_partition(kb.first);
                      obs[kb.first] = wrap_multi_array_not_owned(b);
                  }
                  return obs;
              })
        .def("relabel", &PartitionModeState::relabel)
        .def("entropy", &PartitionModeState::entropy)
        .def("posterior_entropy", &PartitionModeState::posterior_entropy);
}
