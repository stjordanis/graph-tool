// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2020 Tiago de Paula Peixoto <tiago@skewed.de>
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

auto get_bv(python::object ob)
{
    PartitionModeState::bv_t bv;
    for (int i = 0; i < python::len(ob); ++i)
    {
        auto& b = python::extract<PartitionModeState::b_t&>(ob[i])();
        bv.emplace_back(b);
    }
    return bv;
}

void export_partition_mode()
{
    using namespace boost::python;

    class_<PartitionModeState>
        ("PartitionModeState")
        .def("add_partition",
             +[](PartitionModeState& state, object ob, bool relabel)
              {
                  auto bv = get_bv(ob);
                  return state.add_partition(bv, relabel);
              })
        .def("remove_partition",
             +[](PartitionModeState& state, size_t i)
              {
                  state.remove_partition(i);
              })
        .def("replace_partitions", &PartitionModeState::replace_partitions)
        .def("get_B", &PartitionModeState::get_B)
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
        .def("get_map",
             +[](PartitionModeState& state,
                 GraphInterface& gi, boost::any ob)
              {
                  run_action<>()
                      (gi, [&](auto& g, auto b)
                      {
                          state.get_map(g, b);
                      }, writable_vertex_scalar_properties())(ob);
              })
        .def("get_map_bs",
             +[](PartitionModeState& state)
              {
                  python::list bs;
                  PartitionModeState* s = &state;
                  while (s != nullptr)
                  {
                      bs.append(wrap_vector_owned(s->get_map_b()));
                      s = s->get_coupled_state().get();
                  }
                  return bs;
              })
        .def("get_partition",
             +[](PartitionModeState& state, size_t i)
              {
                  return state.get_partition(i);
              })
        .def("get_nested_partition",
             +[](PartitionModeState& state, size_t i)
              {
                  python::list obv;
                  auto bv = state.get_nested_partition(i);
                  for (PartitionModeState::b_t& b : bv)
                      obv.append(b);
                  return obv;
              })
        .def("sample_partition",
             +[](PartitionModeState& state, bool MLE, rng_t& rng)
              {
                  return wrap_vector_owned(state.sample_partition(MLE, rng));
              })
        .def("sample_nested_partition",
             +[](PartitionModeState& state, bool MLE, rng_t& rng)
              {
                  python::list obv;
                  auto bv = state.sample_nested_partition(MLE, rng);
                  for (auto& b : bv)
                      obv.append(wrap_vector_owned(b));
                  return obv;
              })
        .def("get_partitions",
             +[](PartitionModeState& state)
              {
                  python::dict obs;
                  auto& bs = state.get_partitions();
                  for (auto& kb : bs)
                  {
                      auto& b = state.get_partition(kb.first);
                      obs[kb.first] = b;
                  }
                  return obs;
              })
        .def("get_nested_partitions",
             +[](PartitionModeState& state)
              {
                  python::dict obs;
                  auto& bs = state.get_partitions();
                  for (auto& kb : bs)
                  {
                      python::list obv;
                      auto bv = state.get_nested_partition(kb.first);
                      for (PartitionModeState::b_t& b : bv)
                          obv.append(b);
                      obs[kb.first] = obv;
                  }
                  return obs;
              })
        .def("get_coupled_state",
             +[](PartitionModeState& state)
              {
                  auto c = state.get_coupled_state();
                  if (c == nullptr)
                      return python::object();
                  return python::object(*c);
              })
        .def("relabel", &PartitionModeState::relabel)
        .def("entropy", &PartitionModeState::entropy)
        .def("posterior_cerror", &PartitionModeState::posterior_cerror)
        .def("posterior_dev", &PartitionModeState::posterior_dev)
        .def("posterior_entropy", &PartitionModeState::posterior_entropy)
        .def("posterior_lprob",
             +[](PartitionModeState& state, object ob, bool MLE)
              {
                  auto b = get_array<int32_t, 1>(ob);
                  return state.posterior_lprob(b, MLE);
              })
        .def("get_ptr",
             +[](PartitionModeState& state)
              {
                  return size_t(&state);
              });

    def("partition_overlap",
        +[](object ox, object oy)
         {
             auto x = get_array<int32_t, 1>(ox);
             auto y = get_array<int32_t, 1>(oy);
             return partition_overlap(x, y);
         });

    def("partition_shuffle_labels",
        +[](object ox, rng_t& rng)
         {
             auto x = get_array<int32_t, 1>(ox);
             idx_map<int32_t, int32_t> rmap;
             for (auto r : x)
                 rmap[r] = r;
             std::vector<int32_t> rset;
             for (auto& r : rmap)
                 rset.push_back(r.first);
             std::shuffle(rset.begin(), rset.end(), rng);
             size_t pos = 0;
             for (auto& r : rmap)
                 r.second = rset[pos++];
             for (auto& r : x)
                 r = rmap[r];
         });

    def("align_partition_labels",
        +[](object ox, object oy)
         {
             auto x = get_array<int32_t, 1>(ox);
             auto y = get_array<int32_t, 1>(oy);
             partition_align_labels(x, y);
         });

    def("get_contingency_graph",
        +[](GraphInterface& gi, boost::any alabel,
            boost::any amrs, boost::any apartition, object ox, object oy)
         {
             auto x = get_array<int32_t, 1>(ox);
             auto y = get_array<int32_t, 1>(oy);
             auto label = any_cast<vprop_map_t<int32_t>::type>(alabel);
             auto partition = any_cast<vprop_map_t<uint8_t>::type>(apartition);
             auto mrs = any_cast<eprop_map_t<int32_t>::type>(amrs);
             run_action<>()
                 (gi, [&](auto& g)
                  {
                      get_contingency_graph<false>(g, partition, label, mrs, x, y);
                  })();
         });
}
