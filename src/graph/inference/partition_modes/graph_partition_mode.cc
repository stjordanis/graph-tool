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
#include "numpy_bind.hh"

#include <boost/python.hpp>

#include "graph_partition_mode.hh"

using namespace boost;
using namespace graph_tool;

PartitionModeState::bv_t get_bv(python::object ob)
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
        .def("virtual_add_partition",
             +[](PartitionModeState& state, object ob, bool relabel)
              {
                  auto bv = get_bv(ob);
                  return state.virtual_add_partition(bv, relabel);
              })
        .def("virtual_remove_partition",
             +[](PartitionModeState& state, object ob)
              {
                  auto bv = get_bv(ob);
                  return state.virtual_remove_partition(bv);
              })
        .def("replace_partitions",
             +[](PartitionModeState& state, rng_t& rng)
              {
                  return state.replace_partitions(rng);
              })
        .def("relabel_partition",
             +[](PartitionModeState& state, python::object obv)
              {
                  PartitionModeState::bv_t bv;
                  for (int i = 0; i < python::len(obv); ++i)
                  {
                      PartitionModeState::b_t& b =
                          python::extract<PartitionModeState::b_t&>(obv[i]);
                      bv.emplace_back(b);
                  }
                  state.relabel_partition(bv, 0);
              })
        .def("align_mode", &PartitionModeState::align_mode)
        .def("get_B", &PartitionModeState::get_B)
        .def("get_M", &PartitionModeState::get_M)
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
             +[](PartitionModeState& state, bool MLE, bool fix_empty,
                 rng_t& rng)
              {
                  python::list obv;
                  auto bv = state.sample_nested_partition(MLE, fix_empty, rng);
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
        .def("posterior_cdev", &PartitionModeState::posterior_cdev)
        .def("posterior_entropy", &PartitionModeState::posterior_entropy)
        .def("posterior_lprob",
             +[](PartitionModeState& state, object obv, bool MLE)
              {
                  PartitionModeState::bv_t bv;
                  for (int i = 0; i < python::len(obv); ++i)
                  {
                      PartitionModeState::b_t& b =
                          python::extract<PartitionModeState::b_t&>(obv[i]);
                      bv.emplace_back(b);
                  }
                  return state.posterior_lprob(bv, MLE);
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
             partition_shuffle_labels(x, rng);
         });

    def("nested_partition_shuffle_labels",
        +[](object ox, rng_t& rng)
         {
             std::vector<std::vector<int32_t>> x;
             for (int l = 0; l < python::len(ox); ++l)
             {
                 auto a = get_array<int32_t, 1>(ox[l]);
                 x.emplace_back(a.begin(), a.end());
             }

             nested_partition_shuffle_labels(x, rng);

             python::list onx;
             for (auto& xl : x)
                 onx.append(wrap_vector_owned(xl));

             return onx;
         });

    def("partition_order_labels",
        +[](object ox)
         {
             auto x = get_array<int32_t, 1>(ox);
             partition_order_labels(x);
         });

    def("nested_partition_order_labels",
        +[](object ox)
         {
             std::vector<std::vector<int32_t>> x;
             for (int l = 0; l < python::len(ox); ++l)
             {
                 auto a = get_array<int32_t, 1>(ox[l]);
                 x.emplace_back(a.begin(), a.end());
             }

             nested_partition_order_labels(x);

             python::list onx;
             for (auto& xl : x)
                 onx.append(wrap_vector_owned(xl));

             return onx;
         });

    def("align_partition_labels",
        +[](object ox, object oy)
         {
             auto x = get_array<int32_t, 1>(ox);
             auto y = get_array<int32_t, 1>(oy);
             partition_align_labels(x, y);
         });

    def("align_nested_partition_labels",
        +[](object ox, object oy)
         {
             std::vector<std::vector<int32_t>> x, y;
             for (int l = 0; l < python::len(ox); ++l)
             {
                 auto a = get_array<int32_t, 1>(ox[l]);
                 x.emplace_back(a.begin(), a.end());
             }

             for (int l = 0; l < python::len(oy); ++l)
             {
                 auto a = get_array<int32_t, 1>(oy[l]);
                 y.emplace_back(a.begin(), a.end());
             }

             nested_partition_align_labels(x, y);

             python::list onx;
             for (auto& xl : x)
                 onx.append(wrap_vector_owned(xl));

             return onx;
         });

    def("partition_overlap_center",
        +[](object obs, object oc)
         {
             auto c = get_array<int32_t, 1>(oc);
             auto bs = get_array<int32_t, 2>(obs);

             return partition_overlap_center(c, bs);
         });

    def("nested_partition_overlap_center",
        +[](object obs, object oc)
         {
             std::vector<std::vector<int32_t>> c;
             for (int l = 0; l < python::len(oc); ++l)
             {
                 auto a = get_array<int32_t, 1>(oc[l]);
                 c.emplace_back(a.begin(), a.end());
             }

             std::vector<std::vector<std::vector<int32_t>>> bs;

             for (int m = 0; m < python::len(obs); ++m)
             {
                 bs.emplace_back();
                 auto& x = bs.back();
                 for (int l = 0; l < python::len(obs[m]); ++l)
                 {
                     auto a = get_array<int32_t, 1>(obs[m][l]);
                     x.emplace_back(a.begin(), a.end());
                 }
             }

             double r = nested_partition_overlap_center(c, bs);

             python::list onx;
             for (auto& xl : c)
                 onx.append(wrap_vector_owned(xl));

             python::list onbs;
             for (auto& bv : bs)
             {
                 python::list nx;
                 for (auto& xl : bv)
                     nx.append(wrap_vector_owned(xl));
                 onbs.append(nx);
             }

             return python::make_tuple(onx, onbs, r);
         });

    def("nested_partition_clear_null",
        +[](object ox)
         {
             python::list onx;
             for (int l = 0; l < python::len(ox); ++l)
             {
                 auto a = get_array<int32_t, 1>(ox[l]);
                 std::vector<int32_t> x(a.begin(), a.end());
                 while (!x.empty() && x.back() == -1)
                     x.pop_back();
                 for (auto& r : x)
                 {
                     if (r == -1)
                         r = 0;
                 }
                 onx.append(wrap_vector_owned(x));
             }
             return onx;
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
