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

#include "graph_partition_centroid_rmi.hh"
#include "../support/graph_state.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, RMICenterState, BLOCK_STATE_params)

python::object make_rmi_center_state(boost::python::object ostate)
{
    python::object state;
    block_state::make_dispatch(ostate,
                               [&](auto& s){state = python::object(s);});
    return state;
}

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;
    def("make_rmi_center_state", &make_rmi_center_state);

    block_state::dispatch
        ([&](auto* s)
         {
             typedef typename std::remove_reference<decltype(*s)>::type state_t;
             void (state_t::*move_vertex)(size_t, size_t) =
                 &state_t::move_vertex;
             double (state_t::*virtual_move)(size_t, size_t, size_t) =
                 &state_t::virtual_move;

             class_<state_t, bases<>, std::shared_ptr<state_t>>
                 c(name_demangle(typeid(state_t).name()).c_str(),
                   no_init);
             c.def("move_vertex", move_vertex)
                 .def("virtual_move", virtual_move)
                 .def("entropy", &state_t::entropy);
         });

    def("log_omega",
        +[](python::object onr, python::object ons) -> double
         {
             auto nr = get_array<int32_t, 1>(onr);
             auto ns = get_array<int32_t, 1>(ons);
             return log_omega(nr, ns, [](auto& x) {return x;});
         });

    def("expected_MI",
        +[](python::object onr, python::object ons) -> double
         {
             auto nr = get_array<int32_t, 1>(onr);
             auto ns = get_array<int32_t, 1>(ons);

             int32_t N = 0;
             for (auto c : nr)
                 N += c;

             double EMI = 0;

             #pragma omp parallel for reduction(+:EMI) collapse(2)
             for (size_t r = 0; r < nr.size(); ++r)
             {
                 for (size_t s = 0; s < ns.size(); ++s)
                 {
                     auto a = nr[r];
                     auto b = ns[s];
                     for (int32_t m = max(1, a + b - N); m <= min(a, b); ++m)
                     {
                         double T = (m * (safelog_fast(m) + safelog_fast(N) - safelog_fast(a) - safelog_fast(b))) / N;
                         double lT = lgamma_fast(a + 1) + lgamma_fast(b + 1);
                         lT += lgamma_fast(N - a + 1) + lgamma_fast(N - b + 1);
                         lT -= lgamma_fast(N + 1) + lgamma_fast(m + 1);
                         lT -= lgamma_fast(a - m + 1) + lgamma_fast(b - m + 1);
                         lT -= lgamma_fast(N - a - b + m + 1);
                         EMI += T * exp(lT);
                     }
                 }
             }

             return EMI;
         });
});
