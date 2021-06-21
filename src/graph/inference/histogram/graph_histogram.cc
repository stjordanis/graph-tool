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

#include "graph_histogram.hh"
#include "../support/graph_state.hh"

using namespace boost;
using namespace graph_tool;

template <template <class T> class VT>
GEN_DISPATCH(hist_state, HistD<VT>::template HistState, HIST_STATE_params)

template <size_t n>
struct va_t
{
    template <class T>
    using type = std::array<T, n>;
};

template <class T>
using Vec = std::vector<T>;

python::object make_hist_state(boost::python::object ostate, size_t D)
{
    python::object state;
    switch (D)
    {
    case 1:
        {
            hist_state<va_t<1>::type>::make_dispatch(ostate,
                                                     [&](auto& s){state = python::object(s);});
        }
        break;
    case 2:
        {
            hist_state<va_t<2>::type>::make_dispatch(ostate,
                                                     [&](auto& s){state = python::object(s);});
        }
        break;
    case 3:
        {
            hist_state<va_t<3>::type>::make_dispatch(ostate,
                                                     [&](auto& s){state = python::object(s);});
        }
        break;
    case 4:
        {
            hist_state<va_t<4>::type>::make_dispatch(ostate,
                                                     [&](auto& s){state = python::object(s);});
        }
        break;
    default:
        {
            hist_state<Vec>::make_dispatch(ostate,
                                           [&](auto& s){state = python::object(s);});
        }
    }
    return state;
}

template <class State>
void dispatch_state_def(State*)
{
    using namespace boost::python;

    class_<State>
        c(name_demangle(typeid(State).name()).c_str(),
          no_init);
    c.def("entropy", &State::entropy)
        .def("get_mle_lpdf",
             +[](State& state, python::object xo)
             {
                 auto x = get_array<typename State::value_t, 1>(xo);
                 return state.get_mle_lpdf(x);
             })
        .def("sample",
             +[](State& state, size_t n, python::object cxo, rng_t& rng)
             {
                 auto cx = get_array<typename State::value_t, 1>(cxo);
                 auto x = state.sample(n, cx, rng);
                 return wrap_multi_array_owned(x);
             });
}


void export_hist_state()
{
    using namespace boost::python;
    def("make_hist_state", &make_hist_state);

    {
        hist_state<va_t<1>::type>::dispatch
            ([&](auto* s){ dispatch_state_def(s);});
    }
    {
        hist_state<va_t<2>::type>::dispatch
            ([&](auto* s){ dispatch_state_def(s);});
    }
    {
        hist_state<va_t<3>::type>::dispatch
            ([&](auto* s){ dispatch_state_def(s);});
    }
    {
        hist_state<va_t<4>::type>::dispatch
            ([&](auto* s){ dispatch_state_def(s);});
    }
    {
        hist_state<Vec>::dispatch
            ([&](auto* s){ dispatch_state_def(s);});
    }
}
