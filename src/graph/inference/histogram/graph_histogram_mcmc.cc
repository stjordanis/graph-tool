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

#include "graph_histogram.hh"
#include "graph_histogram_mcmc.hh"
#include "../loops/mcmc_loop.hh"

using namespace boost;
using namespace graph_tool;

template <template <class T> class VT>
GEN_DISPATCH(hist_state, HistD<VT>::template HistState, HIST_STATE_params)

template <class State>
GEN_DISPATCH(mcmc_hist_state, MCMC<State>::template MCMCHistState,
             MCMC_HIST_STATE_params(State))

template <size_t n>
struct va_t
{
    template <class T>
    using type = std::array<T, n>;
};

template <class T>
using Vec = std::vector<T>;

python::object hist_mcmc_sweep(python::object omcmc_state,
                               python::object ohist_state,
                               size_t D,
                               rng_t& rng)
{
    python::object ret;
    auto dispatch = [&](auto& hist_state)
    {
        typedef typename std::remove_reference<decltype(hist_state)>::type
            state_t;

        mcmc_hist_state<state_t>::make_dispatch
           (omcmc_state,
            [&](auto& s)
            {
                auto ret_ = mcmc_sweep(*s, rng);
                ret = tuple_apply([&](auto&... args){ return python::make_tuple(args...); }, ret_);
            });
    };


    switch (D)
    {
    case 1:
        {
            hist_state<va_t<1>::type>::dispatch(ohist_state, dispatch);
        }
        break;
    case 2:
        {
            hist_state<va_t<2>::type>::dispatch(ohist_state, dispatch);
        }
        break;
    case 3:
        {
            hist_state<va_t<3>::type>::dispatch(ohist_state, dispatch);
        }
        break;
    case 4:
        {
            hist_state<va_t<4>::type>::dispatch(ohist_state, dispatch);
        }
        break;
    default:
        {
            hist_state<Vec>::dispatch(ohist_state, dispatch);
        }
    }

    return ret;
}

namespace graph_tool
{
ostream& operator<<(ostream& s, hmove_t v)
{
    s << int(v);
    return s;
}
}

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;
    def("hist_mcmc_sweep", &hist_mcmc_sweep);
});
