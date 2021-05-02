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
#include "graph_histogram_mcmc.hh"
#include "../loops/mcmc_loop.hh"

using namespace boost;
using namespace graph_tool;

template <class VT>
GEN_DISPATCH(hist_state, HistD<VT>::template HistState, HIST_STATE_params)

template <class State>
GEN_DISPATCH(mcmc_hist_state, MCMC<State>::template MCMCHistState,
             MCMC_HIST_STATE_params(State))

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
                auto ret_ = mcmc_sweep(s, rng);
                ret = tuple_apply([&](auto&... args){ return python::make_tuple(args...); }, ret_);
            });
    };


    switch (D)
    {
    case 1:
        {
            typedef std::array<double, 1> v_t;
            hist_state<v_t>::dispatch(ohist_state, dispatch);
        }
        break;
    case 2:
        {
            typedef std::array<double, 2> v_t;
            hist_state<v_t>::dispatch(ohist_state, dispatch);
        }
        break;
    case 3:
        {
            typedef std::array<double, 3> v_t;
            hist_state<v_t>::dispatch(ohist_state, dispatch);
        }
        break;
    default:
        {
            typedef std::vector<double> v_t;
            hist_state<v_t>::dispatch(ohist_state, dispatch);
        }
    }

    return ret;
}

ostream& operator<<(ostream& s, move_t v)
{
    s << int(v);
    return s;
}


void export_hist_mcmc()
{
    using namespace boost::python;
    def("hist_mcmc_sweep", &hist_mcmc_sweep);
}
