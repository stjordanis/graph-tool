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

#include <boost/python.hpp>
#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "numpy_bind.hh"

#include "graph_selectors.hh"
#include "graph_properties.hh"

#include "graph_continuous.hh"

#include "random.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

template <class Graph, class State>
class WrappedCState
{
public:
    typedef typename State::smap_t smap_t;

    WrappedCState(Graph& g, smap_t s, smap_t s_diff,
                 python::dict params, rng_t& rng)
        : _state(g, s, s_diff, params, rng), _g(g)
    {
    }

    void get_diff_sync(double t, double dt, rng_t& rng)
    {
        graph_tool::get_diff_sync(_g, _state, t, dt, rng);
    }

    static void python_export()
    {
        python::class_<WrappedCState<Graph,State>>
            (name_demangle(typeid(WrappedCState<Graph,State>).name()).c_str(),
             python::init<Graph&, smap_t, smap_t, python::dict, rng_t&>())
            .def("get_diff_sync", &WrappedCState<Graph,State>::get_diff_sync);
    }

private:
    State _state;
    Graph& _g;
};

template <class State>
python::object make_state(GraphInterface& gi, boost::any as, boost::any as_diff,
                          python::dict params, rng_t& rng)
{
    typedef typename State::smap_t::checked_t smap_t;
    smap_t s = boost::any_cast<smap_t>(as);
    smap_t s_diff = boost::any_cast<smap_t>(as_diff);

    python::object state;
    run_action<>()
        (gi,
         [&](auto& g)
         {
             typedef typename std::remove_reference<decltype(g)>::type g_t;
             state =
                 python::object(WrappedCState<g_t,State>
                                (g, s.get_unchecked(num_vertices(g)),
                                 s_diff.get_unchecked(num_vertices(g)), params,
                                 rng));
         })();
    return state;
}

struct add_ptr
{
    template <class T>
    struct apply
    {
        typedef typename std::add_pointer<T>::type type;
    };
};


template <class State>
void export_cstate()
{
    mpl::for_each<all_graph_views, add_ptr>
        ([](auto g)
         {
             typedef typename std::remove_pointer<decltype(g)>::type g_t;
             WrappedCState<g_t,State>::python_export();
         });
}

void export_continuous()
{
    export_cstate<kuramoto_state>();
    def("make_kuramoto_state", &make_state<kuramoto_state>);
}
