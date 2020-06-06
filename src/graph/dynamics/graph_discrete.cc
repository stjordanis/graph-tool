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

#include "graph_discrete.hh"

#include "random.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

template <class Graph, class State>
class WrappedState
{
public:
    typedef typename State::smap_t smap_t;

    WrappedState(Graph& g, smap_t s, smap_t s_temp,
                 python::dict params, rng_t& rng)
        : _state(g, s, s_temp, params, rng), _g(g)
    {
    }

    void reset_active(rng_t& rng)
    {
        auto& active = *_state._active;
        active.clear();
        for (auto v : vertices_range(_g))
        {
            if (_state.is_absorbing(_g, v))
                continue;
            active.push_back(v);
        }
        std::shuffle(active.begin(), active.end(), rng);
    }

    python::object get_active()
    {
        return wrap_vector_not_owned(*_state._active);
    }

    size_t iterate_sync(size_t niter, rng_t& rng)
    {
        return discrete_iter_sync(_g, _state, niter, rng);
    }

    size_t iterate_async(size_t niter, rng_t& rng)
    {
        return discrete_iter_async(_g, _state, niter, rng);
    }

    static void python_export()
    {
        python::class_<WrappedState<Graph,State>>
            (name_demangle(typeid(WrappedState<Graph,State>).name()).c_str(),
             python::init<Graph&, smap_t, smap_t, python::dict, rng_t&>())
            .def("reset_active", &WrappedState<Graph,State>::reset_active)
            .def("get_active", &WrappedState<Graph,State>::get_active)
            .def("iterate_sync", &WrappedState<Graph,State>::iterate_sync)
            .def("iterate_async", &WrappedState<Graph,State>::iterate_async);
    }

private:
    State _state;
    Graph& _g;
};

template <class State>
python::object make_state(GraphInterface& gi, boost::any as, boost::any as_temp,
                          python::dict params, rng_t& rng)
{
    typedef typename State::smap_t::checked_t smap_t;
    smap_t s = boost::any_cast<smap_t>(as);
    smap_t s_temp = boost::any_cast<smap_t>(as_temp);

    python::object state;
    run_action<>()
        (gi,
         [&](auto& g)
         {
             typedef typename std::remove_reference<decltype(g)>::type g_t;
             state =
                 python::object(WrappedState<g_t,State>
                                (g, s.get_unchecked(num_vertices(g)),
                                 s_temp.get_unchecked(num_vertices(g)), params,
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
void export_state()
{
    mpl::for_each<all_graph_views, add_ptr>
        ([](auto g)
         {
             typedef typename std::remove_pointer<decltype(g)>::type g_t;
             WrappedState<g_t,State>::python_export();
         });
}

template <template <bool...> class Base, bool... As>
void export_SI_state()
{
    export_state<Base<As..., false, false>>();
    export_state<Base<As..., true, false>>();
    export_state<Base<As..., true, true>>();
}

template <template <bool...> class Base, bool... As>
python::object make_SI_state(GraphInterface& gi, boost::any as,
                             boost::any as_temp, python::dict params,
                             rng_t& rng, bool weighted, bool constant_beta)
{
    if (weighted)
    {
        if (constant_beta)
            return make_state<Base<As..., true, true>>(gi, as, as_temp, params, rng);
        else
            return make_state<Base<As..., true, false>>(gi, as, as_temp, params, rng);
    }
    else
    {
        return make_state<Base<As..., false, false>>(gi, as, as_temp, params, rng);
    }
}

void export_discrete()
{
    export_SI_state<SI_state,false>();
    def("make_SI_state", &make_SI_state<SI_state,false>);

    export_SI_state<SIS_state,false,false>();
    def("make_SIS_state", &make_SI_state<SIS_state,false,false>);

    export_SI_state<SIS_state,false,true>();
    def("make_SIR_state", &make_SI_state<SIS_state,false,true>);

    export_SI_state<SIRS_state,false>();
    def("make_SIRS_state", &make_SI_state<SIRS_state,false>);

    export_SI_state<SI_state,true>();
    def("make_SEI_state", &make_SI_state<SI_state,true>);

    export_SI_state<SIS_state,true,false>();
    def("make_SEIS_state", &make_SI_state<SIS_state,true,false>);

    export_SI_state<SIS_state,true,true>();
    def("make_SEIR_state", &make_SI_state<SIS_state,true,true>);

    export_SI_state<SIRS_state,true>();
    def("make_SEIRS_state", &make_SI_state<SIRS_state,true>);

    export_state<voter_state>();
    def("make_voter_state", &make_state<voter_state>);

    export_state<majority_voter_state>();
    def("make_majority_voter_state", &make_state<majority_voter_state>);

    export_state<binary_threshold_state>();
    def("make_binary_threshold_state", &make_state<binary_threshold_state>);

    export_state<ising_glauber_state>();
    def("make_ising_glauber_state", &make_state<ising_glauber_state>);

    export_state<cising_glauber_state>();
    def("make_cising_glauber_state", &make_state<cising_glauber_state>);

    export_state<ising_metropolis_state>();
    def("make_ising_metropolis_state", &make_state<ising_metropolis_state>);

    export_state<potts_glauber_state>();
    def("make_potts_glauber_state", &make_state<potts_glauber_state>);

    export_state<potts_metropolis_state>();
    def("make_potts_metropolis_state", &make_state<potts_metropolis_state>);

    export_state<axelrod_state>();
    def("make_axelrod_state", &make_state<axelrod_state>);

    export_state<boolean_state>();
    def("make_boolean_state", &make_state<boolean_state>);

    export_state<kirman_state>();
    def("make_kirman_state", &make_state<kirman_state>);
}
