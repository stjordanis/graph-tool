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

#define BOOST_PYTHON_MAX_ARITY 20

#define __MOD__ generation
#define DEF_REGISTRY
#include "module_registry.hh"

#include "graph.hh"
#include "graph_util.hh"
#include "graph_filtering.hh"
#include "graph_generation.hh"
#include "sampler.hh"
#include "dynamic_sampler.hh"
#include <boost/python.hpp>

using namespace std;
using namespace boost;
using namespace graph_tool;

class PythonFuncWrap
{
public:
    PythonFuncWrap(boost::python::object o): _o(o) {}

    pair<size_t, size_t> operator()(size_t i) const
    {
        boost::python::object ret = _o(i);
        return boost::python::extract<pair<size_t,size_t> >(ret);
    }

    size_t operator()(size_t i, bool) const
    {
        boost::python::object ret = _o(i);
        return boost::python::extract<size_t>(ret);
    }

private:
    boost::python::object _o;
};

void generate_graph(GraphInterface& gi, size_t N,
                    boost::python::object deg_sample, bool no_parallel,
                    bool no_self_loops, bool undirected, rng_t& rng,
                    bool verbose, bool verify)
{
    typedef graph_tool::detail::get_all_graph_views::apply<
    graph_tool::detail::filt_scalar_type, boost::mpl::bool_<false>,
        boost::mpl::bool_<false>, boost::mpl::bool_<false>,
        boost::mpl::bool_<true>, boost::mpl::bool_<true> >::type graph_views;

    if (undirected)
        gi.set_directed(false);

    run_action<graph_views>()
        (gi,
         [&](auto&& graph)
         {
             return gen_graph()
                 (std::forward<decltype(graph)>(graph), N,
                  PythonFuncWrap(deg_sample), no_parallel, no_self_loops, rng,
                  verbose, verify);
         })();
}


using namespace boost::python;

BOOST_PYTHON_MODULE(libgraph_tool_generation)
{
    docstring_options dopt(true, false);
    def("gen_graph", &generate_graph);

    class_<Sampler<int, boost::mpl::false_>>("Sampler",
                                             init<const vector<int>&, const vector<double>&>())
        .def("sample", &Sampler<int, boost::mpl::false_>::sample<rng_t>,
             return_value_policy<copy_const_reference>());

    class_<DynamicSampler<int>>("DynamicSampler",
                                init<const vector<int>&,
                                     const vector<double>&>())
        .def("sample", &DynamicSampler<int>::sample<rng_t>,
             return_value_policy<copy_const_reference>())
        .def("insert", &DynamicSampler<int>::insert)
        .def("remove", &DynamicSampler<int>::remove)
        .def("clear", &DynamicSampler<int>::clear)
        .def("rebuild", &DynamicSampler<int>::rebuild);

    __MOD__::EvokeRegistry();
}
