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

#include <boost/python.hpp>
#include "numpy_bind.hh"
#include "hash_map_wrap.hh"

#include "support/util.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

template <class Value>
void vector_map(boost::python::object ovals, boost::python::object omap)
{
    multi_array_ref<Value,1> vals = get_array<Value,1>(ovals);
    multi_array_ref<Value,1> map = get_array<Value,1>(omap);

    size_t pos = 0;
    for (size_t i = 0; i < vals.size(); ++i)
    {
        Value v = vals[i];
        if (map[v] == -1)
            map[v] = pos++;
        vals[i] = map[v];
    }
}

template <class Value>
void vector_contiguous_map(boost::python::object ovals)
{
    multi_array_ref<Value,1> vals = get_array<Value,1>(ovals);
    gt_hash_map<Value, size_t> map;

    for (size_t i = 0; i < vals.size(); ++i)
    {
        Value v = vals[i];
        auto iter = map.find(v);
        if (iter == map.end())
            iter = map.insert(make_pair(v, map.size())).first;
        vals[i] = iter->second;
    }
}

template <class Value>
void vector_rmap(boost::python::object ovals, boost::python::object omap)
{
    multi_array_ref<Value,1> vals = get_array<Value,1>(ovals);
    multi_array_ref<Value,1> map = get_array<Value,1>(omap);

    for (size_t i = 0; i < vals.size(); ++i)
    {
        map[vals[i]] = i;
    }
}

#define __MOD__ inference
#define DEF_REGISTRY
#include "module_registry.hh"

BOOST_PYTHON_MODULE(libgraph_tool_inference)
{
    using namespace boost::python;
    docstring_options dopt(true, false);

    def("vector_map", vector_map<int32_t>);
    def("vector_map64", vector_map<int64_t>);
    def("vector_mapdouble", vector_map<double>);
    def("vector_rmap", vector_rmap<int32_t>);
    def("vector_rmap64", vector_rmap<int64_t>);
    def("vector_rmapdouble", vector_rmap<double>);
    def("vector_contiguous_map", vector_contiguous_map<int32_t>);
    def("vector_contiguous_map64", vector_contiguous_map<int64_t>);

    def("lbinom", lbinom<size_t, size_t>);
    def("lbinom_fast", lbinom_fast<true, size_t, size_t>);
    def("log_sum_exp", +[](double x, double y){ return log_sum_exp(x, y); });

    __MOD__::EvokeRegistry();
}
