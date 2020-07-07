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

#include "graph_filtering.hh"
#include "graph_selectors.hh"
#include "graph_properties.hh"
#include "graph.hh"

#include "graph_assortativity.hh"

using namespace std;
using namespace graph_tool;

struct empty_object
{};

struct deleted_object
{};

namespace std
{
template <>
struct hash<boost::python::object>
{
    size_t operator()(boost::python::object const& v) const
    {
        return boost::python::extract<size_t>(v.attr("__hash__")());
    }
};
}

template <>
struct empty_key<boost::python::object>
{
    static boost::python::object get()
    {
        return boost::python::object(empty_object());
    }
};

template <>
struct empty_key<std::string>
{
    static std::string get()
    {
        return "___gt__empty___";
    }
};

template <>
struct deleted_key<boost::python::object>
{
    static boost::python::object get()
    {
        return boost::python::object(deleted_object());
    }
};

template <>
struct deleted_key<std::string>
{
    static std::string get()
    {
        return "___gt__deleted___";
    }
};

pair<double,double>
assortativity_coefficient(GraphInterface& gi, GraphInterface::deg_t deg,
                          boost::any weight)
{
    typedef UnityPropertyMap<size_t,GraphInterface::edge_t> weight_map_t;
    typedef boost::mpl::push_back<edge_scalar_properties, weight_map_t>::type
        weight_props_t;

    if (!weight.empty() && !belongs<edge_scalar_properties>()(weight))
        throw ValueException("weight edge property must have a scalar value type");

    if(weight.empty())
        weight = weight_map_t();

    double a = 0, a_err = 0;
    run_action<>()
        (gi,
         [&](auto&& graph, auto&& a2, auto&& a3)
         {
             return get_assortativity_coefficient()
                 (std::forward<decltype(graph)>(graph),
                  std::forward<decltype(a2)>(a2),
                  std::forward<decltype(a3)>(a3), a, a_err);
         },
         all_selectors(), weight_props_t())(degree_selector(deg), weight);
    return make_pair(a, a_err);
}

pair<double,double>
scalar_assortativity_coefficient(GraphInterface& gi, GraphInterface::deg_t deg,
                                 boost::any weight)
{
    typedef UnityPropertyMap<size_t,GraphInterface::edge_t> weight_map_t;
    typedef boost::mpl::push_back<edge_scalar_properties, weight_map_t>::type
        weight_props_t;

    if (!weight.empty() && !belongs<edge_scalar_properties>()(weight))
        throw ValueException("weight edge property must have a scalar value type");

    if(weight.empty())
        weight = weight_map_t();

    double a = 0, a_err = 0;
    run_action<>()
        (gi,
         [&](auto&& graph, auto&& a2, auto&& a3)
         {
             return get_scalar_assortativity_coefficient()
                 (std::forward<decltype(graph)>(graph),
                  std::forward<decltype(a2)>(a2),
                  std::forward<decltype(a3)>(a3), a, a_err);
         },
         scalar_selectors(), weight_props_t())(degree_selector(deg), weight);
    return make_pair(a, a_err);
}

#include <boost/python.hpp>

using namespace boost::python;

void export_assortativity()
{
    def("assortativity_coefficient", &assortativity_coefficient);
    def("scalar_assortativity_coefficient", &scalar_assortativity_coefficient);

    class_<empty_object>("empty_object");
    class_<deleted_object>("deleted_object");
}
