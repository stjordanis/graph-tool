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
#include "graph.hh"
#include "graph_util.hh"
#include "graph_python_interface.hh"
#include "hash_map_wrap.hh"

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <set>


using namespace std;
using namespace boost;
using namespace graph_tool;

namespace graph_tool
{

template <class ValueList>
struct add_edge_list
{
    template <class Graph>
    void operator()(Graph& g, python::object aedge_list,
                    python::object& eprops, bool& found) const
    {
        boost::mpl::for_each<ValueList>(
            [&](auto&& graph)
            {
                return dispatch()
                    (g, aedge_list, eprops, found,
                     std::forward<decltype(graph)>(graph));
            });
    }

    struct dispatch
    {
        template <class Graph, class Value>
        void operator()(Graph& g, python::object& aedge_list,
                        python::object& oeprops, bool& found, Value) const
        {
            if (found)
                return;
            try
            {
                boost::multi_array_ref<Value, 2> edge_list = get_array<Value, 2>(aedge_list);

                if (edge_list.shape()[1] < 2)
                    throw GraphException("Second dimension in edge list must be of size (at least) two");

                typedef typename graph_traits<Graph>::edge_descriptor edge_t;
                vector<DynamicPropertyMapWrap<Value, edge_t>> eprops;
                python::stl_input_iterator<boost::any> iter(oeprops), end;
                for (; iter != end; ++iter)
                    eprops.emplace_back(*iter, writable_edge_properties());

                size_t n_props = std::min(eprops.size(), edge_list.shape()[1] - 2);

                for (const auto& e : edge_list)
                {
                    size_t s = e[0];
                    size_t t = e[1];
                    while (s >= num_vertices(g) || t >= num_vertices(g))
                        add_vertex(g);
                    auto ne = add_edge(vertex(s, g), vertex(t, g), g).first;
                    for (size_t i = 0; i < n_props; ++i)
                    {
                        try
                        {
                            put(eprops[i], ne, e[i + 2]);
                        }
                        catch(bad_lexical_cast&)
                        {
                            throw ValueException("Invalid edge property value: " +
                                                 lexical_cast<string>(e[i + 2]));
                        }
                    }
                }
                found = true;
            }
            catch (InvalidNumpyConversion& e) {}
        }
    };
};

void do_add_edge_list(GraphInterface& gi, python::object aedge_list,
                      python::object eprops)
{
    typedef mpl::vector<bool, char, uint8_t, uint16_t, uint32_t, uint64_t,
                        int8_t, int16_t, int32_t, int64_t, uint64_t, double,
                        long double> vals_t;
    bool found = false;
    run_action<>()
        (gi,
         [&](auto&& graph)
         {
             return add_edge_list<vals_t>()(
                 std::forward<decltype(graph)>(graph), aedge_list, eprops,
                 found);
         })();
    if (!found)
        throw GraphException("Invalid type for edge list; must be two-dimensional with a scalar type");
}


struct add_edge_list_iter
{
    template <class Graph>
    void operator()(Graph& g, python::object& edge_list,
                    python::object& oeprops) const
    {
        typedef typename graph_traits<Graph>::edge_descriptor edge_t;
        vector<DynamicPropertyMapWrap<python::object, edge_t>> eprops;
        python::stl_input_iterator<boost::any> piter(oeprops), pend;
        for (; piter != pend; ++piter)
            eprops.emplace_back(*piter, writable_edge_properties());

        python::stl_input_iterator<python::object> iter(edge_list), end;
        for (; iter != end; ++iter)
        {
            const auto& row = *iter;
            python::stl_input_iterator<python::object> eiter(row), eend;

            size_t s = 0;
            size_t t = 0;

            typename graph_traits<Graph>::edge_descriptor e;
            size_t i = 0;
            for(; eiter != eend; ++eiter)
            {
                if (i >= eprops.size() + 2)
                    break;
                const auto& val = *eiter;
                switch (i)
                {
                case 0:
                    s = python::extract<size_t>(val);
                    while (s >= num_vertices(g))
                        add_vertex(g);
                    break;
                case 1:
                    t = python::extract<size_t>(val);
                    while (t >= num_vertices(g))
                        add_vertex(g);
                    e = add_edge(vertex(s, g), vertex(t, g), g).first;
                    break;
                default:
                    try
                    {
                        put(eprops[i - 2], e, val);
                    }
                    catch(bad_lexical_cast&)
                    {
                        throw ValueException("Invalid edge property value: " +
                                             python::extract<string>(python::str(val))());
                    }
                }
                i++;
            }
        }
    }
};

void do_add_edge_list_iter(GraphInterface& gi, python::object edge_list,
                           python::object eprops)
{
    run_action<>()
        (gi,
         [&](auto&& graph)
         {
             return add_edge_list_iter()
                 (std::forward<decltype(graph)>(graph), edge_list, eprops);
         })();
}

struct add_edge_list_hash
{
    template <class Graph, class VProp>
    void operator()(Graph& g, python::object aedge_list, VProp vmap,
                    python::object& eprops) const
    {
        typedef typename property_traits<VProp>::value_type val_t;
        if constexpr (is_scalar_v<val_t>)
        {
            try
            {
                numpy_dispatch(g, aedge_list, vmap, eprops);
            }
            catch (InvalidNumpyConversion&)
            {
                dispatch(g, aedge_list, vmap, eprops);
            }
        }
        else
        {
            dispatch(g, aedge_list, vmap, eprops);
        }
    }

    template <class Graph, class VProp>
    void numpy_dispatch(Graph& g, python::object& aedge_list, VProp& vmap,
                        python::object& oeprops) const
    {
        typedef typename property_traits<VProp>::value_type val_t;

        boost::multi_array_ref<val_t, 2> edge_list = get_array<val_t, 2>(aedge_list);
        typedef typename std::conditional_t<std::is_integral_v<val_t>,
                                            gt_hash_map<val_t, size_t>,
                                            unordered_map<val_t, size_t>> vmap_t;
        vmap_t vertices;

        if (edge_list.shape()[1] < 2)
            throw GraphException("Second dimension in edge list must be of size (at least) two");

        typedef typename graph_traits<Graph>::edge_descriptor edge_t;
        vector<DynamicPropertyMapWrap<val_t, edge_t>> eprops;
        python::stl_input_iterator<boost::any> iter(oeprops), end;
        for (; iter != end; ++iter)
            eprops.emplace_back(*iter, writable_edge_properties());

        size_t n_props = std::min(eprops.size(), edge_list.shape()[1] - 2);

        auto get_vertex = [&] (const val_t& r) -> size_t
            {
                auto iter = vertices.find(r);
                if (iter == vertices.end())
                {
                    auto v = add_vertex(g);
                    vertices[r] = v;
                    vmap[v] = r;
                    return v;
                }
                return iter->second;
            };

        for (const auto& e : edge_list)
        {
            size_t s = get_vertex(e[0]);
            size_t t = get_vertex(e[1]);
            auto ne = add_edge(vertex(s, g), vertex(t, g), g).first;
            for (size_t i = 0; i < n_props; ++i)
            {
                try
                {
                    put(eprops[i], ne, e[i + 2]);
                }
                catch(bad_lexical_cast&)
                {
                    throw ValueException("Invalid edge property value: " +
                                         lexical_cast<string>(e[i + 2]));
                }
            }
        }
    }

    template <class Graph, class VProp>
    void dispatch(Graph& g, python::object& edge_list, VProp& vmap,
                  python::object& oeprops) const
    {
        typedef typename property_traits<VProp>::value_type val_t;

        typedef typename std::conditional_t<std::is_integral_v<val_t>,
                                            gt_hash_map<val_t, size_t>,
                                            unordered_map<val_t, size_t>> vmap_t;
        vmap_t vertices;

        typedef typename graph_traits<Graph>::edge_descriptor edge_t;
        vector<DynamicPropertyMapWrap<python::object, edge_t>> eprops;
        python::stl_input_iterator<boost::any> piter(oeprops), pend;
        for (; piter != pend; ++piter)
            eprops.emplace_back(*piter, writable_edge_properties());

        auto get_vertex = [&] (const val_t& r) -> size_t
            {
                auto iter = vertices.find(r);
                if (iter == vertices.end())
                {
                    auto v = add_vertex(g);
                    vertices[r] = v;
                    vmap[v] = r;
                    return v;
                }
                return iter->second;
            };

        python::stl_input_iterator<python::object> iter(edge_list), end;
        for (; iter != end; ++iter)
        {
            const auto& row = *iter;

            python::stl_input_iterator<python::object> eiter(row), eend;

            size_t s = 0;
            typename graph_traits<Graph>::edge_descriptor e;
            size_t i = 0;
            for(; eiter != eend; ++eiter)
            {
                if (i >= eprops.size() + 2)
                    break;
                const auto& val = *eiter;
                if (i < 2)
                {
                    val_t x;
                    if constexpr (std::is_same_v<val_t, python::object>)
                        x = val;
                    else
                        x = python::extract<val_t>(val);
                    auto v = get_vertex(x);
                    while (v >= num_vertices(g))
                        add_vertex(g);
                    if (i == 0)
                        s = v;
                    else
                        e = add_edge(s, v, g).first;
                }
                else
                {
                    try
                    {
                        put(eprops[i - 2], e, val);
                    }
                    catch(bad_lexical_cast&)
                    {
                        throw ValueException("Invalid edge property value: " +
                                             python::extract<string>(python::str(val))());
                    }
                }
                i++;
            }
        }
    };
};

void do_add_edge_list_hashed(GraphInterface& gi, python::object aedge_list,
                             boost::any& vertex_map,
                             python::object eprops)
{
    run_action<graph_tool::all_graph_views, boost::mpl::true_>()
        (gi,
         [&](auto&& graph, auto&& a2)
         {
             return add_edge_list_hash()
                 (std::forward<decltype(graph)>(graph), aedge_list,
                  std::forward<decltype(a2)>(a2), eprops);
         },
         writable_vertex_properties())(vertex_map);
}

} // namespace graph_tool
