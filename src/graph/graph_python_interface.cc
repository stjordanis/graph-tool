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

#include <boost/python.hpp>

#include "coroutine.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

namespace graph_tool
{

struct get_vertex_iterator
{
    template <class Graph>
    void operator()(Graph& g, GraphInterface& gi,
                    python::object& iter) const
    {
        auto gp = retrieve_graph_view<Graph>(gi, g);
        typedef typename graph_traits<Graph>::vertex_iterator vertex_iterator;
        iter = python::object(PythonIterator<Graph, PythonVertex<Graph>,
                                             vertex_iterator>(gp, vertices(g)));
    }
};

python::object get_vertices(GraphInterface& gi)
{
    python::object iter;
    run_action<>()
        (gi,
         [&](auto&& graph)
         {
             return get_vertex_iterator()
                 (std::forward<decltype(graph)>(graph), gi, iter);
         })();
    return iter;
}

struct get_vertex_soft
{
    template <class Graph>
    void operator()(Graph& g, GraphInterface& gi, size_t i, python::object& v) const
    {
        auto gp = retrieve_graph_view<Graph>(gi, g);
        if (i < num_vertices(g))
            v = python::object(PythonVertex<Graph>(gp, vertex(i, g)));
        else
            v = python::object(PythonVertex<Graph>(gp,
                                                   graph_traits<Graph>::null_vertex()));
    }
};

struct get_vertex_hard
{
    template <class Graph>
    void operator()(Graph& g, GraphInterface& gi, size_t i, python::object& v) const
    {
        auto gp = retrieve_graph_view<Graph>(gi, g);
        size_t c = 0;
        for (auto vi : vertices_range(g))
        {
            if (c == i)
            {
                v = python::object(PythonVertex<Graph>(gp, vi));
                return;
            }
            ++c;
        }
        v = python::object(PythonVertex<Graph>(gp,
                                               graph_traits<Graph>::null_vertex()));
    }
};

python::object get_vertex(GraphInterface& gi, size_t i, bool use_index)
{
    python::object v;
    if (!use_index)
        run_action<>()
            (gi,
             [&](auto&& graph)
             {
                 return get_vertex_hard()
                     (std::forward<decltype(graph)>(graph), gi, i, v);
             })();
    else
        run_action<>()
            (gi,
             [&](auto&& graph)
             {
                 return get_vertex_soft()
                     (std::forward<decltype(graph)>(graph), gi, i, v);
             })();
    return v;
}

struct get_edge_iterator
{
    template <class Graph>
    void operator()(Graph& g, GraphInterface& gi, python::object& iter)
        const
    {
        auto gp = retrieve_graph_view<Graph>(gi, g);
        typedef typename graph_traits<Graph>::edge_iterator edge_iterator;
        iter = python::object(PythonIterator<Graph, PythonEdge<Graph>,
                                             edge_iterator>(gp, edges(g)));
    }
};

python::object get_edges(GraphInterface& gi)
{
    python::object iter;
    run_action<>()
        (gi,
         [&](auto&& graph)
         {
             return get_edge_iterator()
                 (std::forward<decltype(graph)>(graph), gi, iter);
         })();
    return iter;
}

struct add_new_vertex
{
    template <class Graph>
    void operator()(Graph& g, GraphInterface& gi, size_t n,
                    python::object& new_v) const
    {
        auto gp = retrieve_graph_view<Graph>(gi, g);
        if (n != 1)
        {
            for (size_t i = 0; i < n; ++i)
                add_vertex(g);
            new_v = python::object();
        }
        else
        {
            new_v = python::object(PythonVertex<Graph>(gp, add_vertex(g)));
        }
    }
};


python::object add_vertex(GraphInterface& gi, size_t n)
{
    python::object v;
    run_action<>()
        (gi,
         [&](auto&& graph)
         {
             return add_new_vertex()
                 (std::forward<decltype(graph)>(graph), gi, n, v);
         })();
    return v;
}


void remove_vertex_array(GraphInterface& gi, const python::object& oindex, bool fast)
{
    boost::multi_array_ref<int64_t,1> index = get_array<int64_t,1>(oindex);
    auto& g = gi.get_graph();
    if (fast)
    {
        for (auto v : index)
            remove_vertex_fast(vertex(v, g), g);
    }
    else
    {
        for (auto v : index)
            remove_vertex(vertex(v, g), g);
    }
}

void remove_vertex(GraphInterface& gi, size_t v, bool fast)
{
    auto& g = gi.get_graph();
    if (fast)
    {
        remove_vertex_fast(vertex(v, g), g);
    }
    else
    {
        remove_vertex(vertex(v, g), g);
    }
}

struct do_clear_vertex
{
    template <class Graph>
    void operator()(Graph& g, size_t v) const
    {
        clear_vertex(vertex(v, g), g);
    }
};

void clear_vertex(GraphInterface& gi, size_t v)
{
    run_action<>()
        (gi,
         [&](auto&& graph)
         {
             return do_clear_vertex()
                 (std::forward<decltype(graph)>(graph), v);
         })();
}

struct add_new_edge
{
    template <class Graph>
    void operator()(Graph& g, GraphInterface& gi, size_t s, size_t t,
                    python::object& new_e) const
    {
        auto gp = retrieve_graph_view<Graph>(gi, g);
        auto e = add_edge(vertex(s, g), vertex(t, g), g).first;
        new_e = python::object(PythonEdge<Graph>(gp, e));
    }
};

python::object add_edge(GraphInterface& gi, size_t s, size_t t)
{
    python::object new_e;
    run_action<>()
        (gi,
         [&](auto&& graph)
         {
             return add_new_edge()
                 (std::forward<decltype(graph)>(graph), gi, s, t, new_e);
         })();
    return new_e;
}

void remove_edge(GraphInterface& gi, EdgeBase& e)
{
    e.check_valid();
    auto edge = e.get_descriptor();
    run_action<>()(gi, [&](auto& g) { remove_edge(edge, g); })();
    e.invalidate();
}

struct get_edge_dispatch
{
    template <class Graph>
    void operator()(Graph& g, GraphInterface& gi, size_t s, size_t t,
                    bool all_edges, boost::python::list& es) const
    {
        auto gp = retrieve_graph_view<Graph>(gi, g);
        size_t k_t = graph_tool::is_directed(g) ?
            in_degreeS()(t, g) : out_degree(t, g);
        if (out_degree(s, g) <= k_t)
        {
            for (auto e : out_edges_range(vertex(s, g), g))
            {
                if (target(e, g) == vertex(t, g))
                {
                    es.append(PythonEdge<Graph>(gp, e));
                    if (!all_edges)
                        break;
                }
            }
        }
        else
        {
            for (auto e : in_or_out_edges_range(vertex(t, g), g))
            {
                auto w = source(e, g);
                if (w == vertex(s, g))
                {
                    if (!graph_tool::is_directed(g) && e.s != s)
                        std::swap(e.s, e.t);
                    es.append(PythonEdge<Graph>(gp, e));
                    if (!all_edges)
                        break;
                }
            }
        }
    }
};

python::object get_edge(GraphInterface& gi, size_t s, size_t t, bool all_edges)
{
    python::list es;
    run_action<>()
        (gi,
         [&](auto&& graph)
         {
             return get_edge_dispatch()
                 (std::forward<decltype(graph)>(graph), gi, s, t, all_edges,
                  es);
         })();
    return std::move(es);
}


struct get_degree_map
{
    template <class Graph, class DegS, class Weight>
    void operator()(const Graph& g, python::object& odeg_map, DegS deg, Weight weight) const
    {
        typedef typename detail::get_weight_type<Weight>::type weight_t;
        typedef typename mpl::if_<std::is_same<weight_t, size_t>, int32_t, weight_t>::type deg_t;

        typedef typename vprop_map_t<deg_t>::type map_t;

        map_t cdeg_map(get(vertex_index, g));
        typename map_t::unchecked_t deg_map = cdeg_map.get_unchecked(num_vertices(g));

        parallel_vertex_loop
            (g,
             [&](auto v)
             {
                 deg_map[v] = deg(v, g, weight);
             });

        odeg_map = python::object(PythonPropertyMap<map_t>(cdeg_map));
    }
};

python::object GraphInterface::degree_map(string deg, boost::any weight) const
{

    python::object deg_map;

    typedef mpl::push_back<edge_scalar_properties,
                           detail::no_weightS>::type weight_t;
    if (weight.empty())
        weight = detail::no_weightS();

    if (deg == "in")
        run_action<>()
            (const_cast<GraphInterface&>(*this),
             [&](auto&& graph, auto&& a2)
             {
                 return get_degree_map()
                     (std::forward<decltype(graph)>(graph), deg_map,
                      in_degreeS(), std::forward<decltype(a2)>(a2));
             },
             weight_t())(weight);
    else if (deg == "out")
        run_action<>()
            (const_cast<GraphInterface&>(*this),
             [&](auto&& graph, auto&& a2)
             {
                 return get_degree_map()
                     (std::forward<decltype(graph)>(graph), deg_map,
                      out_degreeS(), std::forward<decltype(a2)>(a2));
             },
             weight_t())(weight);
    else if (deg == "total")
        run_action<>()
            (const_cast<GraphInterface&>(*this),
             [&](auto&& graph, auto&& a2)
             {
                 return get_degree_map()
                     (std::forward<decltype(graph)>(graph), deg_map,
                      total_degreeS(), std::forward<decltype(a2)>(a2));
             },
             weight_t())(weight);
    return deg_map;
}

template <class PMaps>
int value_type_promotion(std::vector<boost::any>& props)
{
    int type_pos = boost::mpl::find<value_types,int64_t>::type::pos::value;
    for (auto& aep : props)
    {
        gt_dispatch<>()([&](auto& ep)
                        {
                            typedef std::remove_reference_t<decltype(ep)> pmap_t;
                            typedef typename property_traits<pmap_t>::value_type val_t;
                            if (std::is_same<val_t, size_t>::value)
                                return;
                            constexpr int ep_t = boost::mpl::find<value_types,val_t>::type::pos::value;
                            type_pos = std::max(type_pos, ep_t);
                        },
                        PMaps())(aep);
    }
    return type_pos;
}

template <int kind>
python::object get_vertex_list(GraphInterface& gi, size_t v,
                               python::list ovprops)
{
    std::vector<boost::any> avprops;
    for (int i = 0; i < python::len(ovprops); ++i)
    {
        avprops.push_back(python::extract<boost::any>(ovprops[i])());
        if (!belongs<vertex_scalar_properties>()(avprops.back()))
            throw ValueException("vertex property map must be of scalar type");
    }

    int vtype = boost::mpl::find<value_types,int64_t>::type::pos::value;
    if (!avprops.empty())
        vtype = value_type_promotion<vertex_scalar_properties>(avprops);

    python::object ret;
    auto dispatch =
        [&](auto&& vrange)
        {
            mpl::for_each<scalar_types>(
                [&](auto t)
                {
                    typedef decltype(t) t_t;
                    if (vtype != boost::mpl::find<value_types, t_t>::type::pos::value)
                        return;
                    typedef DynamicPropertyMapWrap<t_t, GraphInterface::vertex_t>
                        converted_map_t;
                    std::vector<converted_map_t> vprops;
                    for (auto& aep: avprops)
                        vprops.emplace_back(aep, vertex_scalar_properties());

                    std::vector<t_t> vlist;
                    run_action<>()(gi,
                                   [&](auto& g)
                                   {
                                       for (auto u: vrange(g))
                                       {
                                           vlist.push_back(u);
                                           for (auto& vp : vprops)
                                               vlist.push_back(get(vp, u));
                                       }
                                   })();
                    ret = wrap_vector_owned(vlist);
                });
        };

    switch (kind)
    {
    case 0:
        dispatch([&](auto& g){return vertices_range(g);});
        break;
    case 1:
        dispatch([&](auto& g){return out_neighbors_range(v, g);});
        break;
    case 2:
        dispatch([&](auto& g){return in_neighbors_range(v, g);});
        break;
    case 3:
        dispatch([&](auto& g){return all_neighbors_range(v, g);});
    }
    return ret;
}

enum range_t { FULL, OUT, IN, ALL };

template <int kind>
python::object get_vertex_iter(GraphInterface& gi, int v, python::list ovprops)
{
#ifdef HAVE_BOOST_COROUTINE
    auto dispatch = [&](auto&&vrange)
        {
            auto yield_dispatch = [&](auto& yield)
                {
                    if (python::len(ovprops) == 0)
                    {
                        run_action<>()(gi,
                                       [&](auto& g)
                                       {
                                           for (auto v: vertices_range(g))
                                               yield(python::object(v));
                                       })();
                    }
                    else
                    {
                        typedef DynamicPropertyMapWrap<python::object,
                                                       GraphInterface::vertex_t>
                            converted_map_t;

                        std::vector<converted_map_t> vprops;
                        for (int i = 0; i < python::len(ovprops); ++i)
                            vprops.emplace_back(python::extract<boost::any>(ovprops[i])(),
                                                vertex_properties());
                        run_action<>()(gi,
                                       [&](auto& g)
                                       {
                                           for (auto v: vrange(g))
                                           {
                                               python::list vlist;
                                               vlist.append(python::object(v));
                                               for (auto& vp : vprops)
                                                   vlist.append(get(vp, v));
                                               yield(vlist);
                                           }
                                       })();
                    }
                };
            return boost::python::object(CoroGenerator(yield_dispatch));
        };
    switch (kind)
    {
    case range_t::FULL:
        return dispatch([&](auto& g){return vertices_range(g);});
    case range_t::OUT:
        return dispatch([&](auto& g){return out_neighbors_range(v, g);});
    case range_t::IN:
        return dispatch([&](auto& g){return in_neighbors_range(v, g);});
    case range_t::ALL:
        return dispatch([&](auto& g){return all_neighbors_range(v, g);});
    }
#else
    throw GraphException("This functionality is not available because boost::coroutine was not found at compile-time");
#endif
}

template <int kind>
python::object get_edge_list(GraphInterface& gi, size_t v, python::list oeprops)
{
    std::vector<boost::any> aeprops;
    for (int i = 0; i < python::len(oeprops); ++i)
    {
        aeprops.push_back(python::extract<boost::any>(oeprops[i])());
        if (!belongs<edge_scalar_properties>()(aeprops.back()))
            throw ValueException("edge property map must be of scalar type");
    }

    int etype = boost::mpl::find<value_types,int64_t>::type::pos::value;
    if (!aeprops.empty())
        etype = value_type_promotion<edge_scalar_properties>(aeprops);

    python::object ret;
    auto dispatch =
        [&](auto&& erange)
        {
            mpl::for_each<scalar_types>(
                [&](auto t)
                {
                    typedef decltype(t) t_t;
                    if (etype != boost::mpl::find<value_types, t_t>::type::pos::value)
                        return;
                    typedef DynamicPropertyMapWrap<t_t, GraphInterface::edge_t>
                        converted_map_t;
                    std::vector<converted_map_t> eprops;
                    for (auto& aep: aeprops)
                        eprops.emplace_back(aep, edge_scalar_properties());

                    std::vector<t_t> elist;
                    run_action<>()(gi,
                                   [&](auto& g)
                                   {
                                       for (auto e: erange(g))
                                       {
                                           elist.push_back(source(e, g));
                                           elist.push_back(target(e, g));
                                           for (auto& ep : eprops)
                                               elist.push_back(get(ep,e));
                                       }
                                   })();
                    ret = wrap_vector_owned(elist);
                });
        };
    switch (kind)
    {
    case range_t::FULL:
        dispatch([&](auto& g){return edges_range(g);});
        break;
    case range_t::OUT:
        dispatch([&](auto& g){return out_edges_range(v, g);});
        break;
    case range_t::IN:
        dispatch([&](auto& g){return in_edges_range(v, g);});
        break;
    case range_t::ALL:
        dispatch([&](auto& g){return all_edges_range(v, g);});
    }
    return ret;
}

template <int kind>
python::object get_edge_iter(GraphInterface& gi, size_t v, python::list oeprops)
{
#ifdef HAVE_BOOST_COROUTINE
   auto dispatch = [&](auto&&erange)
        {
            auto yield_dispatch = [&](auto& yield)
                {
                    typedef DynamicPropertyMapWrap<python::object,
                                                   GraphInterface::edge_t>
                        converted_map_t;

                    std::vector<converted_map_t> eprops;
                    for (int i = 0; i < python::len(oeprops); ++i)
                        eprops.emplace_back(python::extract<boost::any>(oeprops[i])(),
                                            edge_properties());
                    run_action<>()(gi,
                                   [&](auto& g)
                                   {
                                       for (auto e: erange(g))
                                       {
                                           python::list elist;
                                           elist.append(python::object(source(e, g)));
                                           elist.append(python::object(target(e, g)));
                                           for (auto& ep : eprops)
                                               elist.append(get(ep, e));
                                           yield(elist);
                                       }
                                   })();
                };
            return boost::python::object(CoroGenerator(yield_dispatch));
        };
   switch (kind)
    {
    case range_t::FULL:
        return dispatch([&](auto& g){return edges_range(g);});
    case range_t::OUT:
        return dispatch([&](auto& g){return out_edges_range(v, g);});
    case range_t::IN:
        return dispatch([&](auto& g){return in_edges_range(v, g);});
    case range_t::ALL:
        return dispatch([&](auto& g){return all_edges_range(v, g);});
    }
#else
    throw GraphException("This functionality is not available because boost::coroutine was not found at compile-time");
#endif
}

python::object get_degree_list(GraphInterface& gi, python::object ovlist,
                               boost::any eprop, int kind)
{
    python::object ret;
    auto vlist = get_array<uint64_t,1>(ovlist);

    typedef UnityPropertyMap<size_t,
                             graph_traits<GraphInterface::multigraph_t>::edge_descriptor>
        empty_t;
    if (eprop.empty())
    {
        eprop = empty_t();
    }
    else
    {
        if (!belongs<edge_scalar_properties>()(eprop))
            throw ValueException("edge weight property map must be of scalar type");
    }

    typedef mpl::push_back<edge_scalar_properties,
                           empty_t>::type eprops_t;

    auto get_degs = [&](auto deg)
        {
            run_action<>()(gi,
                           [&](auto& g, auto& ew)
                           {
                               typedef typename std::remove_reference
                                   <decltype(ew)>::type::value_type val_t;
                               std::vector<val_t> dlist;
                               dlist.reserve(vlist.size());
                               for (auto v : vlist)
                               {
                                   if (!is_valid_vertex(v, g))
                                       throw ValueException("invalid vertex: " +
                                                            lexical_cast<string>(v));
                                   dlist.push_back(val_t(deg(v, g, ew)));
                               }
                               ret = wrap_vector_owned(dlist);
                           }, eprops_t())(eprop);
        };

    switch (kind)
    {
    case 0:
        get_degs(out_degreeS());
        break;
    case 1:
        get_degs(in_degreeS());
        break;
    case 2:
        get_degs(total_degreeS());
        break;
    }
    return ret;
}

//
// Below are the functions with will properly register all the types to python,
// for every filter, type, etc.
//

// this will register all the Vertex/Edge classes to python
struct export_python_interface
{
    template <class Graph, class GraphViews>
    void operator()(Graph* gp, python::list vclasses,
                    python::list eclasses, GraphViews) const
    {
        using namespace boost::python;

        class_<PythonVertex<Graph>, bases<VertexBase>> vclass("Vertex", no_init);
        vclass
            .def("__in_degree", &PythonVertex<Graph>::get_in_degree,
                 "Return the in-degree.")
            .def("__weighted_in_degree", &PythonVertex<Graph>::get_weighted_in_degree,
                 "Return the weighted in-degree.")
            .def("__out_degree", &PythonVertex<Graph>::get_out_degree,
                 "Return the out-degree.")
            .def("__weighted_out_degree", &PythonVertex<Graph>::get_weighted_out_degree,
                 "Return the weighted out-degree.")
            .def("in_edges", &PythonVertex<Graph>::in_edges,
                 "Return an iterator over the in-edges.")
            .def("out_edges", &PythonVertex<Graph>::out_edges,
                 "Return an iterator over the out-edges.")
            .def("is_valid", &PythonVertex<Graph>::is_valid,
                 "Return whether the vertex is valid.")
            .def("graph_ptr", &PythonVertex<Graph>::get_graph_ptr)
            .def("graph_type", &PythonVertex<Graph>::get_graph_type)
            .def("__str__", &PythonVertex<Graph>::get_string)
            .def("__int__", &PythonVertex<Graph>::get_index)
            .def("__hash__", &PythonVertex<Graph>::get_hash);

        vclasses.append(vclass);

        class_<PythonEdge<Graph>, bases<EdgeBase>> eclass("Edge", no_init);
        eclass
            .def("source", &PythonEdge<Graph>::get_source,
                 "Return the source vertex.")
            .def("target", &PythonEdge<Graph>::get_target,
                 "Return the target vertex.")
            .def("is_valid", &PythonEdge<Graph>::is_valid,
                 "Return whether the edge is valid.")
            .def("graph_ptr", &PythonEdge<Graph>::get_graph_ptr)
            .def("graph_type", &PythonEdge<Graph>::get_graph_type)
            .def("__str__", &PythonEdge<Graph>::get_string)
            .def("__hash__", &PythonEdge<Graph>::get_hash);

        boost::mpl::for_each<GraphViews>(
            [&](auto&& graph)
            {
                return export_python_interface()
                    (gp, std::forward<decltype(graph)>(graph), eclass);
            });

        eclasses.append(eclass);

        typedef typename graph_traits<Graph>::vertex_iterator vertex_iterator;
        class_<PythonIterator<Graph, PythonVertex<Graph>, vertex_iterator> >
            ("VertexIterator", no_init)
            .def("__iter__", objects::identity_function())
            .def("__next__", &PythonIterator<Graph, PythonVertex<Graph>,
                                             vertex_iterator>::next)
            .def("next", &PythonIterator<Graph, PythonVertex<Graph>,
                                         vertex_iterator>::next);

        typedef typename graph_traits<Graph>::edge_iterator edge_iterator;
        class_<PythonIterator<Graph, PythonEdge<Graph>,
                              edge_iterator> >("EdgeIterator", no_init)
            .def("__iter__", objects::identity_function())
            .def("__next__", &PythonIterator<Graph, PythonEdge<Graph>,
                                             edge_iterator>::next)
            .def("next", &PythonIterator<Graph, PythonEdge<Graph>,
                                         edge_iterator>::next);

        typedef typename graph_traits<Graph>::out_edge_iterator
            out_edge_iterator;
        class_<PythonIterator<Graph, PythonEdge<Graph>,
                              out_edge_iterator> >("OutEdgeIterator", no_init)
            .def("__iter__", objects::identity_function())
            .def("__next__", &PythonIterator<Graph, PythonEdge<Graph>,
                                             out_edge_iterator>::next)
            .def("next", &PythonIterator<Graph, PythonEdge<Graph>,
                                         out_edge_iterator>::next);

        typedef typename graph_traits<Graph>::directed_category
            directed_category;
        typedef typename std::is_convertible<directed_category,
                                             boost::directed_tag>::type is_directed;
        if (is_directed::value)
        {
            typedef typename in_edge_iteratorS<Graph>::type in_edge_iterator;
            class_<PythonIterator<Graph, PythonEdge<Graph>,
                                  in_edge_iterator> >("InEdgeIterator", no_init)
                .def("__iter__", objects::identity_function())
                .def("__next__", &PythonIterator<Graph, PythonEdge<Graph>,
                                                 in_edge_iterator>::next)
                .def("next", &PythonIterator<Graph, PythonEdge<Graph>,
                                             in_edge_iterator>::next);
        }
    }

    template <class Graph, class OGraph, class Eclass>
    void operator()(Graph*, OGraph*, Eclass& eclass) const
    {
        std::function<bool(const PythonEdge<Graph>&,
                           const PythonEdge<OGraph>&)> eq =
            [] (const PythonEdge<Graph>& e1,
                const PythonEdge<OGraph>& e2) -> bool { return e1 == e2; };
        std::function<bool(const PythonEdge<Graph>& e1,
                           const PythonEdge<OGraph>&)> ne =
            [] (const PythonEdge<Graph>& e1,
                const PythonEdge<OGraph>& e2) -> bool { return e1 != e2; };
        std::function<bool(const PythonEdge<Graph>&,
                           const PythonEdge<OGraph>&)> gt =
            [] (const PythonEdge<Graph>& e1,
                const PythonEdge<OGraph>& e2) -> bool { return e1 > e2; };
        std::function<bool(const PythonEdge<Graph>&,
                           const PythonEdge<OGraph>&)> lt =
            [] (const PythonEdge<Graph>& e1,
                const PythonEdge<OGraph>& e2) -> bool { return e1 < e2; };
        std::function<bool(const PythonEdge<Graph>&,
                           const PythonEdge<OGraph>&)> ge =
            [] (const PythonEdge<Graph>& e1,
                const PythonEdge<OGraph>& e2) -> bool { return e1 >= e2; };
        std::function<bool(const PythonEdge<Graph>&,
                           const PythonEdge<OGraph>&)> le =
            [] (const PythonEdge<Graph>& e1,
                const PythonEdge<OGraph>& e2) -> bool { return e1 <= e2; };

        eclass
            .def("__eq__", eq)
            .def("__ne__", ne)
            .def("__lt__", lt)
            .def("__gt__", gt)
            .def("__le__", le)
            .def("__ge__", ge);
    }
};

PythonPropertyMap<GraphInterface::vertex_index_map_t>
get_vertex_index(GraphInterface& g)
{
    return PythonPropertyMap<GraphInterface::vertex_index_map_t>
        (g.get_vertex_index());
}

PythonPropertyMap<GraphInterface::edge_index_map_t>
do_get_edge_index(GraphInterface& g)
{
    return PythonPropertyMap<GraphInterface::edge_index_map_t>
        (g.get_edge_index());
}

void do_add_edge_list(GraphInterface& gi, python::object aedge_list,
                      python::object eprops);

void do_add_edge_list_hashed(GraphInterface& gi, python::object aedge_list,
                             boost::any& vertex_map,
                             python::object eprops);

void do_add_edge_list_iter(GraphInterface& gi, python::object edge_list,
                           python::object eprops);

bool hasattr(boost::python::object obj, std::string const& attrName)
{
    return PyObject_HasAttrString(obj.ptr(), attrName.c_str());
}

} // namespace graph_tool

// register everything

void export_python_properties();

python::list* _vlist(0);
python::list* _elist(0);

python::list get_vlist()
{
    if (_vlist == nullptr)
        _vlist = new python::list();
    return *_vlist;
}

python::list get_elist()
{
    if (_elist == nullptr)
        _elist = new python::list();
    return *_elist;
}

void export_python_interface()
{
    using namespace boost::python;

    class_<VertexBase>("VertexBase", no_init);
    class_<EdgeBase, boost::noncopyable>("EdgeBase", no_init);

    typedef boost::mpl::transform<graph_tool::all_graph_views,
                                  boost::mpl::quote1<std::add_const> >::type const_graph_views;
    typedef boost::mpl::transform<graph_tool::all_graph_views,
                                  boost::mpl::quote1<std::add_pointer> >::type all_graph_views;
    typedef boost::mpl::transform<const_graph_views,
                                  boost::mpl::quote1<std::add_pointer> >::type all_const_graph_views;
    typedef boost::mpl::joint_view<all_graph_views, all_const_graph_views>::type graph_views;
    boost::mpl::for_each<graph_views>(std::bind(graph_tool::export_python_interface(),
                                                std::placeholders::_1, get_vlist(),
                                                get_elist(), graph_views()));
    export_python_properties();
    def("new_vertex_property",
        &new_property<GraphInterface::vertex_index_map_t>);
    def("new_edge_property", &new_property<GraphInterface::edge_index_map_t>);
    def("new_graph_property",
        &new_property<ConstantPropertyMap<size_t,graph_property_tag> >);

    def("get_vertex", get_vertex);
    def("get_vertices", get_vertices);
    def("get_edges", get_edges);
    def("add_vertex", graph_tool::add_vertex);
    def("add_edge", graph_tool::add_edge);
    def("remove_vertex", graph_tool::remove_vertex);
    def("remove_vertex_array", graph_tool::remove_vertex_array);
    def("clear_vertex", graph_tool::clear_vertex);
    def("remove_edge", graph_tool::remove_edge);
    def("add_edge_list", graph_tool::do_add_edge_list);
    def("add_edge_list_hashed", graph_tool::do_add_edge_list_hashed);
    def("add_edge_list_iter", graph_tool::do_add_edge_list_iter);
    def("get_edge", get_edge);

    def("get_vertex_list", &get_vertex_list<range_t::FULL>);
    def("get_vertex_iter", &get_vertex_iter<range_t::FULL>);
    def("get_edge_list", &get_edge_list<range_t::FULL>);
    def("get_edge_iter", &get_edge_iter<range_t::FULL>);
    def("get_out_edge_list", &get_edge_list<range_t::OUT>);
    def("get_out_edge_iter", &get_edge_iter<range_t::OUT>);
    def("get_in_edge_list", &get_edge_list<range_t::IN>);
    def("get_in_edge_iter", &get_edge_iter<range_t::IN>);
    def("get_all_edge_list", &get_edge_list<range_t::ALL>);
    def("get_all_edge_iter", &get_edge_iter<range_t::ALL>);
    def("get_out_neighbors_list", &get_vertex_list<range_t::OUT>);
    def("get_out_neighbors_iter", &get_vertex_iter<range_t::OUT>);
    def("get_in_neighbors_list", &get_vertex_list<range_t::IN>);
    def("get_in_neighbors_iter", &get_vertex_iter<range_t::IN>);
    def("get_all_neighbors_list", &get_vertex_list<range_t::ALL>);
    def("get_all_neighbors_iter", &get_vertex_iter<range_t::ALL>);
    def("get_degree_list", get_degree_list);

    def("get_vertex_index", get_vertex_index);
    def("get_edge_index", do_get_edge_index);

    def("get_vlist", get_vlist);
    def("get_elist", get_elist);

#ifdef HAVE_BOOST_COROUTINE
    class_<CoroGenerator>("CoroGenerator", no_init)
        .def("__iter__", objects::identity_function())
        .def("next", &CoroGenerator::next)
        .def("__next__", &CoroGenerator::next);
#endif
}
