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

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_properties.hh"
#include "graph_selectors.hh"
#include "graph_python_interface.hh"
#include "numpy_bind.hh"
#include "hash_map_wrap.hh"
#include "coroutine.hh"
#include "graph_python_interface.hh"
#include "../generation/sampler.hh"

#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/dijkstra_shortest_paths_no_color_map.hpp>
#include <boost/graph/bellman_ford_shortest_paths.hpp>
#include <boost/graph/dag_shortest_paths.hpp>
#include <boost/python.hpp>

#if (BOOST_VERSION >= 106000)
# include <boost/math/special_functions/relative_difference.hpp>
#endif

using namespace std;
using namespace boost;
using namespace graph_tool;

struct stop_search {};

template <class DistMap, class PredMap>
class bfs_max_visitor:
    public boost::bfs_visitor<null_visitor>
{
public:
    bfs_max_visitor(DistMap dist_map, PredMap pred, size_t max_dist,
                    size_t source, size_t target, std::vector<size_t>& reached)
        : _dist_map(dist_map), _pred(pred), _max_dist(max_dist),
          _source(source), _target(target), _dist(0), _reached(reached) {}

    ~bfs_max_visitor()
    {
        typedef typename property_traits<DistMap>::value_type dist_t;
        constexpr dist_t inf = std::is_floating_point<dist_t>::value ?
            numeric_limits<dist_t>::infinity() :
            numeric_limits<dist_t>::max();
        for (auto v : _unreached)
            _dist_map[v] = inf;
    }

    template <class Graph>
    void tree_edge(typename graph_traits<Graph>::edge_descriptor e,
                   Graph& g)
    {
        _pred[target(e,g)] = source(e,g);
    }

    template <class Graph>
    void examine_vertex(typename graph_traits<Graph>::vertex_descriptor v,
                        Graph&)
    {
        if ( _dist_map[v] > _max_dist)
            throw stop_search();
    }

    template <class Graph>
    void discover_vertex(typename graph_traits<Graph>::vertex_descriptor v,
                         Graph&)
    {
        auto p = _pred[v];
        if (size_t(p) == v)
            return;
        _dist_map[v] = _dist_map[p] + 1;
        _reached.push_back(v);
        if (_dist_map[v] > _max_dist)
            _unreached.push_back(v);
        if (v == _target)
            throw stop_search();
    }

private:
    DistMap _dist_map;
    PredMap _pred;
    typename property_traits<DistMap>::value_type _max_dist;
    size_t _source;
    size_t _target;
    size_t _dist;
    std::vector<size_t> _unreached;
    std::vector<size_t>& _reached;
};

template <class DistMap, class PredMap>
class bfs_max_multiple_targets_visitor:
    public boost::bfs_visitor<null_visitor>
{
public:
    bfs_max_multiple_targets_visitor(DistMap dist_map, PredMap pred,
                                     size_t max_dist, size_t source,
                                     gt_hash_set<std::size_t> target,
                                     std::vector<size_t>& reached)
        : _dist_map(dist_map), _pred(pred), _max_dist(max_dist),
          _source(source), _target(target), _dist(0), _reached(reached) {}

    ~bfs_max_multiple_targets_visitor()
    {
        typedef typename property_traits<DistMap>::value_type dist_t;
        constexpr dist_t inf = std::is_floating_point<dist_t>::value ?
            numeric_limits<dist_t>::infinity() :
            numeric_limits<dist_t>::max();
        for (auto v : _unreached)
            _dist_map[v] = inf;
    }

    template <class Graph>
    void tree_edge(typename graph_traits<Graph>::edge_descriptor e,
                   Graph& g)
    {
        _pred[target(e,g)] = source(e,g);
    }

    template <class Graph>
    void examine_vertex(typename graph_traits<Graph>::vertex_descriptor v,
                        Graph&)
    {
        if ( _dist_map[v] > _max_dist)
            throw stop_search();
    }

    template <class Graph>
    void discover_vertex(typename graph_traits<Graph>::vertex_descriptor v,
                         Graph&)
    {
        auto p = _pred[v];
        if (size_t(p) == v)
            return;
        _dist_map[v] = _dist_map[p] + 1;

        if (_dist_map[v] > _max_dist)
            _unreached.push_back(v);

        auto iter = _target.find(v);
        if (iter != _target.end())
        {
            _target.erase(iter);
            if (_target.empty())
                throw stop_search();
        };
    }

private:
    DistMap _dist_map;
    PredMap _pred;
    typename property_traits<DistMap>::value_type _max_dist;
    size_t _source;
    gt_hash_set<std::size_t> _target;
    size_t _dist;
    std::vector<size_t> _unreached;
    std::vector<size_t>& _reached;
};


template <class DistMap>
class djk_max_visitor:
    public boost::dijkstra_visitor<null_visitor>
{
public:
    djk_max_visitor(DistMap dist_map,
                    typename property_traits<DistMap>::value_type max_dist,
                    typename property_traits<DistMap>::value_type inf,
                    size_t target, std::vector<size_t>& reached)
        : _dist_map(dist_map), _max_dist(max_dist), _inf(inf),
          _target(target), _reached(reached) {}

    ~djk_max_visitor()
    {
        for (auto v : _unreached)
            _dist_map[v] = _inf;
    }

    template <class Graph>
    void examine_vertex(typename graph_traits<Graph>::vertex_descriptor u,
                        Graph&)
    {
        if (_dist_map[u] > _max_dist)
            throw stop_search();

        if (u == _target)
            throw stop_search();
    }

    template <class Graph>
    void discover_vertex(typename graph_traits<Graph>::vertex_descriptor u,
                         Graph&)
    {
        if (_dist_map[u] > _max_dist)
            _unreached.push_back(u);
        _reached.push_back(u);
    }

private:
    DistMap _dist_map;
    typename property_traits<DistMap>::value_type _max_dist;
    typename property_traits<DistMap>::value_type _inf;
    size_t _target;
    std::vector<size_t> _unreached;
    std::vector<size_t>& _reached;
};


template <class DistMap>
class djk_max_multiple_targets_visitor:
    public boost::dijkstra_visitor<null_visitor>
{
public:
    djk_max_multiple_targets_visitor(DistMap dist_map,
                                     typename property_traits<DistMap>::value_type max_dist, 
                                     typename property_traits<DistMap>::value_type inf, 
                                     gt_hash_set<std::size_t> target,
                                     std::vector<size_t>& reached)
        : _dist_map(dist_map), _max_dist(max_dist), _inf(inf),
          _target(target), _reached(reached) {}

    ~djk_max_multiple_targets_visitor()
    {
        for (auto v : _unreached)
            _dist_map[v] = _inf;
    }

    template <class Graph>
    void examine_vertex(typename graph_traits<Graph>::vertex_descriptor u,
                        Graph&)
    {
        if (_dist_map[u] > _max_dist)
            throw stop_search();

        auto iter = _target.find(u);
        if (iter != _target.end())
        {
            _target.erase(iter);
            if (_target.empty())
                throw stop_search();
        };
    }

    template <class Graph>
    void discover_vertex(typename graph_traits<Graph>::vertex_descriptor u,
                         Graph&)
    {
        if (_dist_map[u] > _max_dist)
            _unreached.push_back(u);
        _reached.push_back(u);
    }

private:
    DistMap _dist_map;
    typename property_traits<DistMap>::value_type _max_dist;
    typename property_traits<DistMap>::value_type _inf;
    gt_hash_set<std::size_t> _target;
    std::vector<size_t> _unreached;
    std::vector<size_t>& _reached;
};

struct do_bfs_search
{
    template <class Graph, class VertexIndexMap, class DistMap, class PredMap>
    void operator()(const Graph& g, size_t source,
                    boost::python::object otarget_list,
                    VertexIndexMap vertex_index, DistMap dist_map,
                    PredMap pred_map, long double max_dist,
                    std::vector<size_t>& reached) const
    {
        typedef typename property_traits<DistMap>::value_type dist_t;

        auto target_list = get_array<int64_t, 1>(otarget_list);
        gt_hash_set<std::size_t> tgt(target_list.begin(),
                                     target_list.end());

        dist_t inf = std::is_floating_point<dist_t>::value ?
            numeric_limits<dist_t>::infinity() :
            numeric_limits<dist_t>::max();

        dist_t max_d = (max_dist > 0) ? max_dist : inf;

        dist_map[source] = 0;

        unchecked_vector_property_map<boost::default_color_type, VertexIndexMap>
        color_map(vertex_index, num_vertices(g));
        try
        {
            if (tgt.size() <= 1)
            {
                size_t target = tgt.empty() ?
                    graph_traits<GraphInterface::multigraph_t>::null_vertex() :
                    *tgt.begin();
                breadth_first_visit(g, vertex(source, g),
                                    visitor(bfs_max_visitor<DistMap, PredMap>
                                            (dist_map, pred_map, max_d,
                                             source, target, reached)).
                                    vertex_index_map(vertex_index).
                                    color_map(color_map));
            }
            else
            {
                breadth_first_visit(g, vertex(source, g),
                                    visitor(bfs_max_multiple_targets_visitor<DistMap, PredMap>
                                            (dist_map, pred_map, max_d,
                                             source, tgt, reached)).
                                    vertex_index_map(vertex_index).
                                    color_map(color_map));
            }

        }
        catch (stop_search&) {}
    }
};

struct do_djk_search
{
    template <class Graph, class VertexIndexMap, class DistMap, class PredMap,
              class WeightMap>
    void operator()(const Graph& g, size_t source,
                    boost::python::object otarget_list,
                    VertexIndexMap vertex_index, DistMap dist_map,
                    PredMap pred_map, WeightMap weight, long double max_dist,
                    std::vector<size_t>& reached, bool dag) const
    {
        auto target_list = get_array<int64_t, 1>(otarget_list);

        typedef typename property_traits<DistMap>::value_type dist_t;

        constexpr dist_t inf = (std::is_floating_point<dist_t>::value) ?
            numeric_limits<dist_t>::infinity() :
            numeric_limits<dist_t>::max();

        dist_t max_d = (max_dist > 0) ? max_dist : inf;

        gt_hash_set<std::size_t> tgt(target_list.begin(),
                                     target_list.end());

        dist_map[source] = 0;

        try
        {
            if (tgt.size() <= 1)
            {
                size_t target = tgt.empty() ?
                    graph_traits<GraphInterface::multigraph_t>::null_vertex() :
                    *tgt.begin();
                if (!dag)
                {
                    dijkstra_shortest_paths_no_color_map_no_init
                        (g, vertex(source, g), pred_map, dist_map, weight,
                         vertex_index, std::less<dist_t>(),
                         boost::closed_plus<dist_t>(), inf, dist_t(),
                         djk_max_visitor<DistMap>(dist_map, max_d, inf, target,
                                                  reached));
                }
                else
                {
                    unchecked_vector_property_map<boost::default_color_type, VertexIndexMap>
                        color_map(vertex_index, num_vertices(g));
                    dag_shortest_paths
                        (g, vertex(source, g), dist_map, weight,
                         color_map, pred_map,
                         djk_max_visitor<DistMap>(dist_map, max_d, inf, target,
                                                  reached),
                         std::less<dist_t>(),
                         boost::closed_plus<dist_t>(), inf, dist_t());
                }
            }
            else
            {
                if (!dag)
                {
                    dijkstra_shortest_paths_no_color_map_no_init
                        (g, vertex(source, g), pred_map, dist_map, weight,
                         vertex_index, std::less<dist_t>(),
                         boost::closed_plus<dist_t>(), inf, dist_t(),
                         djk_max_multiple_targets_visitor<DistMap>(dist_map,
                                                                   max_d, inf,
                                                                   tgt,
                                                                   reached));
                }
                else
                {
                    unchecked_vector_property_map<boost::default_color_type, VertexIndexMap>
                        color_map(vertex_index, num_vertices(g));
                    dag_shortest_paths
                        (g, vertex(source, g), dist_map, weight, color_map,
                         pred_map,
                         djk_max_multiple_targets_visitor<DistMap>(dist_map,
                                                                   max_d, inf,
                                                                   tgt,
                                                                   reached),
                         std::less<dist_t>(), boost::closed_plus<dist_t>(), inf,
                         dist_t());
                }
            }

        }
        catch (stop_search&) {}

    }
};

struct do_bf_search
{
    template <class Graph, class DistMap, class PredMap, class WeightMap>
    void operator()(const Graph& g, size_t source, DistMap dist_map,
                    PredMap pred_map, WeightMap weight) const
    {
        bool ret = bellman_ford_shortest_paths(g, root_vertex(source).
                                               predecessor_map(pred_map).
                                               distance_map(dist_map).
                                               weight_map(weight));
        if (!ret)
            throw ValueException("Graph contains negative loops");

        // consistency with dijkstra
        typedef typename property_traits<DistMap>::value_type dist_t;
        if (std::is_floating_point<dist_t>::value)
        {
            for (auto v : vertices_range(g))
            {
                if (dist_map[v] == numeric_limits<dist_t>::max())
                    dist_map[v] = numeric_limits<dist_t>::infinity();
            }
        }
    }
};

void get_dists(GraphInterface& gi, size_t source, boost::python::object tgt,
               boost::any dist_map, boost::any weight, boost::any pred_map,
               long double max_dist, bool bf, std::vector<size_t>& reached,
               bool dag)
{
    typedef property_map_type
        ::apply<int64_t, GraphInterface::vertex_index_map_t>::type pred_map_t;

    pred_map_t pmap = any_cast<pred_map_t>(pred_map);

    if (weight.empty())
    {
        run_action<>()
            (gi,
             [&](auto&& graph, auto&& a2)
             {
                 return do_bfs_search()
                     (std::forward<decltype(graph)>(graph), source, tgt,
                      gi.get_vertex_index(), std::forward<decltype(a2)>(a2),
                      pmap.get_unchecked(num_vertices(gi.get_graph())),
                      max_dist, reached);
             },
             writable_vertex_scalar_properties())(dist_map);
    }
    else
    {
        if (bf)
        {
            run_action<>()
                (gi,
                 [&](auto&& graph, auto&& a2, auto&& a3)
                 {
                     return do_bf_search()
                         (std::forward<decltype(graph)>(graph), source,
                          std::forward<decltype(a2)>(a2),
                          pmap.get_unchecked(num_vertices(gi.get_graph())),
                          std::forward<decltype(a3)>(a3));
                 },
                 writable_vertex_scalar_properties(),
                 edge_scalar_properties())(dist_map, weight);
        }
        else
        {
            run_action<>()
                (gi,
                 [&](auto&& graph, auto&& a2, auto&& a3)
                 {
                     return do_djk_search()
                         (std::forward<decltype(graph)>(graph), source, tgt,
                          gi.get_vertex_index(), std::forward<decltype(a2)>(a2),
                          pmap.get_unchecked(num_vertices(gi.get_graph())),
                          std::forward<decltype(a3)>(a3), max_dist, reached,
                          dag);
                 },
                 writable_vertex_scalar_properties(),
                 edge_scalar_properties())(dist_map, weight);
        }
    }
}

template <class Graph, class Dist, class Pred, class Weight, class Preds>
void get_all_preds(Graph g, Dist dist, Pred pred, Weight weight, Preds preds,
                   long double epsilon)
{
    parallel_vertex_loop
        (g,
         [&](auto v)
         {
            if (size_t(pred[v]) == v)
                return;
            auto d = dist[v];
            typedef decltype(d) dist_t;
            for (auto e : in_or_out_edges_range(v, g))
            {
                auto u = source(e, g);
                if (!std::is_floating_point<dist_t>::value)
                {
                    if (dist_t(dist[u] + get(weight, e)) == d)
                        preds[v].push_back(u);
                }
                else
                {
#if (BOOST_VERSION >= 106000)
                    if (boost::math::relative_difference(dist_t(dist[u] + get(weight, e)), d) < epsilon)
                        preds[v].push_back(u);
#else
                    if (abs(dist_t(dist[u] + get(weight, e)) - d) < epsilon)
                        preds[v].push_back(u);
#endif
                }
            }
         });
};

void do_get_all_preds(GraphInterface& gi, boost::any adist, boost::any apred,
                      boost::any aweight, boost::any apreds,
                      long double epsilon)
{
    typedef property_map_type
        ::apply<int64_t, GraphInterface::vertex_index_map_t>::type pred_map_t;
    typedef property_map_type
        ::apply<vector<int64_t>, GraphInterface::vertex_index_map_t>::type preds_map_t;

    pred_map_t pred = any_cast<pred_map_t>(apred);
    preds_map_t preds = any_cast<preds_map_t>(apreds);

    typedef UnityPropertyMap<int,GraphInterface::edge_t> weight_map_t;
    typedef boost::mpl::push_back<edge_scalar_properties, weight_map_t>::type
        weight_props_t;

    if (aweight.empty())
        aweight = weight_map_t();

    run_action<>()
        (gi, [&](auto& g, auto dist, auto weight)
             {get_all_preds(g, dist, pred.get_unchecked(num_vertices(g)),
                            weight, preds.get_unchecked(num_vertices(g)),
                            epsilon);},
         vertex_scalar_properties(), weight_props_t())(adist, aweight);
}


template <class Graph, class Pred, class Yield>
void get_all_shortest_paths(GraphInterface& gi, Graph& g, size_t s, size_t t,
                            Pred pred, bool edges, Yield& yield)
{
    vector<size_t> path;
    vector<pair<size_t, size_t>> stack = {{t, 0}};
    while (!stack.empty())
    {
        size_t v, i;
        std::tie(v, i) = stack.back();
        if (v == s)
        {
            if (!edges)
            {
                path.clear();
                for (auto iter = stack.rbegin(); iter != stack.rend(); ++iter)
                    path.push_back(iter->first);
                yield(wrap_vector_owned<size_t>(path));
            }
            else
            {
                auto gp = retrieve_graph_view<Graph>(gi, g);
                boost::python::list opath;
                auto s = graph_traits<Graph>::null_vertex();
                for (auto iter = stack.rbegin(); iter != stack.rend(); ++iter)
                {
                    auto t = iter->first;
                    if (s != graph_traits<Graph>::null_vertex())
                    {
                        auto e = edge(s, t, g).first;
                        opath.append(PythonEdge<Graph>(gp, e));
                    }
                    s = t;
                }
                yield(opath);
            }
        }
        if (pred[v].size() > i)
        {
            stack.emplace_back(pred[v][i], 0);
        }
        else
        {
            stack.pop_back();
            if (!stack.empty())
                ++stack.back().second;
        }
    }
};

python::object do_get_all_shortest_paths(GraphInterface& gi, size_t s, size_t t,
                                         boost::any apred, bool edges)
{
#ifdef HAVE_BOOST_COROUTINE
    auto dispatch = [&](auto& yield)
        {
            run_action<>()
                (gi, [&](auto& g, auto pred)
                     {get_all_shortest_paths(gi, g, s, t, pred, edges, yield);},
                 vertex_scalar_vector_properties())(apred);
        };
    return python::object(CoroGenerator(dispatch));
#else
    throw GraphException("This functionality is not available because boost::coroutine was not found at compile-time");
#endif // HAVE_BOOST_COROUTINE
}

void get_weighted_succs(size_t t, boost::any apred, boost::any asucc,
                        boost::any acount, boost::any avisited)
{
    typedef vprop_map_t<vector<int64_t>>::type pred_map_t;
    typedef vprop_map_t<int64_t>::type count_map_t;
    typedef vprop_map_t<uint8_t>::type visited_map_t;

    pred_map_t pred = any_cast<pred_map_t>(apred);
    pred_map_t succ = any_cast<pred_map_t>(asucc);
    count_map_t count = any_cast<count_map_t>(acount);
    visited_map_t visited = any_cast<visited_map_t>(avisited);

    count[t] = 1;
    visited[t] = true;

    deque<size_t> queue = {t};
    while (!queue.empty())
    {
        size_t v = queue.front();
        queue.pop_front();
        for (auto w : pred[v])
        {
            count[w] += count[v];
            succ[w].push_back(v);

            if (!visited[w])
            {
                visited[w] = true;
                queue.push_back(w);
            }
        }
    }
};

void get_random_shortest_path(size_t s, size_t t, boost::any asucc,
                              boost::any acount, vector<size_t>& path,
                              rng_t& rng)
{
    typedef vprop_map_t<vector<int64_t>>::type pred_map_t;
    typedef vprop_map_t<int64_t>::type count_map_t;

    pred_map_t succ = any_cast<pred_map_t>(asucc);
    count_map_t count = any_cast<count_map_t>(acount);

    vector<double> probs;
    path.clear();

    if (succ[s].empty())
        return;

    path.push_back(s);
    size_t v = s;
    while (v != t)
    {
        if (succ[v].size() == 1)
        {
            v = succ[v].front();
        }
        else
        {
            probs.clear();
            for (auto w : succ[v])
                probs.push_back(count[w]);
            Sampler<int64_t> next(succ[v], probs);
            v = next.sample(rng);
        }
        path.push_back(v);
    }
};

template <bool edges, class Graph, class Yield, class VMap>
void get_all_paths(size_t s, size_t t, size_t cutoff, VMap visited,
                   Yield& yield, Graph& g, GraphInterface& gi)
{
    typedef typename graph_traits<Graph>::out_edge_iterator eiter_t;
    typedef std::pair<eiter_t, eiter_t> item_t;

    visited[s] = true;
    vector<size_t> vs = {s};
    vector<item_t> stack = {out_edges(s, g)};
    while (!stack.empty())
    {
        auto& pos = stack.back();
        if (pos.first == pos.second || stack.size() > cutoff)
        {
            visited[vs.back()] = false;
            vs.pop_back();
            stack.pop_back();
            if (!stack.empty())
                ++stack.back().first;
            continue;
        }

        auto v = target(*pos.first, g);

        if (v == t)
        {
            if (!edges)
            {
                vector<size_t> path = {s};
                for (auto& ei : stack)
                    path.push_back(target(*ei.first, g));

                yield(wrap_vector_owned<size_t>(path));
            }
            else
            {
                auto gp = retrieve_graph_view<Graph>(gi, g);
                boost::python::list path;
                for (auto& ei : stack)
                    path.append(PythonEdge<Graph>(gp, *ei.first));
                yield(path);
            }

            ++pos.first;
        }
        else
        {
            if (!visited[v])
            {
                visited[v] = true;
                vs.push_back(v);
                stack.push_back(out_edges(v, g));
            }
            else
            {
                ++pos.first;
            }
        }
    }
};

python::object do_get_all_paths(GraphInterface& gi, size_t s, size_t t,
                                size_t cutoff, boost::any avisited, bool edges)
{
#ifdef HAVE_BOOST_COROUTINE
    typedef vprop_map_t<uint8_t>::type vprop_t;
    vprop_t visited = boost::any_cast<vprop_t>(avisited);
    auto dispatch = [&](auto& yield)
        {
            run_action<>()
                (gi, [&](auto& g)
                {
                    if (edges)
                        get_all_paths<true>(s, t, cutoff,
                                            visited.get_unchecked(), yield,
                                            g, gi);
                    else
                        get_all_paths<false>(s, t, cutoff,
                                             visited.get_unchecked(), yield,
                                             g, gi);
                })();
        };
    return python::object(CoroGenerator(dispatch));
#else
    throw GraphException("This functionality is not available because boost::coroutine was not found at compile-time");
#endif // HAVE_BOOST_COROUTINE
}

void export_dists()
{
    python::def("get_dists", &get_dists);
    python::def("get_all_preds", &do_get_all_preds);
    python::def("get_all_shortest_paths", &do_get_all_shortest_paths);
    python::def("get_all_paths", &do_get_all_paths);

    python::def("get_weighted_succs", &get_weighted_succs);
    python::def("get_random_shortest_path", &get_random_shortest_path);
};
