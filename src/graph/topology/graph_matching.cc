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
#include "graph_properties.hh"

#include <boost/graph/max_cardinality_matching.hpp>
#include <boost/graph/maximum_weighted_matching.hpp>

#include "graph_bipartite_weighted_matching.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;


void get_max_matching(GraphInterface& gi, std::string initial_matching,
                      boost::any omatching)
{
    typedef typename vprop_map_t<int64_t>::type vprop_t;

    vprop_t::unchecked_t matching = any_cast<vprop_t>(omatching).get_unchecked();

    run_action<graph_tool::detail::never_directed>()
        (gi,
         [&](auto& g)
         {
             auto vindex = get(vertex_index, g);
             typedef decltype(vindex) vindex_t;
             typedef std::remove_reference_t<decltype(g)> g_t;

             if (initial_matching == "empty")
                 boost::matching<g_t, vprop_t::unchecked_t, vindex_t,
                                 edmonds_augmenting_path_finder, empty_matching,
                                 no_matching_verifier>
                     (g, matching, vindex);
             else if (initial_matching == "greedy")
                 boost::matching<g_t, vprop_t::unchecked_t, vindex_t,
                                 edmonds_augmenting_path_finder, greedy_matching,
                                 no_matching_verifier>
                     (g, matching, vindex);
             else if (initial_matching == "extra_greedy")
                 boost::matching<g_t, vprop_t::unchecked_t, vindex_t,
                                 edmonds_augmenting_path_finder, extra_greedy_matching,
                                 no_matching_verifier>
                     (g, matching, vindex);
             else
                 throw ValueException("invalid initial matching: " +
                                      initial_matching);

             for (auto v : vertices_range(g))
             {
                 if (matching[v] == int64_t(graph_traits<g_t>::null_vertex()))
                     matching[v] = std::numeric_limits<int64_t>::max();
             }

         })();
}

void get_max_weighted_matching(GraphInterface& gi, boost::any oweight,
                               boost::any omatching, bool brute_force)
{
    typedef typename vprop_map_t<int64_t>::type vprop_t;

    vprop_t::unchecked_t matching = any_cast<vprop_t>(omatching).get_unchecked();

    run_action<graph_tool::detail::never_directed>()
        (gi,
         [&](auto& g, auto w)
         {
             typedef std::remove_reference_t<decltype(g)> g_t;

             typedef typename graph_traits<g_t>::vertex_descriptor vertex_t;
             typename vprop_map_t<vertex_t>::type match(get(vertex_index,g));

             if (brute_force)
                 brute_force_maximum_weighted_matching(g, w, match);
             else
                 maximum_weighted_matching(g, w, match);

             for (auto v : vertices_range(g))
             {
                 if (match[v] == graph_traits<g_t>::null_vertex())
                     matching[v] = std::numeric_limits<int64_t>::max();
                 else
                     matching[v] = match[v];
             }
         },
         edge_scalar_properties())(oweight);
}

void get_max_bip_weighted_matching(GraphInterface& gi, boost::any opartition,
                                   boost::any oweight, boost::any omatching)
{
    typedef typename vprop_map_t<int64_t>::type vprop_t;

    vprop_t::unchecked_t matching = any_cast<vprop_t>(omatching).get_unchecked();

    typedef UnityPropertyMap<int, GraphInterface::edge_t> ecmap_t;
    typedef boost::mpl::push_back<edge_scalar_properties, ecmap_t>::type
        weight_props_t;

    if (oweight.empty())
        oweight = ecmap_t();

    run_action<graph_tool::detail::never_directed>()
        (gi,
         [&](auto& g, auto part, auto w)
         {
             typedef std::remove_reference_t<decltype(g)> g_t;

             typedef typename graph_traits<g_t>::vertex_descriptor vertex_t;
             typename vprop_map_t<vertex_t>::type match(get(vertex_index,g));

             maximum_bipartite_weighted_matching(g, part, w, match);

             for (auto v : vertices_range(g))
             {
                 if (match[v] == graph_traits<g_t>::null_vertex())
                     matching[v] = std::numeric_limits<int64_t>::max();
                 else
                     matching[v] = match[v];
             }
         },
         vertex_properties(), weight_props_t())(opartition, oweight);
}

void match_edges(GraphInterface& gi, boost::any omatching,
                 boost::any oematching)
{
    typedef typename vprop_map_t<int64_t>::type vprop_t;
    typedef typename eprop_map_t<uint8_t>::type eprop_t;

    vprop_t::unchecked_t matching = any_cast<vprop_t>(omatching).get_unchecked();
    eprop_t::unchecked_t ematching = any_cast<eprop_t>(oematching).get_unchecked();

    run_action<graph_tool::detail::never_directed>()
        (gi,
         [&](auto& g)
         {
             for (auto v : vertices_range(g))
             {
                 auto u = matching[v];
                 if (size_t(u) > num_vertices(g))
                     continue;
                 ematching[edge(v, u, g).first] = true;
             }
         })();
}

#include <boost/python.hpp>
using namespace boost::python;

void export_matching()
{
    def("get_max_matching", &get_max_matching);
    def("get_max_weighted_matching", &get_max_weighted_matching);
    def("get_max_bip_weighted_matching", &get_max_bip_weighted_matching);
    def("match_edges", &match_edges);
}
