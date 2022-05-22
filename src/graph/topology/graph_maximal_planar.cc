// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2022 Tiago de Paula Peixoto <tiago@skewed.de>
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

#include <boost/graph/make_maximal_planar.hpp>
#include <boost/graph/make_biconnected_planar.hpp>
#include <boost/graph/boyer_myrvold_planar_test.hpp>

using namespace std;
using namespace boost;
using namespace graph_tool;

struct mark_planar_edge
{
    template <typename Graph, typename Vertex>
    void visit_vertex_pair(Vertex u, Vertex v, Graph& g)
    {
        if (!is_adjacent(u, v, g))
            add_edge(u, v, g);
    }
};

struct do_maximal_planar
{
    typedef typename eprop_map_t<size_t>::type::unchecked_t eimap_t;
    typedef typename vprop_map_t<size_t>::type::unchecked_t vimap_t;

    template <class Graph>
    eimap_t get_edge_index(const Graph& g) const
    {
        eimap_t::checked_t eidx;
        size_t E = 0;
        for (auto e : edges_range(g))
            eidx[e] = E++;
        return eidx.get_unchecked();
    }

    template <class Graph>
    void operator()(Graph& g) const
    {

        typename vprop_map_t<vector<typename graph_traits<Graph>::edge_descriptor>>::type::unchecked_t
            embedding(num_vertices(g));

        eimap_t edge_index = get_edge_index(g);

        bool is_planar = boyer_myrvold_planarity_test
            (boyer_myrvold_params::graph = g,
             boyer_myrvold_params::edge_index_map = edge_index,
             boyer_myrvold_params::embedding = embedding);

        if (!is_planar)
            throw GraphException("Graph is not planar!");

        mark_planar_edge vis;
        make_biconnected_planar(g, embedding, edge_index, vis);
        boyer_myrvold_planarity_test
            (boyer_myrvold_params::graph = g,
             boyer_myrvold_params::edge_index_map = edge_index,
             boyer_myrvold_params::embedding = embedding);
        make_maximal_planar(g, embedding, get(vertex_index, g), edge_index, vis);
    }

};


void maximal_planar(GraphInterface& gi)
{
    run_action<graph_tool::detail::never_directed, mpl::true_>()
        (gi,
         [&](auto&& graph)
         {
             return do_maximal_planar()
                 (std::forward<decltype(graph)>(graph));
         })();
}
