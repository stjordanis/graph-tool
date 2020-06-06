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

#ifndef GRAPH_NONBACKTRACKING_MATRIX_HH
#define GRAPH_NONBACKTRACKING_MATRIX_HH

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"

namespace graph_tool
{
using namespace boost;

template <class Graph, class Index>
void get_nonbacktracking(Graph& g, Index index,
                         std::vector<int64_t>& i,
                         std::vector<int64_t>& j)
{
    for (auto u : vertices_range(g))
    {
        for (auto e1 : out_edges_range(u, g))
        {
            auto v = target(e1, g);
            int64_t idx1 = index[e1];
            if (!graph_tool::is_directed(g))
                idx1 = (idx1 << 1) + (u > v);
            for (auto e2 : out_edges_range(v, g))
            {
                auto w = target(e2, g);
                if (w == u)
                    continue;
                int64_t idx2 = index[e2];
                if (!graph_tool::is_directed(g))
                    idx2 = (idx2 << 1) + (v > w);
                i.push_back(idx1);
                j.push_back(idx2);
            }
        }
    }
}

template <class Graph>
void get_compact_nonbacktracking(Graph& g,
                                 std::vector<int64_t>& i,
                                 std::vector<int64_t>& j,
                                 std::vector<double>& x)
{
    for (auto e : edges_range(g))
    {
        auto u = source(e, g);
        auto v = target(e, g);
        i.push_back(u);
        j.push_back(v);
        x.push_back(1);

        i.push_back(v);
        j.push_back(u);
        x.push_back(1);
    }

    auto N = num_vertices(g);

    for (auto u : vertices_range(g))
    {
        int32_t k = out_degree(u, g);
        auto idx = u + N;

        i.push_back(u);
        j.push_back(idx);
        x.push_back(-1);

        i.push_back(idx);
        j.push_back(u);
        x.push_back(k-1);
    }
}

} // namespace graph_tool

#endif // GRAPH_NONBACKTRACKING_MATRIX_HH
