// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2021 Tiago de Paula Peixoto <tiago@skewed.de>
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

#ifndef GRAPH_ARF_HH
#define GRAPH_ARF_HH

namespace graph_tool
{
using namespace std;
using namespace boost;

struct get_arf_layout
{
    template <class Graph, class PosMap, class WeightMap>
    void operator()(Graph& g, PosMap pos, WeightMap weight, double a, double d,
                    double dt, double epsilon, size_t max_iter, size_t dim)
        const
    {
        typedef typename property_traits<PosMap>::value_type::value_type pos_t;

        parallel_vertex_loop_no_spawn
            (g,
             [&](auto v)
             {
                 pos[v].resize(dim);
             });

        pos_t delta = epsilon + 1;
        size_t n_iter = 0;
        pos_t r = d*sqrt(pos_t(HardNumVertices()(g)));
        while (delta > epsilon && (max_iter == 0 || n_iter < max_iter))
        {
            delta = 0;
            #pragma omp parallel if (num_vertices(g) > OPENMP_MIN_THRESH) \
                reduction(+:delta)
            parallel_vertex_loop_no_spawn
                (g,
                 [&](auto v)
                 {
                     vector<pos_t> delta_pos(dim,0);

                     for (auto w : vertices_range(g))
                     {
                         if (w == v)
                             continue;
                         pos_t diff = 0;
                         for (size_t j = 0; j < dim; ++j)
                         {
                             pos_t dx = pos[w][j] - pos[v][j];
                             diff += dx*dx;
                             delta_pos[j] += dx;
                         }
                         diff = std::max(sqrt(diff), pos_t(1e-6));
                         pos_t m = r/diff;
                         for (size_t j = 0; j < dim; ++j)
                         {
                             pos_t dx = pos[w][j] - pos[v][j];
                             delta_pos[j] -= m*dx;
                         }
                     }

                     for (auto e : out_edges_range(v, g))
                     {
                         auto u = target(e, g);
                         if (u == v)
                             continue;
                         pos_t m = a * get(weight, e) - 1;
                         for (size_t j = 0; j < dim; ++j)
                         {
                             pos_t dx = pos[u][j] - pos[v][j];
                             delta_pos[j] += m * dx;
                         }
                     }

                     for (size_t j = 0; j < dim; ++j)
                     {
                         delta += abs(delta_pos[j]);
                         auto dt_j = dt * delta_pos[j];
                         #pragma omp atomic
                         pos[v][j] += dt_j;
                     }

                 });
            n_iter++;
        }
    }
};

} // namespace graph_tool


#endif // GRAPH_ARF_HH
