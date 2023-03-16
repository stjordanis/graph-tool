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

#include "graph_filtering.hh"
#include "graph.hh"
#include "graph_properties.hh"
#include "graph_exceptions.hh"

#include "graph_sfdp.hh"
#include "random.hh"
#include "hash_map_wrap.hh"
#include "numpy_bind.hh"

#include <boost/multi_array/multi_array_ref.hpp>

using namespace std;
using namespace boost;
using namespace graph_tool;

void sfdp_layout(GraphInterface& g, boost::any pos, boost::any vweight,
                 boost::any eweight, boost::any pin, python::object spring_parms,
                 double theta, double init_step, double step_schedule,
                 size_t max_level, double epsilon, size_t max_iter,
                 bool adaptive, bool verbose, rng_t& rng)
{
    typedef UnityPropertyMap<int,GraphInterface::vertex_t> vweight_map_t;
    typedef UnityPropertyMap<int,GraphInterface::edge_t> eweight_map_t;
    typedef mpl::push_back<vertex_scalar_properties, vweight_map_t>::type
        vertex_props_t;
    typedef mpl::push_back<edge_scalar_properties, eweight_map_t>::type
        edge_props_t;

    double C = python::extract<double>(spring_parms[0]);
    double K = python::extract<double>(spring_parms[1]);
    double p = python::extract<double>(spring_parms[2]);
    double gamma = python::extract<double>(spring_parms[3]);
    double mu = python::extract<double>(spring_parms[4]);
    double kappa = python::extract<double>(spring_parms[5]);

    typedef multi_array_ref<int32_t,1> group_map_t;
    std::vector<group_map_t> groups;
    for (int i = 0; i < python::len(spring_parms[6]); ++i)
        groups.push_back(get_array<int32_t,1>(spring_parms[6][i]));

    double r = python::extract<double>(spring_parms[7]);
    typedef vprop_map_t<int32_t>::type c_map_t;
    c_map_t c_map = boost::any_cast<c_map_t>(python::extract<any&>(spring_parms[8])());

    double R = python::extract<double>(spring_parms[9]);
    typedef vprop_map_t<double>::type o_map_t;
    o_map_t o_map = boost::any_cast<o_map_t>(python::extract<any&>(spring_parms[10])());

    if(vweight.empty())
        vweight = vweight_map_t();
    if(eweight.empty())
        eweight = eweight_map_t();

    typedef vprop_map_t<uint8_t>::type pin_map_t;
    pin_map_t pin_map = boost::any_cast<pin_map_t>(pin);

    run_action<graph_tool::detail::never_directed>()
        (g,
         [&](auto&& graph, auto&& pos, auto&& vweight, auto&& eweight)
         {
             return get_sfdp_layout(graph, pos, vweight, eweight,
                                    pin_map.get_unchecked(num_vertices(g.get_graph())),
                                    groups, C, K, p, theta, gamma, mu, kappa, r,
                                    c_map.get_unchecked(num_vertices(g.get_graph())),
                                    R, o_map.get_unchecked(num_vertices(g.get_graph())),
                                    init_step,
                                    step_schedule, max_level, epsilon, max_iter,
                                    adaptive, verbose, rng);
         },
         vertex_floating_vector_properties(), vertex_props_t(),
         edge_props_t())(pos, vweight, eweight);
}

struct do_propagate_pos
{
    template <class Graph, class CoarseGraph, class VertexMap, class PosMap,
              class RNG>
    void operator()(Graph& g, CoarseGraph& cg, VertexMap vmap,
                    boost::any acvmap, PosMap pos, boost::any acpos,
                    double delta, RNG& rng) const
    {
        typename PosMap::checked_t cpos =
            boost::any_cast<typename PosMap::checked_t>(acpos);
        typename VertexMap::checked_t cvmap =
            boost::any_cast<typename VertexMap::checked_t>(acvmap);
        typedef typename property_traits<VertexMap>::value_type c_t;
        typedef typename property_traits<PosMap>::value_type pos_t;
        typedef typename pos_t::value_type val_t;

        uniform_real_distribution<val_t> noise(-delta, delta);
        gt_hash_map<c_t, pos_t> cmap;

        for (auto v : vertices_range(cg))
            cmap[cvmap[v]] = cpos[v];

        for (auto v : vertices_range(g))
        {
            pos[v] = cmap[vmap[v]];

            if (delta > 0)
            {
                for (size_t j = 0; j < pos[v].size(); ++j)
                    pos[v][j] += noise(rng);
            }
        }
    }
};

void propagate_pos(GraphInterface& gi, GraphInterface& cgi, boost::any vmap,
                   boost::any cvmap, boost::any pos, boost::any cpos,
                   double delta, rng_t& rng)
{
    typedef mpl::vector<property_map_type::apply
                            <int32_t,
                             GraphInterface::vertex_index_map_t>::type>::type
        vmaps_t;

    gt_dispatch<>()
        (
            [&](auto&& graph, auto&& a2, auto&& a3, auto&& a4)
            {
                return do_propagate_pos()
                    (std::forward<decltype(graph)>(graph),
                     std::forward<decltype(a2)>(a2),
                     std::forward<decltype(a3)>(a3), cvmap,
                     std::forward<decltype(a4)>(a4), cpos, delta, rng);
            },
            all_graph_views(), all_graph_views(), vmaps_t(),
            vertex_floating_vector_properties())(
            gi.get_graph_view(), cgi.get_graph_view(), vmap, pos);
}

struct do_propagate_pos_mivs
{
    template <class Graph, class MIVSMap, class PosMap,
              class RNG>
    void operator()(Graph& g, MIVSMap mivs, PosMap pos, double delta, RNG& rng) const
    {
        typedef typename property_traits<PosMap>::value_type pos_t;
        typedef typename pos_t::value_type val_t;

        uniform_real_distribution<val_t> noise(-delta, delta);

        for (auto v : vertices_range(g))
        {
            if (mivs[v])
                continue;
            size_t count = 0;
            for (auto a : adjacent_vertices_range(v, g))
            {
                if (!mivs[a])
                    continue;
                pos[v].resize(pos[a].size(), 0);
                for (size_t j = 0; j < pos[a].size(); ++j)
                    pos[v][j] += pos[a][j];
                ++count;
            }

            if (count == 0)
                throw ValueException("invalid MIVS! Vertex has no neighbors "
                                     "belonging to the set!");

            if (count == 1)
            {
                if (delta > 0)
                {
                    for (size_t j = 0; j < pos[v].size(); ++j)
                        pos[v][j] += noise(rng);
                }
            }
            else
            {
                for (size_t j = 0; j < pos[v].size(); ++j)
                    pos[v][j] /= count;
            }
        }
    }
};


void propagate_pos_mivs(GraphInterface& gi, boost::any mivs, boost::any pos,
                        double delta, rng_t& rng)
{
    run_action<>()
        (gi,
         [&](auto&& graph, auto&& a2, auto&& a3)
         {
             return do_propagate_pos_mivs()
                 (std::forward<decltype(graph)>(graph),
                  std::forward<decltype(a2)>(a2),
                  std::forward<decltype(a3)>(a3), delta, rng);
         },
         vertex_scalar_properties(),
         vertex_floating_vector_properties())(mivs, pos);
}


struct do_avg_dist
{
    template <class Graph, class PosMap>
    void operator()(Graph& g, PosMap pos, double& ad) const
    {
        size_t count = 0;
        double d = 0;
        #pragma omp parallel if (num_vertices(g) > OPENMP_MIN_THRESH) \
            reduction(+: d, count)
        parallel_vertex_loop_no_spawn
                (g,
                 [&](auto v)
                 {
                     for (auto a : adjacent_vertices_range(v, g))
                     {
                         d += dist(pos[v], pos[a]);
                         count++;
                     }
                 });
        if (count > 0)
            d /= count;
        ad = d;
    }
};


double avg_dist(GraphInterface& gi, boost::any pos)
{
    double d;
    run_action<>()
        (gi,
         [&](auto&& graph, auto&& a2)
         {
             return do_avg_dist()
                 (std::forward<decltype(graph)>(graph),
                  std::forward<decltype(a2)>(a2), d);
         },
         vertex_scalar_vector_properties())(pos);
    return d;
}


struct do_sanitize_pos
{
    template <class Graph, class PosMap>
    void operator()(Graph& g, PosMap pos) const
    {
        parallel_vertex_loop
                (g,
                 [&](auto v)
                 {
                     pos[v].resize(2);
                 });
    }
};


void sanitize_pos(GraphInterface& gi, boost::any pos)
{
    run_action<>()
        (gi,
         [&](auto&& graph, auto&& a2)
         {
             return do_sanitize_pos()
                 (std::forward<decltype(graph)>(graph),
                  std::forward<decltype(a2)>(a2));
         },
         vertex_scalar_vector_properties())(pos);
}

#include <boost/python.hpp>

#define __MOD__ layout
#include "module_registry.hh"
REGISTER_MOD
([]
 {
     python::def("sfdp_layout", &sfdp_layout);
     python::def("propagate_pos", &propagate_pos);
     python::def("propagate_pos_mivs", &propagate_pos_mivs);
     python::def("avg_dist", &avg_dist);
     python::def("sanitize_pos", &sanitize_pos);
 });
