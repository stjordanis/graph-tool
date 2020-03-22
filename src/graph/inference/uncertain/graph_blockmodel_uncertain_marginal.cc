// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2020 Tiago de Paula Peixoto <tiago@skewed.de>
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "graph_tool.hh"
#include "graph_blockmodel_uncertain_marginal.hh"

using namespace boost;
using namespace graph_tool;

void collect_marginal_dispatch(GraphInterface& gi, GraphInterface& ui,
                               boost::any aecount)
{
    typedef eprop_map_t<int32_t>::type emap_t;
    auto ecount = any_cast<emap_t>(aecount);

    gt_dispatch<>()
        ([&](auto& g, auto& u) { collect_marginal(g, u, ecount,
                                                  dummy_property(),
                                                  dummy_property(),
                                                  dummy_property(),
                                                  dummy_property(),
                                                  dummy_property()); },
         all_graph_views(), all_graph_views())(gi.get_graph_view(),
                                               ui.get_graph_view());
}

void collect_xmarginal_dispatch(GraphInterface& gi, GraphInterface& ui,
                                boost::any ax, boost::any aecount,
                                boost::any axsum, boost::any ax2sum)
{
    typedef eprop_map_t<int32_t>::type emap_t;
    auto ecount = any_cast<emap_t>(aecount);

    typedef eprop_map_t<double>::type xmap_t;
    auto x = any_cast<xmap_t>(ax);
    auto xsum = any_cast<xmap_t>(axsum);
    auto x2sum = any_cast<xmap_t>(ax2sum);

    gt_dispatch<>()
        ([&](auto& g, auto& u) { collect_marginal(g, u, ecount,
                                                  x, xsum, x2sum,
                                                  dummy_property(),
                                                  dummy_property()); },
         all_graph_views(), all_graph_views())(gi.get_graph_view(),
                                               ui.get_graph_view());
}

void collect_marginal_count_dispatch(GraphInterface& gi, GraphInterface& ui,
                                     boost::any aex, boost::any aexs,
                                     boost::any aexc)
{
    typedef eprop_map_t<int32_t>::type ecmap_t;
    auto ex = any_cast<ecmap_t>(aex);

    typedef eprop_map_t<std::vector<int32_t>>::type emap_t;
    auto exs = any_cast<emap_t>(aexs);
    auto exc = any_cast<emap_t>(aexc);

    gt_dispatch<>()
        ([&](auto& g, auto& u) { collect_marginal(g, u,
                                                  dummy_property(),
                                                  ex,
                                                  dummy_property(),
                                                  dummy_property(),
                                                  exs, exc); },
         all_graph_views(), all_graph_views())(gi.get_graph_view(),
                                               ui.get_graph_view());
}

void marginal_count_entropy(GraphInterface& gi, boost::any aexc, boost::any aeh)
{
    typedef eprop_map_t<double>::type ehmap_t;
    auto eh = any_cast<ehmap_t>(aeh);

    gt_dispatch<>()
        ([&](auto& g, auto exc)
         {
             for (auto e : edges_range(g))
             {
                 auto& S = eh[e];
                 S = 0;
                 size_t N = 0;
                 for (auto n : exc[e])
                 {
                     S -= xlogx_fast(n);
                     N += n;
                 }
                 if (N == 0)
                     continue;
                 S /= N;
                 S += safelog_fast(N);
             }
         },
        all_graph_views(), edge_scalar_vector_properties())
        (gi.get_graph_view(), aexc);
}
