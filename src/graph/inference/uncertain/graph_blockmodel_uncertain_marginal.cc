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

#include "graph_tool.hh"
#include "graph_blockmodel_uncertain_marginal.hh"
#include "parallel_rng.hh"

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

double marginal_count_entropy(GraphInterface& gi, boost::any aexc, boost::any aeh)
{
    typedef eprop_map_t<double>::type ehmap_t;
    auto eh = any_cast<ehmap_t>(aeh);

    double S_tot = 0;
    gt_dispatch<>()
        ([&](auto& g, auto exc)
         {
             parallel_edge_loop
                 (g,
                  [&](auto& e)
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
                          return;
                      S /= N;
                      S += safelog_fast(N);

                      #pragma omp atomic
                      S_tot += S;
                  });
         },
        all_graph_views(), edge_scalar_vector_properties())
        (gi.get_graph_view(), aexc);
    return S_tot;
}

void marginal_multigraph_sample(GraphInterface& gi, boost::any axs, boost::any axc,
                                boost::any ax, rng_t& rng_)
{
    gt_dispatch<>()
        ([&](auto& g, auto& xs, auto& xc, auto& x)
         {
             parallel_rng<rng_t>::init(rng_);
             parallel_edge_loop
                 (g,
                  [&](auto& e)
                  {
                      typedef std::remove_reference_t<decltype(xs[e][0])> val_t;
                      std::vector<double> probs(xc[e].begin(), xc[e].end());
                      Sampler<val_t> sample(xs[e], probs);

                      auto& rng = parallel_rng<rng_t>::get(rng_);
                      x[e] = sample.sample(rng);
                  });
         },
         all_graph_views(), edge_scalar_vector_properties(),
         edge_scalar_vector_properties(), writable_edge_scalar_properties())
        (gi.get_graph_view(), axs, axc, ax);
}

double marginal_multigraph_lprob(GraphInterface& gi, boost::any axs, boost::any axc,
                                 boost::any ax)
{
    double L = 0;
    gt_dispatch<>()
        ([&](auto& g, auto& xs, auto& xc, auto& x)
         {
             for (auto e : edges_range(g))
             {
                 size_t Z = 0;
                 size_t p = 0;
                 for (size_t i = 0; i < xs[e].size(); ++i)
                 {
                     size_t m = xs[e][i];
                     if (m == size_t(x[e]))
                         p = xc[e][i];
                     Z += xc[e][i];
                 }
                 if (p == 0)
                 {
                     L = -numeric_limits<double>::infinity();
                     break;
                 }
                 L += std::log(p) - std::log(Z);
             }
         },
         all_graph_views(), edge_scalar_vector_properties(),
         edge_scalar_vector_properties(), edge_scalar_properties())
        (gi.get_graph_view(), axs, axc, ax);
    return L;
}

void marginal_graph_sample(GraphInterface& gi, boost::any ap,
                           boost::any ax, rng_t& rng_)
{
    gt_dispatch<>()
        ([&](auto& g, auto& p, auto& x)
         {
             parallel_rng<rng_t>::init(rng_);
             parallel_edge_loop
                 (g,
                  [&](auto& e)
                  {
                      std::bernoulli_distribution sample(p[e]);
                      auto& rng = parallel_rng<rng_t>::get(rng_);
                      x[e] = sample(rng);
                  });
         },
         all_graph_views(), edge_scalar_properties(),
         writable_edge_scalar_properties())
        (gi.get_graph_view(), ap, ax);
}

double marginal_graph_lprob(GraphInterface& gi, boost::any ap,
                            boost::any ax)
{
    double L = 0;
    gt_dispatch<>()
        ([&](auto& g, auto& p, auto& x)
         {
             for (auto e : edges_range(g))
             {
                 if (x[e] == 1)
                     L += std::log(p[e]);
                 else
                     L += std::log1p(-p[e]);
             }
         },
         all_graph_views(), edge_scalar_properties(),
         edge_scalar_properties())
        (gi.get_graph_view(), ap, ax);
    return L;
}
