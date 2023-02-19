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

#include "graph_tool.hh"
#include "../support/util.hh"
#include "random.hh"
#include "numpy_bind.hh"
#include "graph_clique_decomposition.hh"

#include <boost/python.hpp>

using namespace graph_tool;

boost::python::object
clique_iter_mh(GraphInterface& gi, boost::any ac, boost::any ax,
               boost::any ais_fac, boost::any ais_max, boost::python::object oEd,
               int N, int E, double beta, size_t niter, rng_t& rng)
{
    typedef typename vprop_map_t<int32_t>::type vprop_t;
    typedef typename vprop_map_t<uint8_t>::type vbprop_t;
    typedef typename vprop_map_t<std::vector<int32_t>>::type vvprop_t;

    vprop_t x = boost::any_cast<vprop_t>(ax);
    vvprop_t c = boost::any_cast<vvprop_t>(ac);
    vbprop_t is_fac = boost::any_cast<vbprop_t>(ais_fac);
    vbprop_t is_max = boost::any_cast<vbprop_t>(ais_max);

    multi_array_ref<int32_t, 1> Ed = get_array<int32_t, 1>(oEd);

    boost::python::object pret;
    gt_dispatch<>()
        ([&](auto& g){
             auto ret = iter_mh(g, x.get_unchecked(), c.get_unchecked(),
                                is_fac.get_unchecked(), is_max.get_unchecked(),
                                Ed, N, E, beta, niter, rng);
             pret = boost::python::make_tuple(get<0>(ret), get<1>(ret));
         },
         all_graph_views())
        (gi.get_graph_view());
    return pret;
};

void export_clique_decomposition()
{
    using namespace boost::python;
    def("clique_iter_mh", clique_iter_mh);
    def("clique_L_over", L_over);
}
