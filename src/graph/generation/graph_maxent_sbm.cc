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

#include "graph_maxent_sbm.hh"
#include "numpy_bind.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void generate_maxent_sbm(GraphInterface& gi, boost::any ab,
                         boost::python::object ors, boost::python::object oss,
                         boost::python::object omrs, boost::any ain_theta,
                         boost::any aout_theta, bool multi, bool self_loops,
                         rng_t& rng)
{
    auto rs = get_array<int64_t, 1>(ors);
    auto ss = get_array<int64_t, 1>(oss);
    auto mrs = get_array<double, 1>(omrs);

    typedef vprop_map_t<int32_t>::type bmap_t;
    auto b = any_cast<bmap_t>(ab).get_unchecked();

    typedef vprop_map_t<double>::type dmap_t;
    auto in_theta = any_cast<dmap_t>(ain_theta).get_unchecked();
    auto out_theta = any_cast<dmap_t>(aout_theta).get_unchecked();

    if (multi)
    {
        run_action<>()
            (gi, [&](auto& g) { gen_maxent_sbm<true>(g, b, rs, ss, mrs, in_theta,
                                                     out_theta, self_loops, rng); })();
    }
    else
    {
        run_action<>()
            (gi, [&](auto& g) { gen_maxent_sbm<false>(g, b, rs, ss, mrs, in_theta,
                                                      out_theta, self_loops, rng); })();
    }
}

SBMFugacities get_sbm_fugacities(boost::python::object ors,
                                 boost::python::object oss,
                                 boost::python::object oers,
                                 boost::python::object odegs_in,
                                 boost::python::object odegs_out,
                                 boost::python::object ob, bool directed,
                                 bool multigraph, bool self_loops)
{
    auto rs = get_array<int64_t,1>(ors);
    auto ss = get_array<int64_t,1>(oss);
    auto ers = get_array<double,1>(oers);
    auto degs_in = get_array<double,1>(odegs_in);
    auto degs_out = get_array<double,1>(odegs_out);
    auto b = get_array<int32_t,1>(ob);
    return SBMFugacities(rs, ss, ers, degs_in, degs_out, b, directed,
                         multigraph, self_loops);
}

using namespace boost::python;

void export_maxent_sbm()
{
    def("get_sbm_fugacities", &get_sbm_fugacities);
    def("gen_maxent_sbm", &generate_maxent_sbm);

    class_<SBMFugacities>("SBMFugacities", no_init)
        .def("pack", &SBMFugacities::pack)
        .def("unpack", &SBMFugacities::unpack)
        .def("get_f", &SBMFugacities::get_f)
        .def("get_diff", &SBMFugacities::get_diff)
        .def("new_x", &SBMFugacities::new_x)
        .def("norm", &SBMFugacities::norm)
        .def("export_args",
             +[](SBMFugacities& state, python::object ors, python::object oss,
                 python::object omrs, python::object odegs_in,
                 python::object odegs_out, python::object otheta_in,
                 python::object otheta_out, python::object ob)
              {
                  auto rs = get_array<int64_t,1>(ors);
                  auto ss = get_array<int64_t,1>(oss);
                  auto mrs = get_array<double,1>(omrs);
                  auto degs_in = get_array<double,1>(odegs_in);
                  auto degs_out = get_array<double,1>(odegs_out);
                  auto theta_in = get_array<double,1>(otheta_in);
                  auto theta_out = get_array<double,1>(otheta_out);
                  auto b = get_array<int32_t,1>(ob);
                  state.export_args(rs, ss, mrs, degs_in, degs_out, theta_in,
                                    theta_out, b);
              });
}
