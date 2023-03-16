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
#include "random.hh"

#include <boost/python.hpp>

#include "graph_blockmodel_util.hh"
#include "graph_blockmodel.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(block_state, BlockState, BLOCK_STATE_params)

python::object make_block_state(boost::python::object ostate);

degs_map_t get_empty_degs(GraphInterface& gi)
{
    return degs_map_t(gi.get_num_vertices(false));
}

template <class Prop>
boost::any get_any(Prop& p)
{
    return boost::any(p);
}

degs_map_t copy_degs(degs_map_t& degs)
{
    return degs.copy();
}

simple_degs_t copy_simple_degs(simple_degs_t& degs)
{
    return degs;
}

double spence(double);

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;

    class_<vcmap_t>("unity_vprop_t").def("_get_any", &get_any<vcmap_t>);
    class_<ecmap_t>("unity_eprop_t").def("_get_any", &get_any<ecmap_t>);

    class_<entropy_args_t>("entropy_args")
        .def_readwrite("exact", &entropy_args_t::exact)
        .def_readwrite("dense", &entropy_args_t::dense)
        .def_readwrite("multigraph", &entropy_args_t::multigraph)
        .def_readwrite("adjacency", &entropy_args_t::adjacency)
        .def_readwrite("deg_entropy", &entropy_args_t::deg_entropy)
        .def_readwrite("recs", &entropy_args_t::recs)
        .def_readwrite("partition_dl", &entropy_args_t::partition_dl)
        .def_readwrite("degree_dl", &entropy_args_t::degree_dl)
        .def_readwrite("degree_dl_kind", &entropy_args_t::degree_dl_kind)
        .def_readwrite("edges_dl", &entropy_args_t::edges_dl)
        .def_readwrite("recs_dl", &entropy_args_t::recs_dl)
        .def_readwrite("beta_dl", &entropy_args_t::beta_dl)
        .def_readwrite("Bfield", &entropy_args_t::Bfield);

    enum_<deg_dl_kind>("deg_dl_kind")
        .value("ent", deg_dl_kind::ENT)
        .value("uniform", deg_dl_kind::UNIFORM)
        .value("dist", deg_dl_kind::DIST);

    enum_<weight_type>("rec_type")
        .value("none", weight_type::NONE)
        .value("count", weight_type::COUNT)
        .value("real_exponential", weight_type::REAL_EXPONENTIAL)
        .value("real_normal", weight_type::REAL_NORMAL)
        .value("discrete_geometric", weight_type::DISCRETE_GEOMETRIC)
        .value("discrete_poisson", weight_type::DISCRETE_POISSON)
        .value("discrete_binomial", weight_type::DISCRETE_BINOMIAL);

    def("make_block_state", &make_block_state);

    def("get_empty_degs", &get_empty_degs);
    class_<degs_map_t>("degs_map_t")
        .def("copy", &copy_degs)
        .def("_get_any", &get_any<degs_map_t>);
    class_<simple_degs_t>("simple_degs_t")
        .def("copy", &copy_simple_degs)
        .def("_get_any", &get_any<simple_degs_t>);

    def("init_q_cache", init_q_cache);
    def("clear_q_cache", clear_q_cache);
    def("log_q", log_q<size_t>);
    def("q_rec", q_rec);
    def("q_rec_memo", q_rec_memo);
    def("log_q_approx", log_q_approx);
    def("log_q_approx_big", log_q_approx_big);
    def("log_q_approx_small", log_q_approx_small);
    def("spence", spence);

    def("positive_w_log_P", positive_w_log_P<size_t>);
    def("signed_w_log_P", signed_w_log_P<size_t>);
    def("geometric_w_log_P", geometric_w_log_P<size_t>);
    def("binomial_w_log_P", binomial_w_log_P<size_t>);
    def("poisson_w_log_P", poisson_w_log_P<size_t>);
}, 3);
