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

#include <boost/python.hpp>
#include "graph.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void adjacency(GraphInterface& g, boost::any index, boost::any weight,
               python::object odata, python::object oi,
               python::object oj);

void adjacency_matvec(GraphInterface& g, boost::any index, boost::any weight,
                      python::object ov, python::object oret);

void adjacency_matmat(GraphInterface& g, boost::any index, boost::any weight,
                      python::object ov, python::object oret);

void laplacian(GraphInterface& g, boost::any index, boost::any weight,
               string sdeg,
               python::object odata, python::object oi,
               python::object oj);

void laplacian_matvec(GraphInterface& g, boost::any index, boost::any weight,
                      boost::any deg, python::object ov, python::object oret);

void laplacian_matmat(GraphInterface& g, boost::any index, boost::any weight,
                      boost::any deg, python::object ov, python::object oret);

void norm_laplacian(GraphInterface& g, boost::any index, boost::any weight,
                    string sdeg,
                    python::object odata, python::object oi,
                    python::object oj);

void norm_laplacian_matvec(GraphInterface& g, boost::any index, boost::any weight,
                           boost::any deg, python::object ov, python::object oret);

void norm_laplacian_matmat(GraphInterface& g, boost::any index, boost::any weight,
                           boost::any deg, python::object ov, python::object oret);

void incidence(GraphInterface& g, boost::any vindex, boost::any eindex,
               python::object odata, python::object oi,
               python::object oj);

void incidence_matvec(GraphInterface& g, boost::any vindex, boost::any eindex,
                      python::object ov, python::object oret, bool transpose);

void incidence_matmat(GraphInterface& g, boost::any vindex, boost::any eindex,
                      python::object ov, python::object oret, bool tranpose);

void transition(GraphInterface& g, boost::any index, boost::any weight,
                python::object odata, python::object oi,
                python::object oj);

void transition_matvec(GraphInterface& g, boost::any index, boost::any weight,
                       boost::any deg, python::object ov, python::object oret,
                       bool transpose);

void transition_matmat(GraphInterface& g, boost::any index, boost::any weight,
                       boost::any deg, python::object ov, python::object oret,
                       bool transpose);

void nonbacktracking(GraphInterface& gi, boost::any index,
                     std::vector<int64_t>& i, std::vector<int64_t>& j);

void nonbacktracking_matvec(GraphInterface& g, boost::any index,
                            python::object ov, python::object oret,
                            bool transpose);

void nonbacktracking_matmat(GraphInterface& g, boost::any index,
                            python::object ov, python::object oret,
                            bool transpose);

void compact_nonbacktracking(GraphInterface& gi, std::vector<int64_t>& i,
                             std::vector<int64_t>& j, std::vector<double>& x);

void compact_nonbacktracking_matvec(GraphInterface& g, boost::any index,
                                    python::object ov, python::object oret,
                                    bool transpose);

void compact_nonbacktracking_matmat(GraphInterface& g, boost::any index,
                                    python::object ov, python::object oret,
                                    bool transpose);

BOOST_PYTHON_MODULE(libgraph_tool_spectral)
{
    using namespace boost::python;
    docstring_options dopt(true, false);
    def("adjacency", &adjacency);
    def("adjacency_matvec", &adjacency_matvec);
    def("adjacency_matmat", &adjacency_matmat);
    def("laplacian", &laplacian);
    def("laplacian_matvec", &laplacian_matvec);
    def("laplacian_matmat", &laplacian_matmat);
    def("norm_laplacian", &norm_laplacian);
    def("norm_laplacian_matvec", &norm_laplacian_matvec);
    def("norm_laplacian_matmat", &norm_laplacian_matmat);
    def("incidence", &incidence);
    def("incidence_matvec", &incidence_matvec);
    def("incidence_matmat", &incidence_matmat);
    def("transition", &transition);
    def("transition_matvec", &transition_matvec);
    def("transition_matmat", &transition_matmat);
    def("nonbacktracking", &nonbacktracking);
    def("nonbacktracking_matvec", &nonbacktracking_matvec);
    def("nonbacktracking_matmat", &nonbacktracking_matmat);
    def("compact_nonbacktracking", &compact_nonbacktracking);
    def("compact_nonbacktracking_matvec", &compact_nonbacktracking_matvec);
    def("compact_nonbacktracking_matmat", &compact_nonbacktracking_matmat);
}
