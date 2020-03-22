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

#include <boost/python.hpp>
#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "numpy_bind.hh"

#include "graph_selectors.hh"
#include "graph_properties.hh"

#include "graph_nonbacktracking.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void nonbacktracking(GraphInterface& gi, boost::any index,
                     std::vector<int64_t>& i, std::vector<int64_t>& j)
{
    if (!belongs<edge_scalar_properties>()(index))
        throw ValueException("index vertex property must have a scalar value type");

    run_action<>()
        (gi, [&](auto& g, auto idx){ get_nonbacktracking(g, idx, i, j);},
         edge_scalar_properties())(index);

}

void compact_nonbacktracking(GraphInterface& gi, std::vector<int64_t>& i,
                             std::vector<int64_t>& j, std::vector<double>& x)
{
    run_action<>()
        (gi, [&](auto& g){ get_compact_nonbacktracking(g, i, j, x);})();

}
