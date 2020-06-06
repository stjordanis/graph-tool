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

#include "graph_sbm.hh"
#include "numpy_bind.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void generate_sbm(GraphInterface& gi, boost::any ab, boost::python::object ors,
                  boost::python::object oss, boost::python::object oprobs,
                  boost::any ain_deg, boost::any aout_deg, bool micro_ers,
                  bool micro_degs, rng_t& rng)
{
    auto rs = get_array<int64_t, 1>(ors);
    auto ss = get_array<int64_t, 1>(oss);

    typedef vprop_map_t<int32_t>::type bmap_t;
    auto b = any_cast<bmap_t>(ab).get_unchecked();

    if (micro_degs)
    {
        auto probs = get_array<uint64_t, 1>(oprobs);
        typedef vprop_map_t<int64_t>::type dmap_t;
        auto in_deg = any_cast<dmap_t>(ain_deg).get_unchecked();
        auto out_deg = any_cast<dmap_t>(aout_deg).get_unchecked();
        run_action<>()
            (gi, [&](auto& g) { gen_sbm<true>(g, b, rs, ss, probs, in_deg,
                                              out_deg, true, rng); })();
    }
    else
    {
        auto probs = get_array<double, 1>(oprobs);
        typedef vprop_map_t<double>::type dmap_t;
        auto in_deg = any_cast<dmap_t>(ain_deg).get_unchecked();
        auto out_deg = any_cast<dmap_t>(aout_deg).get_unchecked();
        run_action<>()
            (gi, [&](auto& g) { gen_sbm<false>(g, b, rs, ss, probs, in_deg,
                                               out_deg, micro_ers, rng); })();
    }
}
