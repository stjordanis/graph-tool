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

#include "graph_triadic_closure.hh"
#include "numpy_bind.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

void generate_triadic_closure(GraphInterface& gi,
                              boost::any acurr,
                              boost::any aego,
                              boost::any aEs,
                              bool probs, rng_t& rng)
{
    typedef eprop_map_t<uint8_t>::type emap_t;
    auto curr = any_cast<emap_t>(acurr).get_unchecked();

    typedef eprop_map_t<int64_t>::type eemap_t;
    auto ego = any_cast<eemap_t>(aego);

    run_action<>()(gi,
                   [&](auto& g, auto Es) { gen_triadic_closure(g, curr, ego, Es,
                                                               probs, rng); },
                   vertex_scalar_properties())(aEs);
}
