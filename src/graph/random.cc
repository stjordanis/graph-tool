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

#include "graph.hh"
#include "random.hh"
#include "parallel_rng.hh"

rng_t get_rng(size_t seed)
{
    parallel_rng<rng_t>::clear();
    if (seed == 0)
    {
        pcg_extras::seed_seq_from<std::random_device> seed_source;
        return rng_t(seed_source);
    }
    std::seed_seq seq{seed, seed + 1, seed + 2, seed + 3, seed + 4};
    return rng_t(seq);
}
