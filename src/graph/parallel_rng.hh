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

#ifndef PARALLEL_RNG_HH
#define PARALLEL_RNG_HH

#include "config.h"
#include <vector>

#ifdef _OPENMP
# include <omp.h>
#endif

template <class RNG>
class parallel_rng
{
public:
    static void init(RNG& rng)
    {
        size_t num_threads = 1;
    #ifdef _OPENMP
        num_threads = omp_get_max_threads();
    #endif
        for (size_t i = _rngs.size(); i < num_threads - 1; ++i)
        {
            // std::array<int, RNG::state_size> seed_data;
            // std::generate_n(seed_data.data(), seed_data.size(), std::ref(rng));
            // std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
            // rngs.emplace_back(seq);
            _rngs.emplace_back(rng);
            _rngs.back().set_stream(i + 1);
        }
    }

    static void clear()
    {
        _rngs.clear();
    }

    static RNG& get(RNG& rng)
    {
        size_t tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        if (tid == 0)
            return rng;
        return _rngs[tid - 1];
    }

private:
    static std::vector<RNG> _rngs;
};

template <class RNG>
std::vector<RNG> parallel_rng<RNG>::_rngs;


#endif // PARALLEL_RNG_HH
