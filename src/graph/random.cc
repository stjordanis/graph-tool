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

#include "graph.hh"
#include "random.hh"
#include "parallel_rng.hh"

#include <random>
#include <mutex>
#include <thread>
#include <unordered_map>

rng_t _rng;

std::unordered_map<std::thread::id, rng_t> _rngs;
size_t _rng_stream = 0;
std::mutex _rng_mutex;

rng_t& get_rng()
{
    auto tid = std::this_thread::get_id();
    auto iter = _rngs.find(tid);
    if (iter == _rngs.end())
    {
        auto& rng = _rngs[tid] = _rng;
        rng.set_stream(get_rng_stream());
        return rng;
    }
    return iter->second;
}

size_t get_rng_stream()
{
    std::lock_guard<std::mutex> lock(_rng_mutex);
    return _rng_stream++;
}

void seed_rng(size_t seed)
{
    std::lock_guard<std::mutex> lock(_rng_mutex);

    parallel_rng<rng_t>::clear();
    _rngs.clear();
    _rng_stream = 0;

    if (seed == 0)
    {
        pcg_extras::seed_seq_from<std::random_device> seed_source;
        _rng = rng_t(seed_source);
    }
    else
    {
        std::seed_seq seq{seed, seed + 1, seed + 2, seed + 3, seed + 4};
        _rng = rng_t(seq);
    }
}
