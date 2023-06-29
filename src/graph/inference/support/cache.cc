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

#include "cache.hh"

namespace graph_tool
{

using namespace std;

vector<vector<double>> __safelog_cache;
vector<vector<double>> __xlogx_cache;
vector<vector<double>> __lgamma_cache;

void clear_safelog()
{
    __safelog_cache.clear();
}


void clear_xlogx()
{
    __xlogx_cache.clear();
}


void clear_lgamma()
{
    __lgamma_cache.clear();
}

void init_cache()
{
    auto nt = get_num_threads();
    __lgamma_cache.resize(nt);
    __xlogx_cache.resize(nt);
    __safelog_cache.resize(nt);
}


} // namespace graph_tool
