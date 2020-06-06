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

#ifndef GRAPH_SBM_SAMPLE_EDGE_HH
#define GRAPH_SBM_SAMPLE_EDGE_HH

#include <tuple>
#include <iostream>

#include "graph.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"
#include "../../generation/sampler.hh"
#include "../../generation/urn_sampler.hh"

#include "random.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class State>
class SBMEdgeSampler
{
public:
    SBMEdgeSampler(State& state, bool edges_only=false)
        : _state(state),
          _v_in_sampler(graph_tool::is_directed(state._g) ?
                        __v_in_sampler : _v_out_sampler),
          _edges_only(edges_only)
    {
        _N = num_vertices(state._g);

        for (auto e : edges_range(state._g))
        {
            size_t u = source(e, state._g);
            size_t v = target(e, state._g);
            _edges.push_back(get_edge(u, v));
            _edge_pos[_edges.back()] = _edges.size() - 1;
        }

        if (_edges_only)
            return;

        for (auto me : edges_range(state._bg))
        {
            auto r = source(me, state._bg);
            auto s = target(me, state._bg);
            size_t mrs = state._mrs[me];
            if (mrs == 0)
                continue;
            _rs_pos[me] = _rs_sampler.insert({r, s}, mrs);
            _E += mrs;
        }

        for (auto v : vertices_range(state._g))
        {
            size_t r = state._b[v];
            if (r >= _v_out_sampler.size())
            {
                if (graph_tool::is_directed(state._g))
                    _v_in_sampler.resize(r+1);
                _v_out_sampler.resize(r+1);
            }

            size_t kin = 0, kout = 0;
            if (state._deg_corr)
            {
                degs_op(v, state._vweight, state._eweight, state._degs,
                        state._g,
                        [&] (size_t din, size_t dout, auto c)
                        {
                            kin += din * c;
                            kout += dout * c;
                        });
            }

            if (graph_tool::is_directed(state._g))
                _v_in_pos[v] = _v_in_sampler[r].insert(v, kin + 1);
            _v_out_pos[v] = _v_out_sampler[r].insert(v, kout + 1);
        }

        for (auto r : vertices_range(state._bg))
            if (state._wr[r] > 0)
                _groups.push_back(r);

        auto B = _groups.size();
        _NB = B * B;
    }

    std::tuple<size_t, size_t> get_edge(size_t u, size_t v)
    {
        if (!graph_tool::is_directed(_state._g) && u > v)
            return {v, u};
        return {u, v};
    }


    template <bool add>
    void update_edge(size_t u, size_t v, size_t m)
    {
        if (_edges_only)
            return;

        if (add)
        {
            if (m == 0)
            {
                _edges.push_back(get_edge(u, v));
                _edge_pos[_edges.back()] = _edges.size() - 1;
            }
        }
        else
        {
            if (m == 1)
            {
                auto iter = _edge_pos.find(get_edge(u, v));
                size_t pos = iter->second;
                _edge_pos.erase(iter);
                if (pos < _edges.size() - 1)
                {
                    std::swap(_edges[pos], _edges.back());
                    _edge_pos[_edges[pos]] = pos;
                }
                _edges.pop_back();
            }
        }

        constexpr int delta = (add) ? 0 : -1;
        _E += (add) ? 1 : -1;
        size_t r = _state._b[u];
        size_t s = _state._b[v];

        auto me = _state._emat.get_me(r, s);
        if (me != _state._emat.get_null_edge())
        {
            auto ers = _state._mrs[me] + delta;
            if (!add || ers > 1)
                _rs_sampler.remove(_rs_pos[me]);
            if (ers > 0)
                _rs_pos[me] = _rs_sampler.insert({r,s}, ers);
            else
                _rs_pos[me] = std::numeric_limits<size_t>::max();
        }

        if (_state._deg_corr)
        {
            size_t ku = 0, kv = 0;
            degs_op(u, _state._vweight, _state._eweight, _state._degs, _state._g,
                    [&] (size_t, size_t dout, auto c)
                    {
                        ku += dout * c;
                    });
            degs_op(v, _state._vweight, _state._eweight, _state._degs, _state._g,
                    [&] (size_t din, size_t dout, auto c)
                    {
                        if (graph_tool::is_directed(_state._g))
                            kv += din * c;
                        else
                            kv += dout * c;
                    });

            if (u != v || graph_tool::is_directed(_state._g))
            {
                ku += delta;
                kv += delta;
            }
            else
            {
                ku += 2 * delta;
                kv += 2 * delta;
            }

            _v_out_sampler[r].remove(_v_out_pos[u]);
            _v_out_pos[u] = _v_out_sampler[r].insert(u, ku + 1);

            if (u != v || graph_tool::is_directed(_state._g))
            {
                if (graph_tool::is_directed(_state._g))
                {
                    _v_in_sampler[s].remove(_v_in_pos[v]);
                    _v_in_pos[v] = _v_in_sampler[s].insert(v, kv + 1);
                }
                else
                {
                    _v_out_sampler[s].remove(_v_out_pos[v]);
                    _v_out_pos[v] = _v_out_sampler[s].insert(v, kv + 1);
                }
            }
        }
    }

    template <class RNG>
    std::tuple<size_t, size_t> sample(RNG& rng)
    {
        // std::uniform_int_distribution<size_t> sample(0, _N-1);
        // return {sample(rng), sample(rng)};

        if (_edges_only)
        {
            std::bernoulli_distribution coin(_edges.size() /
                                             double(_edges.size() + _N));
            if (coin(rng))
            {
                return uniform_sample(_edges, rng);
            }
            else
            {
                std::uniform_int_distribution<size_t> vsample(0, _N-1);
                auto v = vsample(rng);
                return {v, v};
            }
        }

        if (!_edges.empty())
        {
            std::bernoulli_distribution coin(.5);
            if (coin(rng))
                return uniform_sample(_edges, rng);
        }

        std::tuple<size_t, size_t> rs;
        auto E = graph_tool::is_directed(_state._g) ? _E : 2 * _E;
        std::bernoulli_distribution random(_NB/double(E + _NB));
        if (random(rng))
        {
            get<0>(rs) = uniform_sample(_groups, rng);
            get<1>(rs) = uniform_sample(_groups, rng);
        }
        else
        {
            rs = _rs_sampler.sample(rng);
        }

        auto& r_sampler = _v_out_sampler[get<0>(rs)];
        auto& s_sampler = _v_in_sampler[get<1>(rs)];
        return std::make_tuple(r_sampler.sample(rng),
                               s_sampler.sample(rng));
    }

    double log_prob(size_t u, size_t v, size_t m, int delta)
    {
        if (_edges_only)
            return 0;

        auto& g = _state._g;
        size_t r = _state._b[u];
        size_t s = _state._b[v];

        size_t ku = 0, kv = 0;
        if (_state._deg_corr)
        {
            degs_op(u, _state._vweight, _state._eweight, _state._degs, g,
                    [&] (size_t, size_t dout, auto c)
                    {
                        ku += dout * c;
                    });
            degs_op(v, _state._vweight, _state._eweight, _state._degs, g,
                    [&] (size_t din, size_t dout, auto c)
                    {
                        if (graph_tool::is_directed(g))
                            kv += din * c;
                        else
                            kv += dout * c;
                    });
        }

        auto&& me = _state._emat.get_me(r, s);
        size_t ers = (me == _state._emat.get_null_edge()) ? 0 : _state._mrs[me];
        ers += delta;

        if (!graph_tool::is_directed(g) && r == s)
            ers *= 2;

        size_t nr = _state._wr[r];
        size_t ns = _state._wr[s];
        size_t er = _state._mrp[r];
        size_t es = graph_tool::is_directed(g) ? _state._mrm[s] : _state._mrp[s];

        if (_state._deg_corr)
        {
            if (r != s || graph_tool::is_directed(g))
            {
                er += delta;
                es += delta;
            }
            else
            {
                er += 2 * delta;
                es += 2 * delta;
            }

            if (u != v || graph_tool::is_directed(g))
            {
                ku += delta;
                kv += delta;
            }
            else
            {
                ku += 2 * delta;
                kv += 2 * delta;
            }
        }
        else
        {
            er = es = 0;
        }

        auto E = graph_tool::is_directed(g) ? (_E + delta) : 2 * (_E + delta);

        double lp = (log(ers + 1) - log(E + _NB) +
                     log(ku + 1) - log(er + nr) +
                     log(kv + 1) - log(es + ns));

        if (!graph_tool::is_directed(g) && u != v)
            lp += log(2);

        if (m > 0)
        {
            double rp;
            if (m == 1 && delta == 1)
                rp = -log(_edges.size() + delta);
            else
                rp = -log(_edges.size());
            double x = std::max(rp, lp);
            double y = std::min(rp, lp);
            return x + log1p(exp(y - x)) - log(2);
        }
        else
        {
            return lp;
        }
    }

private:
    State& _state;

    typedef DynamicSampler<std::tuple<size_t, size_t>> rs_sampler_t;
    rs_sampler_t _rs_sampler;
    eprop_map_t<size_t>::type _rs_pos;

    typedef DynamicSampler<size_t> vsampler_t;
    vector<vsampler_t> __v_in_sampler, _v_out_sampler;
    vector<vsampler_t>& _v_in_sampler;

    vprop_map_t<size_t>::type _v_in_pos;
    vprop_map_t<size_t>::type _v_out_pos;

    std::vector<size_t> _groups;

    std::vector<std::tuple<size_t, size_t>> _edges;
    gt_hash_map<std::tuple<size_t, size_t>, size_t> _edge_pos;

    size_t _NB = 0;
    size_t _E = 0;
    size_t _N = 0;

    bool _edges_only;
};


} // graph_tool namespace

#endif // GRAPH_SBM_SAMPLE_EDGE_HH
