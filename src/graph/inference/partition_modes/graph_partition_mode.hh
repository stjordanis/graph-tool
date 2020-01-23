// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2019 Tiago de Paula Peixoto <tiago@skewed.de>
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

#ifndef GRAPH_PARTITION_MODE_HH
#define GRAPH_PARTITION_MODE_HH

#include "config.h"

#include <vector>

#include "../blockmodel/graph_blockmodel_util.hh"
#include <boost/graph/maximum_weighted_matching.hpp>

#include "idx_map.hh"

namespace graph_tool
{

template <class Graph, class VMap, class EMap, class VX, class VY>
void get_contingency_graph(Graph& g, VMap label, EMap mrs, VX& x, VY& y)
{
    typedef typename graph_traits<Graph>::vertex_descriptor vertex_t;
    typedef typename VX::value_type xv_t;
    typedef typename VY::value_type yv_t;

    idx_map<int, vertex_t> x_vertices, y_vertices;

    auto get_v =
        [&](auto& vs, auto val)
        {
            auto iter = vs.find(val);
            if (iter == vs.end())
            {
                auto v = add_vertex(g);
                vs[val] = v;
                return v;
            }
            return iter->second;
        };

    for (auto&& r : x)
    {
        if constexpr (std::is_scalar_v<xv_t>)
        {
            label[get_v(x_vertices, r)] = r;
        }
        else
        {
            for (auto& rn : r)
                label[get_v(x_vertices, rn.first)] = rn.first;
        }
    }

    for (auto&& s : y)
    {
        if constexpr (std::is_scalar_v<yv_t>)
        {
            label[get_v(y_vertices, s)] = s;
        }
        else
        {
            for (auto& sn : s)
                label[get_v(y_vertices, sn.first)] = sn.first;
        }
    }

    auto add_mrs
        = [&](auto i, auto u)
          {
              if constexpr (std::is_scalar_v<yv_t>)
              {
                  auto s = y[i];
                  auto v = get_v(y_vertices, s);
                  auto e = edge(u, v, g);
                  if (!e.second)
                      e = add_edge(u, v, g);
                  mrs[e.first]++;
              }
              else
              {
                  for (auto& sn : y[i])
                  {
                      auto v = get_v(y_vertices, sn.first);
                      auto e = edge(u, v, g);
                      if (!e.second)
                          e = add_edge(u, v, g);
                      mrs[e.first] += safelog_fast(sn.second + 1);
                  }
              }
          };

    for (size_t i = 0; i < x.size(); ++i)
    {
        if constexpr (std::is_scalar_v<xv_t>)
        {
            auto r = x[i];
            auto u = get_v(x_vertices, r);
            add_mrs(i, u);
        }
        else
        {
            for (auto& rn : x[i])
            {
                auto u = get_v(x_vertices, rn.first);
                add_mrs(i, u);
            }
        }
    }
}


class PartitionModeState
{
public:
    PartitionModeState(size_t N)
        : _N(N), _nr(N), _count(N) {}

    typedef multi_array_ref<int32_t,1> b_t;

    size_t add_partition(b_t& b, bool relabel)
    {
        if (relabel)
            relabel_partition(b);
        for (size_t i = 0; i < b.size(); ++i)
        {
            size_t r = b[i];
            _nr[i][r]++;
            if (r >= _count.size())
                _count.resize(r + 1);
            _count[r]++;
            if (_count[r] == 1)
            {
                _B++;
                _free_idxs.erase(r);
            }
            if (r > _rmax)
                _rmax = r;
        }
        size_t j;
        if (_free_pos.empty())
        {
            j = _max_pos++;
        }
        else
        {
            j = _free_pos.back();
            _free_pos.pop_back();
        }
        _bs[j] = b.data();
        return j;
    }

    void remove_partition(size_t j)
    {
        assert(_bs.find(j) != _bs.end());
        auto b = b_t(_bs[j], extents[_N]);
        for (size_t i = 0; i < b.size(); ++i)
        {
            auto r = b[i];
            auto iter = _nr[i].find(r);
            iter->second--;
            if (iter->second == 0)
                _nr[i].erase(iter);
            _count[r]--;
            if (_count[r] == 0)
            {
                _B--;
                _free_idxs.insert(r);
            }
        }
        _bs.erase(j);
        _free_pos.push_back(j);
    }

    void replace_partitions()
    {
        std::vector<size_t> pos;
        for (auto ib : _bs)
            pos.push_back(ib.first);
        for (auto j : pos)
        {
            auto b = get_partition(j);
            double dS = virtual_remove_partition(b);
            remove_partition(j);
            dS += virtual_add_partition(b);
            add_partition(b, dS < 0);
        }
    }

    bool has_partition(size_t j, b_t& b)
    {
        if (_bs.find(j) == _bs.end())
            return false;
        return (b == get_partition(j));
    }

    void rebuild_nr()
    {
        _B = 0;
        _rmax = 0;
        for (auto& x : _nr)
            x.clear();
        std::fill(_count.begin(), _count.end(), 0);
        for (auto jb : _bs)
        {
            auto b = get_partition(jb.first);
            for (size_t i = 0; i < b.size(); ++i)
            {
                size_t r = b[i];
                _nr[i][r]++;
                _count[r]++;
                if (_count[r] == 1)
                {
                    _B++;
                    _free_idxs.erase(r);
                }
                if (r > _rmax)
                    _rmax = r;
            }
        }
        for (size_t r = 0; r < _rmax; ++r)
        {
            if (_count[r] == 0)
                _free_idxs.insert(r);
        }
    }

    void relabel()
    {
        std::vector<int> labels(_N), map(_N);
        std::iota(labels.begin(), labels.end(), 0);
        std::sort(labels.begin(), labels.end(),
                  [&](auto r, auto s) { return _count[r] > _count[s]; });

        for (size_t i = 0; i < _N; ++i)
            map[labels[i]] = i;

        for (auto jb : _bs)
        {
            auto b = get_partition(jb.first);
            for (size_t i = 0; i < _N; ++i)
                b[i] = map[b[i]];
        }
        rebuild_nr();
    }


    template <class BT>
    void relabel_partition(BT& b)
    {
        idx_map<int32_t, size_t> rpos;
        for (size_t i = 0; i < b.size(); ++i)
        {
            auto r = b[i];
            auto iter = rpos.find(r);
            if (iter == rpos.end())
                b[i] = rpos[r] = rpos.size();
            else
                b[i] = iter->second;
        }

        if (_bs.empty())
            return;

        adj_list<> g;
        typename vprop_map_t<int32_t>::type label(get(vertex_index_t(), g));
        typename eprop_map_t<long double>::type mrs(get(edge_index_t(), g));

        get_contingency_graph(g, label, mrs, b, _nr);

        typedef typename graph_traits<adj_list<>>::vertex_descriptor vertex_t;
        typename vprop_map_t<vertex_t>::type match(get(vertex_index_t(), g));

        auto u = undirected_adaptor<adj_list<>>(g);
        maximum_weighted_matching(u, mrs, match);

        idx_map<int32_t, size_t> b_vertices;
        for (auto v : vertices_range(g))
        {
            if (v >= rpos.size())
                break;
            b_vertices[label[v]] = v;
        }

        auto ipos = _free_idxs.begin();
        rpos.clear();
        for (size_t i = 0; i < b.size(); ++i)
        {
            auto r = b[i];
            auto v = match[b_vertices[r]];
            if (v != graph_traits<adj_list<>>::null_vertex())
            {
                b[i] = label[v];
            }
            else
            {
                auto iter = rpos.find(r);
                if (iter == rpos.end())
                {
                    if (ipos == _free_idxs.end())
                        ipos = _free_idxs.insert(++_rmax).first;
                    rpos[r] = b[i] = *(ipos++);
                }
                else
                {
                    b[i] = iter->second;
                }
            }
        }
    }

    b_t get_partition(size_t i)
    {
        return b_t(_bs[i], extents[_N]);
    }

    auto& get_partitions()
    {
        return _bs;
    }

    template <class Graph, class VM>
    void get_marginal(Graph& g, VM bm)
    {
        for (auto v : vertices_range(g))
        {
            auto& h = bm[v];
            for (auto rn : _nr[v])
            {
                if (rn.first >= h.size())
                    h.resize(rn.first + 1);
                h[rn.first] = rn.second;
            }
        }
    }

    double entropy()
    {
        if (_bs.empty())
            return 0;

        double L = 0;
        for (auto& x : _nr)
        {
            size_t n = 0;
            for (auto& rn : x)
            {
                L += lgamma(rn.second + 1);
                n += rn.second;
            }
            assert(n == _bs.size());
            L += lgamma(_B) - lgamma(n + _B);
        }
        L -= log(_N);
        return -L;
    }

    double posterior_entropy()
    {
        size_t M = _bs.size();
        if (M == 0)
            return 0;
        double S = 0;
        for (auto& x : _nr)
        {
            double Si = 0;
            for (auto& rn : x)
                Si -= xlogx(rn.second + 1);
            Si /= (M + _B);
            Si += log(M + _B);
            S += Si;
        }
        return S;
    }

    template <bool add>
    double virtual_change_partition(b_t& x)
    {
        multi_array<int32_t, 1> b(extents[_N]);
        b = x;

        if constexpr (add)
            relabel_partition(b);

        double dL = 0;
        std::vector<int> delta_nr(_N);
        for (size_t i = 0; i < b.size(); ++i)
        {
            size_t r = b[i];
            size_t nir = 0;

            auto& x = _nr[i];
            auto iter = x.find(r);
            if (iter != x.end())
                nir = iter->second;

            dL -= lgamma_fast(nir + 1);
            if constexpr (add)
            {
                dL += lgamma_fast(nir + 2);
                delta_nr[r]++;
            }
            else
            {
                dL += lgamma_fast(nir);
                delta_nr[r]--;
            }
        }

        int dB = 0;
        for (size_t r = 0; r < _N; ++r)
        {
            if constexpr (add)
            {
                if (_count[r] == 0 && delta_nr[r] > 0)
                    dB++;
            }
            else
            {
                if (_count[r] > 0 && _count[r] + delta_nr[r] == 0)
                    dB--;
            }
        }

        size_t M = _bs.size();
        double dLi = 0;
        if constexpr (add)
            dLi += lgamma_fast(_B + dB) - lgamma_fast(M + _B + dB + 1);
        else if (M > 1)
            dLi += lgamma_fast(_B + dB) - lgamma_fast(M + _B + dB - 1);
        if (M > 0)
            dLi -= lgamma_fast(_B) - lgamma_fast(M + _B);
        dL += _N * dLi;

        if (!add && M == 1)
            dL += log(_N);
        if (add && M == 0)
            dL -= log(_N);

        return -dL;
    }

    double virtual_remove_partition(b_t& x)
    {
        return virtual_change_partition<false>(x);
    }

    double virtual_add_partition(b_t& x)
    {
        return virtual_change_partition<true>(x);
    }

//private:

    size_t _N;
    idx_map<size_t, int32_t*> _bs;
    std::vector<gt_hash_map<size_t, size_t>> _nr;
    std::vector<size_t> _count;
    size_t _B = 0;
    size_t _rmax = 0;
    std::set<size_t> _free_idxs;
    std::vector<size_t> _free_pos;
    size_t _max_pos = 0;
};

} // graph_tool namespace

#endif //GRAPH_PARTITION_MODE_HH
