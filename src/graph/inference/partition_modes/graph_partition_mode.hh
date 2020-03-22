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

#ifndef GRAPH_PARTITION_MODE_HH
#define GRAPH_PARTITION_MODE_HH

#include "config.h"

#include <vector>

#include <boost/graph/maximum_weighted_matching.hpp>
#include "../blockmodel/graph_blockmodel_util.hh"
#include "../../topology/graph_bipartite_weighted_matching.hh"

#include "idx_map.hh"

namespace graph_tool
{

template <bool log_sum, class Graph, class PMap, class VMap, class EMap, class VX, class VY>
void get_contingency_graph(Graph& g, PMap partition, VMap label, EMap mrs,
                           VX& x, VY& y)
{
    typedef typename graph_traits<Graph>::vertex_descriptor vertex_t;
    typedef typename VX::value_type xv_t;
    typedef typename VY::value_type yv_t;

    idx_map<int, vertex_t> x_vertices, y_vertices;

    auto get_v =
        [&](auto& vs, auto val, auto pval)
        {
            auto iter = vs.find(val);
            if (iter == vs.end())
            {
                auto v = add_vertex(g);
                vs[val] = v;
                partition[v] = pval;
                return v;
            }
            return iter->second;
        };

    for (auto& r : x)
    {
        if constexpr (std::is_scalar_v<xv_t>)
        {
            if (r != -1)
                label[get_v(x_vertices, r, 0)] = r;
        }
        else
        {
            for (auto& rn : r)
                label[get_v(x_vertices, rn.first, 0)] = rn.first;
        }
    }

    for (auto& s : y)
    {
        if constexpr (std::is_scalar_v<yv_t>)
        {
            if (s != -1)
                label[get_v(y_vertices, s, 1)] = s;
        }
        else
        {
            for (auto& sn : s)
                label[get_v(y_vertices, sn.first, 1)] = sn.first;
        }
    }

    auto add_mrs
        = [&](auto i, auto u, auto c)
          {
              if constexpr (std::is_scalar_v<yv_t>)
              {
                  auto s = y[i];
                  if (s == -1)
                      return;

                  auto v = get_v(y_vertices, s, 1);
                  auto e = edge(u, v, g);
                  if (!e.second)
                      e = add_edge(u, v, g);
                  mrs[e.first] += c;
              }
              else
              {
                  for (auto& sn : y[i])
                  {
                      auto v = get_v(y_vertices, sn.first, 1);
                      auto e = edge(u, v, g);
                      if (!e.second)
                          e = add_edge(u, v, g);
                      if constexpr (log_sum)
                          mrs[e.first] += (lgamma_fast(sn.second + c + 1) -
                                           lgamma_fast(sn.second + 1) -
                                           lgamma_fast(c + 1));
                      else
                          mrs[e.first] += sn.second + c;
                  }
              }
          };

    for (size_t i = 0; i < x.size(); ++i)
    {
        if constexpr (std::is_scalar_v<xv_t>)
        {
            auto r = x[i];
            if (r == -1)
                continue;
            auto u = get_v(x_vertices, r, 0);
            add_mrs(i, u, 1);
        }
        else
        {
            for (auto& rn : x[i])
            {
                auto u = get_v(x_vertices, rn.first, 0);
                add_mrs(i, u, rn.second);
            }
        }
    }
}

template <class BV>
size_t partition_overlap(BV& x, BV& y)
{
        adj_list<> g;
        typename vprop_map_t<int32_t>::type label(get(vertex_index_t(), g));
        typename vprop_map_t<bool>::type partition(get(vertex_index_t(), g));
        typename eprop_map_t<double>::type mrs(get(edge_index_t(), g));

        get_contingency_graph<false>(g, partition, label, mrs, x, y);

        typedef typename graph_traits<adj_list<>>::vertex_descriptor vertex_t;
        typename vprop_map_t<vertex_t>::type match(get(vertex_index_t(), g));

        auto u = undirected_adaptor<adj_list<>>(g);
        //maximum_weighted_matching(u, mrs, match);
        maximum_bipartite_weighted_matching(u, partition, mrs, match);

        size_t m = 0;
        for (auto v : vertices_range(g))
        {
            if (partition[v])
                continue;
            auto w = match[v];
            if (w == graph_traits<adj_list<>>::null_vertex())
                continue;
            m += mrs[edge(v, w, u).first];
        }
        return m;
}

template <class BV>
void partition_align_labels(BV& x, BV& y)
{
        adj_list<> g;
        typename vprop_map_t<int32_t>::type label(get(vertex_index_t(), g));
        typename vprop_map_t<bool>::type partition(get(vertex_index_t(), g));
        typename eprop_map_t<double>::type mrs(get(edge_index_t(), g));

        get_contingency_graph<false>(g, partition, label, mrs, x, y);

        typedef typename graph_traits<adj_list<>>::vertex_descriptor vertex_t;
        typename vprop_map_t<vertex_t>::type match(get(vertex_index_t(), g));

        auto u = undirected_adaptor<adj_list<>>(g);
        //maximum_weighted_matching(u, mrs, match);
        maximum_bipartite_weighted_matching(u, partition, mrs, match);

        idx_set<int> used;
        idx_map<size_t, size_t> vertices, umatch;
        for (auto v : vertices_range(u))
        {
            if (partition[v])
                continue;
            vertices[label[v]] = v;
            auto w = match[v];
            if (w == graph_traits<adj_list<>>::null_vertex())
                continue;
            used.insert(label[w]);
        }

        std::vector<int> unused;
        for (size_t r = 0; r < x.size(); ++r)
        {
            if (used.find(r) != used.end())
                continue;
            unused.push_back(r);
        }
        auto iter = unused.begin();
        for (auto v : vertices_range(u))
        {
            if (partition[v])
                continue;
            auto w = match[v];
            if (w == graph_traits<adj_list<>>::null_vertex())
                umatch[label[v]] = *iter++;
        }

        for (auto& r : x)
        {
            auto w = match[vertices[r]];
            if (w == graph_traits<adj_list<>>::null_vertex())
                r = umatch[r];
            else
                r = label[w];
        }
}

class PartitionModeState
{
public:
    PartitionModeState() {}

    typedef std::vector<int32_t> b_t;
    typedef std::vector<std::reference_wrapper<b_t>> bv_t;

    template <class BT>
    void check_size(BT& b)
    {
        size_t n = std::max(_nr.size(), b.size());
        b.resize(n, -1);
        _nr.resize(n);
        _count.resize(n);
    }

    size_t add_partition(bv_t& bv, bool relabel)
    {
        if (_coupled_state == nullptr && bv.size() > 1)
        {
            auto* s = this;
            for (size_t i = 0; i < bv.size() - 1; ++i)
            {
                s->_coupled_state = std::make_shared<PartitionModeState>();
                s = s->_coupled_state.get();
            }
        }
        clean_labels(bv, 0);
        return add_partition(bv, 0, relabel);
    }

    void clean_labels(bv_t& bv, size_t pos)
    {
        if (bv.size() - pos == 1)
            return;
        idx_set<int> rs;
        for (auto r : bv[pos].get())
        {
            if (r == -1)
                continue;
            rs.insert(r);
        }
        b_t& c = bv[pos + 1];
        for (size_t r = 0; r < c.size(); ++r)
        {
            if (rs.find(r) == rs.end())
                c[r] = -1;
        }
        clean_labels(bv, pos + 1);
    }

    size_t add_partition(bv_t& bv, size_t pos, bool relabel)
    {
        auto& b = bv[pos].get();
        check_size(b);

        if (relabel && pos == 0)
            relabel_partition(bv, 0);

        for (size_t i = 0; i < b.size(); ++i)
        {
            auto r = b[i];
            if (r == -1)
                continue;
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
        _bs.insert(std::make_pair(j, std::ref(b)));

        if (_coupled_state != nullptr)
            _coupled_pos[j] = _coupled_state->add_partition(bv, pos + 1, false);

        return j;
    }

    void remove_partition(size_t j)
    {
        assert(_bs.find(j) != _bs.end());
        b_t& b = _bs.find(j)->second.get();
        for (size_t i = 0; i < b.size(); ++i)
        {
            auto r = b[i];
            if (r == -1)
                continue;
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
        if (_coupled_state != nullptr)
            _coupled_state->remove_partition(_coupled_pos[j]);
    }

    void replace_partitions()
    {
        std::vector<size_t> pos;
        for (auto ib : _bs)
            pos.push_back(ib.first);
        for (auto j : pos)
        {
            auto bv = get_nested_partition(j);
            double dS = virtual_remove_partition(bv);
            remove_partition(j);
            dS += virtual_add_partition(bv);
            add_partition(bv, dS < 0);
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
            auto& b = get_partition(jb.first);
            check_size(b);

            for (size_t i = 0; i < b.size(); ++i)
            {
                auto r = b[i];
                if (r == -1)
                    continue;
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
        for (int r = 0; r < _rmax; ++r)
        {
            if (_count[r] == 0)
                _free_idxs.insert(r);
        }
        if (_coupled_state != nullptr)
            _coupled_state->rebuild_nr();
    }

    void relabel()
    {
        std::vector<int32_t> labels(_count.size()), map(_count.size());
        std::iota(labels.begin(), labels.end(), 0);
        std::sort(labels.begin(), labels.end(),
                  [&](auto r, auto s) { return _count[r] > _count[s]; });

        for (size_t r = 0; r < _count.size(); ++r)
            map[labels[r]] = r;

        for (auto& jb : _bs)
        {
            auto& b = get_partition(jb.first);
            check_size(b);
            b_t b_orig = b;
            for (auto& r : b)
            {
                if (r == -1)
                    continue;
                r = map[r];
            }

            if (_coupled_state != nullptr)
            {
                auto& c = _coupled_state->get_partition(_coupled_pos[jb.first]);
                relabel_nested(b, b_orig, c);
            }
        }
        rebuild_nr();
        if (_coupled_state != nullptr)
            _coupled_state->relabel();
    }

    template <class BV>
    void relabel_partition(BV& bv, size_t pos)
    {
        b_t& b = bv[pos];

        check_size(b);

        b_t b_orig = b;

        idx_map<int32_t, size_t> rpos;
        for (size_t i = 0; i < b.size(); ++i)
        {
            auto r = b[i];
            if (r == -1)
                continue;
            auto iter = rpos.find(r);
            if (iter == rpos.end())
                b[i] = rpos[r] = rpos.size();
            else
                b[i] = iter->second;
        }

        if (_bs.empty())
        {
            if (_coupled_state != nullptr)
            {
                relabel_nested(b, b_orig, bv[pos + 1]);
                _coupled_state->relabel_partition(bv, pos + 1);
            }
            return;
        }

        adj_list<> g;
        typename vprop_map_t<int32_t>::type label(get(vertex_index_t(), g));
        typename vprop_map_t<bool>::type partition(get(vertex_index_t(), g));
        typename eprop_map_t<double>::type mrs(get(edge_index_t(), g));

        get_contingency_graph<true>(g, partition, label, mrs, b, _nr);

        typedef typename graph_traits<adj_list<>>::vertex_descriptor vertex_t;
        typename vprop_map_t<vertex_t>::type match(get(vertex_index_t(), g));

        auto u = undirected_adaptor<adj_list<>>(g);
        //maximum_weighted_matching(u, mrs, match);
        maximum_bipartite_weighted_matching(u, partition, mrs, match);

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
            if (r == -1)
                continue;
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

        if (_coupled_state != nullptr)
        {
            relabel_nested(b, b_orig, bv[pos + 1]);
            _coupled_state->relabel_partition(bv, pos + 1);
        }
    }

    void relabel_nested(b_t& b, b_t& b_orig, b_t& c)
    {
        b_t temp = c;
        std::fill(c.begin(), c.end(), -1);
        idx_map<int, int> rmap;
        for (size_t j = 0; j < b_orig.size(); ++j)
        {
            if (b_orig[j] == -1)
                continue;
            rmap[b_orig[j]] = b[j];
        }
        for (auto rs : rmap)
        {
            if (size_t(rs.second) >= c.size())
                c.resize(rs.second + 1, -1);
            c[rs.second] = temp[rs.first];
        }
    }

    b_t& get_partition(size_t i)
    {
        return _bs.find(i)->second.get();
    }

    auto& get_partitions()
    {
        return _bs;
    }

    bv_t get_nested_partition(size_t i)
    {
        bv_t bv;
        bv.push_back(std::ref(get_partition(i)));
        if (_coupled_state != nullptr)
        {
            auto bv_l =_coupled_state->get_nested_partition(_coupled_pos[i]);
            bv.insert(bv.end(), bv_l.begin(), bv_l.end());
        }
        return bv;
    }

    template <class Graph, class VM>
    void get_marginal(Graph& g, VM bm)
    {
        for (auto v : vertices_range(g))
        {
            if (v >= _nr.size())
                break;
            auto& h = bm[v];
            for (auto rn : _nr[v])
            {
                if (rn.first >= h.size())
                    h.resize(rn.first + 1);
                h[rn.first] = rn.second;
            }
        }
    }

    template <class Graph, class VM>
    void get_map(Graph& g, VM b)
    {
        for (auto v : vertices_range(g))
        {
            if (v >= _nr.size())
                break;
            size_t cmax = 0;
            int r = -1;
            for (auto rn : _nr[v])
            {
                if (rn.second > cmax)
                {
                    cmax = rn.second;
                    r = rn.first;
                }
            }
            b[v] = r;
        }
    }

    std::vector<int> get_map_b()
    {
        std::vector<int> b;
        for (auto& nrv : _nr)
        {
            size_t cmax = 0;
            int r = 0;
            for (auto& rn : nrv)
            {
                if (rn.second > cmax)
                {
                    cmax = rn.second;
                    r = rn.first;
                }
            }
            b.push_back(r);
        }
        return b;
    }

    double entropy()
    {
        double L = 0;
        size_t N = 0;
        for (auto& x : _nr)
        {
            size_t n = 0;
            for (auto& rn : x)
            {
                L += lgamma(rn.second + 1);
                n += rn.second;
            }
            if (n > 0)
            {
                N++;
                L += lgamma(_B) - lgamma(n + _B);
            }
        }
        L -= safelog(N);
        if (_coupled_state != nullptr)
            L -= _coupled_state->entropy();
        return -L;
    }

    double posterior_entropy(bool MLE)
    {
        if (_bs.size() == 0)
            return 0;
        double S = 0;
        for (auto& x : _nr)
        {
            if (_nr.empty())
                continue;
            double Si = 0;
            size_t n = 0;
            for (auto& rn : x)
            {
                Si -= xlogx(MLE ? rn.second : rn.second + 1);
                n += rn.second;
            }
            auto B = MLE ? 0 : _B;
            Si /= (n + B);
            Si += log(n + B);
            S += Si;
        }
        if (_coupled_state != nullptr)
            S += _coupled_state->posterior_entropy(MLE);
        return S;
    }

    double posterior_dev(bool MLE)
    {
        if (_bs.size() == 0)
            return 0;
        double ce = 0;
        size_t N = 0;
        for (auto& x : _nr)
        {
            if (_nr.empty())
                continue;
            size_t n = 0;
            for (auto& rn : x)
                n += rn.second;
            if (n == 0)
                continue;
            for (auto& rn : x)
            {
                double p;
                if (MLE)
                    p = rn.second / double(n);
                else
                    p = (rn.second + 1) / double(n + _B);
                ce += p * p;
            }
            N++;
        }
        return 1. - ce / N;
    }

    double posterior_cross_dev(PartitionModeState& m,
                               bool MLE)
    {
        if (_bs.size() == 0)
            return 0;
        double ce = 0;
        size_t N = 0;
        for (size_t i = 0; i < _nr.size(); ++i)
        {
            auto& x1 = _nr[i];
            auto& x2 = m._nr[i];
            if (x1.empty() || x2.empty())
                continue;
            size_t n1 = 0, n2 = 0;
            for (auto& rn : x1)
                n1 += rn.second;
            for (auto& rn : x2)
                n2 += rn.second;
            if (n1 == 0 || n2 == 0)
                continue;
            for (auto& rn : x1)
            {
                double p1, p2;
                size_t c1 = rn.second;
                size_t c2 = 0;
                auto iter = x2.find(rn.first);
                if (iter != x2.end())
                    c2 = iter->second;
                if (MLE)
                {
                    p1 = c1 / double(n1);
                    p2 = c2 / double(n2);
                }
                else
                {
                    p1 = (c1 + 1) / double(n1 + _B);
                    p2 = (c2 + 1) / double(n2 + m._B);
                }
                ce += p1 * p2;
            }
            N++;
        }
        return 1. - ce / N;
    }

    double posterior_cerror(bool MLE)
    {
        if (_bs.size() == 0)
            return 0;
        double ce = 0;
        size_t N = 0;
        for (auto& x : _nr)
        {
            if (x.empty())
                continue;
            size_t n = 0;
            size_t nmax = 0;
            for (auto& rn : x)
            {
                nmax = std::max(rn.second, nmax);
                n += rn.second;
            }
            if (n == 0)
                continue;
            if (MLE)
                ce += nmax / double(n);
            else
                ce += (nmax + 1) / double(n + _B);
            N++;
        }
        return 1. - ce / N;
    }

    template <class BV>
    double posterior_lprob(BV&& b, bool MLE)
    {
        if (_bs.size() == 0)
            return 0;
        double L = 0;
        for (size_t i = 0; i < _nr.size(); ++i)
        {
            auto& x  = _nr[i];
            if (x.empty())
                continue;
            size_t n = 0;
            for (auto& rn : x)
                n += rn.second;
            size_t nr = 0;
            auto iter = x.find(b[i]);
            if (iter != x.end())
                nr = iter->second;
            if (MLE)
                L += log(nr) - log(n);
            else
                L += log1p(nr) - log(n + _B);
        }
        return L;
    }

    template <bool add, class BV>
    double virtual_change_partition(const BV& x, size_t pos, bool relabel=false)
    {
        std::vector<b_t> bv;
        for (auto& b : x)
            bv.emplace_back(b);

        if (add && bv.size() - pos > 1 && _coupled_state == nullptr)
            _coupled_state = std::make_shared<PartitionModeState>();

        if (add && pos == 0 && relabel)
            relabel_partition(bv, 0);

        auto& b = bv[pos];

        check_size(b);

        double dL = 0;
        std::vector<int> delta_nr(_count.size());
        for (size_t i = 0; i < b.size(); ++i)
        {
            auto r = b[i];
            if (r == -1)
                continue;
            auto& x = _nr[i];

            int nir = 0;
            auto iter = x.find(r);
            if (iter != x.end())
                nir = iter->second;
            assert(nir >= 0);

            dL -= lgamma_fast(nir + 1);
            if (size_t(r) >= delta_nr.size())
                delta_nr.resize(r + 1);
            if constexpr (add)
            {
                dL += lgamma_fast(nir + 2);
                delta_nr[r]++;
            }
            else
            {
                assert(nir > 0);
                dL += lgamma_fast(nir);
                delta_nr[r]--;
            }
        }

        int dB = 0;
        for (size_t r = 0; r < delta_nr.size(); ++r)
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

        int dN = 0;
        size_t N = 0;
        for (size_t i = 0; i < _nr.size(); ++i)
        {
            int n = 0;
            for (auto rn : _nr[i])
                n += rn.second;

            assert(n >= 0);
            if (n > 0)
            {
                dL -= lgamma_fast(_B) - lgamma_fast(n + _B);
                N++;
            }
            if (b[i] != -1)
            {
                if constexpr (add)
                {
                    if (n == 0)
                        dN++;
                    n++;
                }
                else
                {
                    n--;
                    if (n == 0)
                        dN--;
                }
            }
            if (n > 0)
                dL += lgamma_fast(_B + dB) - lgamma_fast(n + _B + dB);
            assert(n >= 0);
        }

        dL -= safelog_fast(N + dN);
        dL += safelog_fast(N);

        if (_coupled_state != nullptr)
            dL -= _coupled_state->virtual_change_partition<add>(bv, pos + 1);

        return -dL;
    }

    double virtual_remove_partition(bv_t& x)
    {
        return virtual_change_partition<false>(x, 0, false);
    }

    double virtual_add_partition(bv_t& x, bool relabel = true)
    {
        return virtual_change_partition<true>(x, 0, relabel);
    }

    size_t get_N()
    {
        size_t N = 0;
        for (auto& x : _nr)
        {
            if (!x.empty())
                N++;
        }
        return N;
    }

    size_t get_B()
    {
        return _B;
    }

    std::shared_ptr<PartitionModeState> get_coupled_state()
    {
        return _coupled_state;
    }

    template <class RNG>
    b_t sample_partition(bool MLE, RNG& rng)
    {
        b_t b;
        std::vector<int> rs_base;
        std::vector<double> probs_base;
        if (not MLE)
        {
            for (size_t r = 0; r < _count.size(); ++r)
            {
                if (_count[r] == 0)
                    continue;
                rs_base.push_back(r);
                probs_base.push_back(1);
            }
        }

        for (auto& nrv : _nr)
        {
            if (nrv.empty())
                continue;
            auto rs = rs_base;
            auto probs = probs_base;
            for (auto rn : nrv)
            {
                rs.push_back(rn.first);
                probs.push_back(rn.second);
            }
            Sampler<int> sample(rs, probs);
            b.push_back(sample.sample(rng));
        }
        return b;
    }

    template <class RNG>
    std::vector<b_t> sample_nested_partition(bool MLE, RNG& rng)
    {
        std::vector<b_t> bv;
        bv.push_back(sample_partition(MLE, rng));
        if (_coupled_state != nullptr)
        {
            auto nbv = _coupled_state->sample_nested_partition(MLE, rng);
            bv.insert(bv.end(), nbv.begin(), nbv.end());
        }
        return bv;
    }

//private:

    idx_map<size_t, std::reference_wrapper<b_t>> _bs;
    std::vector<gt_hash_map<size_t, size_t>> _nr;
    std::vector<size_t> _count;
    size_t _B = 0;
    int _rmax = 0;
    std::set<size_t> _free_idxs;
    std::vector<size_t> _free_pos;
    size_t _max_pos = 0;

    std::shared_ptr<PartitionModeState> _coupled_state;
    idx_map<size_t, size_t> _coupled_pos;

};

} // graph_tool namespace

#endif //GRAPH_PARTITION_MODE_HH
