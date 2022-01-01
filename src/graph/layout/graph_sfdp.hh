// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2022 Tiago de Paula Peixoto <tiago@skewed.de>
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

#ifndef GRAPH_FDP_HH
#define GRAPH_FDP_HH

#include <limits>
#include <iostream>

#include "idx_map.hh"

namespace graph_tool
{
using namespace std;
using namespace boost;

template <class Value>
Value pow2(Value x)
{
    return x * x;
}

template <class Val, class Weight>
class QuadTree
{
public:

    typedef typename std::array<Val, 2> pos_t;

    class TreeNode
    {
    public:
        template <class Pos>
        TreeNode(const Pos& ll, const Pos& ur, size_t level)
            : _ll(ll), _ur(ur), _cm{0,0}, _level(level), _count(0) {}

        double get_w()
        {
            return sqrt(pow2(_ur[0] - _ll[0]) +
                        pow2(_ur[1] - _ll[1]));
        }

        template <class Pos>
        void get_cm(Pos& cm)
        {
            for (size_t i = 0; i < 2; ++i)
                cm[i] = _cm[i] / _count;
        }

        Weight get_count()
        {
            return _count;
        }

        friend class QuadTree;
    private:
        pos_t _ll;
        pos_t _ur;
        std::array<double,2> _cm;
        size_t _level;
        Weight _count;
        size_t _leafs = std::numeric_limits<size_t>::max();
    };

    QuadTree(): _max_level(0) {}

    template <class Pos>
    QuadTree(const Pos& ll, const Pos& ur, int max_level, size_t n)
        : _tree(1, {ll, ur, 0}), _dense_leafs(1), _max_level(max_level)
    {
        _tree.reserve(n);
        _dense_leafs.reserve(n);
    }

    auto& operator[](size_t pos)
    {
        return _tree[pos];
    }

    size_t get_leafs(size_t pos)
    {
        auto& node = _tree[pos];

        size_t level = _tree[pos]._level;
        if (level >= _max_level)
            return _tree.size();

        if (node._leafs >= _tree.size())
        {
            auto ll = node._ll;
            auto ur = node._ur;
            auto level = node._level;

            node._leafs = _tree.size();
            //_tree.reserve(_tree.size() + 4);
            for (size_t i = 0; i < 4; ++i)
            {
                pos_t lll = ll, lur = ur;
                if (i % 2)
                    lll[0] += (ur[0] - ll[0]) / 2;
                else
                    lur[0] -= (ur[0] - ll[0]) / 2;
                if (i / 2)
                    lll[1] += (ur[1] - ll[1]) / 2;
                else
                    lur[1] -= (ur[1] - ll[1]) / 2;
                _tree.emplace_back(lll, lur, level + 1);
            }
            _dense_leafs.resize(_tree.size());
        }

        return _tree[pos]._leafs;
    }

    auto& get_dense_leafs(size_t pos)
    {
        return _dense_leafs[pos];
    }

    template <class Pos>
    size_t get_branch(size_t pos, Pos& p)
    {
        auto& n = _tree[pos];
        int i = p[0] > (n._ll[0] + (n._ur[0] - n._ll[0]) / 2);
        int j = p[1] > (n._ll[1] + (n._ur[1] - n._ll[1]) / 2);
        return i + 2 * j;
    }

    template <class Pos>
    void put_pos(size_t pos, Pos& p, Weight w)
    {
        while (pos < _tree.size())
        {
            auto& node = _tree[pos];

            node._count += w;
            node._cm[0] += p[0] * w;
            node._cm[1] += p[1] * w;

            if (node._level >= _max_level || node._count == w)
            {
                _dense_leafs[pos].emplace_back(pos_t{p[0], p[1]}, w);
                pos = _tree.size();
            }
            else
            {
                auto leafs = get_leafs(pos);

                if (!_dense_leafs[pos].empty())
                {
                    // move dense leafs down
                    for (auto& leaf : _dense_leafs[pos])
                    {
                        auto& lp = get<0>(leaf);
                        auto& lw = get<1>(leaf);
                        put_pos(leafs + get_branch(pos, lp), lp, lw);
                    }
                    _dense_leafs[pos].clear();
                }
                pos = leafs + get_branch(pos, p);
            }
        }
    }

    size_t size()
    {
        return _tree.size();
    }

private:
    vector<TreeNode> _tree;
    vector<vector<std::tuple<pos_t,Weight>>> _dense_leafs;
    size_t _max_level;
};

template <class Pos1, class Pos2>
static double dist(const Pos1& p1, const Pos2& p2)
{
    double r = 0;
    for (size_t i = 0; i < 2; ++i)
        r += pow2(double(p1[i] - p2[i]));
    return sqrt(r);
}

template <class Pos1, class Pos2>
static double f_r(double C, double K, double p, const Pos1& p1, const Pos2& p2)
{
    double d = dist(p1, p2);
    if (d == 0)
        return 0;
    return -C * pow(K, 1 + p) / pow(d, p);
}

template <class Pos1, class Pos2>
static double f_a(double K, const Pos1& p1, const Pos2& p2)
{
    return pow2(dist(p1, p2)) / K;
}

template <class Pos1, class Pos2, class Pos3>
static double get_diff(const Pos1& p1, const Pos2& p2, Pos3& r)
{
    double abs = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        r[i] = p1[i] - p2[i];
        abs += r[i] * r[i];
    }
    if (abs == 0)
        abs = 1;
    abs = sqrt(abs);
    for (size_t i = 0; i < 2; ++i)
        r[i] /= abs;
    return abs;
}

template <class Pos>
static double norm(Pos& x)
{
    double abs = 0;
    for (size_t i = 0; i < 2; ++i)
        abs += pow2(x[i]);
    for (size_t i = 0; i < 2; ++i)
        x[i] /= sqrt(abs);
    return sqrt(abs);
}


template <class Graph, class PosMap, class VertexWeightMap,
          class EdgeWeightMap, class PinMap, class GroupMaps, class CMap,
          class RNG>
void  get_sfdp_layout(Graph& g, PosMap pos, VertexWeightMap vweight,
                      EdgeWeightMap eweight, PinMap pin, GroupMaps& groups,
                      double C, double K, double p, double theta, double gamma,
                      double mu, double kappa, double r, CMap c,
                      double init_step, double step_schedule, size_t max_level,
                      double epsilon, size_t max_iter, bool simple,
                      bool verbose, RNG& rng)
{
    typedef typename property_traits<PosMap>::value_type::value_type val_t;
    typedef std::array<val_t, 2> pos_t;

    typedef typename property_traits<VertexWeightMap>::value_type vweight_t;

    vector<size_t> vertices;
    idx_map<size_t, size_t> rs;

    int HN = 0;
    for (auto v : vertices_range(g))
    {
        if (pin[v] == 0)
            vertices.push_back(v);
        pos[v].resize(2, 0);
        HN++;

        rs[groups[0][v]]++;
    }

    val_t delta = epsilon * K + 1, E = 0, E0;
    E0 = numeric_limits<val_t>::max();
    size_t n_iter = 0;
    val_t step = init_step;
    size_t progress = 0;

    vector<pos_t> ccm;
    vector<size_t> csize;

    while (delta > epsilon * K && (max_iter == 0 || n_iter < max_iter))
    {
        delta = 0;
        E0 = E;
        E = 0;

        pos_t ll{numeric_limits<val_t>::max(), numeric_limits<val_t>::max()},
            ur{-numeric_limits<val_t>::max(), -numeric_limits<val_t>::max()};

        ccm.clear();
        csize.clear();
        for (auto v : vertices_range(g))
        {
            for (size_t j = 0; j < 2; ++j)
            {
                ll[j] = min(pos[v][j], ll[j]);
                ur[j] = max(pos[v][j], ur[j]);
            }

            size_t s = c[v];
            if (s >= ccm.size())
            {
                ccm.resize(s + 1);
                csize.resize(s + 1, 0);
            }

            csize[s] += get(vweight, v);
            for (size_t j = 0; j < 2; ++j)
                ccm[s][j] += pos[v][j] * get(vweight, v);
        }

        for (size_t s = 0; s < ccm.size(); ++s)
        {
            if (csize[s] == 0)
                continue;
            for (size_t j = 0; j < 2; ++j)
                ccm[s][j] /= csize[s];
        }

        QuadTree<val_t, vweight_t> qt(ll, ur, max_level, num_vertices(g));
        idx_map<size_t, QuadTree<val_t, vweight_t>> qtr;
        for (auto& [r, nr] : rs)
            qtr[r] = QuadTree<val_t, vweight_t>(ll, ur, max_level, nr);

        for (auto v : vertices_range(g))
        {
            if (rs.size() > 1)
                qtr[groups[0][v]].put_pos(0, pos[v], vweight[v]);
            qt.put_pos(0, pos[v], vweight[v]);
        }

        std::shuffle(vertices.begin(), vertices.end(), rng);

        size_t nmoves = 0;
        vector<size_t> Q;
        Q.reserve(num_vertices(g));

        size_t nopen = 0;

        auto get_rf_bh =
            [&](auto v, auto& qt, auto& Q, auto& ftot, bool groups, bool intra)
            {
                pos_t cm{0, 0}, diff{0, 0};

                Q.push_back(0);
                while (!Q.empty())
                {
                    size_t q = Q.back();
                    Q.pop_back();

                    auto& dleafs = qt.get_dense_leafs(q);
                    if (!dleafs.empty())
                    {
                        for (auto& dleaf : dleafs)
                        {
                            val_t d = get_diff(get<0>(dleaf), pos[v], diff);
                            if (d == 0)
                                continue;
                            val_t f;
                            if (groups)
                            {
                                if (intra)
                                    f = -f_r(mu, K, gamma, pos[v], get<0>(dleaf));
                                else
                                    f = f_r(mu, K, gamma, pos[v], get<0>(dleaf)) + f_r(C, K, p, pos[v], get<0>(dleaf));
                            }
                            else
                            {
                                f = f_r(C, K, p, pos[v], get<0>(dleaf));
                            }
                            f *= get<1>(dleaf) * get(vweight, v);
                            for (size_t l = 0; l < 2; ++l)
                                ftot[l] += f * diff[l];

                            nopen++;
                        }
                    }
                    else
                    {
                        double w = qt[q].get_w();
                        qt[q].get_cm(cm);
                        double d = get_diff(cm, pos[v], diff);
                        if (w > theta * d)
                        {
                            auto leaf = qt.get_leafs(q);
                            for (size_t i = 0; i < 4; ++i)
                            {
                                if (qt[leaf].get_count() > 0)
                                    Q.push_back(leaf);
                                ++leaf;
                            }
                        }
                        else
                        {
                            if (d > 0)
                            {
                                val_t f;
                                if (groups)
                                {
                                    if (intra)
                                        f = -f_r(mu, K, gamma, pos[v], cm);
                                    else
                                        f = f_r(mu, K, gamma, pos[v], cm) + f_r(C, K, p, pos[v], cm);
                                }
                                else
                                {
                                    f = f_r(C, K, p, pos[v], cm);
                                }
                                f *= qt[q].get_count() * get(vweight, v);
                                for (size_t l = 0; l < 2; ++l)
                                    ftot[l] += f * diff[l];

                                nopen++;
                            }
                        }
                    }
                }
            };

        #pragma omp parallel if (num_vertices(g) > OPENMP_MIN_THRESH)   \
            private(Q) reduction(+:E, delta, nmoves)
        parallel_loop_no_spawn
            (vertices,
             [&](size_t, auto v)
             {
                 pos_t diff{0, 0}, ftot{0, 0};

                 // global repulsive forces
                 if (rs.size() == 1)
                 {
                     get_rf_bh(v, qt, Q, ftot, false, false);
                 }
                 else
                 {
                     get_rf_bh(v, qt, Q, ftot, true, false);
                     get_rf_bh(v, qtr[groups[0][v]], Q, ftot, true, true);
                 }

                 // local attractive forces
                 auto& pos_v = pos[v];
                 for (auto e : out_edges_range(v, g))
                 {
                     auto u = target(e, g);
                     if (u == v)
                         continue;
                     auto& pos_u = pos[u];
                     get_diff(pos_u, pos_v, diff);
                     val_t f = f_a(K, pos_u, pos_v);
                     f *= get(eweight, e) * get(vweight, u) * get(vweight, v);
                     if (groups[0][v] == groups[0][u])
                         f *= kappa;
                     for (size_t l = 0; l < 2; ++l)
                         ftot[l] += f * diff[l];
                 }

                 // inter-component attractive forces
                 if (r > 0)
                 {
                     for (size_t s = 0; s < ccm.size(); ++s)
                     {
                         if (csize[s] == 0)
                             continue;
                         if (s == size_t(c[v]))
                             continue;
                         val_t d = get_diff(ccm[s], pos[v], diff);
                         if (d == 0)
                             continue;
                         double Kp = K * pow2(HN);
                         val_t f = f_a(Kp, ccm[s], pos[v]) * r * csize[s] * get(vweight, v);
                         for (size_t l = 0; l < 2; ++l)
                             ftot[l] += f * diff[l];
                     }
                 }

                 E += pow2(norm(ftot));

                 for (size_t l = 0; l < 2; ++l)
                 {
                     ftot[l] *= step;
                     pos[v][l] += ftot[l];
                 }

                 delta += norm(ftot);
                 nmoves++;
             });

        n_iter++;
        delta /= nmoves;

        if (verbose)
            cout << n_iter << " " << E << " " << step << " "
                 << delta << " " << max_level << " "
                 << nopen / double(HN) << endl;

        if (simple)
        {
            step *= step_schedule;
        }
        else
        {
            if (E < E0)
            {
                ++progress;
                if (progress >= 5)
                {
                    progress = 0;
                    step /= step_schedule;
                }
            }
            else
            {
                progress = 0;
                step *= step_schedule;
            }
        }
    }
}

} // namespace graph_tool


#endif // GRAPH_FDP_HH
