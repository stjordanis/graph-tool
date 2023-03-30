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

#ifndef MULTILEVEL_HH
#define MULTILEVEL_HH

#include <queue>
#include <cmath>

#include "mcmc_loop.hh"

#include <boost/range/combine.hpp>
#include "../../topology/graph_bipartite_weighted_matching.hh"
#include "../support/contingency.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

template <class State, class Node, class Group,
          template <class> class VSet,
          template <class, class> class VMap,
          template <class> class GSet,
          template <class, class> class GMap,
          class GSMap, bool allow_empty=false,
          bool labelled=false>
struct Multilevel: public State
{
    enum class move_t { single = 0, multilevel, null };

    template <class... TS>
    Multilevel(TS&&... as)
        : State(as...)
        {
            State::iter_nodes
                ([&](const auto& v)
                {
                    auto r = State::get_group(v);
                    _groups[r].insert(v);
                    _N++;
                    _nodes.insert(v);
                });

            State::iter_groups
                ([&](const auto& r)
                {
                    _rlist.insert(r);
                });

            std::vector<move_t> moves
                = {move_t::single, move_t::multilevel};
            std::vector<double> probs
                = {State::_psingle, State::_pmultilevel};
            _move_sampler = Sampler<move_t, mpl::false_>(moves, probs);
        }

    VSet<Node> _nodes;
    GSMap _groups;

    using State::_state;
    using State::_merge_sweeps;
    using State::_mh_sweeps;
    using State::_init_r;
    using State::_gibbs;
    using State::_null_group;
    using State::_beta;
    using State::_init_beta;
    using State::_cache_states;
    using State::_verbose;

    using State::_B_min;
    using State::_B_max;
    using State::_b_min;
    using State::_b_max;

    size_t _nmoves = 0;

    std::vector<std::vector<std::tuple<Node,Group>>> _bstack;

    Sampler<move_t, mpl::false_> _move_sampler;

    template <class Vs>
    void push_b(Vs& vs)
    {
        _bstack.emplace_back();
        auto& back = _bstack.back();
        for (const auto& v : vs)
            back.emplace_back(v, State::get_group(v));
        State::push_state(vs);
    }

    void pop_b()
    {
        auto& back = _bstack.back();
        for (auto& vb : back)
        {
            auto& v = get<0>(vb);
            auto& s = get<1>(vb);
            move_node(v, s);
        }
        _bstack.pop_back();
        State::pop_state();
    }

    GSet<Group> _rlist;

    std::vector<Node> _vs;
    move_t _move;
    GSet<Group> _rs, _rs_prev;
    GMap<Group, pair<Group, double>> _best_merge;
    GSet<Group> _past_merges;
    GMap<Group, Group> _root;

    VSet<Node> _visited;

    VMap<Node, Group> _bnext, _bprev, _btemp;

    constexpr static size_t _null_move = 1;

    size_t _N = 0;

    double _dS;
    double _a;

    Group node_state(const Node&)
    {
        return Group();
    }

    constexpr bool skip_node(const Node&)
    {
        return false;
    }

    void move_node(const Node& v, const Group& r, bool cache = false)
    {
        Group s = State::get_group(v);
        if (s == r)
            return;
        State::move_node(v, r, cache);
        auto& vs = _groups[s];
        vs.erase(v);
        if (vs.empty())
            _groups.erase(s);
        _groups[r].insert(v);
        _nmoves++;
    }

    template <bool clear=true>
    void get_group_vs(const Group& r, std::vector<Node>& vs)
    {
        if constexpr (clear)
            vs.clear();
        auto iter = _groups.find(r);
        if (iter != _groups.end())
            vs.insert(vs.end(), iter->second.begin(), iter->second.end());
    }

    size_t get_wr(const Group& r)
    {
        auto iter = _groups.find(r);
        if (iter != _groups.end())
            return iter->second.size();
        return 0;
    }

    std::vector<size_t> _vis;

    template <bool smart, class RNG>
    std::pair<double, double>
    mh_sweep(std::vector<Node>& vs, GSet<Group>& rs, double beta, RNG& rng,
             size_t B_min = std::numeric_limits<size_t>::max(),
             bool init_heuristic = false)
    {
        if (rs.size() == 1 || (rs.size() == vs.size() && B_min == rs.size()))
            return {0, 0};

        _vis.resize(vs.size());
        std::iota(_vis.begin(), _vis.end(), 0);
        std::shuffle(_vis.begin(), _vis.end(), rng);

        double S = 0;
        double lp = 0;
        for (size_t vi : _vis)
        {
            const auto& v = vs[vi];

            auto r = State::get_group(v);
            Group s;

            if constexpr (smart)
            {
                s = State::sample_group(v, false, false, init_heuristic, rng); // c == 0!
                if (rs.find(s) == rs.end())
                    continue;
            }
            else
            {
                rs.erase(r);
                s = uniform_sample(rs, rng);
                rs.insert(r);
            }

            double dS;
            if (s != r && (get_wr(r) == 1 && rs.size() <= B_min))
                dS = std::numeric_limits<double>::infinity();
            else
                dS = State::virtual_move(v, r, s);

            double pf = 0, pb = 0;
            if constexpr (smart)
            {
                if (!std::isinf(beta) && s != r)
                {
                    pf = State::get_move_prob(v, r, s, false, rs.size() > B_min, false);
                    pb = State::get_move_prob(v, s, r, false, rs.size() > B_min, true);
                }
            }

            double ap = 0, rp = 0;
            bool accept;
            if constexpr (smart)
            {
                accept = metropolis_accept(dS, pb - pf, beta, rng);
            }
            else
            {
                if (!std::isinf(beta))
                {
                    double logZ = log_sum_exp(-beta * dS, 0.);
                    ap = -beta * dS - logZ;
                    rp = -logZ;

                    std::bernoulli_distribution u(exp(ap));
                    accept = u(rng);
                }
                else
                {
                    accept = dS < 0;
                    ap = 0;
                    rp = -std::numeric_limits<double>::infinity();
                    if (!accept)
                        std::swap(ap, rp);
                }
            }

            if (accept)
            {
                move_node(v, s, true);
                S += dS;
                lp += ap;

                if constexpr (!smart)
                    lp -= safelog_fast(rs.size() - 1);

                if (get_wr(r) == 0)
                    rs.erase(r);

                assert(r != s || dS == 0);
            }
            else
            {
                lp += rp;
            }
        }

        return {S, lp};
    }


    template <class RNG>
    double mh_sweep_prob(std::vector<Node>& vs, GSet<Group>& rs, double beta,
                         RNG& rng)
    {
        if (rs.size() == 1 || rs.size() == vs.size())
            return 0;

        _vis.resize(vs.size());
        std::iota(_vis.begin(), _vis.end(), 0);
        std::shuffle(_vis.begin(), _vis.end(), rng);

        gt_hash_map<Group, Group> mu;
        if constexpr (!labelled)
            mu = relabel_rs(vs, _bprev);

        double lp = 0;

        for (auto& v : vs)
            _btemp[v] = State::get_group(v);

        for (size_t vi : _vis)
        {
            const auto& v = vs[vi];

            auto r = State::get_group(v);
            Group s = labelled ? _bprev[v] : mu[_bprev[v]];

            if (s != r && get_wr(r) == 1)
            {
                lp = -std::numeric_limits<double>::infinity();
                break;
            }

            bool accept = true;
            if (s == r)
            {
                rs.erase(r);
                s = uniform_sample(rs, rng);
                rs.insert(r);
                accept = false;
            }
            else
            {
                lp -= safelog_fast(rs.size()-1);
            }

            double dS;
            if (s != r && get_wr(State::get_group(v)) == 1)
                dS = std::numeric_limits<double>::infinity();
            else
                dS = State::virtual_move(v, r, s);

            double ap, rp;
            if (!std::isinf(beta))
            {
                double logZ = log_sum_exp(-beta * dS, 0.);
                ap = -beta * dS - logZ;
                rp = -logZ;
            }
            else
            {
                if (dS < 0)
                {
                    ap = 0;
                    rp = -std::numeric_limits<double>::infinity();
                }
                else
                {
                    ap = -std::numeric_limits<double>::infinity();
                    rp = 0;
                }
            }

            if (accept)
            {
                move_node(v, s, true);
                lp += ap;
            }
            else
            {
                lp += rp;
            }
        }

        for (auto& v : vs)
            move_node(v, _btemp[v]);

        return lp;
    }


    template <bool forward, class RNG>
    std::pair<double, double>
    gibbs_sweep(std::vector<Node>& vs, GSet<Group>& rs, RNG& rng)
    {
        if (rs.size() == 1 || rs.size() == vs.size())
            return {0, 0};

        _vis.resize(vs.size());
        std::iota(_vis.begin(), _vis.end(), 0);
        std::shuffle(_vis.begin(), _vis.end(), rng);

        gt_hash_map<Group, Group> mu;

        if constexpr (!forward && !labelled)
            mu = relabel_rs(vs, _bprev);

        double S = 0;
        double lp = 0;

        std::vector<double> dS(rs.size());
        std::vector<double> probs(rs.size());
        std::vector<double> ps(rs.size());
        std::vector<size_t> ris(rs.size());
        std::iota(ris.begin(), ris.end(), 0);

        if constexpr (!forward)
        {
            for (auto& v : vs)
                _btemp[v] = State::get_group(v);
        }

        for (size_t vi : _vis)
        {
            const auto& v = vs[vi];

            auto r = State::get_group(v);

            for (size_t j = 0; j < rs.size(); ++j)
            {
                auto iter = rs.begin();
                std::advance(iter, j);
                Group s = *iter;
                if (s != r && get_wr(State::get_group(v)) == 1)
                    dS[j] = std::numeric_limits<double>::infinity();
                else
                    dS[j] = State::virtual_move(v, r, s);
            }

            double Z = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < rs.size(); ++j)
            {
                if (!std::isinf(_beta) && !std::isinf(dS[j]))
                {
                    ps[j] = -dS[j] * _beta;
                }
                else
                {
                    if (dS[j] < 0)
                        ps[j] = 0;
                    else
                        ps[j] = -std::numeric_limits<double>::infinity();
                }
                Z = log_sum_exp(Z, ps[j]);
            }

            size_t si = rs.size();
            if constexpr (forward)
            {
                for (size_t j = 0; j < rs.size(); ++j)
                    probs[j] = exp(ps[j] - Z);

                Sampler<size_t> sampler(ris, probs);
                si = sampler.sample(rng);
            }
            else
            {
                auto s = (!labelled) ? mu[_bprev[v]] : _bprev[v];
                auto iter = rs.begin();
                for (size_t j = 0; j < rs.size(); ++j)
                {
                    if (*iter++ == s)
                    {
                        si = j;
                        break;
                    }
                }
            }

            if (si >= rs.size() || std::isinf(dS[si]))
            {
                lp = -std::numeric_limits<double>::infinity();
                break;
            }

            auto iter = rs.begin();
            std::advance(iter, si);
            move_node(v, *iter);
            lp += ps[si] - Z;
            S += dS[si];
        }

        if constexpr (!forward)
        {
            for (auto& v : vs)
                move_node(v, _btemp[v]);
        }

        return {S, lp};
    }

    template <class VB>
    auto relabel_rs(std::vector<Node>& vs, VB& b)
    {
        // label matching

        if constexpr (!labelled)
        {
            adj_list<> g;
            typename vprop_map_t<Group>::type label(get(vertex_index_t(), g));
            typename vprop_map_t<bool>::type partition(get(vertex_index_t(), g));
            typename eprop_map_t<double>::type mrs(get(edge_index_t(), g));

            std::vector<int> x, y;
            for (auto& v: vs)
            {
                x.push_back(State::get_group(v));
                y.push_back(b[v]);
            }

            get_contingency_graph<false>(g, partition, label, mrs, x, y);

            typedef typename graph_traits<adj_list<>>::vertex_descriptor vertex_t;
            typename vprop_map_t<vertex_t>::type match(get(vertex_index_t(), g));

            auto u = undirected_adaptor<adj_list<>>(g);
            maximum_bipartite_weighted_matching(u, partition, mrs, match);

            vector<size_t> unmatched;
            for (auto v : vertices_range(g))
            {
                if (!partition[v])
                    continue;
                if (match[v] == graph_traits<adj_list<>>::null_vertex())
                    unmatched.push_back(v);
            }

            gt_hash_map<Group, Group> mu;
            for (auto v : vertices_range(g))
            {
                if (partition[v])
                    continue;
                auto w = match[v];
                if (w == graph_traits<adj_list<>>::null_vertex())
                {
                    w = unmatched.back();
                    unmatched.pop_back();
                }
                mu[label[w]] = label[v];
            }

            return mu;
        }
        else
        {
            gt_hash_map<Group, Group> mu;
            return mu;
        }
    }

    std::vector<Node> _mvs;
    double virtual_merge_dS(const Group& r, const Group& s)
    {
        assert(r != s);

        State::relax_update(true);

        _mvs.clear();
        double dS = 0;
        for (auto& v : _groups[r])
        {
            assert(State::get_group(v) == r);
            double ddS = State::virtual_move(v, r, s);
            dS += ddS;
            if (std::isinf(ddS))
                break;
            State::move_node(v, s, true);
            _mvs.push_back(v);
        }

        for (auto& v : _mvs)
            State::move_node(v, r, false);

        State::relax_update(false);

        return dS;
    }

    void merge(const Group& r, const Group& s)
    {
        assert(r != s);

        get_group_vs(r, _mvs);
        for (auto& v : _mvs)
            move_node(v, s);
    }

    template <class RNG>
    double merge_sweep(GSet<Group>& rs, size_t B, size_t niter, RNG& rng)
    {
        double S = 0;

        _best_merge.clear();
        for (const auto& r : rs)
            _best_merge[r] = std::make_pair(r, numeric_limits<double>::infinity());

        _past_merges.clear();
        for (const auto& r : rs)
        {
            auto find_candidates = [&](bool allow_random)
                {
                    for (size_t i = 0; i < niter; ++i)
                    {
                        auto v = uniform_sample(_groups[r], rng);
                        auto s = State::sample_group(v, allow_random, false, false, rng);
                        if (s != r &&
                            rs.find(s) != rs.end() &&
                            _past_merges.find(s) == _past_merges.end())
                        {
                            double dS = virtual_merge_dS(r, s);
                            if (!std::isinf(dS) && dS < _best_merge[r].second)
                                _best_merge[r] = std::make_pair(s, dS);
                            _past_merges.insert(s);
                        }
                    }
                };

            // Prefer smart constrained moves. If no candidates were found, the
            // group is likely to be "stuck" (e.g. isolated or constrained by
            // clabel); attempt random movements instead.

            find_candidates(false);

            if (_best_merge[r].first == r)
                find_candidates(true);

            _past_merges.clear();
        }

        std::vector<std::pair<Group, Group>> pairs;
        std::vector<double> dS;
        for (const auto& r : rs)
        {
            auto& m = _best_merge[r];
            if (m.first == r)
                continue;
            pairs.emplace_back(r, m.first);
            dS.emplace_back(m.second);
        }

        auto cmp = [&](size_t i, size_t j) { return dS[i] > dS[j]; };
        std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)>
            queue(cmp);

        std::vector<size_t> pis(pairs.size());
        std::iota(pis.begin(), pis.end(), 0);
        std::shuffle(pis.begin(), pis.end(), rng);

        for (auto i : pis)
            queue.push(i);

        _root.clear();
        auto get_root = [&](Group r)
        {
            Group s = r;
            if (_root.find(r) == _root.end())
                _root[r] = r;
            while (_root[r] != r)
                r = _root[r];
            _root[s] = r;
            return r;
        };

        while (rs.size() > B && !queue.empty())
        {
            auto i = queue.top();
            queue.pop();

            std::pair<Group, Group>& m = pairs[i];
            m.first = get_root(m.first);
            m.second = get_root(m.second);
            if (m.first == m.second)
                continue;

            double ndS = virtual_merge_dS(m.first, m.second);
            if (!queue.empty() && ndS > dS[queue.top()])
            {
                dS[i] = ndS;
                queue.push(i);
                continue;
            }

            _root[m.first] = m.second;
            merge(m.first, m.second);
            S += ndS;
            rs.erase(m.first);
            assert(get_wr(m.first) == 0);
        }

        assert(rs.size() >= B);
        return S;
    }

#ifndef __clang__
    constexpr static
#endif
    double _phi = (1 + sqrt(5)) / 2;

    size_t fibo(size_t n)
    {
        return size_t(round(std::pow(_phi, n) / sqrt(5)));
    }

    size_t fibo_n_floor(size_t x)
    {
        return floor(log(x * sqrt(5) + .5) / log(_phi));
    }

    template <class RNG>
    size_t get_mid(size_t a, size_t b, RNG& rng)
    {
        if (a == b)
            return a;
        if (State::_random_bisect)
        {
            std::uniform_int_distribution<size_t> random(a, b - 1);
            return random(rng);
        }
        auto n = fibo_n_floor(b - a);
        return b - fibo(n - 1);
    }

    template <bool forward=true, class RNG>
    std::pair<double, double>
    stage_multilevel(GSet<Group>& rs, std::vector<size_t>& vs, RNG& rng)
    {
        size_t N = vs.size();

        if (N == 1)
            return {0, 0};

        if (_verbose)
            cout << "staging multilevel, N = " << N << endl;

        size_t B_max = State::_global_moves ? std::min(N, _B_max) : std::min(N, State::_M);
        size_t B_min = State::_global_moves ? std::max(size_t(1), _B_min) : 1;

        B_min = std::min(std::max(B_min, State::get_Bmin(vs)), B_max);

        size_t B_mid;

        size_t B_init = rs.size();
        size_t B_max_init = B_max;
        size_t B_min_init = B_min;

        if constexpr (!forward)
        {
            _rs_prev.clear();
            for (auto& v : vs)
                _rs_prev.insert(_bprev[v]);
        }

        double S_best = std::numeric_limits<double>::infinity();

        map<size_t, std::pair<double, std::vector<Group>>> cache;

        auto put_cache = [&](size_t B, double S)
        {
            assert(cache.find(B) == cache.end());

            auto& c = cache[B];
            c.first = S;
            c.second.resize(vs.size());
            for (size_t i = 0; i < vs.size(); ++i)
                c.second[i] = State::get_group(vs[i]);
            if (S < S_best)
                S_best = S;
        };

        auto get_cache = [&](size_t B, GSet<Group>& rs)
        {
            assert(cache.find(B) != cache.end());

            rs.clear();
            auto& c = cache[B];
            for (size_t i = 0; i < vs.size(); ++i)
            {
                auto& s = c.second[i];
                move_node(vs[i], s);
                rs.insert(s);
            }
            assert(rs.size() == B);
            return c.first;
        };

        auto clean_cache = [&](size_t Bmin, size_t Bmax, bool keep_best=true)
        {
            for (auto iter = cache.begin(), last = cache.end(); iter != last;)
            {
                const auto& [B, b] = *iter;
                if ((B < Bmin || B > Bmax) && (!keep_best || b.first > S_best))
                    iter = cache.erase(iter);
                else
                    ++iter;
            }
        };

        auto get_S = [&](size_t B, bool keep_cache=true)
        {
            auto iter = cache.lower_bound(B);
            if (iter->first == B)
                return iter->second.first;
            assert(iter != cache.end());

            double S = get_cache(iter->first, rs);

            if (_verbose)
            {
                cout << "bracket B = [ " << B_min
                     << ", " << B_mid
                     << ", " << B_max << " ]"
                     << endl;
                cout << "shrinking from: " << iter->first << " -> " << B << endl;
            }

            // merge & sweep
            while (rs.size() > B)
            {
                size_t Bprev = rs.size();
                auto Bnext =
                    std::max(std::min(rs.size() - 1,
                                      size_t(round(rs.size() * State::_r))),
                             B);
                while (rs.size() != Bnext)
                    S += merge_sweep(rs, Bnext, _merge_sweeps, rng);
                for (size_t i = 0; i < _mh_sweeps; ++i)
                    S += mh_sweep<true>(vs, rs, _beta, rng, B).first;
                if ((keep_cache && _cache_states) || rs.size() == B)
                    put_cache(rs.size(), S);

                if (_verbose)
                    cout << "    " << Bprev << " -> "
                         << rs.size() << ": " << S << endl;
            }

            assert(rs.size() == B);
            return S;
        };

        if (State::_has_b_min)
        {
            if (B_min == _B_min)
            {
                push_b(vs);
                double S = 0;
                if (rs.size() != B_min)
                {
                    for (auto& v : vs)
                    {
                        auto r = State::get_group(v);
                        Group t = _b_min[v];
                        if (r == t)
                            continue;
                        S += State::virtual_move(v, r, t);
                        move_node(v, t, true);
                    }
                }
                put_cache(B_min, S);
                pop_b();
            }
        }
        else if (B_min == 1)
        {
            push_b(vs);
            double S = 0;
            if (rs.size() > 1)
            {
                auto u = uniform_sample(vs, rng);
                auto t = State::get_group(u);
                for (auto& v : vs)
                {
                    auto r = State::get_group(v);
                    if (r == t)
                        continue;
                    S += State::virtual_move(v, r, t);
                    move_node(v, t, true);
                }
            }
            put_cache(B_min, S);
            pop_b();
        }

        if (!State::_has_b_max)
        {
            double S = 0;
            rs.clear();
            push_b(vs);
            State::relax_update(true);
            for (auto& v : vs)
            {
                auto s = State::get_group(v);
                if (get_wr(s) == 1)
                {
                    rs.insert(s);
                    continue;
                }
                auto t = State::get_new_group(v, true, rng);
                S += State::virtual_move(v, s, t);
                move_node(v, t, true);
                rs.insert(t);
            }

            // single-node sweep initialization with B = N. This is faster than
            // using merges!
            if (std::isinf(_beta) && _init_r < 1.)
            {
                size_t Bprev;
                size_t i = 0;
                do
                {
                    Bprev = rs.size();
                    double dS = mh_sweep<true>(vs, rs, (i++ == 0) ? _init_beta : _beta,
                                               rng, B_min, true).first;
                    S += dS;
                    if (_verbose)
                        cout << i - 1 << " " << ((i - 1 == 0) ? _init_beta : _beta)
                             << " " << dS << " " << rs.size() << " "
                             << rs.size()/double(Bprev) << endl;
                }
                while (rs.size()/double(Bprev) < _init_r);

                B_max = B_max_init = std::min(rs.size(), B_max);
            }

            if (cache.find(rs.size()) == cache.end())
                put_cache(rs.size(), S);

            pop_b();
            State::relax_update(false);
            get_cache(rs.size(), rs);
        }
        else if (B_max != B_min)
        {
            double S = 0;
            for (auto& v : vs)
            {
                auto r = State::get_group(v);
                Group t = _b_max[v];
                if (r == t)
                    continue;
                S += State::virtual_move(v, r, t);
                move_node(v, t, true);
            }
            put_cache(B_max, S);
        }


        B_mid = get_mid(B_min, B_max, rng);

        // initial bracketing
        double S_max = get_S(B_max);
        double S_mid = get_S(B_mid);
        double S_min = get_S(B_min);
        while (S_mid > S_min || S_mid > S_max)
        {
            if (S_min < S_max)
            {
                B_max = B_mid;
                S_max = S_mid;
                B_mid = get_mid(B_min, B_mid, rng);
                S_mid = get_S(B_mid);
            }
            else
            {
                B_min = B_mid;
                S_min = S_mid;
                B_mid = get_mid(B_mid, B_max, rng);
            }

            if (std::isinf(_beta))
                clean_cache(B_min, B_max);

            if (B_min == B_mid && B_max == B_mid + 1)
                break;
        }

        // Fibonnaci search
        while (B_max - B_mid > 1)
        {
            size_t x;
            if (B_max - B_mid > B_mid - B_min)
                x = get_mid(B_mid, B_max, rng);
            else
                x = get_mid(B_min, B_mid, rng);

            double S_x = get_S(x);
            double S_mid = get_S(B_mid);

            if (S_x < S_mid)
            {
                if (B_max - B_mid > B_mid - B_min)
                    B_min = B_mid;
                else
                    B_max = B_mid;
                B_mid = x;
            }
            else
            {
                if (B_max - B_mid > B_mid - B_min)
                    B_max = x;
                else
                    B_min = x;
            }

            if (std::isinf(_beta))
                clean_cache(B_min, B_max);
        }

        clean_cache(B_min_init, B_max_init, false);

        // add midpoint
        if (!std::isinf(_beta))
        {
            size_t Br;
            if constexpr (forward)
                Br = get_mid(B_min_init, B_max_init, rng);
            else
                Br = _rs_prev.size();
            get_S(Br, false);
        }

        // Sample partition
        auto iter = max_element(cache.begin(), cache.end(),
                                [&](auto& x, auto& y)
                                {
                                    return x.second.first > y.second.first;
                                });

        std::vector<size_t> Bs;
        std::vector<double> probs;
        for (auto& BS : cache)
        {
            Bs.push_back(BS.first);
            if (!std::isinf(_beta))
            {
                probs.push_back(exp(_beta * (-BS.second.first +
                                             iter->second.first)));
            }
            else
            {
                if (BS.second.first == iter->second.first)
                    probs.push_back(1);
                else
                    probs.push_back(0);
            }
        }

        size_t B;
        if constexpr (forward)
        {
            Sampler<size_t> B_sampler(Bs, probs);
            B = B_sampler.sample(rng);
        }
        else
        {
            B = _rs_prev.size();
        }

        double lp = 0;
        double S = get_cache(B, rs);

        if (!std::isinf(_beta))
        {
            double Z = -numeric_limits<double>::infinity();
            for (auto& BS : cache)
                Z = log_sum_exp(Z, -_beta * BS.second.first);
            lp += -_beta * S - Z;
        }

        if (!State::_has_b_min)
            B_min_init = 1;
        if (!State::_has_b_max)
            B_max_init = N;

        if ((rs.size() > B_min_init) && (rs.size() < B_max_init))
        {
            if (State::_gibbs)
            {
                double g_lp, g_dS;
                std::tie(g_dS, g_lp) = gibbs_sweep<forward>(vs, rs, rng);
                S += g_dS;
                lp += g_lp;
            }
            else
            {
                double mh_lp, dS = 0;
                if constexpr (forward)
                {
                    //for relabel
                    if (!std::isinf(_beta))
                    {
                        for (auto& v : vs)
                            _btemp[v] = State::get_group(v);
                    }

                    std::tie(dS, mh_lp) = mh_sweep<false>(vs, rs, _beta, rng);

                    if (!std::isinf(_beta))
                    {
                        auto mu = relabel_rs(vs, _btemp);
                        for (auto& [r, s] : mu)
                        {
                            if (r != s)
                            {
                                mh_lp = numeric_limits<double>::infinity();
                                break;
                            }
                        }
                    }
                }
                else
                {
                    mh_lp = mh_sweep_prob(vs, rs, _beta, rng);
                }
                S += dS;
                lp += mh_lp;
            }
        }

        assert(rs.size() == B);

        if (forward && State::_global_moves &&
            (B_init < _B_min || B_init > _B_max))
        {
            S = -numeric_limits<double>::infinity();
            lp = 0;
        }

        return {S, lp};
    }

    template <class RNG>
    void sample_rs(GSet<Group>& rs, RNG& rng)
    {
        if (State::_global_moves)
        {
            rs.clear();
            for (auto r : _rlist)
                rs.insert(r);
        }
        else
        {
            std::uniform_int_distribution<size_t>
                sample(1, std::min(State::_M, _rlist.size()));

            auto M = sample(rng);

            rs.clear();
            while (rs.size() < M)
            {
                auto r = uniform_sample(_rlist, rng);
                _rlist.erase(r);
                rs.insert(r);

                if (get_wr(r) == 0)
                    abort();
            }

            for (const auto& r : rs)
                _rlist.insert(r);
        }
    }

    template <class RNG>
    std::tuple<size_t, size_t>
    move_proposal(const Node&, RNG& rng)
    {
        _dS = _a = 0;
        _vs.clear();
        _nmoves = 0;

        auto move = _move_sampler.sample(rng);

        switch (move)
        {
        case move_t::single:
            {
                auto v = uniform_sample(_nodes, rng);
                auto r = State::get_group(v);
                auto s = State::sample_group(v, true, true, false, rng);
                if (r == s)
                {
                    move = move_t::null;
                    break;
                }
                _dS = State::virtual_move(v, r, s);
                if (!std::isinf(_beta))
                {
                    double pf = State::get_move_prob(v, r, s, true, true, false);
                    double pb = State::get_move_prob(v, s, r, true, true, true);
                    _a = pb - pf;
                }
                _vs.clear();
                _vs.push_back(v);
                _bnext[v] = s;
                _nmoves++;
            }
            break;

        case move_t::multilevel:
            {
                sample_rs(_rs, rng);
                size_t nr = _rs.size();

                _vs.clear();
                for (const auto& r : _rs)
                    get_group_vs<false>(r, _vs);

                //push_b(_vs);
                for (auto& v : _vs)
                    _bprev[v] = State::get_group(v);

                double pf = 0, pb = 0;

                auto ret = stage_multilevel(_rs, _vs, rng);
                _dS = get<0>(ret);
                pf += get<1>(ret);

                size_t nnr = _rs.size();

                for (auto& v : _vs)
                    _bnext[v] = State::get_group(v);

                if (!std::isinf(_beta))
                {
                    if (!std::isinf(pf))
                    {
                        pb += stage_multilevel<false>(_rs, _vs, rng).second;

                        if (!State::_global_moves && !std::isinf(pb))
                        {
                            int dB = int(nnr) - int(nr);
                            pf += -safelog_fast(std::min(_rlist.size()     , State::_M))
                                - lbinom_fast(_rlist.size(), nr);
                            pb += -safelog_fast(std::min(_rlist.size() + dB, State::_M))
                                - lbinom_fast(_rlist.size() + dB, nnr);
                        }
                    }

                    _a = pb - pf;
                }

                if (_verbose)
                    cout << "multilevel proposal: " << nr << "->" << nnr
                         << " (" << _vs.size() << "), dS: " << _dS << ", pf: "
                         << pf << ", pb: " << pb << ", a: " << -_beta * _dS + pb - pf
                         << endl;

                for (auto& v : _vs)
                    move_node(v, _bprev[v]);
            }
            break;

        default:
            move = move_t::null;
            break;
        }

        if (move == move_t::null)
            return {_null_move, _nmoves ? _nmoves : 1};

        _move = move;

        return {0, _nmoves};
    }

    std::tuple<double, double>
    virtual_move_dS(const Node&, size_t)
    {
        return {_dS, _a};
    }

    void perform_move(const Node&, size_t)
    {
        for (auto& v : _vs)
        {
            auto r = State::get_group(v);
            auto s = _bnext[v];
            if (r == s)
                continue;

            if (get_wr(s) == 0)
                _rlist.insert(s);

            move_node(v, s);

            if (get_wr(r) == 0)
                _rlist.erase(r);
        }
    }

    constexpr bool is_deterministic()
    {
        return true;
    }

    constexpr bool is_sequential()
    {
        return false;
    }

    std::array<Node, 1> _vlist = {Node()};

    auto& get_vlist()
    {
        return _vlist;
    }

    size_t get_N()
    {
        return 1; //_N;
    }

    double get_beta()
    {
        return _beta;
    }

    size_t get_niter()
    {
        return State::_niter;
    }

    constexpr void step(const Node&, size_t)
    {
    }
};

} // graph_tool namespace

#endif // MULTILEVEL_HH
