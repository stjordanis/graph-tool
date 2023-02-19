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

#ifndef GRAPH_HISTOGRAM_HH
#define GRAPH_HISTOGRAM_HH

#include "config.h"

#include <vector>

#include "../blockmodel/graph_blockmodel_util.hh"
#include "../support/graph_state.hh"

#include <boost/container/static_vector.hpp>

template <class T, size_t D>
struct empty_key<boost::container::static_vector<T,D>>
{
    static boost::container::static_vector<T,D> get()
    {
        boost::container::static_vector<T,D> x(D);
        for (size_t i = 0; i < D; ++i)
            x[i] = empty_key<T>::get();
        return x;
    }
};

template <class T, size_t D>
struct deleted_key<boost::container::static_vector<T,D>>
{
    static boost::container::static_vector<T,D> get()
    {
        boost::container::static_vector<T,D> x(D);
        for (size_t i = 0; i < D; ++i)
            x[i] = deleted_key<T>::get();
        return x;
    }
};


namespace std
{
template <class Value, size_t D>
struct hash<boost::container::static_vector<Value, D>>
{
    size_t operator()(const boost::container::static_vector<Value, D>& v) const
    {
        size_t seed = 0;
        for (const auto& x : v)
            std::_hash_combine(seed, x);
        return seed;
    }
};

}

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef mpl::vector2<multi_array_ref<double,2>,
                     multi_array_ref<int64_t,2>> x_ts;

#define HIST_STATE_params                                                     \
    ((__class__,&, mpl::vector<python::object>, 1))                           \
    ((x,, x_ts, 1))                                                           \
    ((obins,, python::list, 0))                                               \
    ((obounded,, python::list, 0))                                            \
    ((odiscrete,, python::list, 0))                                           \
    ((alpha,, double, 0))                                                     \
    ((conditional,, size_t, 0))

GEN_STATE_BASE(HistStateBase, HIST_STATE_params)

template <template <class T> class VT>
class HistD
{
public:
    template <class... Ts>
    class HistState
        : public HistStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(HistStateBase<Ts...>, HIST_STATE_params)
        GET_PARAMS_TYPEDEF(Ts, HIST_STATE_params)

        typedef typename x_t::element value_t;
        typedef std::vector<value_t> bins_t;

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
        HistState(ATs&&... args)
            : HistStateBase<Ts...>(std::forward<ATs>(args)...),
              _N(_x.shape()[0]),
              _D(_x.shape()[1]),
              _mgroups(_D)
        {
            for (size_t j = 0; j < _D; ++j)
                _bins.push_back(&python::extract<bins_t&>(_obins[j])());

            for (size_t i = 0; i < _N; ++i)
                update_hist<true>(i);

            for (size_t j = 0; j < _D; ++j)
                _bounded.push_back({python::extract<bool>(_obounded[j][0])(),
                                    python::extract<bool>(_obounded[j][1])()});

            for (size_t j = 0; j < _D; ++j)
                _discrete.push_back(python::extract<bool>(_odiscrete[j])());
        }

        size_t _N;
        size_t _D;

        std::vector<bins_t*> _bins;
        std::vector<std::pair<value_t, value_t>> _bounds;
        std::vector<std::pair<bool, bool>> _bounded;
        std::vector<bool> _discrete;

        typedef VT<value_t> group_t;
        typedef is_instance<group_t, std::vector> is_vec;
        typedef std::conditional_t
            <is_vec{},
             group_t,
             boost::container::static_vector
                 <value_t,
                  std::tuple_size<std::conditional_t
                                    <is_vec{},
                                     std::array<value_t,1>,
                                     group_t>>::value>>
            cgroup_t;

        gt_hash_map<group_t, size_t> _hist;
        gt_hash_map<cgroup_t, size_t> _chist;
        std::vector<gt_hash_map<value_t, gt_hash_set<size_t>>> _mgroups;

        group_t _r_temp;

        template <class V>
        group_t get_bin(V&& x)
        {
            group_t r = group_t();
            for (size_t j = 0; j < _D; ++j)
            {
                auto iter = std::upper_bound(_bins[j]->begin(),
                                             _bins[j]->end(),
                                             x[j]);
                --iter;
                if constexpr (is_instance<VT<value_t>, std::vector>{})
                    r.push_back(*iter);
                else
                    r[j] = *iter;
            }
            return r;
        }

        size_t get_hist(const group_t& r)
        {
            auto iter = _hist.find(r);
            if (iter != _hist.end())
                return iter->second;
            return 0;
        }

        size_t get_chist(const cgroup_t& cr)
        {
            auto iter = _chist.find(cr);
            if (iter != _chist.end())
                return iter->second;
            return 0;
        }

        template <class V>
        const group_t& to_group(V&& r)
        {
            auto& nr = _r_temp;
            if constexpr (is_instance<VT<value_t>, std::vector>{})
            {
                nr.clear();
                nr.insert(nr.end(), r.begin(), r.end());
            }
            else
            {
                for (size_t i = 0; i < r.size(); ++i)
                    nr[i] = r[i];
            }
            return nr;
        }

        cgroup_t to_cgroup(const group_t& r)
        {
            cgroup_t cr;
            cr.insert(cr.end(), r.begin() + _conditional, r.end());
            return cr;
        }

        template <class V>
        size_t get_hist(V&& r)
        {
            return get_hist(to_group(r));
        }

        size_t get_hist(size_t i)
        {
            return get_hist(get_bin(_x[i]));
        }

        template <bool Add>
        void update_hist(size_t i, const group_t& r)
        {
            if constexpr (Add)
            {
                _hist[r]++;
                for (size_t j = 0; j < _D; ++j)
                {
                    auto& vs = _mgroups[j][r[j]];
                    vs.insert(i);
                }

                if (_conditional < _D)
                    _chist[to_cgroup(r)]++;
            }
            else
            {
                auto iter = _hist.find(r);
                iter->second--;
                if (iter->second == 0)
                    _hist.erase(iter);

                for (size_t j = 0; j < _D; ++j)
                {
                    auto& vs = _mgroups[j][r[j]];
                    vs.erase(i);
                    if (vs.empty())
                        _mgroups[j].erase(r[j]);
                }

                if (_conditional < _D)
                {
                    auto iter = _chist.find(to_cgroup(r));
                    iter->second--;
                    if (iter->second == 0)
                        _chist.erase(iter);
                }
            }
        }

        template <bool Add, class V>
        void update_hist(size_t i, V&& r)
        {
            update_hist<Add>(i, to_group(r));
        }

        template <bool Add>
        void update_hist(size_t i)
        {
            auto r = get_bin(_x[i]);
            update_hist<Add>(i, r);
        }

        double entropy_group(const group_t& r, size_t n)
        {
            double S = 0;

            S -= lgamma_fast(n + 1);
            double lw = 0;
            for (size_t j = 0; j < _conditional; ++j)
            {
                auto x = r[j];
                auto& bin = *_bins[j];
                auto iter = std::lower_bound(bin.begin(),
                                             bin.end(), x);
                assert(*(iter+1) > *iter);
                lw += log(*(iter+1) - *iter);
            }
            S += n * lw;

            return S;
        }

        double entropy_cgroup(size_t n)
        {
            size_t Mx = 1;
            for (size_t j = 0; j < _conditional; ++j)
                Mx *= _bins[j]->size() - 1;
            return lgamma_fast(Mx + n) - lgamma_fast(Mx);
        }

        double entropy()
        {
            double S = 0;

            S += _D * safelog(_N);

            size_t M = 1;
            for (size_t j = 0; j < _D; ++j)
            {
                size_t Md = _bins[j]->size() - 1;
                double delta = *(_bins[j]->end()-1) - *_bins[j]->begin();
                assert(delta > 0);
                if (_discrete[j])
                    S += lbinom(delta - 1, Md - 1);
                else
                    S += (Md + _alpha + 1) * log(delta) + lgamma_fast(Md);
                M *= Md;
            }

            if (_conditional >= _D)
            {
                S += lgamma_fast(_N + M) - lgamma_fast(M);
            }
            else
            {
                for (auto& [cr, n] : _chist)
                    S += entropy_cgroup(n);
            }

            for (auto& nrc : _hist)
                S += entropy_group(nrc.first, nrc.second);

            return S;
        }

        // =========================================================================
        // State modification
        // =========================================================================

        void move_edge(size_t j, size_t i, value_t y)
        {
            auto x = (*_bins[j])[i];
            auto& mvs = _mgroups[j][x];
            std::vector<size_t> vs(mvs.begin(), mvs.end());

            if (i > 0)
            {
                auto xn = (*_bins[j])[i-1];
                vs.insert(vs.end(),
                          _mgroups[j][xn].begin(),
                          _mgroups[j][xn].end());
            }

            for (auto& v : vs)
                update_hist<false>(v);

            (*_bins[j])[i] = y;

            for (auto& v : vs)
                update_hist<true>(v);
        }

        void remove_edge(size_t j, size_t i)
        {
            auto x = (*_bins[j])[i];
            auto& mvs = _mgroups[j][x];
            std::vector<size_t> vs(mvs.begin(), mvs.end());

            for (auto& v : vs)
                update_hist<false>(v);

            auto& b = *_bins[j];
            b.erase(b.begin() + i);

            for (auto& v : vs)
                update_hist<true>(v);
        }

        void add_edge(size_t j, size_t i, value_t y)
        {
            auto x = (*_bins[j])[i];
            auto& mvs = _mgroups[j][x];
            std::vector<size_t> vs(mvs.begin(), mvs.end());

            for (auto& v : vs)
                update_hist<false>(v);

            auto& b = *_bins[j];
            b.insert(b.begin() + i + 1, y);

            for (auto& v : vs)
                update_hist<true>(v);
        }

        template <class V>
        void get_rs(V& vs, gt_hash_set<group_t>& rs)
        {
            for (auto v : vs)
                rs.insert(get_bin(_x[v]));
        }

        gt_hash_set<group_t> _rs;
        gt_hash_set<cgroup_t> _crs;

        double virtual_move_edge(size_t j, size_t i, value_t y)
        {
            auto x = (*_bins[j])[i];

            _rs.clear();
            get_rs(_mgroups[j][x], _rs);
            if (i > 0)
                get_rs(_mgroups[j][(*_bins[j])[i-1]], _rs);

            auto S_terms =
                [&]()
                {
                    double S = 0;
                    for (auto& r : _rs)
                        S += entropy_group(r, get_hist(r));

                    if (_conditional < _D)
                    {
                        _crs.clear();
                        for (auto& r : _rs)
                            _crs.insert(to_cgroup(r));
                        for (auto& cr : _crs)
                            S += entropy_cgroup(get_chist(cr));
                    }

                    if (i == 0 || i == _bins[j]->size() - 1)
                    {
                        size_t Md = _bins[j]->size() - 1;
                        auto delta = *(_bins[j]->end()-1) - *_bins[j]->begin();
                        if (_discrete[j])
                            S += lbinom(size_t(delta - 1), Md - 1);
                        else
                            S += (Md + _alpha + 1) * log(delta);
                    }
                    return S;
                };

            double Sb = S_terms();

            move_edge(j, i, y);

            _rs.clear();
            get_rs(_mgroups[j][y], _rs);
            if (i > 0)
                get_rs(_mgroups[j][(*_bins[j])[i-1]], _rs);

            double Sa = S_terms();

            move_edge(j, i, x);

            return Sa - Sb;
        }

        template <bool Add>
        double virtual_change_edge(size_t j, size_t i, value_t y)
        {
            auto x = (*_bins[j])[i];

            if constexpr (!Add)
                y = (*_bins[j])[i-1];

            _rs.clear();
            get_rs(_mgroups[j][x], _rs);
            if constexpr (!Add)
                get_rs(_mgroups[j][y], _rs);

            size_t M = 1;
            for (auto& b : _bins)
                M *= b->size() - 1;
            size_t Md = _bins[j]->size() - 1;
            auto delta = *(_bins[j]->end()-1) - *_bins[j]->begin();

            auto S_terms =
                [&]()
                {
                    double S = 0;
                    for (auto& r : _rs)
                        S += entropy_group(r, get_hist(r));

                    if (_discrete[j])
                        S += lbinom(size_t(delta - 1), Md - 1);
                    else
                        S += (Md + _alpha + 1) * log(delta) + lgamma_fast(Md);

                    if (_conditional < _D)
                    {
                        if (j < _D)
                        {
                            for (auto& [cr, n] : _chist)
                                S += entropy_cgroup(n);
                        }
                        else
                        {
                            _crs.clear();
                            for (auto& r : _rs)
                                _crs.insert(to_cgroup(r));
                            for (auto& cr : _crs)
                                S += entropy_cgroup(get_chist(cr));
                        }
                    }
                    else
                    {
                        S += lgamma_fast(_N + M) - lgamma_fast(M);
                    }

                    return S;
                };

            double Sb = S_terms();

            if constexpr (Add)
                add_edge(j, i, y);
            else
                remove_edge(j, i);

            _rs.clear();
            if constexpr (Add)
            {
                get_rs(_mgroups[j][x], _rs);
                get_rs(_mgroups[j][y], _rs);
            }
            else
            {
                get_rs(_mgroups[j][y], _rs);
            }

            M /= Md;
            Md = _bins[j]->size() - 1;
            M *= Md;

            double Sa = S_terms();

            if constexpr (Add)
                remove_edge(j, i + 1);
            else
                add_edge(j, i - 1, x);

            return Sa - Sb;
        }

        template <class V>
        void replace_point(size_t i, V&& xn, bool move_edges = false)
        {
            if (!_bounds.empty())
            {
                for (size_t j = 0; j < _D; ++j)
                {
                    if (_x[i][j] == _bounds[j].first ||
                        _x[i][j] == _bounds[j].second ||
                        xn[j] <= _bounds[j].first ||
                        xn[j] >= _bounds[j].second)
                    {
                        _bounds.clear();
                        break;
                    }
                }
            }

            if (move_edges)
            {
                for (size_t j = 0; j < _D; ++j)
                {
                    if (*_bins[j]->begin() > xn[j])
                        move_edge(j, 0, xn[j]);
                    if (*_bins[j]->rbegin() <= xn[j])
                    {
                        if (_discrete[j])
                            move_edge(j, _bins[j]->size() - 1, xn[j] + 1);
                        else
                            move_edge(j, _bins[j]->size() - 1,
                                      std::nextafter(xn[j], numeric_limits<value_t>::max()));
                    }
                }
            }

            update_hist<false>(i);
            for (size_t j = 0; j < _D; ++j)
                _x[i][j] = xn[j];
            update_hist<true>(i);
        }

        template <class V>
        double virtual_replace_point_dS(size_t i, V&& xn)
        {
            for (size_t j = 0; j < _D; ++j)
            {
                if (xn[j] < *_bins[j]->begin() || xn[j] >= *_bins[j]->rbegin())
                    return numeric_limits<double>::infinity();
            }

            group_t r = get_bin(_x[i]);
            group_t nr = get_bin(xn);
            if (r == nr)
                return 0.0;

            size_t count_r = get_hist(r);
            size_t count_nr = get_hist(nr);

            double Sb = entropy_group(r, count_r) + entropy_group(nr, count_nr);
            double Sa = entropy_group(r, count_r - 1) + entropy_group(nr, count_nr + 1);

            if (_conditional < _D)
            {
                cgroup_t cr = to_cgroup(r);
                cgroup_t cnr = to_cgroup(nr);
                if (cr != cnr)
                {
                    size_t count_cr = get_chist(cr);
                    size_t count_cnr = get_chist(cnr);
                    Sb += entropy_cgroup(count_cr) + entropy_cgroup(count_cnr);
                    Sa += entropy_cgroup(count_cr - 1) + entropy_cgroup(count_cnr + 1);
                }
            }

            return Sa - Sb;
        }

        void update_bounds()
        {
            if (_bounds.empty())
            {
                _bounds.resize(_D, {std::numeric_limits<value_t>::max(),
                                    std::numeric_limits<double>::lowest()});
                for (size_t i = 0; i < _N; ++i)
                {
                    for (size_t j = 0; j < _D; ++j)
                    {
                        _bounds[j].first = std::min(_bounds[j].first, _x[i][j]);
                        _bounds[j].second = std::max(_bounds[j].second, _x[i][j]);
                    }
                }
            }
        }

        // sampling and querying

        template <class V>
        double get_mle_lpdf(const V& x)
        {
            auto r = get_bin(x);

            double lw = 0;
            for (size_t j = 0; j < _conditional; ++j)
            {
                auto& bin = *_bins[j];
                auto iter = std::lower_bound(bin.begin(),
                                             bin.end(), r[j]);
                if (iter == bin.end() || iter == bin.end() - 1)
                    return -numeric_limits<double>::infinity();
                lw += log(*(iter+1) - *iter);
            }

            double L = -lw + safelog_fast(get_hist(r));

            if (_conditional >= _D)
                L -= safelog_fast(_N);
            else
                L -= safelog_fast(get_chist(to_cgroup(r)));

            return L;
        }

        template <class RNG>
        multi_array<value_t, 2> sample(size_t n, multi_array_ref<value_t, 1> cx,
                                       RNG& rng)
        {
            multi_array<value_t, 2> x(extents[n][_conditional]);

            std::vector<value_t> y;
            y.push_back(*_bins[0]->begin());
            y.insert(y.end(), cx.begin(), cx.end());

            auto cr = to_cgroup(get_bin(y));

            std::vector<group_t> nrs;
            std::vector<double> counts;

            for (auto& [r, count] : _hist)
            {
                if (to_cgroup(r) != cr)
                    continue;
                nrs.emplace_back(r);
                counts.push_back(count);
            };

            Sampler<group_t> idx_sampler(nrs, counts);

            for (size_t i = 0; i < n; ++i)
            {
                auto& r = idx_sampler.sample(rng);
                for (size_t j = 0; j < _conditional; ++j)
                {
                    auto& bin = *_bins[j];
                    auto iter = std::lower_bound(bin.begin(),
                                                 bin.end(), r[j]);
                    if (_discrete[j])
                    {
                        std::uniform_int_distribution<int64_t> d(*iter, *(iter+1)-1);
                        x[i][j] = d(rng);
                    }
                    else
                    {
                        std::uniform_real_distribution<double> d(*iter, *(iter+1));
                        x[i][j] = d(rng);
                    }
                }
            }

            return x;
        }

    };
};
} // graph_tool namespace

#endif //GRAPH_HISTOGRAM_HH
