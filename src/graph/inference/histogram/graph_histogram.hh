// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2021 Tiago de Paula Peixoto <tiago@skewed.de>
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

template <class T, size_t D>
struct empty_key<std::array<T,D>>
{
    static std::array<T,D> get()
    {
        std::array<T,D> x;
        for (size_t i = 0; i < D; ++i)
            x[i] = empty_key<T>::get();
        return x;
    }
};

template <class T, size_t D>
struct deleted_key<std::array<T,D>>
{
    static std::array<T,D> get()
    {
        std::array<T,D> x;
        for (size_t i = 0; i < D; ++i)
            x[i] = deleted_key<T>::get();
        return x;
    }
};


namespace std
{
template <class Value, size_t D>
struct hash<std::array<Value, D>>
{
    size_t operator()(const std::array<Value, D>& v) const
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

typedef multi_array_types::index_range range;

namespace
{
    template <typename, template <typename...> typename>
    struct is_instance_impl : public std::false_type {};

    template <template <typename...> typename U, typename...Ts>
    struct is_instance_impl<U<Ts...>, U> : public std::true_type {};
}

template <typename T, template <typename ...> typename U>
using is_instance = is_instance_impl<std::decay_t<T>, U>;

typedef multi_array_ref<double,2> x_t;
typedef std::vector<double> bins_t;

#define HIST_STATE_params                                                     \
    ((__class__,&, mpl::vector<python::object>, 1))                           \
    ((x,, x_t, 0))                                                            \
    ((obins,, python::list, 0))                                               \
    ((obounded,, python::list, 0))                                            \
    ((odiscrete,, python::list, 0))                                           \
    ((alpha,, double, 0))

GEN_STATE_BASE(HistStateBase, HIST_STATE_params)

template <class VT>
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

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
        HistState(ATs&&... args)
            : HistStateBase<Ts...>(std::forward<ATs>(args)...),
              _N(_x.shape()[0]),
              _D(_x.shape()[1]),
              _bounds(_D, {std::numeric_limits<double>::infinity(),
                           -std::numeric_limits<double>::infinity()}),
              _mgroups(_D)
        {
            for (size_t j = 0; j < _D; ++j)
                _bins.push_back(&python::extract<bins_t&>(_obins[j])());

            for (size_t i = 0; i < _N; ++i)
            {
                update_hist<true>(i);
                for (size_t j = 0; j < _D; ++j)
                {
                    _bounds[j].first = std::min(_bounds[j].first, _x[i][j]);
                    _bounds[j].second = std::max(_bounds[j].second, _x[i][j]);
                }
            }

            for (size_t j = 0; j < _D; ++j)
                _bounded.push_back({python::extract<bool>(_obounded[j][0])(),
                                    python::extract<bool>(_obounded[j][1])()});

            for (size_t j = 0; j < _D; ++j)
                _discrete.push_back(python::extract<bool>(_odiscrete[j])());
        }

        size_t _N;
        size_t _D;

        std::vector<bins_t*> _bins;
        std::vector<std::pair<double, double>> _bounds;
        std::vector<std::pair<bool, bool>> _bounded;
        std::vector<bool> _discrete;

        typedef VT group_t;
        gt_hash_map<group_t, size_t> _hist;
        std::vector<gt_hash_map<double, gt_hash_set<size_t>>> _mgroups;

        group_t _r_temp;

        template <class V>
        group_t get_bin(V&& x)
        {
            group_t r;
            for (size_t j = 0; j < _D; ++j)
            {
                auto iter = std::upper_bound(_bins[j]->begin(),
                                             _bins[j]->end(),
                                             x[j]);
                --iter;
                if constexpr (is_instance<VT, std::vector>{})
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

        template <class V>
        const group_t& to_group(V&& r)
        {
            auto& nr = _r_temp;
            if constexpr (is_instance<VT, std::vector>{})
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
            update_hist<Add>(i, get_bin(_x[i]));
        }

        double entropy_group(const group_t& r, size_t n)
        {
            double S = 0;

            S -= lgamma_fast(n + 1);
            double lw = 0;
            for (size_t j = 0; j < _D; ++j)
            {
                double x = r[j];
                auto& bin = *_bins[j];
                auto iter = std::lower_bound(bin.begin(),
                                             bin.end(), x);
                assert(*(iter+1) > *iter);
                lw += log(*(iter+1) - *iter);
            }
            S += n * lw;

            return S;
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
                {
                    S += lbinom(delta - 1, Md - 1);
                }
                else
                {
                    S += (Md + _alpha + 1) * log(delta);
                    S += lgamma_fast(Md);
                }
                M *= Md;
            }

            S += lgamma_fast(_N + M) - lgamma_fast(M);

            for (auto& nrc : _hist)
                S += entropy_group(nrc.first, nrc.second);

            return S;
        }

        template <class V, class X>
        double lpdf(V&& n, V&& r, X&& x)
        {
            double L = 0;
            for (size_t j = 0; j < _D; ++j)
                L += lhist_pdf(n[j], r[j], x[j]);
            return L;
        }

        // =========================================================================
        // State modification
        // =========================================================================

        void move_edge(size_t j, size_t i, double y)
        {
            double x = (*_bins[j])[i];
            auto& mvs = _mgroups[j][x];
            std::vector<size_t> vs(mvs.begin(), mvs.end());

            if (i > 0)
            {
                double xn = (*_bins[j])[i-1];
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
            double x = (*_bins[j])[i];
            auto& mvs = _mgroups[j][x];
            std::vector<size_t> vs(mvs.begin(), mvs.end());

            for (auto& v : vs)
                update_hist<false>(v);

            auto& b = *_bins[j];
            b.erase(b.begin() + i);

            for (auto& v : vs)
                update_hist<true>(v);
        }

        void add_edge(size_t j, size_t i, double y)
        {
            double x = (*_bins[j])[i];
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

        double virtual_move_edge(size_t j, size_t i, double y)
        {
            double x = (*_bins[j])[i];

            gt_hash_set<group_t> rs;
            get_rs(_mgroups[j][x], rs);
            if (i > 0)
                get_rs(_mgroups[j][(*_bins[j])[i-1]], rs);

            double Sb = 0;

            for (auto& r : rs)
                Sb += entropy_group(r, get_hist(r));

            if (i == 0 || i == _bins[j]->size() - 1)
            {
                size_t Md = _bins[j]->size() - 1;
                double delta = *(_bins[j]->end()-1) - *_bins[j]->begin();
                if (_discrete[j])
                {
                    Sb += lbinom(delta - 1, Md - 1);
                }
                else
                {
                    Sb += (Md + _alpha + 1) * log(delta);
                }
            }

            move_edge(j, i, y);

            rs.clear();
            get_rs(_mgroups[j][y], rs);
            if (i > 0)
                get_rs(_mgroups[j][(*_bins[j])[i-1]], rs);

            double Sa = 0;
            for (auto& r : rs)
                Sa += entropy_group(r, get_hist(r));

            if (i == 0 || i == _bins[j]->size() - 1)
            {
                size_t Md = _bins[j]->size() - 1;
                double delta;
                if (i == 0)
                    delta = *(_bins[j]->end()-1) - y;
                else
                    delta = y - *_bins[j]->begin();
                if (_discrete[j])
                {
                    Sa += lbinom(delta - 1, Md - 1);
                }
                else
                {
                    Sa += (Md + _alpha + 1) * log(delta);
                }
            }

            move_edge(j, i, x);

            return Sa - Sb;
        }

        double virtual_remove_edge(size_t j, size_t i)
        {
            double x = (*_bins[j])[i];
            double xn = (*_bins[j])[i-1];

            gt_hash_set<group_t> rs;
            get_rs(_mgroups[j][x], rs);
            get_rs(_mgroups[j][xn], rs);

            double Sb = 0;
            for (auto& r : rs)
                Sb += entropy_group(r, get_hist(r));

            size_t M = 1;
            for (auto& b : _bins)
                M *= b->size() - 1;
            size_t Md = _bins[j]->size() - 1;
            double delta = *(_bins[j]->end()-1) - *_bins[j]->begin();

            if (_discrete[j])
            {
                Sb += lbinom(delta - 1, Md - 1);
            }
            else
            {
                Sb += (Md + _alpha + 1) * log(delta);
                Sb += lgamma_fast(Md);
            }
            Sb += lgamma_fast(_N + M) - lgamma_fast(M);

            remove_edge(j, i);

            rs.clear();
            get_rs(_mgroups[j][xn], rs);

            double Sa = 0;
            for (auto& r : rs)
                Sa += entropy_group(r, get_hist(r));

            M /= Md;
            Md = _bins[j]->size() - 1;
            M *= Md;

            if (_discrete[j])
            {
                Sa += lbinom(delta - 1, Md - 1);
            }
            else
            {
                Sa += (Md + _alpha + 1) * log(delta);
                Sa += lgamma_fast(Md);
            }
            Sa += lgamma_fast(_N + M) - lgamma_fast(M);

            add_edge(j, i - 1, x);

            return Sa - Sb;
        }

        double virtual_add_edge(size_t j, size_t i, double x)
        {
            double y = (*_bins[j])[i];

            gt_hash_set<group_t> rs;
            get_rs(_mgroups[j][y], rs);

            double Sb = 0;
            for (auto& r : rs)
                Sb += entropy_group(r, get_hist(r));

            size_t M = 1;
            for (auto& b : _bins)
                M *= b->size() - 1;
            size_t Md = _bins[j]->size() - 1;
            double delta = *(_bins[j]->end()-1) - *_bins[j]->begin();

            if (_discrete[j])
            {
                Sb += lbinom(delta - 1, Md - 1);
            }
            else
            {
                Sb += (Md + _alpha + 1) * log(delta);
                Sb += lgamma_fast(Md);
            }
            Sb += lgamma_fast(_N + M) - lgamma_fast(M);

            add_edge(j, i, x);

            double Sa = 0;

            rs.clear();
            get_rs(_mgroups[j][y], rs);
            get_rs(_mgroups[j][x], rs);

            for (auto& r : rs)
                Sa += entropy_group(r, get_hist(r));

            M /= Md;
            Md = _bins[j]->size() - 1;
            M *= Md;

            if (_discrete[j])
            {
                Sa += lbinom(delta - 1, Md - 1);
            }
            else
            {
                Sa += (Md + _alpha + 1) * log(delta);
                Sa += lgamma_fast(Md);
            }
            Sa += lgamma_fast(_N + M) - lgamma_fast(M);

            remove_edge(j, i + 1);

            return Sa - Sb;
        }

        // sampling

        double get_mle_lpdf(const multi_array_ref<double, 1>& x)
        {
            auto r = get_bin(x);

            double lw = 0;
            for (size_t j = 0; j < _D; ++j)
            {
                auto& bin = *_bins[j];
                auto iter = std::lower_bound(bin.begin(),
                                             bin.end(), r[j]);
                if (iter == bin.end() || iter == bin.end() - 1)
                    return -numeric_limits<double>::infinity();
                lw += log(*(iter+1) - *iter);
            }

            double L = -lw + log(get_hist(r)) - log(_N);
            return L;
        }

        template <class RNG>
        multi_array<double, 2> sample(size_t n, RNG& rng)
        {
            multi_array<double, 2> x(extents[n][_D]);

            std::vector<group_t> nrs;
            std::vector<double> counts;

            for (auto& [r, count] : _hist)
            {
                nrs.emplace_back(r);
                counts.push_back(count);
            };

            Sampler<group_t> idx_sampler(nrs, counts);

            for (size_t i = 0; i < n; ++i)
            {
                auto& r = idx_sampler.sample(rng);
                for (size_t j = 0; j < _D; ++j)
                {
                    auto& bin = *_bins[j];
                    auto iter = std::lower_bound(bin.begin(),
                                                 bin.end(), r[j]);
                    if (_discrete[j])
                    {
                        std::uniform_int_distribution<int> d(*iter, *(iter+1)-1);
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
