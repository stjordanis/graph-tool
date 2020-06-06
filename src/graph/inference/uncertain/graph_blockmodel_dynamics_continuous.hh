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

#ifndef GRAPH_BLOCKMODEL_DYNAMICS_CONTIUOUS_HH
#define GRAPH_BLOCKMODEL_DYNAMICS_CONTIUOUS_HH

#include "config.h"

#include <vector>

#include "graph_blockmodel_dynamics.hh"
#include "graph_python_interface.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

template <class Base, bool tshift>
class ContinuousStateBase
{
public:

    typedef vprop_map_t<std::vector<int32_t>>::type tmap_t;
    typedef vprop_map_t<std::vector<double>>::type smap_t;

    template <class State>
    ContinuousStateBase(State& s)
        : _s(s._s), _st(num_vertices(s._u))
    {
        for (auto sn : _s)
        {
            size_t T = numeric_limits<size_t>::max();
            for (auto v : vertices_range(s._u))
            {
                if (T == numeric_limits<size_t>::max())
                    T = sn[v].size();
                if (sn[v].size() != T)
                    throw ValueException("invalid time series: all vertices must "
                                         "have the same number of states");
            }
        }

        for (auto sn : _s)
            _m.emplace_back(num_vertices(s._u));

        for (auto v : vertices_range(s._u))
        {
            iter_time<false, false>
                (in_or_out_neighbors_range(v, s._u), v,
                 [&](auto, size_t n, size_t, auto& su)
                 {
                     double m = 0;
                     for (auto e : in_or_out_edges_range(v, s._u))
                     {
                         auto u = target(e, s._u);
                         if (u == v && !s._self_loops)
                             continue;
                         m += su[u] * s._x[e];
                     }
                     _m[n][v].emplace_back(m);
                 });
            for (auto& m : _m)
            {
                if (m[v].empty())
                    m[v].emplace_back(0);
            }

        }
        _m_temp.resize(_s.size());
    };


    template <bool follow_m, bool follow_v, class US, class F>
    void iter_time(US&& us, size_t v, F&& f)
    {
        for (size_t n = 0; n < _s.size(); ++n)
        {
            auto& sn = _s[n];
            auto& snv = _s[n][v];
            auto& mnv = _m[n][v];

            for (size_t t = 0; t < snv.size() - int(tshift); ++t)
            {
                [[maybe_unused]] auto m = follow_m ? mnv[t] : .0;
                [[maybe_unused]] auto s_v = follow_v ? snv[t] : 0;
                [[maybe_unused]] auto s_nv = follow_v ? (tshift ? snv[t+1] : snv[t]) : 0;

                for (auto u : us)
                    _st[u] = sn[u][t];

                if constexpr (follow_m)
                {
                    if constexpr (follow_v)
                    {
                        if constexpr (tshift)
                            f(v, n, t, _st, m, s_v, s_nv);
                        else
                            f(v, n, t, _st, m, s_v);
                    }
                    else
                    {
                        f(v, n, t, _st, m);
                    }
                }
                else
                {
                    if constexpr (follow_v)
                    {
                        if constexpr (tshift)
                            f(v, n, t, _st, s_v, s_nv);
                        else
                            f(v, n, t, _st, s_v);
                    }
                    else
                    {
                        f(v, n, t, _st);
                    }
                }
            }
        }
    }

    template <class State>
    bool check_m(State& s, size_t v)
    {
        auto xc = s._x.get_checked();
        bool check = true;
        iter_time<true, false>
            (in_or_out_neighbors_range(v, s._u), v,
             [&](auto, size_t n, size_t t, auto& su, auto om)
             {
                 double m = 0;
                 for (auto e : in_or_out_edges_range(v, s._u))
                 {
                     auto u = source(e, s._u);
                     if (u == v && !s._self_loops)
                         continue;
                     m += xc[e] * su[u];
                 }
                 if (abs(om - m) > 1e-8)
                 {
                     cout << n << " " << t << " " << m << " " << om << endl;
                     check = false;
                 }
             });
        return check;
    }

    template <bool add>
    void update_edge(size_t u, size_t v, double dx)
    {
        for (auto& m : _m_temp)
            m.clear();
        iter_time<true, false>
            (std::array<size_t,1>({u}), v,
             [&](auto, size_t n, auto, auto& su, auto m)
             {
                 m += su[u] * ((add) ? dx : -dx);
                 auto& mn = _m_temp[n];
                 mn.emplace_back(m);
             });
        for (size_t n = 0; n < _m_temp.size(); ++n)
        {
            auto& m = _m[n][v];
            m.swap(_m_temp[n]);
            if (m.empty())
                m.emplace_back(0);
        }
    }

    template <bool add>
    double get_edge_dS(size_t u, size_t v, double dx)
    {
        double dL = 0;
        iter_time<true, true>
            (std::array<size_t,1>({u}), v,
             [&](size_t v, size_t n, int, auto& su, auto&& m, auto... s_v)
             {
                 double dm = su[u] * ((add) ? dx : -dx);
                 if (dm != 0)
                 {
                     dL += (static_cast<Base*>(this)->log_P(v, n, m + dm, s_v...) -
                            static_cast<Base*>(this)->log_P(v, n, m, s_v...));
                 }
             });

        return -dL;
    }

    double get_node_prob(size_t v)
    {
        double L = 0;
        iter_time<true, true>
            (std::array<size_t,0>({}), v,
             [&](size_t v, size_t n, int, auto&, auto&& m, auto... s_v)
             {
                 L += static_cast<Base*>(this)->log_P(v, n, m, s_v...);
             });
        return L;
    }

    typedef vprop_map_t<std::vector<double>>::type::unchecked_t mmap_t;

protected:
    std::vector<smap_t::unchecked_t>& _s;

    vprop_map_t<double>::type::unchecked_t _st;

    std::vector<mmap_t> _m;
    std::vector<std::vector<double>> _m_temp;
};


template <class T>
double l2sinha(T x) // log((exp(x) - exp(-x))/x)
{
    x = abs(x);
    if (x < 1e-8)
        return log(2);
    return x + log1p(-exp(-2*x)) - log(x);
}

class CIsingBaseState
{
public:
    template <class S>
    CIsingBaseState(S& s, python::dict params)
        : _M(s._s.size())
    {
        set_params(params);
    };

    typedef typename vprop_map_t<double>::type hmap_t;

    void set_params(python::dict params)
    {
        int n = python::extract<int>(params.get("n", -1));
        if (n == -1 ||
            python::extract<double>(params["beta"]).check())
        {
            _beta.resize(_M);
            _h.resize(_M);
            for (size_t n = 0; n < _M; ++n)
                set_params(params, n);
        }
        else
        {
            set_params(params, n);
        }

        if (params.has_key("n"))
            python::api::delitem(params, "n");
    }

    void set_params(python::dict params, size_t n)
    {
        python::extract<double> ebeta(params["beta"]);
        if (ebeta.check())
            _beta[n] = ebeta();
        else
            _beta[n] = python::extract<double>(params["beta"][n]);

        if (hasattr(params["h"], "_get_any"))
            _h[n] = boost::any_cast<hmap_t>(python::extract<boost::any>(params["h"].attr("_get_any")())()).get_unchecked();
        else
            _h[n] = boost::any_cast<hmap_t>(python::extract<boost::any>(params["h"][n].attr("_get_any")())()).get_unchecked();
    }

    double log_P(size_t v, size_t n, double m, double s)
    {
        double x = _h[n][v] + _beta[n] * m;
        return s * x - l2sinha(x);
    }

    std::vector<hmap_t::unchecked_t> _h;

private:
    size_t _M;
    std::vector<double> _beta;
};

class PseudoCIsingState
    : public ContinuousStateBase<PseudoCIsingState, false>,
      public CIsingBaseState
{
public:
    template <class S>
    PseudoCIsingState(S& s, python::dict params)
        : ContinuousStateBase<PseudoCIsingState, false>(s),
        CIsingBaseState(s, params) {}
};

class CIsingGlauberState
    : public ContinuousStateBase<CIsingGlauberState, true>,
      public CIsingBaseState
{
public:
    template <class S>
    CIsingGlauberState(S& s, python::dict params)
        : ContinuousStateBase<CIsingGlauberState, true>(s),
        CIsingBaseState(s, params) {}

    double log_P(size_t v, size_t n, double m, double, double sn)
    {
        return CIsingBaseState::log_P(v, n, m, sn);
    }
};

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_DYNAMICS_CONTIUOUS_HH
