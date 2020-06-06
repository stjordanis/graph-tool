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

#ifndef GRAPH_BLOCKMODEL_DYNAMICS_DISCRETE_HH
#define GRAPH_BLOCKMODEL_DYNAMICS_DISCRETE_HH

#include "config.h"

#include <vector>

#include "graph_blockmodel_dynamics.hh"
#include "graph_python_interface.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

template <class Base, bool spin, bool epidemic, bool tshift>
class DiscreteStateBase
{
public:
    typedef std::conditional_t<spin, double, std::vector<double>> m_t;

    typedef vprop_map_t<std::vector<int32_t>>::type tmap_t;
    typedef vprop_map_t<std::vector<int32_t>>::type smap_t;

    template <class State>
    DiscreteStateBase(State& s)
        : _t(s._t), _s(s._s), _st(num_vertices(s._u)), _pos(num_vertices(s._u))
    {
        if (_t.empty())
        {
            for (auto sn : _s)
            {
                size_t T = numeric_limits<size_t>::max();
                for (auto v : vertices_range(s._u))
                {
                    if (T == numeric_limits<size_t>::max())
                        T = sn[v].size();
                    if (sn[v].size() != T)
                        throw ValueException("invalid uncompressed time "
                                             "series: all vertices must "
                                             "have the same number of "
                                             "states");
                }
            }
        }
        else
        {
            for (size_t n = 0; n < _t.size(); ++n)
            {
                auto& sn = _s[n];
                auto& tn = _t[n];
                for (auto v : vertices_range(s._u))
                {
                    if (sn[v].size() != tn[v].size())
                        throw ValueException("invalid compressed time "
                                             "series: all vertices must "
                                             "have the same number of "
                                             "states and times");
                    if (sn[v].empty())
                        throw ValueException("invalid compressed time "
                                             "series: all vertices must "
                                             "have nonempty states and times");
                }
            }
        }

        for (auto sn : _s)
            _m.emplace_back(num_vertices(s._u));

        for (size_t n = 0; n < _t.size(); ++n)
        {
            auto& sn = _s[n];
            auto& tn = _t[n];
            int T = 0;
            for (auto v : vertices_range(s._u))
                T = std::max(tn[v].back(), T);
            for (auto v : vertices_range(s._u))
            {
                auto& sv = sn[v];
                auto& tv = tn[v];
                if (tv.back() < T)
                {
                    tv.push_back(T);
                    sv.push_back(sv.back());
                }
            }
            _T.push_back(T);
        }

        reset_m(s);
        _m_temp.resize(_s.size());
    };

    template <class State>
    void reset_m(State& s)
    {
        for (auto v : vertices_range(s._u))
        {
            for (auto& m : _m)
                m[v].clear();
        }

        auto xc = s._x.get_checked();
        for (auto v : vertices_range(s._u))
        {
            iter_time<false, false>
                (in_or_out_neighbors_range(v, s._u), v,
                 [&](auto, size_t n, size_t t, auto& su)
                 {
                     m_t m = m_t();
                     if constexpr (!spin)
                        m.resize(static_cast<Base*>(this)->get_q());
                     for (auto e : in_or_out_edges_range(v, s._u))
                     {
                         auto u = source(e, s._u);
                         if (u == v && !s._self_loops)
                             continue;
                         if constexpr (spin)
                         {
                             if constexpr (epidemic)
                                 m += (su[u] == 1) ? xc[e] : 0;
                             else
                                 m += xc[e] * su[u];
                         }
                         else
                         {
                             m[su[u]] += xc[e];
                         }
                     }
                     if (_t.empty() || t == 0 || m != get<1>(_m[n][v].back()))
                         _m[n][v].emplace_back(t, m);
                 });
            for (auto& m : _m)
            {
                if (m[v].empty())
                {
                    if constexpr (spin)
                        m[v].emplace_back(0, m_t());
                    else
                        m[v].emplace_back(0, m_t(static_cast<Base*>(this)->get_q()));
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
                 m_t m = m_t();
                 if constexpr (!spin)
                    m.resize(static_cast<Base*>(this)->get_q());
                 for (auto e : in_or_out_edges_range(v, s._u))
                 {
                     auto u = source(e, s._u);
                     if (u == v && !s._self_loops)
                         continue;
                     if constexpr (spin)
                     {
                         if constexpr (epidemic)
                             m += (su[u] == 1) ? xc[e] : 0;
                         else
                             m += xc[e] * su[u];
                     }
                     else
                     {
                         m[su[u]] += xc[e];
                     }
                 }
                 if (abs(om - m) > 1e-8)
                 {
                     cout << n << " " << t << " " << m << " " << om << endl;
                     check = false;
                 }
             });
        return check;
    }

    template <bool follow_m, bool follow_v, class US, class F>
    void iter_time_compressed(US&& us, size_t v, F&& f)
    {
        for (size_t n = 0; n < _s.size(); ++n)
        {
            auto& sn = _s[n];
            auto& tn = _t[n];

            if (tshift && sn[v].size() <= 1)
                continue;

            for (auto u : us)
            {
                _pos[u] = 0;
                _st[u] = sn[u][0];
            }

            size_t pos_m = 0;
            auto& mnv = _m[n][v];
            [[maybe_unused]] auto m = (follow_m) ? get<1>(mnv[0]) : m_t();

            auto& tnv = _t[n][v];
            size_t pos_v = 0;
            [[maybe_unused]] int s_v = (follow_v) ? sn[v][0] : 0;
            size_t pos_nv = 0;
            [[maybe_unused]] int s_nv = (follow_v) ? sn[v][0] : 0;
            if (tshift && pos_nv + 1 < tnv.size() && tnv[pos_nv + 1] == 1 && follow_v)
                s_nv = sn[v][++pos_nv];

            int t = 0;
            while (t <= _T[n] - int(tshift))
            {
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

                if (!tshift && t == _T[n])
                    break;

                // determine next time point
                auto nt = _T[n];
                for (auto u : us)
                {
                    auto pos = _pos[u];
                    auto& tnu = tn[u];
                    if (pos + 1 < tnu.size())
                        nt = std::min(nt, tnu[pos + 1]);
                }

                if (pos_m + 1 < mnv.size() && follow_m)
                    nt = std::min(nt, get<0>(mnv[pos_m + 1]));

                if (pos_v + 1 < tnv.size() && follow_v)
                    nt = std::min(nt, tnv[pos_v + 1]);

                // need to be t-1 w.r.t. nv
                if (pos_nv + 1 < tnv.size() && follow_v)
                    nt = std::min(nt, tnv[pos_nv + 1] - int(tshift));

                if (tshift)
                {
                    // if nothing happens until the end, we need to update one
                    // last time
                    if (t < _T[n] - 1 && nt == _T[n])
                        nt = _T[n] - 1;
                }

                t = nt;

                // update current states at time t
                for (auto u : us)
                {
                    auto& pos = _pos[u];
                    auto& tnu = tn[u];
                    auto npos = pos + 1;
                    if (npos < tnu.size() && t == tnu[npos])
                    {
                        _st[u] = sn[u][npos];
                        pos = npos;
                    }
                }

                // update current m
                auto npos_m = pos_m + 1;
                if (npos_m < mnv.size() && t == get<0>(mnv[npos_m]) && follow_m)
                {
                    m = get<1>(mnv[npos_m]);
                    pos_m = npos_m;
                }

                // update v state
                auto npos_v = pos_v + 1;
                if (npos_v < tnv.size() && t == tnv[npos_v] && follow_v)
                {
                    s_v = sn[v][npos_v];
                    pos_v = npos_v;
                }

                // update nv state
                auto npos_nv = pos_nv + 1;
                if (npos_nv < tnv.size() && t == tnv[npos_nv] - int(tshift) && follow_v)
                {
                    s_nv = sn[v][npos_nv];
                    pos_nv = npos_nv;
                }
            }
        }
    }

    template <bool follow_m, bool follow_v, class US, class F>
    void iter_time_uncompressed(US&& us, size_t v, F&& f)
    {
        for (size_t n = 0; n < _s.size(); ++n)
        {
            auto& sn = _s[n];
            auto& snv = _s[n][v];
            auto& mnv = _m[n][v];

            for (size_t t = 0; t < snv.size() - int(tshift); ++t)
            {
                [[maybe_unused]] auto m = follow_m ? get<1>(mnv[t]) : m_t();
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

    template <bool follow_m, bool follow_v, class US, class F>
    void iter_time(US&& us, size_t v, F&& f)
    {
        if (!_t.empty())
            iter_time_compressed<follow_m, follow_v>(std::forward<US&&>(us), v,
                                                     std::forward<F&&>(f));
        else
            iter_time_uncompressed<follow_m, follow_v>(std::forward<US&&>(us), v,
                                                       std::forward<F&&>(f));
    }

    template <bool add>
    void update_edge(size_t u, size_t v, double dx)
    {
        for (auto& m : _m_temp)
            m.clear();
        iter_time<true, false>
            (std::array<size_t,1>({u}), v,
             [&](auto, size_t n, auto t, auto& su, auto m)
             {
                 if constexpr (spin)
                 {
                     if constexpr (epidemic)
                         m += (su[u] == 1) ? ((add) ? dx : -dx) : 0;
                     else
                         m += su[u] * ((add) ? dx : -dx);
                 }
                 else
                 {
                     if (add)
                         m[su[u]] += dx;
                     else
                         m[su[u]] -= dx;
                 }
                 auto& mn = _m_temp[n];
                 if (_t.empty() || mn.empty() || get<1>(mn.back()) != m)
                     mn.emplace_back(t, m);
             });
        for (size_t n = 0; n < _m_temp.size(); ++n)
        {
            auto& m = _m[n][v];
            m.swap(_m_temp[n]);
            if (m.empty())
            {
                if constexpr (spin)
                    m.emplace_back(0, 0);
                else
                    m.emplace_back(0, m_t(static_cast<Base*>(this)->get_q()));
            }
        }
    }

    template <bool add>
    double get_edge_dS(size_t u, size_t v, double dx)
    {
        double dL = 0;
        if (!_t.empty())
        {
            size_t ln = 0;
            int lt = 0;
            double ldL = 0;
            iter_time_compressed<true, true>
                (std::array<size_t,1>({u}), v,
                 [&](size_t v, size_t n, int t, auto& su, auto&& m, auto... s_v)
                 {
                     if (n != ln)
                     {
                         ln = n;
                         lt = 0;
                         ldL = 0;
                     }

                     auto dt = t - lt;
                     dL += dt * ldL;

                     if constexpr (spin)
                     {
                         m_t dm;
                         if constexpr (epidemic)
                             dm = (su[u] == 1) ? ((add) ? dx : -dx) : 0;
                         else
                             dm = su[u] * ((add) ? dx : -dx);

                         if (dm != 0)
                         {
                             ldL = (static_cast<Base*>(this)->log_P(v, n, m + dm, s_v...) -
                                    static_cast<Base*>(this)->log_P(v, n, m, s_v...));
                         }
                         else
                         {
                             ldL = 0;
                         }
                     }
                     else
                     {
                         auto nm = m;
                         if (add)
                             nm[su[u]] += dx;
                         else
                             nm[su[u]] -= dx;
                         ldL = (static_cast<Base*>(this)->log_P(v, n, nm, s_v...) -
                                static_cast<Base*>(this)->log_P(v, n, m, s_v...));
                     }

                     lt = t;

                     if (t == _T[n] - 1)
                         dL += ldL;
                 });
        }
        else
        {
            iter_time_uncompressed<true, true>
                (std::array<size_t,1>({u}), v,
                 [&](size_t v, size_t n, int t, auto& su, auto&& m, auto... s_v)
                 {
                     if constexpr (spin)
                     {
                         m_t dm;
                         if constexpr (epidemic)
                             dm = (su[u] == 1) ? ((add) ? dx : -dx) : 0;
                         else
                             dm = su[u] * ((add) ? dx : -dx);

                         if (dm != 0)
                         {
                             dL += (static_cast<Base*>(this)->log_P(t, v, n, m + dm, s_v...) -
                                    static_cast<Base*>(this)->log_P(t, v, n, m, s_v...));
                         }
                     }
                     else
                     {
                         auto nm = m;
                         if (add)
                             nm[su[u]] += dx;
                         else
                             nm[su[u]] -= dx;
                         dL += (static_cast<Base*>(this)->log_P(t, v, n, nm, s_v...) -
                                static_cast<Base*>(this)->log_P(t, v, n, m, s_v...));

                     }
                 });
        }

        return -dL;
    }

    double get_node_prob(size_t v)
    {
        double L = 0;
        if (!_t.empty())
        {
            size_t ln = 0;
            int lt = 0;
            double lL = 0;
            iter_time_compressed<true, true>
                (std::array<size_t,0>({}), v,
                 [&](size_t v, size_t n, int t, auto&, auto&& m, auto... s_v)
                 {
                     if (n != ln)
                     {
                         ln = n;
                         lt = 0;
                         lL = 0;
                     }

                     auto dt = t - lt;
                     L += dt * lL;

                     lL = static_cast<Base*>(this)->log_P(v, n, m, s_v...);

                     lt = t;

                     if (t == _T[n] - 1)
                         L += lL;
                 });
        }
        else
        {
            iter_time_uncompressed<true, true>
                (std::array<size_t,0>({}), v,
                 [&](size_t v, size_t n, int t, auto&, auto&& m, auto... s_v)
                 {
                     L += static_cast<Base*>(this)->log_P(t, v, n, m, s_v...);
                 });
        }
        return L;
    }

    typedef typename vprop_map_t<std::vector<std::tuple<int, m_t>>>::type::unchecked_t mmap_t;

protected:
    std::vector<tmap_t::unchecked_t>& _t;
    std::vector<smap_t::unchecked_t>& _s;
    std::vector<int> _T;

    vprop_map_t<int>::type::unchecked_t _st;
    vprop_map_t<size_t>::type::unchecked_t _pos;

    std::vector<mmap_t> _m;
    std::vector<std::vector<std::tuple<int, m_t>>> _m_temp;
};

class SIState: public DiscreteStateBase<SIState, true, true, true>
{
public:
    enum State { S, I, R, E };

    template <class S>
    SIState(S& s, python::dict params)
        : DiscreteStateBase<SIState, true, true, true>(s),
          _exposed(python::extract<bool>(params["exposed"])),
          _E(_exposed ? State::E : State::I)
    {
        set_params(params);
    };

    typedef typename vprop_map_t<double>::type hmap_t;
    typedef typename vprop_map_t<std::vector<uint8_t>>::type amap_t;

    void set_params(python::dict params)
    {
        int n = python::extract<int>(params.get("n", -1));
        if (n == -1 ||
            python::extract<double>(params["r"]).check())
        {
            _r.resize(_s.size());
            for (size_t n = 0; n < _s.size(); ++n)
                set_params(params, n);

            auto active = params["active"];
            if (active != python::object())
            {
                for (int i = 0; i < python::len(active); ++i)
                    _active.push_back(boost::any_cast<amap_t>(python::extract<boost::any>(active[i].attr("_get_any")())()).get_unchecked());
            }
        }
        else
        {
            set_params(params, n);
        }

        if (params.has_key("n"))
            python::api::delitem(params, "n");

        _has_r_v = false;
        if (params["r_v"] != python::object())
        {
            _r_v = boost::any_cast<hmap_t>(python::extract<boost::any>(params["r_v"].attr("_get_any")())()).get_unchecked();
            _has_r_v = true;
        }
    }

    void set_params(python::dict params, size_t n)
    {
        python::extract<double> er(params["r"]);
        if (er.check())
            _r[n] = er();
        else
            _r[n] = python::extract<double>(params["r"][n]);
    }

    double log_P(size_t v, size_t n, double m, int s, int ns)
    {
        if (s != State::S)
            return 0;

        double r = _r[n] * ((_has_r_v) ? _r_v[v] : 1);
        double p = (1-r) * (1 - exp(m)) + r;
        return (ns == _E) ? log(p) : log1p(-p);
    }

    double log_P(int t, size_t v, size_t n, double m, int s, int ns)
    {
        if (_active.empty() || _active[n][v][t])
        {
            return log_P(v, n, m, s, ns);
        }
        else
        {
            if (ns == State::I && s != ns)
                return -numeric_limits<double>::infinity();
            return 0;
        }
    }

    hmap_t::unchecked_t _r_v;

private:
    std::vector<double> _r;
    std::vector<typename amap_t::unchecked_t> _active;
    bool _has_r_v;
    bool _exposed;
    int _E;
};

template <class T>
double l2cosh(T x) // log(exp(x) + exp(-x))
{
    if (x > 0)
        return x + log1p(exp(-2*x)); // exp(x) * (1 + exp(-2x))
    else
        return -x + log1p(exp(2*x)); // exp(-x) * (1 + exp(2x))
}

template <class T>
double l1p2cosh(T x) // log(1 + exp(x) + exp(-x))
{
    if (x > 0)
        return x + log1p(exp(-x + log1p(exp(-x)))); // exp(x) * (1 + exp(-x)(1 + exp(-x)))
    else
        return -x + log1p(exp(x + log1p(exp(x))));  // exp(-x) * (1 + exp(x)(1 + exp(x)))
}


class IsingBaseState
{
public:
    template <class S>
    IsingBaseState(S& s, python::dict params)
        : _M(s._s.size())
    {
        set_params(params);
    };

    typedef typename vprop_map_t<double>::type hmap_t;

    void set_params(python::dict params)
    {
        _has_zero = python::extract<double>(params["has_zero"]);

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

    double log_P(size_t v, size_t n, double m, int s)
    {
        double x = _h[n][v] + _beta[n] * m;
        if (_has_zero)
            return s * x - l1p2cosh(x);
        else
            return s * x - l2cosh(x);
    }

    double log_P(int, size_t v, size_t n, double m, int s)
    {
        return log_P(v, n, m, s);
    }

    std::vector<hmap_t::unchecked_t> _h;

private:
    size_t _M;
    std::vector<double> _beta;
    bool _has_zero;
};

class PseudoIsingState
    : public DiscreteStateBase<PseudoIsingState, true, false, false>,
      public IsingBaseState
{
public:
    template <class S>
    PseudoIsingState(S& s, python::dict params)
        : DiscreteStateBase<PseudoIsingState, true, false, false>(s),
          IsingBaseState(s, params) {};
};

class IsingGlauberState
    : public DiscreteStateBase<IsingGlauberState, true, false, true>,
      public IsingBaseState
{
public:
    template <class S>
    IsingGlauberState(S& s, python::dict params)
        : DiscreteStateBase<IsingGlauberState, true, false, true>(s),
          IsingBaseState(s, params) {};

    double log_P(size_t v, size_t n, double m, int, int ns)
    {
        return IsingBaseState::log_P(v, n, m, ns);
    }

    double log_P(int, size_t v, size_t n, double m, int s, int ns)
    {
        return log_P(v, n, m, s, ns);
    }
};

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_DYNAMICS_DISCRETE_HH
