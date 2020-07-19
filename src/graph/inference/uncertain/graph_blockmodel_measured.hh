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

#ifndef GRAPH_BLOCKMODEL_MEASURED_HH
#define GRAPH_BLOCKMODEL_MEASURED_HH

#include "config.h"

#include <vector>

#include "../support/graph_state.hh"
#include "../blockmodel/graph_blockmodel_util.hh"
#include "graph_blockmodel_uncertain.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MEASURED_STATE_params                                                  \
    ((g, &, all_graph_views, 1))                                               \
    ((n,, eprop_map_t<int>::type, 0))                                          \
    ((x,, eprop_map_t<int>::type, 0))                                          \
    ((n_default,, int, 0))                                                     \
    ((x_default,, int, 0))                                                     \
    ((alpha,, long double, 0))                                                 \
    ((beta,, long double, 0))                                                  \
    ((mu,, long double, 0))                                                    \
    ((nu,, long double, 0))                                                    \
    ((aE,, double, 0))                                                         \
    ((E_prior,, bool, 0))                                                      \
    ((self_loops,, bool, 0))

template <class BlockState>
struct Measured
{
    GEN_STATE_BASE(MeasuredStateBase, MEASURED_STATE_params)

    template <class... Ts>
    class MeasuredState
        : public MeasuredStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(MeasuredStateBase<Ts...>,
                         MEASURED_STATE_params)
        GET_PARAMS_TYPEDEF(Ts, MEASURED_STATE_params)

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
        MeasuredState(BlockState& block_state, ATs&&... args)
            : MeasuredStateBase<Ts...>(std::forward<ATs>(args)...),
              _block_state(block_state)
        {
            _u_edges.resize(num_vertices(_u));
            for (auto e : edges_range(_u))
            {
                get_u_edge<true>(source(e, _u), target(e, _u)) = e;
                _E += _eweight[e];
            }

            size_t gE = 0;
            _edges.resize(num_vertices(_g));
            for (auto e : edges_range(_g))
            {
                get_edge<true>(source(e, _g), target(e, _g)) = e;
                _N += _n[e];
                _X += _x[e];
                gE++;
            }

            for (auto e : edges_range(_u))
            {
                if (_eweight[e] == 0 || (!_self_loops &&
                                         (source(e, _u) == target(e, _u))))
                    continue;
                auto ge = get_edge<false>(source(e, _u), target(e, _u));
                _T += (ge != _null_edge) ? _x[ge] : _x_default;
                _M += (ge != _null_edge) ? _n[ge] : _n_default;
            }

            uint64_t N = num_vertices(_g);
            if (_self_loops)
                _NP = graph_tool::is_directed(_g) ? N * N : (N * (N + 1)) / 2;
            else
                _NP = graph_tool::is_directed(_g) ? N * (N - 1) : (N * (N - 1)) / 2;

            _N += (_NP - gE) * _n_default;
            _X += (_NP - gE) * _x_default;

            _block_state.enable_partition_stats();
        }

        MeasuredState(const MeasuredState& other)
            : MeasuredStateBase<Ts...>(static_cast<const MeasuredStateBase<Ts...>&>(other)),
              _block_state(other._block_state),
              _u_edges(other._u_edges),
              _edges(other._edges),
              _pe(other._pe),
              _NP(other._NP),
              _E(other._E),
              _N(other._N),
              _X(other._X),
              _T(other._T),
              _M(other._M)
        {
            _block_state.enable_partition_stats();
        }

        typedef BlockState block_state_t;
        BlockState& _block_state;
        typename BlockState::g_t& _u = _block_state._g;
        typename BlockState::eweight_t& _eweight = _block_state._eweight;
        GraphInterface::edge_t _null_edge;
        std::vector<double> _recs;

        std::vector<gt_hash_map<size_t, GraphInterface::edge_t>> _u_edges;
        std::vector<gt_hash_map<size_t, GraphInterface::edge_t>> _edges;

        double _pe = log(_aE);
        uint64_t _NP = 0;
        uint64_t _E = 0;

        uint64_t _N = 0;
        uint64_t _X = 0;
        uint64_t _T = 0;
        uint64_t _M = 0;

        template <bool insert, class Graph, class Elist>
        auto& _get_edge(size_t u, size_t v, Graph& g, Elist& edges)
        {
            if (!graph_tool::is_directed(g) && u > v)
                std::swap(u, v);
            auto& qe = edges[u];
            if (insert)
                return qe[v];
            auto iter = qe.find(v);
            if (iter != qe.end())
                return iter->second;
            return _null_edge;
        }

        template <bool insert=false>
        auto& get_u_edge(size_t u, size_t v)
        {
            return _get_edge<insert>(u, v, _u, _u_edges);
        }

        template <bool insert=false>
        auto& get_edge(size_t u, size_t v)
        {
            return _get_edge<insert>(u, v, _g, _edges);
        }

        long double get_MP(size_t T, size_t M, bool complete = true)
        {
            long double S = 0;
            S += lbeta((M - T) + _alpha, T + _beta);
            S += lbeta((_X - T) + _mu, ((_N - _X) - (M - T)) + _nu);
            if (complete)
            {
                S -= lbeta(_alpha, _beta);
                S -= lbeta(_mu, _nu);
            }
            return S;
        }

        double entropy(bool latent_edges, bool density)
        {
            double S = 0;
            size_t gE = 0;
            if (latent_edges)
            {
                for (auto e : edges_range(_g))
                {
                    S += lbinom(_n[e], _x[e]);
                    gE++;
                }
                S += (_NP - gE) * lbinom(_n_default, _x_default);

                S += get_MP(_T, _M);
            }

            if (density && _E_prior)
                S += _E * _pe - lgamma_fast(_E + 1) - exp(_pe);

            return -S;
        }

        double get_dS(int dT, int dM)
        {
            // FIXME: Can be faster!
            long double Si = get_MP(_T, _M, false);
            long double Sf = get_MP(_T + dT, _M + dM, false);
            return -(Sf - Si);
        }

        double remove_edge_dS(size_t u, size_t v, const uentropy_args_t& ea)
        {
            auto& e = get_u_edge(u, v);
            double dS = _block_state.template modify_edge_dS<false>(source(e, _u),
                                                                    target(e, _u),
                                                                    e, _recs, ea);
            if (ea.density && _E_prior)
            {
                dS += _pe;
                dS += lgamma_fast(_E) - lgamma_fast(_E + 1);
            }

            if (ea.latent_edges)
            {
                if (_eweight[e] == 1 && (_self_loops || u != v))
                {
                    auto& m = get_edge<false>(u, v);
                    int dT = (m == _null_edge) ? _x_default : _x[m];
                    int dM = (m == _null_edge) ? _n_default : _n[m];
                    dS += get_dS(-dT, -dM);
                }
            }

            return dS;
        }

        double add_edge_dS(size_t u, size_t v, const uentropy_args_t& ea)
        {
            auto& e = get_u_edge(u, v);
            double dS = _block_state.template modify_edge_dS<true>(u, v, e,
                                                                   _recs, ea);
            if (ea.density && _E_prior)
            {
                dS -= _pe;
                dS += lgamma_fast(_E + 2) - lgamma_fast(_E + 1);
            }

            if (ea.latent_edges)
            {
                if ((e == _null_edge || _eweight[e] == 0) && (_self_loops || u != v))
                {
                    auto& m = get_edge<false>(u, v);
                    int dT = (m == _null_edge) ? _x_default : _x[m];
                    int dM = (m == _null_edge) ? _n_default : _n[m];
                    dS += get_dS(dT, dM);
                }
            }
            return dS;
        }

        void remove_edge(size_t u, size_t v)
        {
            auto& e = get_u_edge(u, v);

            if (_eweight[e] == 1 && (_self_loops || u != v))
            {
                auto& m = get_edge<false>(u, v);
                int dT = (m == _null_edge) ? _x_default : _x[m];
                int dM = (m == _null_edge) ? _n_default : _n[m];
                _T -= dT;
                _M -= dM;
            }

            _block_state.template modify_edge<false>(u, v, e,
                                                     _recs);
            _E--;
        }

        void add_edge(size_t u, size_t v)
        {
            auto& e = get_u_edge<true>(u, v);

            if ((e == _null_edge || _eweight[e] == 0) && (_self_loops || u != v))
            {
                auto& m = get_edge<false>(u, v);
                int dT = (m == _null_edge) ? _x_default : _x[m];
                int dM = (m == _null_edge) ? _n_default : _n[m];
                _T += dT;
                _M += dM;
            }

            _block_state.template modify_edge<true>(u, v, e,
                                                    _recs);
            _E++;
        }

        void set_hparams(double alpha, double beta, double mu, double nu)
        {
            _alpha = alpha;
            _beta = beta;
            _mu = mu;
            _nu = nu;
        }

        uint64_t get_N() { return _N; }
        uint64_t get_X() { return _X; }
        uint64_t get_T() { return _T; }
        uint64_t get_M() { return _M; }
    };
};

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_MEASURED_HH
