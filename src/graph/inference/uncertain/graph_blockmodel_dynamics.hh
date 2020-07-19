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

#ifndef GRAPH_BLOCKMODEL_DYNAMICS_HH
#define GRAPH_BLOCKMODEL_DYNAMICS_HH

#include "config.h"

#include <vector>

#include "../support/graph_state.hh"
#include "../blockmodel/graph_blockmodel_util.hh"
#include "graph_blockmodel_uncertain_util.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef typename eprop_map_t<double>::type xmap_t;


#define DYNAMICS_STATE_params                                                  \
    ((g, &, all_graph_views, 1))                                               \
    ((params,, python::dict, 0))                                               \
    ((ot,, python::list, 0))                                                   \
    ((os,, python::list, 0))                                                   \
    ((x,, xmap_t, 0))                                                          \
    ((aE,, double, 0))                                                         \
    ((E_prior,, bool, 0))                                                      \
    ((self_loops,, bool, 0))


template <class Type>
std::vector<typename Type::unchecked_t> from_list(python::object list)
{
    vector<typename Type::unchecked_t> v;
    for (int i = 0; i < python::len(list); ++i)
    {
        boost::any a = python::extract<boost::any>(list[i])();
        v.push_back(boost::any_cast<Type>(a).get_unchecked());
    }
    return v;
};

template <class BlockState, class DState>
struct Dynamics
{
    GEN_STATE_BASE(DynamicsStateBase, DYNAMICS_STATE_params)

    template <class... Ts>
    class DynamicsState
        : public DynamicsStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(DynamicsStateBase<Ts...>,
                         DYNAMICS_STATE_params)
        GET_PARAMS_TYPEDEF(Ts, DYNAMICS_STATE_params)

        typedef typename DState::tmap_t tmap_t;
        typedef typename DState::smap_t smap_t;

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
        DynamicsState(BlockState& block_state, ATs&&... args)
            : DynamicsStateBase<Ts...>(std::forward<ATs>(args)...),
              _block_state(block_state),
              _t(from_list<tmap_t>(_ot)),
              _s(from_list<smap_t>(_os)),
              _dstate(*this, _params),
              _xc(_x.get_checked())
        {
            _u_edges.resize(num_vertices(_u));
            for (auto e : edges_range(_u))
            {
                get_u_edge<true>(source(e, _u), target(e, _u)) = e;
                _E += _eweight[e];
            }
            _block_state.enable_partition_stats();
        }

        DynamicsState(const DynamicsState& other)
            : DynamicsStateBase<Ts...>(static_cast<const DynamicsStateBase<Ts...>&>(other)),
              _block_state(other._block_state),
              _t(other._t),
              _s(other._s),
              _u_edges(other._u_edges),
              _pe(other._pe),
              _E(other._E),
              _dstate(*this, _params),
              _xc(_x.get_checked())
        {
            _block_state.enable_partition_stats();
        }

        typedef BlockState block_state_t;
        BlockState& _block_state;
        std::vector<typename tmap_t::unchecked_t> _t;
        std::vector<typename smap_t::unchecked_t> _s;
        typename BlockState::g_t& _u = _block_state._g;
        typename BlockState::eweight_t& _eweight = _block_state._eweight;
        GraphInterface::edge_t _null_edge;
        std::vector<double> _recs;

        std::vector<gt_hash_map<size_t, GraphInterface::edge_t>> _u_edges;

        double _pe = log(_aE);
        size_t _E = 0;

        DState _dstate;

        xmap_t _xc;

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

        double get_node_prob(size_t u)
        {
            return _dstate.get_node_prob(u);
        }

        double entropy(bool latent_edges, bool density)
        {
            double S = 0;
            if (latent_edges)
            {
                for (auto v : vertices_range(_u))
                    S += _dstate.get_node_prob(v);
            }

            if (density && _E_prior)
                S += _E * _pe - lgamma_fast(_E + 1) - exp(_pe);

            return -S;
        }

        double remove_edge_dS(size_t u, size_t v, const uentropy_args_t& ea)
        {
            auto& e = get_u_edge(u, v);
            auto x = _xc[e];
            double dS = _block_state.template modify_edge_dS<false>(source(e, _u),
                                                                    target(e, _u),
                                                                    e, _recs, ea);
            _xc[e] = x;

            if (ea.density && _E_prior)
            {
                dS += _pe;
                dS += lgamma_fast(_E) - lgamma_fast(_E + 1);
            }

            if (ea.latent_edges)
            {
                if (_eweight[e] == 1 && (_self_loops || u != v))
                {
                    dS += _dstate.template get_edge_dS<false>(u, v, _xc[e]);
                    if (u != v && !graph_tool::is_directed(_u))
                        dS += _dstate.template get_edge_dS<false>(v, u, _xc[e]);
                }
            }
            return dS;
        }

        double add_edge_dS(size_t u, size_t v, double x, const uentropy_args_t& ea)
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
                    dS += _dstate.template get_edge_dS<true>(u, v, x);
                    if (u != v && !graph_tool::is_directed(_u))
                        dS += _dstate.template get_edge_dS<true>(v, u, x);
                }
            }
            return dS;
        }

        double update_edge_dS(size_t u, size_t v, double dx, const uentropy_args_t& ea)
        {
            double dS = 0;
            if (ea.latent_edges)
            {
                if (_self_loops || u != v)
                {
                    dS += _dstate.template get_edge_dS<true>(u, v, dx);
                    if (u != v && !graph_tool::is_directed(_u))
                        dS += _dstate.template get_edge_dS<true>(v, u, dx);
                }
            }
            return dS;
        }

        void remove_edge(size_t u, size_t v)
        {
            auto& e = get_u_edge(u, v);
            auto x = _xc[e];

            _block_state.template modify_edge<false>(u, v, e,
                                                     _recs);

            if ((e == _null_edge || _eweight[e] == 0) && (_self_loops || u != v))
            {
                _dstate.template update_edge<false>(u, v, x);
                if (u != v && !graph_tool::is_directed(_u))
                    _dstate.template update_edge<false>(v, u, x);
            }

            _E--;
        }

        void add_edge(size_t u, size_t v, double x)
        {
            auto& e = get_u_edge<true>(u, v);
            _block_state.template modify_edge<true>(u, v, e,
                                                    _recs);

            if (_eweight[e] == 1 && (_self_loops || u != v))
            {
                _xc[e] = x;
                _dstate.template update_edge<true>(u, v, x);
                if (u != v && !graph_tool::is_directed(_u))
                    _dstate.template update_edge<true>(v, u, x);
            }
            _E++;
        }

        void update_edge(size_t u, size_t v, double dx)
        {
            if (_self_loops || u != v)
            {
                auto& e = get_u_edge(u, v);
                _xc[e] += dx;
                _dstate.template update_edge<true>(u, v, dx);
                if (u != v && !graph_tool::is_directed(_u))
                    _dstate.template update_edge<true>(v, u, dx);
            }
        }

        void set_params(python::dict params)
        {
            _dstate.set_params(params);
        }
    };
};

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_DYNAMICS_HH
