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

#ifndef GRAPH_RANKED_HH
#define GRAPH_RANKED_HH

#include "config.h"

#include <vector>

#include "../blockmodel/graph_blockmodel_util.hh"
#include "../support/graph_state.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef vprop_map_t<int32_t>::type vmap_t;
typedef eprop_map_t<int32_t>::type emap_t;
typedef vprop_map_t<double>::type umap_t;

#define RANKED_STATE_params                                                    \
   ((__class__,&, mpl::vector<python::object>, 1))                             \
   ((u,, umap_t, 0))

GEN_STATE_BASE(RankedStateBase, RANKED_STATE_params)

template <class BState>
class OState
{
public:

    template <class... Ts>
    class RankedState
        : public RankedStateBase<Ts...>
    {
    public:
        GET_PARAMS_USING(RankedStateBase<Ts...>, RANKED_STATE_params)
        GET_PARAMS_TYPEDEF(Ts, RANKED_STATE_params)

        typedef BState bstate_t;

        typedef typename bstate_t::_entropy_args_t _entropy_args_t;

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
        RankedState(BState& ustate, ATs&&... args)
            : RankedStateBase<Ts...>(std::forward<ATs>(args)...),
              _ustate(ustate),
              _g(ustate._g),
              _b(ustate._b),
              _eweight(ustate._eweight),
              _u_c(_u.get_checked())
        {
            for (auto e : edges_range(_g))
            {
                auto u = source(e, _g);
                auto v = target(e, _g);
                _E[stream_dir(_b[u], _b[v])] += _eweight[e];
            }
        }

        bstate_t& _ustate;
        typename bstate_t::g_t& _g;
        typename bstate_t::b_t& _b;
        typename bstate_t::eweight_t& _eweight;

        typedef typename bstate_t::m_entries_t m_entries_t;

        std::array<size_t, 3> _E = {0, 0, 0};

        typename u_t::checked_t _u_c;

        int stream_dir(size_t r, size_t s)
        {
            auto x = _u[r];
            auto y = _u[s];
            if (x < y)
                return 0;   // upstream
            if (x > y)
                return 2;   // downstream
            return 1;
        }

        void add_block(size_t n)
        {
            _ustate.add_block(n);
        }

        gt_hash_map<size_t, int> _delta;
        std::array<int, 3> _dE = {0, 0, 0}; // FIXME: should go in m_entries!

        void get_dE(size_t v, size_t r, size_t nr)
        {
            _dE = {0, 0, 0};

            for (auto e : out_edges_range(v, _g))
            {
                auto s = _b[target(e, _g)];
                auto w = _eweight[e];
                _dE[stream_dir(r, s)] -= w;
                if (target(e, _g) == v)
                    s = nr;
                _dE[stream_dir(nr, s)] += w;
            }

            for (auto e : in_edges_range(v, _g))
            {
                auto s = _b[source(e, _g)];
                auto w = _eweight[e];
                _dE[stream_dir(s, r)] -= w;
                if (source(e, _g) == v)
                    s = nr;
                _dE[stream_dir(s, nr)] += w;
            }
        }

        template <class ME>
        double virtual_move(size_t v, size_t r, size_t nr, entropy_args_t& ea,
                            ME& m_entries)
        {
            if (r == nr)
                return 0;

            entropy_args_t uea = ea;
            uea.edges_dl = false;

            double dS = _ustate.virtual_move(v, r, nr, uea, m_entries);

            if (ea.edges_dl)
            {
                dS -= get_edges_dl({0, 0, 0}, 0);

                get_dE(v, r, nr);

                int dB = 0;
                if (_ustate._wr[r] == 1)
                    dB--;
                if (_ustate._wr[nr] == 0)
                    dB++;

                dS += get_edges_dl(_dE, dB);

                _delta.clear();
                size_t B = num_vertices(_ustate._bg) + 1;
                entries_op(m_entries, _ustate._emat,
                           [&](auto t, auto u, auto&, auto delta)
                           {
                               if (delta == 0 || t == u)
                                   return;
                               _delta[t + B * u] = delta;
                           });

                entries_op(m_entries, _ustate._emat,
                           [&](auto t, auto u, auto& me, auto delta)
                           {
                               if (delta == 0 || t == u)
                                   return;

                               size_t etu = _ustate._mrs[me];
                               size_t eut = get_beprop(u, t, _ustate._mrs,
                                                       _ustate._emat);

                               int delta_r = 0;
                               auto iter = _delta.find(u + B * t);
                               if (iter != _delta.end())
                                   delta_r = iter->second;

                               if (delta_r != 0 && t > u)
                                   return;

                               dS += lbinom_fast(etu + eut, etu);
                               dS -= lbinom_fast(etu + delta + eut + delta_r,
                                                 etu + delta);
                           });
            }
            else
            {
                get_dE(v, r, nr);
            }

            return dS;
        }

        double virtual_move(size_t v, size_t r, size_t nr, entropy_args_t& ea)
        {
            return virtual_move(v, r, nr, ea, _ustate._m_entries);
        }

        template <class ME>
        void move_vertex(size_t v, size_t nr, ME& m_entries)
        {
            auto r = _b[v];
            if (r == nr)
                return;

            for (size_t i = 0; i < 3; ++i)     // FIXME: should go in m_entries!
                _E[i] += _dE[i];

            _ustate.move_vertex(v, nr, m_entries);
        }

        void move_vertex(size_t v, size_t nr)
        {
            auto r = _b[v];

            if (r == nr)
                return;

            get_dE(v, r, nr);

            for (size_t i = 0; i < 3; ++i)
                _E[i] += _dE[i];

            _ustate.move_vertex(v, nr);
        }

        size_t get_empty_block(size_t v, bool force_add = false)
        {
            return _ustate.get_empty_block(v, force_add);
        }

        double sample_u(rng_t& rng)
        {
            uniform_real_distribution<double> usample;
            return usample(rng);
        }

        size_t sample_block(size_t v, double c, double d, rng_t& rng)
        {
            auto s = _ustate.sample_block(v, c, d, rng);

            if (_ustate._wr[s] == 0)
            {
                uniform_real_distribution<double> usample;
                _u_c[s] = usample(rng);
            }

            return s;
        }

        size_t sample_block_local(size_t v, rng_t& rng)
        {
            return _ustate.sample_block_local(v, rng);
        }

        // Computes the move proposal probability
        double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                             bool reverse)
        {
            return _ustate.get_move_prob(v, r, s, c, d, reverse);
        }

        double get_edges_dl(const std::array<int,3>& dE, int dB)
        {
            size_t B = _ustate._candidate_groups.size() + dB;

            double S = 0;

            if (_ustate._coupled_state == nullptr)
                S += lbinom_fast<false>((B * (B + 1)) / 2 + _ustate._E - 1,
                                        _ustate._E);

            std::array<size_t, 3> E;
            for (int i = 0; i < 3; ++i)
                E[i] = _E[i] + dE[i];

            S += lgamma_fast(E[0] + E[2] + 2);
            S -= lgamma_fast(E[0] + 1) + lgamma_fast(E[2] + 1);

            return S;
        }

        double entropy(entropy_args_t ea)
        {
            double S = 0;

            bool edges_dl = ea.edges_dl;
            ea.edges_dl = false;
            S += _ustate.entropy(ea);

            if (edges_dl)
            {
                S += get_edges_dl({0, 0, 0}, 0);

                for (auto e : edges_range(_ustate._bg))
                {
                    auto r = source(e, _ustate._bg);
                    auto s = target(e, _ustate._bg);

                    if (r >= s)
                        continue;

                    size_t ers = _ustate._mrs[e];
                    size_t esr = get_beprop(s, r, _ustate._mrs, _ustate._emat);

                    S -= lbinom_fast(ers + esr, ers);
                }
            }

            return S;
        }

        template <class MCMCState>
        void init_mcmc(MCMCState& state)
        {
            _ustate.init_mcmc(state);
        }

        size_t node_weight(size_t v)
        {
            return _ustate.node_weight(v);
        }

        bool is_last(size_t v)
        {
            return _ustate.is_last(v);
        }

        bool allow_move(size_t v, size_t r)
        {
            return _ustate.allow_move(v, r);
        }

        size_t virtual_remove_size(size_t v)
        {
            return _ustate.virtual_remove_size(v);
        }

        template <class V>
        void push_state(V&) {}
        void pop_state() {}
        void store_next_state(size_t) {}
        void clear_next_state() {}

        void relax_update(bool relax)
        {
            _ustate.relax_update(relax);
        }

        void couple_state(BlockStateVirtualBase& us,
                          const entropy_args_t& ea)
        {
            _ustate.couple_state(us, ea);
        }

        void decouple_state()
        {
            _ustate.decouple_state();
        }
    };
};
} // graph_tool namespace

#endif //GRAPH_RANKED_HH
