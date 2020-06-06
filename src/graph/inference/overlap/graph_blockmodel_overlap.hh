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

#ifndef GRAPH_BLOCKMODEL_OVERLAP_HH
#define GRAPH_BLOCKMODEL_OVERLAP_HH

#include "config.h"
#include <tuple>

#include "../support/graph_state.hh"
#include "../blockmodel/graph_blockmodel_util.hh"
#include "graph_blockmodel_overlap_util.hh"

namespace graph_tool
{

using namespace boost;
using namespace std;

typedef vprop_map_t<int32_t>::type vmap_t;
typedef eprop_map_t<int32_t>::type emap_t;

typedef vprop_map_t<int64_t>::type vimap_t;
typedef vprop_map_t<vector<int64_t>>::type vvmap_t;

typedef mpl::vector2<std::true_type, std::false_type> use_hash_tr;

#define OVERLAP_BLOCK_STATE_params                                             \
    ((g, &, never_filtered_never_reversed, 1))                                 \
    ((use_hash,, use_hash_tr, 1))                                              \
    ((_abg, &, boost::any&, 0))                                                \
    ((node_index,, vimap_t, 0))                                                \
    ((half_edges,, vvmap_t, 0))                                                \
    ((mrs,, emap_t, 0))                                                        \
    ((mrp,, vmap_t, 0))                                                        \
    ((mrm,, vmap_t, 0))                                                        \
    ((wr,, vmap_t, 0))                                                         \
    ((b,, vmap_t, 0))                                                          \
    ((empty_blocks, & ,std::vector<size_t>&, 0))                               \
    ((empty_pos,, vmap_t, 0))                                                  \
    ((candidate_blocks, &, std::vector<size_t>&, 0))                           \
    ((candidate_pos,, vmap_t, 0))                                              \
    ((bclabel,, vmap_t, 0))                                                    \
    ((pclabel,, vmap_t, 0))                                                    \
    ((bfield,, vprop_map_t<std::vector<double>>::type, 0))                     \
    ((Bfield, &, std::vector<double>&, 0))                                     \
    ((deg_corr,, bool, 0))                                                     \
    ((rec_types,, std::vector<int>, 0))                                        \
    ((rec,, std::vector<eprop_map_t<double>::type>, 0))                        \
    ((drec,, std::vector<eprop_map_t<double>::type>, 0))                       \
    ((brec,, std::vector<eprop_map_t<double>::type>, 0))                       \
    ((bdrec,, std::vector<eprop_map_t<double>::type>, 0))                      \
    ((brecsum,, vprop_map_t<double>::type, 0))                                 \
    ((wparams,, std::vector<std::vector<double>>, 0))                          \
    ((recdx, &, std::vector<double>&, 0))                                      \
    ((Lrecdx, &, std::vector<double>&, 0))                                     \
    ((epsilon, &, std::vector<double>&, 0))

GEN_STATE_BASE(OverlapBlockStateVirtualBase, OVERLAP_BLOCK_STATE_params)

template <class... Ts>
class OverlapBlockState
    : public OverlapBlockStateVirtualBase<Ts...>,
      public BlockStateVirtualBase
{
public:
    GET_PARAMS_USING(OverlapBlockStateVirtualBase<Ts...>, OVERLAP_BLOCK_STATE_params)
    GET_PARAMS_TYPEDEF(Ts, OVERLAP_BLOCK_STATE_params)

    template <class RNG, class... ATs,
              typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
    OverlapBlockState(RNG& rng, ATs&&... args)
        : OverlapBlockStateVirtualBase<Ts...>(std::forward<ATs>(args)...),
          _bg(boost::any_cast<std::reference_wrapper<bg_t>>(__abg)),
          _c_mrs(_mrs.get_checked()),
          _emat(_g, _bg, rng),
          _egroups_update(true),
          _overlap_stats(_g, _b, _half_edges, _node_index, num_vertices(_bg)),
          _coupled_state(nullptr)
    {
        _empty_blocks.clear();
        _candidate_blocks.clear();
        _candidate_blocks.push_back(null_group);
        for (auto r : vertices_range(_bg))
        {
            _wr[r] = _overlap_stats.get_block_size(r);
            if (_wr[r] == 0)
                add_element(_empty_blocks, _empty_pos, r);
            else
                add_element(_candidate_blocks, _candidate_pos, r);
        }

        for (auto& p : _brec)
        {
            _c_brec.push_back(p.get_checked());
            double x = 0;
            for (auto me : edges_range(_bg))
                x += p[me];
            _recsum.push_back(x);
        }
        for (auto& p : _bdrec)
            _c_bdrec.push_back(p.get_checked());
        if (!_rec_types.empty())
        {
            _recx2.resize(this->_rec_types.size());
            _recdx.resize(this->_rec_types.size());
            for (auto me : edges_range(_bg))
            {
                if (_brec[0][me] > 0)
                {
                    _B_E++;
                    for (size_t i = 0; i < _rec_types.size(); ++i)
                    {
                        if (this->_rec_types[i] == weight_type::REAL_NORMAL)
                        {
                            _recx2[i] += std::pow(_brec[i][me], 2);
                            if (_brec[0][me] > 1)
                                _recdx[i] += \
                                    (_bdrec[i][me] -
                                     std::pow(_brec[i][me], 2) / _brec[0][me]);
                        }
                    }
                }
                if (_brec[0][me] > 1)
                    _B_E_D++;
            }
        }

        _rt = weight_type::NONE;
        for (auto rt : _rec_types)
        {
            _rt = rt;
            if (rt == weight_type::REAL_NORMAL)
                break;
        }
        _dBdx.resize(_rec_types.size());
        _LdBdx.resize(_rec_types.size());
    }

    OverlapBlockState(const OverlapBlockState& other)
        : OverlapBlockStateVirtualBase<Ts...>
             (static_cast<const OverlapBlockStateVirtualBase<Ts...>&>(other)),
          _bg(other._bg),
          _c_mrs(other._c_mrs),
          _c_brec(other._c_brec),
          _c_bdrec(other._c_bdrec),
          _recsum(other._recsum),
          _recx2(other._recx2),
          _dBdx(other._dBdx),
          _LdBdx(other._LdBdx),
          _B_E(other._B_E),
          _B_E_D(other._B_E_D),
          _rt(other._rt),
          _emat(other._emat),
          _egroups_update(other._egroups_update),
          _overlap_stats(other._overlap_stats),
          _coupled_state(nullptr)
    {
        if (other.is_partition_stats_enabled())
            enable_partition_stats();
    }

    template <bool Add>
    void modify_vertex(size_t v, size_t r)
    {
        if (Add && _wr[r] == 0)
        {
            remove_element(_empty_blocks, _empty_pos, r);
            add_element(_candidate_blocks, _candidate_pos, r);
        }

        if (Add)
            get_move_entries(v, null_group, r, _m_entries);
        else
            get_move_entries(v, r, null_group, _m_entries);

        apply_delta<Add,!Add>(*this, _m_entries);

        if (Add)
        {
            _overlap_stats.add_half_edge(v, r, _b, _g);
            _b[v] = r;
            if (!_egroups.empty() && _egroups_update)
                _egroups.add_vertex(v, _b, _eweight, _g);
        }
        else
        {
            _overlap_stats.remove_half_edge(v, r, _b, _g);
            if (!_egroups.empty() && _egroups_update)
                _egroups.remove_vertex(v, _b, _g);
        }

        _wr[r] = _overlap_stats.get_block_size(r);

        if (!Add && _wr[r] == 0)
        {
            remove_element(_candidate_blocks, _candidate_pos, r);
            add_element(_empty_blocks, _empty_pos, r);
        }
    }

    size_t get_B_E()
    {
        return _B_E;
    }

    size_t get_B_E_D()
    {
        return _B_E_D;
    }

    void remove_vertex(size_t v)
    {
        modify_vertex<false>(v, _b[v]);
    }

    void add_vertex(size_t v, size_t r)
    {
        modify_vertex<true>(v, r);
    }

    bool allow_move(size_t r, size_t nr)
    {
        if (_coupled_state != nullptr)
        {
            auto& hb = _coupled_state->get_b();
            auto rr = hb[r];
            auto ss = hb[nr];
            if (rr != ss && !_coupled_state->allow_move(rr, ss))
                return false;
        }
        return _bclabel[r] == _bclabel[nr];
    }

    // move a vertex from its current block to block nr
    void move_vertex(size_t v, size_t nr)
    {
        size_t r = _b[v];

        if (r == nr)
            return;

        if (!allow_move(r, nr))
            throw ValueException("cannot move vertex across clabel barriers");

        bool r_vacate = (_overlap_stats.virtual_remove_size(v, r) == 0);
        bool nr_occupy = (_wr[nr] == 0);

        remove_vertex(v);
        add_vertex(v, nr);

        if (_coupled_state != nullptr)
        {
            auto& hb = _coupled_state->get_b();

            if (r_vacate)
            {
                _coupled_state->remove_partition_node(r, hb[r]);
                _coupled_state->set_vertex_weight(r, 0);
            }

            if (nr_occupy)
            {
                _coupled_state->set_vertex_weight(nr, 1);
                _coupled_state->add_partition_node(nr, hb[nr]);
            }
        }


        if (is_partition_stats_enabled())
            get_partition_stats(v).move_vertex(v, r, nr, _g);

    }

    template <class Vec>
    void move_vertices(Vec& v, Vec& nr)
    {
        for (size_t i = 0; i < std::min(v.size(), nr.size()); ++i)
            move_vertex(v[i], nr[i]);
    }

    void move_vertices(python::object ovs, python::object ors)
    {
        multi_array_ref<uint64_t, 1> vs = get_array<uint64_t, 1>(ovs);
        multi_array_ref<uint64_t, 1> rs = get_array<uint64_t, 1>(ors);
        if (vs.size() != rs.size())
            throw ValueException("vertex and group lists do not have the same size");
        move_vertices(vs, rs);
    }

    void add_edge(size_t, size_t, GraphInterface::edge_t&,
                  const std::vector<double>&)
    {
    }

    void remove_edge(size_t, size_t, GraphInterface::edge_t&,
                     const std::vector<double>&)
    {
    }

    template <class VMap>
    void set_partition(VMap&& b)
    {
        for (auto v : vertices_range(_g))
            move_vertex(v, b[v]);
    }

    void set_partition(boost::any& ab)
    {
        vmap_t& b = boost::any_cast<vmap_t&>(ab);
        set_partition<typename vmap_t::unchecked_t>(b.get_unchecked());
    }

    size_t virtual_remove_size(size_t v)
    {
        return _overlap_stats.virtual_remove_size(v, _b[v]);
    }

    template <class MEntries>
    void get_move_entries(size_t v, size_t r, size_t nr, MEntries& m_entries)
    {
        auto mv_entries = [&](auto&&... args)
            {
                move_entries(v, r, nr, _b, _g, _eweight,
                             num_vertices(_bg), m_entries,
                             [](auto) {return false;},
                             is_loop_overlap(_overlap_stats), args...);
            };

        if (_rt == weight_type::NONE)
        {
            mv_entries();
        }
        else
        {
            if (_rt == weight_type::REAL_NORMAL)
                mv_entries(_rec, _drec);
            else
                mv_entries(_rec);
        }
    }

    // compute the entropy difference of a virtual move of vertex from block r to nr
    template <bool exact, class MEntries>
    double virtual_move_sparse(size_t v, size_t nr, bool multigraph,
                               MEntries& m_entries) const
    {
        size_t r = _b[v];

        if (r == nr)
            return 0.;

        size_t kout = out_degreeS()(v, _g);
        size_t kin = 0;
        if (graph_tool::is_directed(_g))
            kin = in_degreeS()(v, _g);

        double dS = entries_dS<exact>(m_entries, _mrs, _emat, _bg);

        int dwr = _wr[r] - _overlap_stats.virtual_remove_size(v, r, kin, kout);
        int dwnr = _overlap_stats.virtual_add_size(v, nr) - _wr[nr];

        if (multigraph)
            dS += _overlap_stats.virtual_move_parallel_dS(v, r, nr, _b, _g);

        if (!graph_tool::is_directed(_g))
            kin = kout;

        auto vt = [&](auto mrp, auto mrm, auto nr)
            {
                if (exact)
                    return vterm_exact(mrp, mrm, nr, _deg_corr, _bg);
                else
                    return vterm(mrp, mrm, nr, _deg_corr, _bg);
            };

        dS += vt(_mrp[r]  - kout, _mrm[r]  - kin, _wr[r]  - dwr );
        dS += vt(_mrp[nr] + kout, _mrm[nr] + kin, _wr[nr] + dwnr);
        dS -= vt(_mrp[r]        , _mrm[r]       , _wr[r]        );
        dS -= vt(_mrp[nr]       , _mrm[nr]      , _wr[nr]       );

        return dS;
    }

    template <bool exact>
    double virtual_move_sparse(size_t v, size_t nr, bool multigraph)
    {
        return virtual_move_sparse<exact>(v, nr, multigraph, _m_entries);
    }

    template <class MEntries>
    double virtual_move_dense(size_t, size_t, bool, MEntries&) const
    {
        throw GraphException("Dense entropy for overlapping model not implemented!");
    }

    double virtual_move_dense(size_t v, size_t nr, bool multigraph)
    {
        return virtual_move_dense(v, nr, multigraph, _m_entries);
    }

    template <class MEntries>
    double virtual_move(size_t v, size_t r, size_t nr, const entropy_args_t& ea,
                        MEntries& m_entries)
    {
        if (r == nr)
        {
            m_entries.set_move(r, nr, num_vertices(_bg));
            return 0;
        }

        if (!allow_move(r, nr))
            return std::numeric_limits<double>::infinity();

        get_move_entries(v, r, nr, m_entries);

        double dS = 0;
        if (ea.adjacency)
        {
            if (ea.exact)
                dS = virtual_move_sparse<true>(v, nr, ea.multigraph, m_entries);
            else
                dS = virtual_move_sparse<false>(v, nr, ea.multigraph, m_entries);

            if (_deg_corr && ea.deg_entropy)
                dS += _overlap_stats.virtual_move_deg_dS(v, r, nr, _g);
        }

        double dS_dl = 0;
        dS_dl += get_delta_partition_dl(v, r, nr, ea);
        if (ea.partition_dl || ea.degree_dl || ea.edges_dl)
        {
            enable_partition_stats();
            auto& ps = get_partition_stats(v);
            if (_deg_corr && ea.degree_dl)
                dS_dl += ps.get_delta_deg_dl(v, r, nr, _eweight, _g);
            if (ea.edges_dl)
            {
                size_t actual_B = 0;
                for (auto& ps : _partition_stats)
                    actual_B += ps.get_actual_B();
                dS_dl += ps.get_delta_edges_dl(v, r, nr, actual_B, _g);
            }
        }

        int dL = 0;
        if (ea.recs)
        {
            std::fill(_LdBdx.begin(), _LdBdx.end(), 0);
            auto rdS = rec_entries_dS(*this, m_entries, ea, _LdBdx, dL);
            dS += get<0>(rdS);
            dS_dl += get<1>(rdS);
        }

        if (_coupled_state != nullptr)
        {
            m_entries._p_entries.clear();

            if (_rt == weight_type::NONE)
            {
                std::vector<double> dummy;
                entries_op(m_entries, _emat,
                           [&](auto t, auto u, auto& me, auto delta)
                           {
                               if (delta == 0)
                                   return;
                               m_entries._p_entries.emplace_back(t, u, me,
                                                                 delta, dummy);
                           });
            }
            else
            {
                wentries_op(m_entries, _emat,
                            [&](auto t, auto u, auto& me, auto delta, auto& edelta)
                            {
                                m_entries._p_entries.emplace_back(t, u, me,
                                                                  delta,
                                                                  get<0>(edelta));
                            });
            }

            int dr = (_overlap_stats.virtual_remove_size(v, r) == 0) ? -1 : 0;
            int dnr = (_wr[nr] == 0) ? 1 : 0;
            if (!m_entries._p_entries.empty() || dr != 0 || dnr != 0)
                dS_dl += _coupled_state->propagate_entries_dS(r, nr, dr, dnr,
                                                              m_entries._p_entries,
                                                              _coupled_entropy_args,
                                                              _LdBdx, dL);
        }

        return dS + ea.beta_dl * dS_dl;
    }

    double virtual_move(size_t v, size_t r, size_t nr, const entropy_args_t& ea)
    {
        return virtual_move(v, r, nr, ea, _m_entries);
    }

    double get_delta_partition_dl(size_t v, size_t r, size_t nr,
                                  const entropy_args_t& ea)
    {
        if (r == nr)
            return 0;
        double dS = 0;

        if (ea.partition_dl)
        {
            enable_partition_stats();
            auto& ps = get_partition_stats(v);
            dS += ps.get_delta_partition_dl(v, r, nr, _g);
        }

        if (_coupled_state != nullptr)
        {
            bool r_vacate = (_overlap_stats.virtual_remove_size(v, r) == 0);
            bool nr_occupy = (_wr[nr] == 0);

            auto& bh = _coupled_state->get_b();
            if (r_vacate && nr_occupy)
            {
                dS += _coupled_state->get_delta_partition_dl(r, bh[r], bh[nr],
                                                             _coupled_entropy_args);
            }
            else
            {
                if (r_vacate)
                    dS += _coupled_state->get_delta_partition_dl(r, bh[r], null_group,
                                                                 _coupled_entropy_args);
                if (nr_occupy)
                    dS += _coupled_state->get_delta_partition_dl(nr, null_group, bh[nr],
                                                                 _coupled_entropy_args);
            }
        }
        return dS;
    }

    size_t get_empty_block(size_t v, bool force_add = true)
    {
        if (_empty_blocks.empty() || force_add)
        {
            add_block();
            auto s = _empty_blocks.back();
            auto r = _b[v];
            _bclabel[s] = _bclabel[r];
            if (_coupled_state != nullptr)
            {
                auto& hb = _coupled_state->get_b();
                hb[s] = hb[r];
            }
        }
        return _empty_blocks.back();
    }

    // Sample node placement
    template <class RNG>
    size_t sample_block(size_t v, double c, double d, RNG& rng)
    {
        // attempt new block
        std::bernoulli_distribution new_r(d);
        if (d > 0 && new_r(rng) && (_candidate_blocks.size() - 1 < num_vertices(_g)))
        {
            get_empty_block(v);
            auto s = uniform_sample(_empty_blocks, rng);
            auto r = _b[v];
            if (_coupled_state != nullptr)
                _coupled_state->sample_branch(s, r, rng);
            _bclabel[s] = _bclabel[r];
            return s;
        }

        // attempt random block
        size_t s = uniform_sample(_candidate_blocks.begin() + 1,
                                  _candidate_blocks.end(), rng);

        if (!std::isinf(c))
        {
            size_t w = get_lateral_half_edge(v, rng);

            size_t u = _overlap_stats.get_out_neighbor(w);
            if (u >= num_vertices(_g))
                u = _overlap_stats.get_in_neighbor(w);

            size_t t = _b[u];
            double p_rand = 0;
            if (c > 0)
            {
                size_t B = _candidate_blocks.size() - 1;
                if (graph_tool::is_directed(_g))
                    p_rand = c * B / double(_mrp[t] + _mrm[t] + c * B);
                else
                    p_rand = c * B / double(_mrp[t] + c * B);
            }

            typedef std::uniform_real_distribution<> rdist_t;
            if (c == 0 || rdist_t()(rng) >= p_rand)
            {
                if (_egroups.empty())
                    _egroups.init(_b, _eweight, _g, _bg);
                const auto& e = _egroups.sample_edge(t, rng);
                s = _b[target(e, _g)];
                if (s == t)
                    s = _b[source(e, _g)];
            }
        }

        return s;
    }

    size_t sample_block(size_t v, double c, double d, rng_t& rng)
    {
        return sample_block<rng_t>(v, c, d, rng);
    }

    void sample_branch(size_t, size_t, rng_t&)
    {
    }

    template <class RNG>
    size_t get_lateral_half_edge(size_t v, RNG& rng)
    {
        size_t vv = _overlap_stats.get_node(v);
        size_t w = _overlap_stats.sample_half_edge(vv, rng);
        return w;
    }

    template <class RNG>
    size_t random_neighbor(size_t v,  RNG& rng)
    {
        size_t w = get_lateral_half_edge(v, _overlap_stats, rng);

        size_t u = _overlap_stats.get_out_neighbor(w);
        if (u >= num_vertices(_g))
            u = _overlap_stats.get_in_neighbor(w);
        return u;
    }

    // Computes the move proposal probability
    template <class MEntries>
    double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                         bool reverse, MEntries& m_entries)
    {
        size_t B = _candidate_blocks.size() - 1;

        if (reverse)
        {
            if (_overlap_stats.virtual_remove_size(v, s) == 0)
                return d;
            if (_wr[r] == 0)
                B++;
        }
        else
        {
            if (_wr[s] == 0)
                return d;
        }

        if (B == num_vertices(_g))
            d = 0;

        if (std::isinf(c))
            return (1. - d) / B;


        typedef typename graph_traits<g_t>::vertex_descriptor vertex_t;
        double p = 0;
        size_t w = 0;

        size_t kout = out_degreeS()(v, _g, _eweight);
        size_t kin = kout;
        if (graph_tool::is_directed(_g))
            kin = in_degreeS()(v, _g, _eweight);

        size_t vi = _overlap_stats.get_node(v);
        auto& ns = _overlap_stats.get_half_edges(vi);

        for (size_t v: ns)
        {
            for (auto e : all_edges_range(v, _g))
            {
                vertex_t u = target(e, _g);
                if (graph_tool::is_directed(_g) && u == v)
                    u = source(e, _g);
                vertex_t t = _b[u];
                if (u == v)
                    t = r;
                size_t ew = _eweight[e];
                w += ew;

                int mts = 0;
                const auto& me = m_entries.get_me(t, s, _emat);
                if (me != _emat.get_null_edge())
                    mts = _mrs[me];
                int mtp = _mrp[t];
                int mst = mts;
                int mtm = mtp;

                if (graph_tool::is_directed(_g))
                {
                    mst = 0;
                    const auto& me = m_entries.get_me(s, t, _emat);
                    if (me != _emat.get_null_edge())
                        mst = _mrs[me];
                    mtm = _mrm[t];
                }

                if (reverse)
                {
                    int dts = m_entries.get_delta(t, s);
                    int dst = dts;
                    if (graph_tool::is_directed(_g))
                        dst = m_entries.get_delta(s, t);

                    mts += dts;
                    mst += dst;

                    if (t == s)
                    {
                        mtp -= kout;
                        mtm -= kin;
                    }

                    if (t == r)
                    {
                        mtp += kout;
                        mtm += kin;
                    }
                }

                if (graph_tool::is_directed(_g))
                {
                    p += ew * ((mts + mst + c) / (mtp + mtm + c * B));
                }
                else
                {
                    if (t == s)
                        mts *= 2;
                    p += ew * (mts + c) / (mtp + c * B);
                }
            }
        }
        if (w > 0)
            return (1. - d) * p / w;
        else
            return (1. - d) / B;
    }

    double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                         bool reverse)
    {
        return get_move_prob(v, r, s, c, d, reverse, _m_entries);
    }

    double get_move_prob(size_t, size_t, size_t, double, double,
                         bool,
                         std::vector<std::tuple<size_t, size_t, int>>&)
    {
        return 0;
    }

    bool is_last(size_t v)
    {
        auto r = _b[v];
        return _overlap_stats.virtual_remove_size(v, r) == 0;
    }

    size_t node_weight(size_t)
    {
        return 1;
    }

    double sparse_entropy(bool multigraph, bool deg_entropy, bool exact) const
    {
        double S = 0;
        if (exact)
        {
            for (auto e : edges_range(_bg))
                S += eterm_exact(source(e, _bg), target(e, _bg), _mrs[e], _bg);
            for (auto v : vertices_range(_bg))
                S += vterm_exact(_mrp[v], _mrm[v], _wr[v], _deg_corr, _bg);
        }
        else
        {
            for (auto e : edges_range(_bg))
                S += eterm(source(e, _bg), target(e, _bg), _mrs[e], _bg);
            for (auto v : vertices_range(_bg))
                S += vterm(_mrp[v], _mrm[v], _wr[v], _deg_corr, _bg);
        }

        if (_deg_corr && deg_entropy)
        {
            typedef gt_hash_map<int, int> map_t;

            map_t in_hist, out_hist;
            size_t N = _overlap_stats.get_N();

            for (size_t v = 0; v < N; ++v)
            {
                in_hist.clear();
                out_hist.clear();

                const auto& half_edges = _overlap_stats.get_half_edges(v);
                for (size_t u : half_edges)
                {
                    in_hist[_b[u]] += in_degreeS()(u, _g);
                    out_hist[_b[u]] += out_degree(u, _g);
                }

                for (auto& k_c : in_hist)
                    S -= lgamma_fast(k_c.second + 1);
                for (auto& k_c : out_hist)
                    S -= lgamma_fast(k_c.second + 1);
            }
        }

        if (multigraph)
            S += get_parallel_entropy();
        return S;
    }

    double dense_entropy(bool)
    {
        throw GraphException("Dense entropy for overlapping model not implemented!");
    }

    double entropy(const entropy_args_t& ea, bool propagate=false)
    {
        double S = 0, S_dl = 0;
        if (ea.adjacency)
        {
            if (ea.dense)
                S = dense_entropy(ea.multigraph);
            else
                S = sparse_entropy(ea.multigraph, ea.deg_entropy, ea.exact);

            if (!ea.dense && !ea.exact)
            {
                size_t E = 0;
                for (auto e : edges_range(_g))
                    E += _eweight[e];
                if (ea.multigraph)
                    S -= E;
                else
                    S += E;
            }
        }

        if (ea.partition_dl)
            S_dl += get_partition_dl();

        if (_deg_corr && ea.degree_dl)
            S_dl += get_deg_dl(ea.degree_dl_kind);

        if (ea.edges_dl)
        {
            enable_partition_stats();
            size_t actual_B = 0;
            for (auto& ps : _partition_stats)
                actual_B += ps.get_actual_B();
            S_dl += get_edges_dl(actual_B, _partition_stats.front().get_E(), _g);
        }

        if (ea.recs)
        {
            auto rdS = rec_entropy(*this, ea);
            S += get<0>(rdS);
            S_dl += get<1>(rdS);
        }

        if (_coupled_state != nullptr && propagate)
            S_dl += _coupled_state->entropy(_coupled_entropy_args, true);

        return S + S_dl * ea.beta_dl;
    }

    double get_partition_dl()
    {
        if (!is_partition_stats_enabled())
            enable_partition_stats();
        double S = 0;
        for (auto& ps : _partition_stats)
            S += ps.get_partition_dl();
        return S;
    }

    double get_deg_dl(int kind)
    {
        if (!is_partition_stats_enabled())
            enable_partition_stats();
        double S = 0;
        for (auto& ps : _partition_stats)
            S += ps.get_deg_dl(kind);
        return S;
    }

    double get_parallel_entropy() const
    {
        double S = 0;
        for (const auto& h : _overlap_stats.get_parallel_bundles())
        {
            for (const auto& kc : h)
            {
                bool is_loop = get<2>(kc.first);
                auto m = kc.second;
                if (is_loop)
                {
                    assert(m % 2 == 0);
                    S += lgamma_fast(m/2 + 1) + m * log(2) / 2;
                }
                else
                {
                    S += lgamma_fast(m + 1);
                }
            }
        }
        return S;
    }

    double edge_entropy_term(size_t, size_t, const entropy_args_t&)
    {
        return 0;
    }
    double propagate_entries_dS(size_t, size_t, int, int,
                                std::vector<std::tuple<size_t, size_t,
                                                       GraphInterface::edge_t, int,
                                                       std::vector<double>>>&,
                                const entropy_args_t&, std::vector<double>&, int)
    {
        return 0;
    }
    void propagate_delta(size_t, size_t,
                         std::vector<std::tuple<size_t, size_t,
                                                GraphInterface::edge_t, int,
                                                std::vector<double>>>&)
    {
    }

    void enable_partition_stats()
    {
        openmp_scoped_lock lock(_partition_lock);
        if (_partition_stats.empty())
        {

            size_t E = num_vertices(_g) / 2;
            size_t B = num_vertices(_bg);

            auto vi = std::max_element(vertices(_g).first, vertices(_g).second,
                                       [&](auto u, auto v)
                                       { return this->_pclabel[u] < this->_pclabel[v];});
            size_t C = _pclabel[*vi] + 1;

            vector<gt_hash_set<size_t>> vcs(C);
            vector<size_t> rc(num_vertices(_bg));
            for (auto v : vertices_range(_g))
            {
                vcs[_pclabel[v]].insert(_overlap_stats.get_node(v));
                rc[_b[v]] = _pclabel[v];
            }

            for (size_t c = 0; c < C; ++c)
                _partition_stats.emplace_back(_g, _b, vcs[c], E, B,
                                              _eweight, _overlap_stats,
                                              _bmap, _vmap);

            for (size_t r = 0; r < num_vertices(_bg); ++r)
                _partition_stats[rc[r]].get_r(r);
        }
    }

    void disable_partition_stats()
    {
        _partition_stats.clear();
    }

    bool is_partition_stats_enabled() const
    {
        return !_partition_stats.empty();
    }

    overlap_partition_stats_t& get_partition_stats(size_t v)
    {
        return _partition_stats[_pclabel[v]];
    }

    void couple_state(BlockStateVirtualBase& s, const entropy_args_t& ea)
    {
        _coupled_state = &s;
        _coupled_entropy_args = ea;
    }

    void decouple_state()
    {
        _coupled_state = nullptr;
    }

    void clear_egroups()
    {
        _egroups.clear();
    }

    void rebuild_neighbor_sampler()
    {
    }

    void sync_emat()
    {
        _emat.sync(_bg);
    }

    size_t get_N()
    {
        return _overlap_stats.get_N();
    }

    template <class Graph, class EMap>
    void get_be_overlap(Graph& g, EMap be)
    {
        for (auto ei : edges_range(_g))
        {
            auto u = source(ei, _g);
            auto v = target(ei, _g);

            auto s = vertex(_node_index[u], g);
            auto t = vertex(_node_index[v], g);

            for (auto e : out_edges_range(s, g))
            {
                if (!be[e].empty() || target(e, g) != t)
                    continue;
                if (graph_tool::is_directed(g) || s < target(e, g))
                    be[e] = {_b[u], _b[v]};
                else
                    be[e] = {_b[v], _b[u]};
                break;
            }

            if (graph_tool::is_directed(g))
            {
                for (auto e : in_edges_range(t, g))
                {
                    if (!be[e].empty() || source(e, g) != s)
                        continue;
                    be[e] = {_b[u], _b[v]};
                    break;
                }
            }
        }
    }

    template <class Graph, class VMap>
    void get_bv_overlap(Graph& g, VMap bv, VMap bc_in, VMap bc_out,
                        VMap bc_total)
    {
        typedef gt_hash_map<int, int> map_t;
        vector<map_t> hist_in;
        vector<map_t> hist_out;

        for (auto v : vertices_range(_g))
        {
            if (out_degree(v, _g) > 0)
            {
                size_t s = _node_index[v];
                if (s >= hist_out.size())
                    hist_out.resize(s + 1);
                hist_out[s][_b[v]]++;
            }

            if (in_degreeS()(v, _g) > 0)
            {
                size_t t = _node_index[v];
                if (t >= hist_in.size())
                    hist_in.resize(t + 1);
                hist_in[t][_b[v]]++;
            }
        }

        hist_in.resize(num_vertices(g));
        hist_out.resize(num_vertices(g));

        set<size_t> rs;
        for (auto i : vertices_range(g))
        {
            rs.clear();
            for (auto iter = hist_out[i].begin(); iter != hist_out[i].end(); ++iter)
                rs.insert(iter->first);
            for (auto iter = hist_in[i].begin(); iter != hist_in[i].end(); ++iter)
                rs.insert(iter->first);
            // if (rs.empty())
            //     throw GraphException("Cannot have empty overlapping block membership!");
            for (size_t r : rs)
            {
                bv[i].push_back(r);

                auto iter_in = hist_in[i].find(r);
                if (iter_in != hist_in[i].end())
                    bc_in[i].push_back(iter_in->second);
                else
                    bc_in[i].push_back(0);

                auto iter_out = hist_out[i].find(r);
                if (iter_out != hist_out[i].end())
                    bc_out[i].push_back(iter_out->second);
                else
                    bc_out[i].push_back(0);

                bc_total[i].push_back(bc_in[i].back() +
                                      bc_out[i].back());
            }
        }
    }

    template <class Graph, class VVProp, class VProp>
    void get_overlap_split(Graph& g, VVProp bv, VProp b) const
    {
        gt_hash_map<vector<int>, size_t> bvset;

        for (auto v : vertices_range(g))
        {
            auto r = bv[v];
            auto iter = bvset.find(r);
            if (iter == bvset.end())
                iter = bvset.insert(make_pair(r, bvset.size())).first;
            b[v] = iter->second;
        }
    }

    size_t add_block(size_t n = 1)
    {
        _wr.resize(num_vertices(_bg) + n);
        _mrm.resize(num_vertices(_bg) + n);
        _mrp.resize(num_vertices(_bg) + n);
        _bclabel.resize(num_vertices(_bg) + n);
        _empty_pos.resize(num_vertices(_bg) + n);
        _candidate_pos.resize(num_vertices(_bg) + n);
        size_t r = null_group;
        for (size_t i = 0; i < n; ++i)
        {
            r = boost::add_vertex(_bg);
            _wr[r] = _mrm[r] = _mrp[r] = 0;
            add_element(_empty_blocks, _empty_pos, r);
            _overlap_stats.add_block();
            for (auto& p : _partition_stats)
                p.add_block();
            if (!_egroups.empty())
                _egroups.add_block();
            if (_coupled_state != nullptr)
                _coupled_state->coupled_resize_vertex(r);
        }
        _emat.add_block(_bg);
        return r;
    }

    void add_edge(const GraphInterface::edge_t&)
    {
    }

    void remove_edge(const GraphInterface::edge_t&)
    {
    }

    void add_edge_rec(const GraphInterface::edge_t&)
    {
    }

    void remove_edge_rec(const GraphInterface::edge_t&)
    {
    }

    void update_edge_rec(const GraphInterface::edge_t&,
                         const std::vector<double>&)
    {
    }

    vprop_map_t<int32_t>::type::unchecked_t& get_b()
    {
        return _b;
    }

    vprop_map_t<int32_t>::type::unchecked_t& get_pclabel()
    {
        return _pclabel;
    }

    template <class MCMCState>
    void init_mcmc(MCMCState& state)
    {
        auto c = state._c;
        auto& entropy_args = state._entropy_args;
        bool dl = (entropy_args.partition_dl ||
                   entropy_args.degree_dl ||
                   entropy_args.edges_dl);
        if (!std::isinf(c))
        {
            if (_egroups.empty())
                _egroups.init(_b, _eweight, _g, _bg);
        }
        else
        {
            _egroups.clear();
        }

        if (dl)
            enable_partition_stats();
        else
            disable_partition_stats();
    }

    bool check_edge_counts(bool emat=true)
    {
        gt_hash_map<std::pair<size_t, size_t>, size_t> mrs;
        for (auto e : edges_range(_g))
        {
            size_t r = _b[source(e, _g)];
            size_t s = _b[target(e, _g)];
            if (!graph_tool::is_directed(_g) && s < r)
                std::swap(r, s);
            mrs[std::make_pair(r, s)] += _eweight[e];
        }

        for (auto& rs_m : mrs)
        {
            auto r = rs_m.first.first;
            auto s = rs_m.first.second;
            if (rs_m.second == 0)
                continue;
            typename graph_traits<bg_t>::edge_descriptor me;
            if (emat)
            {
                me = _emat.get_me(r, s);
                if (me == _emat.get_null_edge())
                {
                    assert(false);
                    return false;
                }
            }
            else
            {
                auto ret = boost::edge(r, s, _bg);
                assert(ret.second);
                if (!ret.second)
                    return false;
                me = ret.first;
            }
            if (size_t(_mrs[me]) != rs_m.second)
            {
                assert(false);
                return false;
            }
        }
        if (_coupled_state != nullptr)
            if (!_coupled_state->check_edge_counts(false))
                return false;
        return true;
    }

    void add_partition_node(size_t, size_t) { }
    void remove_partition_node(size_t, size_t) { }
    void set_vertex_weight(size_t, int) { }
    void coupled_resize_vertex(size_t) { }
    void update_block_edge(const GraphInterface::edge_t&,
                           const std::vector<double>&) { }
    template <class V>
    void push_state(V&) {}
    void pop_state() {}
    void store_next_state(size_t) {}
    void clear_next_state() {}

//private:
    typedef typename
        std::conditional<is_directed_::apply<g_t>::type::value,
                         GraphInterface::multigraph_t,
                         undirected_adaptor<GraphInterface::multigraph_t>>::type
        bg_t;
    bg_t& _bg;

    typename mrs_t::checked_t _c_mrs;
    std::vector<typename brec_t::value_type::checked_t> _c_brec;
    std::vector<typename brec_t::value_type::checked_t> _c_bdrec;
    std::vector<double> _recsum;
    std::vector<double> _recx2;
    std::vector<double> _dBdx;
    std::vector<double> _LdBdx;
    std::vector<double> _rdelta;
    size_t _B_E = 0;
    size_t _B_E_D = 0;
    int _rt = weight_type::NONE;

    typedef typename std::conditional<use_hash_t::value,
                                      EHash<bg_t>,
                                      EMat<bg_t>>::type
        emat_t;
    emat_t _emat;

    EGroups<g_t, mpl::false_> _egroups;
    bool _egroups_update;

    overlap_stats_t _overlap_stats;
    std::vector<overlap_partition_stats_t> _partition_stats;
    std::vector<size_t> _bmap;
    std::vector<size_t> _vmap;

    typedef SingleEntrySet<g_t, bg_t, std::vector<double>,
                           std::vector<double>> m_entries_t;
    m_entries_t _m_entries;

    std::vector<std::tuple<size_t, size_t, int, std::vector<double>>>
        _p_entries;

    UnityPropertyMap<int,GraphInterface::edge_t> _eweight;
    UnityPropertyMap<int,GraphInterface::vertex_t> _vweight;

    typedef entropy_args_t _entropy_args_t;
    BlockStateVirtualBase* _coupled_state;
    entropy_args_t _coupled_entropy_args;

    openmp_mutex _lock;
    openmp_mutex _partition_lock;
};

} // namespace graph_tool

#endif // GRAPH_BLOCKMODEL_OVERLAP_HH
