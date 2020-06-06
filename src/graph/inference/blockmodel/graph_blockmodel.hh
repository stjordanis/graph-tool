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

#ifndef GRAPH_BLOCKMODEL_HH
#define GRAPH_BLOCKMODEL_HH

#include "config.h"

#include <vector>

#ifdef __clang__
#include <boost/algorithm/minmax_element.hpp>
#endif

#include "../support/graph_state.hh"
#include "graph_blockmodel_util.hh"

#include "openmp_lock.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

typedef vprop_map_t<int32_t>::type vmap_t;
typedef eprop_map_t<int32_t>::type emap_t;
typedef UnityPropertyMap<int,GraphInterface::vertex_t> vcmap_t;
typedef UnityPropertyMap<int,GraphInterface::edge_t> ecmap_t;

template <class PMap>
auto uncheck(boost::any& amap, PMap*)
{
    return any_cast<typename PMap::checked_t&>(amap).get_unchecked();
}

template <class T, class V>
auto& uncheck(boost::any& amap, UnityPropertyMap<T,V>*)
{
    return any_cast<UnityPropertyMap<T,V>&>(amap);
}

inline simple_degs_t uncheck(boost::any& amap, simple_degs_t*)
{
    return any_cast<simple_degs_t>(amap);
}

typedef mpl::vector2<std::true_type, std::false_type> bool_tr;
typedef mpl::vector2<vcmap_t, vmap_t> vweight_tr;
typedef mpl::vector2<ecmap_t, emap_t> eweight_tr;

#ifdef GRAPH_BLOCKMODEL_RMAP_ENABLE
#    ifdef GRAPH_BLOCKMODEL_RMAP_ALL_ENABLE
typedef mpl::vector2<std::true_type, std::false_type> rmap_tr;
#    else
typedef mpl::vector1<std::true_type> rmap_tr;
#    endif
#else
typedef mpl::vector1<std::false_type> rmap_tr;
#endif

#define BLOCK_STATE_params                                                     \
    ((g, &, all_graph_views, 1))                                               \
    ((is_weighted,, bool_tr, 1))                                               \
    ((use_hash,, bool_tr, 1))                                                  \
    ((use_rmap,, rmap_tr, 1))                                                  \
    ((_abg, &, boost::any&, 0))                                                \
    ((_aeweight, &, boost::any&, 0))                                           \
    ((_avweight, &, boost::any&, 0))                                           \
    ((_adegs, &, boost::any&, 0))                                              \
    ((mrs,, emap_t, 0))                                                        \
    ((mrp,, vmap_t, 0))                                                        \
    ((mrm,, vmap_t, 0))                                                        \
    ((wr,, vmap_t, 0))                                                         \
    ((b,, vmap_t, 0))                                                          \
    ((empty_blocks, &, std::vector<size_t>&, 0))                               \
    ((empty_pos,, vmap_t, 0))                                                  \
    ((candidate_blocks, &, std::vector<size_t>&, 0))                           \
    ((candidate_pos,, vmap_t, 0))                                              \
    ((bclabel,, vmap_t, 0))                                                    \
    ((pclabel,, vmap_t, 0))                                                    \
    ((bfield,, vprop_map_t<std::vector<double>>::type, 0))                     \
    ((Bfield, &, std::vector<double>&, 0))                                     \
    ((merge_map,, vmap_t, 0))                                                  \
    ((deg_corr,, bool, 0))                                                     \
    ((rec_types,, std::vector<int32_t>, 0))                                    \
    ((rec,, std::vector<eprop_map_t<double>::type>, 0))                        \
    ((drec,, std::vector<eprop_map_t<double>::type>, 0))                       \
    ((brec,, std::vector<eprop_map_t<double>::type>, 0))                       \
    ((bdrec,, std::vector<eprop_map_t<double>::type>, 0))                      \
    ((brecsum,, vprop_map_t<double>::type, 0))                                 \
    ((wparams, &, std::vector<std::vector<double>>&, 0))                       \
    ((recdx, &, std::vector<double>&, 0))                                      \
    ((Lrecdx, &, std::vector<double>&, 0))                                     \
    ((epsilon, &, std::vector<double>&, 0))

GEN_STATE_BASE(BlockStateBase, BLOCK_STATE_params)

template <class... Ts>
class BlockState
    : public BlockStateBase<Ts...>, public BlockStateVirtualBase
{
public:
    GET_PARAMS_USING(BlockStateBase<Ts...>, BLOCK_STATE_params)
    GET_PARAMS_TYPEDEF(Ts, BLOCK_STATE_params)

    typedef partition_stats<use_rmap_t::value> partition_stats_t;

    template <class RNG, class... ATs,
              typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
    BlockState(RNG& rng, ATs&&... args)
        : BlockStateBase<Ts...>(std::forward<ATs>(args)...),
          _bg(boost::any_cast<std::reference_wrapper<bg_t>>(__abg)),
          _c_mrs(_mrs.get_checked()),
          _vweight(uncheck(__avweight, typename std::add_pointer<vweight_t>::type())),
          _eweight(uncheck(__aeweight, typename std::add_pointer<eweight_t>::type())),
          _degs(uncheck(__adegs, typename std::add_pointer<degs_t>::type())),
          _emat(_g, _bg, rng),
          _neighbor_sampler(_g, _eweight),
          _m_entries(num_vertices(_bg))
    {
        _empty_blocks.clear();
        _candidate_blocks.clear();
        _candidate_blocks.push_back(null_group);
        for (auto r : vertices_range(_bg))
        {
            if (_wr[r] == 0)
                add_element(_empty_blocks, _empty_pos, r);
            else
                add_element(_candidate_blocks, _candidate_pos, r);
        }
        for (auto& p : _rec)
            _c_rec.push_back(p.get_checked());
        for (auto& p : _drec)
            _c_drec.push_back(p.get_checked());
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
            _recx2.resize(_rec_types.size());
            _recdx.resize(_rec_types.size());
            for (auto me : edges_range(_bg))
            {
                if (_brec[0][me] > 0)
                {
                    _B_E++;
                    for (size_t i = 0; i < _rec_types.size(); ++i)
                    {
                        if (_rec_types[i] == weight_type::REAL_NORMAL)
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

        _N = 0;
        for (auto v : vertices_range(_g))
            _N += _vweight[v];
    }

    BlockState(const BlockState& other)
        : BlockStateBase<Ts...>(static_cast<const BlockStateBase<Ts...>&>(other)),
          _bg(boost::any_cast<std::reference_wrapper<bg_t>>(__abg)),
          _c_mrs(_mrs.get_checked()),
          _c_rec(other._c_rec),
          _c_drec(other._c_drec),
          _c_brec(other._c_brec),
          _c_bdrec(other._c_bdrec),
          _recsum(other._recsum),
          _recx2(other._recx2),
          _dBdx(other._dBdx),
          _LdBdx(other._LdBdx),
          _B_E(other._B_E),
          _B_E_D(other._B_E_D),
          _rt(other._rt),
          _N(other._N),
          _vweight(uncheck(__avweight, typename std::add_pointer<vweight_t>::type())),
          _eweight(uncheck(__aeweight, typename std::add_pointer<eweight_t>::type())),
          _degs(uncheck(__adegs, typename std::add_pointer<degs_t>::type())),
          _emat(other._emat),
          _egroups_update(other._egroups_update),
          _neighbor_sampler(other._neighbor_sampler),
          _m_entries(num_vertices(_bg))
    {
        if (other.is_partition_stats_enabled())
            enable_partition_stats();
    }

    // =========================================================================
    // State modification
    // =========================================================================

    template <class MEntries, class EFilt>
    void get_move_entries(size_t v, size_t r, size_t nr, MEntries& m_entries,
                          EFilt&& efilt)
    {
        auto mv_entries = [&](auto&&... args)
            {
                move_entries(v, r, nr, _b, _g, _eweight, num_vertices(_bg),
                             m_entries, std::forward<EFilt>(efilt),
                             is_loop_nop(),
                             std::forward<decltype(args)>(args)...);
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

    template <class MEntries>
    void get_move_entries(size_t v, size_t r, size_t nr, MEntries& m_entries)
    {
        get_move_entries(v, r, nr, m_entries, [](auto) {return false;});
    }


    template <bool Add, class EFilt>
    void modify_vertex(size_t v, size_t r, EFilt&& efilt)
    {
        if (Add)
            get_move_entries(v, null_group, r, _m_entries,
                             std::forward<EFilt>(efilt));
        else
            get_move_entries(v, r, null_group, _m_entries,
                             std::forward<EFilt>(efilt));

        apply_delta<Add,!Add>(*this, _m_entries);

        if (Add)
            BlockState::add_partition_node(v, r);
        else
            BlockState::remove_partition_node(v, r);
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

    template <class EFilt>
    void move_vertex(size_t v, size_t r, size_t nr, EFilt&& efilt)
    {
        if (r == nr)
            return;

        if (!allow_move(r, nr))
            throw ValueException("cannot move vertex across clabel barriers");

        get_move_entries(v, r, nr, _m_entries, std::forward<EFilt>(efilt));

        apply_delta<true, true>(*this, _m_entries);

        BlockState::remove_partition_node(v, r);
        BlockState::add_partition_node(v, nr);
    }

    // move a vertex from its current block to block nr
    void move_vertex(size_t v, size_t r, size_t nr)
    {
        move_vertex(v, r, nr, [](auto&) constexpr {return false;});
    }

    void move_vertex(size_t v, size_t nr)
    {
        size_t r = _b[v];
        move_vertex(v, r, nr);
    }

    void propagate_delta(size_t u, size_t v,
                         std::vector<std::tuple<size_t, size_t,
                                                GraphInterface::edge_t, int,
                                                std::vector<double>>>& entries)
    {
        size_t r = _b[u];
        size_t s = _b[v];
        _m_entries.set_move(r, s, num_vertices(_bg));

        if (_rt == weight_type::NONE)
        {
            for (auto& rsd : entries)
            {
                _m_entries.template insert_delta<true>(_b[get<0>(rsd)],
                                                       _b[get<1>(rsd)],
                                                       get<3>(rsd));
            }
        }
        else
        {
            for (auto& rsd : entries)
            {
                recs_propagate_insert(*this, _b[get<0>(rsd)], _b[get<1>(rsd)],
                                      get<2>(rsd), get<3>(rsd), get<4>(rsd),
                                      _m_entries);
            }
        }
        apply_delta<true, true>(*this, _m_entries);
    }

    void add_edge(const GraphInterface::edge_t& e)
    {
        size_t r = _b[source(e, _g)];
        size_t s = _b[target(e, _g)];
        auto me = _emat.get_me(r, s);
        if (me == _emat.get_null_edge())
        {
            me = boost::add_edge(r, s, _bg).first;
            _emat.put_me(r, s, me);
            _c_mrs[me] = 0;
            for (size_t i = 0; i < _rec_types.size(); ++i)
            {
                _c_brec[i][me] = 0;
                _c_bdrec[i][me] = 0;
            }
            if (_coupled_state != nullptr)
                _coupled_state->add_edge(me);
        }
    }

    void remove_edge(const GraphInterface::edge_t& e)
    {
        size_t r = _b[source(e, _g)];
        size_t s = _b[target(e, _g)];
        auto me = _emat.get_me(r, s);
        if (me != _emat.get_null_edge() && _mrs[me] == 0)
        {
            _emat.remove_me(me, _bg);
            if (_coupled_state != nullptr)
                _coupled_state->remove_edge(me);
        }
        assert(e != _emat.get_null_edge());
        boost::remove_edge(e, _g);
    }

    void add_edge_rec(const GraphInterface::edge_t& e)
    {
        if (_rec_types.empty())
            return;
        auto crec = _rec[0].get_checked();
        crec[e] = 1;
        for (size_t i = 1; i < _rec_types.size(); ++i)
        {
            auto drec = _drec[i].get_checked();
            drec[e] = 0;
        }
    }

    void remove_edge_rec(const GraphInterface::edge_t& e)
    {
        if (_rec_types.empty())
            return;
        _rec[0][e] = 0;
    }

    void update_edge_rec(const GraphInterface::edge_t& e,
                         const std::vector<double>& delta)
    {
        if (_rec_types.empty())
            return;

        for (size_t i = 0; i < _rec_types.size(); ++i)
        {
            if (_rec_types[i] != weight_type::REAL_NORMAL)
                continue;

            auto rec = _c_rec[i][e];
            auto d = (std::pow(rec, 2) -
                      std::pow(rec - delta[i], 2));
            _c_drec[i][e] += d;
        }
    }

    void remove_partition_node(size_t v, size_t r)
    {
        assert(size_t(_b[v]) == r);

        if (_vweight[v] > 0 && _wr[r] == _vweight[v])
        {
            remove_element(_candidate_blocks, _candidate_pos, r);
            add_element(_empty_blocks, _empty_pos, r);

            if (_coupled_state != nullptr)
            {
                auto& hb = _coupled_state->get_b();
                _coupled_state->remove_partition_node(r, hb[r]);
                _coupled_state->set_vertex_weight(r, 0);
            }
        }

        _wr[r] -= _vweight[v];

        if (!_egroups.empty() && _egroups_update)
            _egroups.remove_vertex(v, _b, _g);

        if (is_partition_stats_enabled())
            get_partition_stats(v).remove_vertex(v, r, _deg_corr, _g,
                                                 _vweight, _eweight,
                                                 _degs);
    }

    void add_partition_node(size_t v, size_t r)
    {
        _b[v] = r;

        _wr[r] += _vweight[v];

        if (!_egroups.empty() && _egroups_update)
            _egroups.add_vertex(v, _b, _eweight, _g);

        if (is_partition_stats_enabled())
            get_partition_stats(v).add_vertex(v, r, _deg_corr, _g, _vweight,
                                              _eweight, _degs);

        if (_vweight[v] > 0 && _wr[r] == _vweight[v])
        {
            remove_element(_empty_blocks, _empty_pos, r);
            add_element(_candidate_blocks, _candidate_pos, r);

            if (_coupled_state != nullptr)
            {
                auto& hb = _coupled_state->get_b();
                _coupled_state->set_vertex_weight(r, 1);
                _coupled_state->add_partition_node(r, hb[r]);
            }
        }
    }

    template <class EFilt>
    void remove_vertex(size_t v, size_t r, EFilt&& efilt)
    {
        modify_vertex<false>(v, r, std::forward<EFilt>(efilt));
    }

    void remove_vertex(size_t v, size_t r)
    {
        remove_vertex(v, r, [](auto&) { return false; });
    }

    void remove_vertex(size_t v)
    {
        size_t r = _b[v];
        remove_vertex(v, r);
    }

    template <class Vlist>
    void remove_vertices(Vlist& vs)
    {
        typedef typename graph_traits<g_t>::vertex_descriptor vertex_t;
        typedef typename graph_traits<g_t>::edge_descriptor edges_t;

        gt_hash_set<vertex_t> vset(vs.begin(), vs.end());
        gt_hash_set<edges_t> eset;

        for (auto v : vset)
        {
            for (auto e : all_edges_range(v, _g))
            {
                auto u = (source(e, _g) == v) ? target(e, _g) : source(e, _g);
                if (vset.find(u) != vset.end())
                    eset.insert(e);
            }
        }

        for (auto v : vset)
            remove_vertex(v, _b[v],
                          [&](auto& e) { return eset.find(e) != eset.end(); });

        for (auto& e : eset)
        {
            vertex_t v = source(e, _g);
            vertex_t u = target(e, _g);
            vertex_t r = _b[v];
            vertex_t s = _b[u];

            auto me = _emat.get_me(r, s);

            auto ew = _eweight[e];
            _mrs[me] -= ew;

            assert(_mrs[me] >= 0);

            _mrp[r] -= ew;
            _mrm[s] -= ew;

            for (size_t i = 0; i < _rec_types.size(); ++i)
            {
                switch (_rec_types[i])
                {
                case weight_type::REAL_NORMAL: // signed weights
                    _bdrec[i][me] -= _drec[i][e];
                    [[gnu::fallthrough]];
                default:
                    _brec[i][me] -= _rec[i][e];
                }
            }

            if (_mrs[me] == 0)
            {
                _emat.remove_me(me, _bg);
                if (_coupled_state != nullptr)
                    _coupled_state->remove_edge(me);
                else
                    boost::remove_edge(me, this->_bg);
            }
        }
    }

    void remove_vertices(python::object ovs)
    {
        multi_array_ref<uint64_t, 1> vs = get_array<uint64_t, 1>(ovs);
        remove_vertices(vs);
    }

    template <class EFilt>
    void add_vertex(size_t v, size_t r, EFilt&& efilt)
    {
        modify_vertex<true>(v, r, std::forward<EFilt>(efilt));
    }

    void add_vertex(size_t v, size_t r)
    {
        add_vertex(v, r, [](auto&){ return false; });
    }

    template <class Vlist, class Blist>
    void add_vertices(Vlist& vs, Blist& rs)
    {
        if (vs.size() != rs.size())
            throw ValueException("vertex and group lists do not have the same size");

        typedef typename graph_traits<g_t>::vertex_descriptor vertex_t;

        gt_hash_map<vertex_t, size_t> vset;
        for (size_t i = 0; i < vs.size(); ++i)
            vset[vs[i]] = rs[i];

        typedef typename graph_traits<g_t>::edge_descriptor edges_t;

        gt_hash_set<edges_t> eset;
        for (auto vr : vset)
        {
            auto v = vr.first;
            for (auto e : all_edges_range(v, _g))
            {
                auto u = (source(e, _g) == v) ? target(e, _g) : source(e, _g);
                if (vset.find(u) != vset.end())
                    eset.insert(e);
            }
        }

        for (auto vr : vset)
            add_vertex(vr.first, vr.second,
                       [&](auto& e){ return eset.find(e) != eset.end(); });

        for (auto e : eset)
        {
            vertex_t v = source(e, _g);
            vertex_t u = target(e, _g);
            vertex_t r = vset[v];
            vertex_t s = vset[u];

            auto me = _emat.get_me(r, s);

            if (me == _emat.get_null_edge())
            {
                me = boost::add_edge(r, s, _bg).first;
                _emat.put_me(r, s, me);
                _c_mrs[me] = 0;
                for (size_t i = 0; i < _rec_types.size(); ++i)
                {
                    _c_brec[i][me] = 0;
                    _c_bdrec[i][me] = 0;
                }

                if (_coupled_state != nullptr)
                    _coupled_state->add_edge(me);
            }

            assert(me == _emat.get_me(r, s));

            auto ew = _eweight[e];

            _mrs[me] += ew;
            _mrp[r] += ew;
            _mrm[s] += ew;

            for (size_t i = 0; i < _rec_types.size(); ++i)
            {
                switch (_rec_types[i])
                {
                case weight_type::REAL_NORMAL: // signed weights
                    _bdrec[i][me] += _drec[i][e];
                    [[gnu::fallthrough]];
                default:
                    _brec[i][me] += _rec[i][e];
                }
            }
        }
    }

    void add_vertices(python::object ovs, python::object ors)
    {
        multi_array_ref<uint64_t, 1> vs = get_array<uint64_t, 1>(ovs);
        multi_array_ref<uint64_t, 1> rs = get_array<uint64_t, 1>(ors);
        add_vertices(vs, rs);
    }

    template <bool Add, bool Deplete=true>
    void modify_edge(size_t u, size_t v, GraphInterface::edge_t& e,
                     const std::vector<double>& rec)
    {
        size_t r = _b[u];
        size_t s = _b[v];

        if (is_partition_stats_enabled())
        {
            get_partition_stats(u).remove_vertex(u, r, _deg_corr, _g,
                                                 _vweight, _eweight,
                                                 _degs);
            if (u != v)
                get_partition_stats(v).remove_vertex(v, s, _deg_corr, _g,
                                                     _vweight, _eweight,
                                                     _degs);
        }

        auto me = _emat.get_me(r, s);
        if (Add)
        {
            if (me == _emat.get_null_edge())
            {
                me = boost::add_edge(r, s, _bg).first;
                _emat.put_me(r, s, me);
                _c_mrs[me] = 0;
                for (size_t i = 0; i < _rec_types.size(); ++i)
                {
                    _c_brec[i][me] = 0;
                    _c_bdrec[i][me] = 0;
                }
            }

            if (_coupled_state == nullptr)
                _mrs[me]++;
            _mrp[r]++;
            _mrm[s]++;
        }
        else
        {
            assert(me != _emat.get_null_edge());
            if (_coupled_state == nullptr)
                _mrs[me]--;
            _mrp[r]--;
            _mrm[s]--;
        }

        // constexpr auto one = (Add) ? 1 : -1;
        // for (size_t i = 0; i < _rec_types.size(); ++i)
        // {
        //     switch (_rec_types[i])
        //     {
        //     case weight_type::REAL_NORMAL: // signed weights
        //         _bdrec[i][me] += one * std::pow(rec[i], 2);
        //         throw GraphException("Lrecdx, etc...");
        //         [[gnu::fallthrough]];
        //     default:
        //         _brec[i][me] += one * rec[i];
        //     }
        // }

        modify_edge<Add, Deplete>(u, v, e, _is_weighted);

        if (is_partition_stats_enabled())
        {
            get_partition_stats(u).add_vertex(u, r, _deg_corr, _g,
                                              _vweight, _eweight,
                                              _degs);
            if (u != v)
                get_partition_stats(v).add_vertex(v, s, _deg_corr, _g,
                                                  _vweight, _eweight,
                                                  _degs);
            get_partition_stats(u).change_E(Add ? 1 : -1); // FIXME: wrong for multiple partition stats
        }

        if (_coupled_state != nullptr)
        {
            if (Add)
                _coupled_state->add_edge(r, s, me, rec);
            else
                _coupled_state->remove_edge(r, s, me, rec);
        }
    }

    template <bool Add, bool Deplete>
    void modify_edge(size_t u, size_t v, GraphInterface::edge_t& e,
                     std::false_type)
    {
        if (Add)
        {
            e = boost::add_edge(u, v, _g).first;
        }
        else
        {
            if (Deplete)
            {
                boost::remove_edge(e, _g);
                e = GraphInterface::edge_t();
            }
        }
    }

    template <bool Add, bool Deplete>
    void modify_edge(size_t u, size_t v, GraphInterface::edge_t& e,
                     std::true_type)
    {
        if (Add)
        {
            if (e == GraphInterface::edge_t())
            {
                e = boost::add_edge(u, v, _g).first;
                auto c_eweight = _eweight.get_checked();
                c_eweight[e] = 1;
            }
            else
            {
                _eweight[e]++;
            }
            if (_deg_corr)
            {
                get<1>(_degs[u].front())++;
                if constexpr (is_directed_::apply<g_t>::type::value)
                    get<0>(_degs[v].front())++;
                else
                    get<1>(_degs[v].front())++;
            }
        }
        else
        {
            _eweight[e]--;
            if (_eweight[e] == 0 && Deplete)
            {
                boost::remove_edge(e, _g);
                e = GraphInterface::edge_t();
            }
            if (_deg_corr)
            {
                get<1>(_degs[u].front())--;
                if constexpr (is_directed_::apply<g_t>::type::value)
                    get<0>(_degs[v].front())--;
                else
                    get<1>(_degs[v].front())--;
            }
        }
    }

    void add_edge(size_t u, size_t v, GraphInterface::edge_t& e,
                  const std::vector<double>& rec)
    {
        modify_edge<true, false>(u, v, e, rec);
    }

    void remove_edge(size_t u, size_t v, GraphInterface::edge_t& e,
                     const std::vector<double>& rec)
    {
        modify_edge<false, false>(u, v, e, rec);
    }

    void set_vertex_weight(size_t v, int w)
    {
        set_vertex_weight(v, w, _vweight);
    }

    void set_vertex_weight(size_t, int, vcmap_t&)
    {
        throw ValueException("Cannot set the weight of an unweighted state");
    }

    template <class VMap>
    void set_vertex_weight(size_t v, int w, VMap&& vweight)
    {
        _N -= vweight[v];
        vweight[v] = w;
        _N += w;
    }

    void init_vertex_weight(size_t v)
    {
        init_vertex_weight(v, _vweight);
    }

    void init_vertex_weight(size_t, vcmap_t&)
    {
    }

    template <class VMap>
    void init_vertex_weight(size_t v, VMap&& vweight)
    {
        vweight.resize(num_vertices(_g));
        vweight[v] = 0;
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

    template <class VMap>
    void set_partition(VMap&& b)
    {
        for (auto v : vertices_range(_g))
        {
            size_t r = b[v];
            while (r >= num_vertices(_bg))
                add_block();
            move_vertex(v, r);
        }
    }

    void set_partition(boost::any& ab)
    {
        vmap_t& b = boost::any_cast<vmap_t&>(ab);
        set_partition<typename vmap_t::unchecked_t>(b.get_unchecked());
    }

    size_t virtual_remove_size(size_t v)
    {
        return _wr[_b[v]] - _vweight[v];
    }

    // merge vertex u into v
    void merge_vertices(size_t u, size_t v)
    {
        typedef typename graph_traits<g_t>::edge_descriptor edge_t;
        UnityPropertyMap<int, edge_t> dummy;
        merge_vertices(u, v, dummy);
    }

    template <class Emap>
    void merge_vertices(size_t u, size_t v, Emap&& ec)
    {
        merge_vertices(u, v, ec, _is_weighted);
    }

    template <class Emap>
    void merge_vertices(size_t, size_t, Emap&&, std::false_type)
    {
        throw ValueException("cannot merge vertices of unweighted graph");
    }

    template <class Emap>
    void merge_vertices(size_t u, size_t v, Emap&& ec, std::true_type)
    {
        if (u == v)
            return;

        auto eweight_c = _eweight.get_checked();
        std::vector<typename rec_t::value_type::checked_t> c_rec;
        std::vector<typename brec_t::value_type::checked_t> c_drec;
        for (auto& p : _rec)
            c_rec.push_back(p.get_checked());
        for (auto& p : _drec)
            c_drec.push_back(p.get_checked());

        typedef typename graph_traits<g_t>::vertex_descriptor vertex_t;
        typedef typename graph_traits<g_t>::edge_descriptor edge_t;

        gt_hash_map<std::tuple<vertex_t, int>, vector<edge_t>> ns_u, ns_v;
        for(auto e : out_edges_range(u, _g))
            ns_u[std::make_tuple(target(e, _g), ec[e])].push_back(e);
        for(auto e : out_edges_range(v, _g))
            ns_v[std::make_tuple(target(e, _g), ec[e])].push_back(e);

        size_t nrec = this->_rec_types.size();
        std::vector<double> ecc(nrec), decc(nrec);

        for(auto& kv : ns_u)
        {
            vertex_t t = get<0>(kv.first);
            int l = get<1>(kv.first);
            auto& es = kv.second;

            size_t w = 0;

            std::fill(ecc.begin(), ecc.end(), 0);
            std::fill(decc.begin(), decc.end(), 0);
            for (auto& e : es)
            {
                w += _eweight[e];
                for (size_t i = 0; i < nrec; ++i)
                {
                    ecc[i] += _rec[i][e];
                    decc[i] += _drec[i][e];
                }
            }

            if (t == u)
            {
                t = v;
                if constexpr (!is_directed_::apply<g_t>::type::value)
                {
                    assert(w % 2 == 0);
                    w /= 2;
                    for (size_t i = 0; i < nrec; ++i)
                    {
                        ecc[i] /= 2;
                        decc[i] /= 2;
                    }
                }
            }

            auto iter = ns_v.find(std::make_tuple(t, l));
            if (iter != ns_v.end())
            {
                auto& e = iter->second.front();
                _eweight[e] += w;
                for (size_t i = 0; i < nrec; ++i)
                {
                    _rec[i][e] += ecc[i];
                    _drec[i][e] += decc[i];
                }
                assert(ec[e] == l);
            }
            else
            {
                auto e = boost::add_edge(v, t, _g).first;
                ns_v[std::make_tuple(t, l)].push_back(e);
                eweight_c[e] = w;
                for (size_t i = 0; i < nrec; ++i)
                {
                    c_rec[i][e] = ecc[i];
                    c_drec[i][e] = decc[i];
                }
                set_prop(ec, e, l);
                assert(ec[e] == l);
            }
        }

        if constexpr (is_directed_::apply<g_t>::type::value)
        {
            ns_u.clear();
            ns_v.clear();

            for(auto e : in_edges_range(v, _g))
                ns_v[std::make_tuple(source(e, _g), ec[e])].push_back(e);
            for(auto e : in_edges_range(u, _g))
                ns_u[std::make_tuple(source(e, _g), ec[e])].push_back(e);

            for(auto& kv : ns_u)
            {
                vertex_t s = get<0>(kv.first);
                int l = get<1>(kv.first);
                auto& es = kv.second;

                if (s == u)
                    continue;

                size_t w = 0;
                std::fill(ecc.begin(), ecc.end(), 0);
                std::fill(decc.begin(), decc.end(), 0);
                for (auto& e : es)
                {
                    w += _eweight[e];
                    for (size_t i = 0; i < nrec; ++i)
                    {
                        ecc[i] += _rec[i][e];
                        decc[i] += _drec[i][e];
                    }
                }

                auto iter = ns_v.find(std::make_tuple(s, l));
                if (iter != ns_v.end())
                {
                    auto& e = iter->second.front();
                    _eweight[e] += w;
                    for (size_t i = 0; i < nrec; ++i)
                    {
                        _rec[i][e] += ecc[i];
                        _drec[i][e] += decc[i];
                    }
                    assert(ec[e] == l);
                }
                else
                {
                    auto e = boost::add_edge(s, v, _g).first;
                    ns_v[std::make_tuple(s, l)].push_back(e);
                    eweight_c[e] = w;
                    for (size_t i = 0; i < nrec; ++i)
                    {
                        c_rec[i][e] = ecc[i];
                        c_drec[i][e] = decc[i];
                    }
                    set_prop(ec, e, l);
                    assert(ec[e] == l);
                }
            }
        }

        _vweight[v] +=_vweight[u];
        _vweight[u] = 0;
        for (auto e : all_edges_range(u, _g))
        {
            _eweight[e] = 0;
            set_prop(ec, e, 0);
            for (size_t i = 0; i < nrec; ++i)
            {
                _rec[i][e] = 0;
                _drec[i][e] = 0;
            }
        }
        clear_vertex(u, _g);
        _merge_map[u] = v;
        merge_degs(u, v, _degs);
    }

    template <class EMap, class Edge, class Val>
    void set_prop(EMap& ec, Edge& e, Val&& val)
    {
        ec[e] = val;
    }

    template <class Edge, class Val>
    void set_prop(UnityPropertyMap<typename std::remove_reference<Val>::type, Edge>&,
                  Edge&, Val&&)
    {
    }

    void merge_degs(size_t, size_t, const simple_degs_t&) {}

    void merge_degs(size_t u, size_t v, typename degs_map_t::unchecked_t& degs)
    {
        gt_hash_map<std::tuple<size_t, size_t>, size_t> hist;
        for (auto& kn : degs[u])
            hist[make_tuple(get<0>(kn), get<1>(kn))] += get<2>(kn);
        for (auto& kn : degs[v])
            hist[make_tuple(get<0>(kn), get<1>(kn))] += get<2>(kn);
        degs[u].clear();
        degs[v].clear();
        auto& d = degs[v];
        for (auto& kn : hist)
            d.emplace_back(get<0>(kn.first), get<1>(kn.first), kn.second);
    }

    size_t add_block(size_t n = 1)
    {
        _wr.resize(num_vertices(_bg) + n);
        _mrm.resize(num_vertices(_bg) + n);
        _mrp.resize(num_vertices(_bg) + n);
        _bclabel.resize(num_vertices(_bg) + n);
        _brecsum.resize(num_vertices(_bg) + n);
        _empty_pos.resize(num_vertices(_bg) + n);
        _candidate_pos.resize(num_vertices(_bg) + n);
        size_t r = null_group;
        for (size_t i = 0; i < n; ++i)
        {
            r = boost::add_vertex(_bg);
            _wr[r] = _mrm[r] = _mrp[r] = 0;
            add_element(_empty_blocks, _empty_pos, r);
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

    void coupled_resize_vertex(size_t v)
    {
        _b.resize(num_vertices(_g));
        _bfield.resize(num_vertices(_g));
        init_vertex_weight(v);
        _pclabel.resize(num_vertices(_g));
        resize_degs(_degs);
        _neighbor_sampler.resize(num_vertices(_g));
    }

    void resize_degs(const simple_degs_t&) {}

    void resize_degs(typename degs_map_t::unchecked_t& degs)
    {
        degs.resize(num_vertices(_g));
    }

    // =========================================================================
    // Virtual state modification
    // =========================================================================

    // compute the entropy difference of a virtual move of vertex from block r
    // to nr
    template <bool exact, class MEntries>
    double virtual_move_sparse(size_t v, size_t r, size_t nr,
                               MEntries& m_entries)
    {
        if (r == nr)
            return 0.;

        double dS = entries_dS<exact>(m_entries, _mrs, _emat, _bg);

        size_t kout = out_degreeS()(v, _g, _eweight);
        size_t kin = kout;
        if constexpr (is_directed_::apply<g_t>::type::value)
            kin = in_degreeS()(v, _g, _eweight);

        int dwr = _vweight[v];
        int dwnr = dwr;

        if (r == null_group && dwnr == 0)
            dwnr = 1;

        auto vt = [&](auto mrp, auto mrm, auto nr)
            {
                assert(mrp >= 0 && mrm >=0 && nr >= 0);
                if constexpr (exact)
                    return vterm_exact(mrp, mrm, nr, _deg_corr, _bg);
                else
                    return vterm(mrp, mrm, nr, _deg_corr, _bg);
            };

        if (r != null_group)
        {
            auto mrp_r = _mrp[r];
            auto mrm_r = _mrm[r];
            auto wr_r = _wr[r];
            dS += vt(mrp_r - kout, mrm_r - kin, wr_r - dwr);
            dS -= vt(mrp_r       , mrm_r      , wr_r      );
        }

        if (nr != null_group)
        {
            auto mrp_nr = _mrp[nr];
            auto mrm_nr = _mrm[nr];
            auto wr_nr = _wr[nr];
            dS += vt(mrp_nr + kout, mrm_nr + kin, wr_nr + dwnr);
            dS -= vt(mrp_nr       , mrm_nr      , wr_nr       );
        }

        return dS;
    }

    template <bool exact>
    double virtual_move_sparse(size_t v, size_t r, size_t nr)
    {
        return virtual_move_sparse<exact>(v, r, nr);
    }

    double virtual_move_dense(size_t v, size_t r, size_t nr, bool multigraph)
    {
        if (_deg_corr)
            throw GraphException("Dense entropy for degree corrected model not implemented!");

        typedef typename graph_traits<g_t>::vertex_descriptor vertex_t;

        if (r == nr)
            return 0;

        vector<int> deltap(num_vertices(_bg), 0);
        int deltal = 0;
        for (auto e : out_edges_range(v, _g))
        {
            vertex_t u = target(e, _g);
            vertex_t s = _b[u];
            if (u == v)
                deltal += _eweight[e];
            else
                deltap[s] += _eweight[e];
        }
        if constexpr (!is_directed_::apply<g_t>::type::value)
            deltal /= 2;

        vector<int> deltam(num_vertices(_bg), 0);
        if (is_directed_::apply<g_t>::type::value)
        {
            for (auto e : in_edges_range(v, _g))
            {
                vertex_t u = source(e, _g);
                if (u == v)
                    continue;
                vertex_t s = _b[u];
                deltam[s] += _eweight[e];
            }
        }

        double dS = 0;
        int dwr = _vweight[v];
        int dwnr = dwr;

        if (r == null_group && dwnr == 0)
            dwnr = 1;

        if (nr == null_group)
        {
            std::fill(deltap.begin(), deltap.end(), 0);
            std::fill(deltam.begin(), deltam.end(), 0);
            if (dwr != _wr[r])
                deltal = 0;
        }

        double Si = 0, Sf = 0;
        for (vertex_t s = 0; s < num_vertices(_bg); ++s)
        {
            if (_wr[s] == 0 && s != r && s != nr)
                continue;

            int ers = (r != null_group) ? get_beprop(r, s, _mrs, _emat) : 0;
            int enrs = (nr != null_group) ? get_beprop(nr, s, _mrs, _emat) : 0;

            if (!is_directed_::apply<g_t>::type::value)
            {
                if (s != nr && s != r)
                {
                    if (r != null_group)
                    {
                        Si += eterm_dense(r,  s, ers,              _wr[r],         _wr[s], multigraph, _bg);
                        Sf += eterm_dense(r,  s, ers - deltap[s],  _wr[r] - dwr,   _wr[s], multigraph, _bg);
                    }

                    if (nr != null_group)
                    {
                        Si += eterm_dense(nr, s, enrs,             _wr[nr],        _wr[s], multigraph, _bg);
                        Sf += eterm_dense(nr, s, enrs + deltap[s], _wr[nr] + dwnr, _wr[s], multigraph, _bg);
                    }
                }

                if (s == r)
                {
                    Si += eterm_dense(r, r, ers,                      _wr[r],       _wr[r],       multigraph, _bg);
                    Sf += eterm_dense(r, r, ers - deltap[r] - deltal, _wr[r] - dwr, _wr[r] - dwr, multigraph, _bg);
                }

                if (s == nr)
                {
                    Si += eterm_dense(nr, nr, enrs,                       _wr[nr],        _wr[nr],        multigraph, _bg);
                    Sf += eterm_dense(nr, nr, enrs + deltap[nr] + deltal, _wr[nr] + dwnr, _wr[nr] + dwnr, multigraph, _bg);

                    if (r != null_group)
                    {
                        Si += eterm_dense(r, nr, ers,                          _wr[r],       _wr[nr],        multigraph, _bg);
                        Sf += eterm_dense(r, nr, ers - deltap[nr] + deltap[r], _wr[r] - dwr, _wr[nr] + dwnr, multigraph, _bg);
                    }
                }
            }
            else
            {
                int esr = (r != null_group) ? get_beprop(s, r, _mrs, _emat) : 0;
                int esnr  = (nr != null_group) ? get_beprop(s, nr, _mrs, _emat) : 0;

                if (s != nr && s != r)
                {
                    if (r != null_group)
                    {
                        Si += eterm_dense(r, s, ers            , _wr[r]      , _wr[s]      , multigraph, _bg);
                        Sf += eterm_dense(r, s, ers - deltap[s], _wr[r] - dwr, _wr[s]      , multigraph, _bg);
                        Si += eterm_dense(s, r, esr            , _wr[s]      , _wr[r]      , multigraph, _bg);
                        Sf += eterm_dense(s, r, esr - deltam[s], _wr[s]      , _wr[r] - dwr, multigraph, _bg);
                    }

                    if (nr != null_group)
                    {
                        Si += eterm_dense(nr, s, enrs            , _wr[nr]       , _wr[s]        , multigraph, _bg);
                        Sf += eterm_dense(nr, s, enrs + deltap[s], _wr[nr] + dwnr, _wr[s]        , multigraph, _bg);
                        Si += eterm_dense(s, nr, esnr            , _wr[s]        , _wr[nr]       , multigraph, _bg);
                        Sf += eterm_dense(s, nr, esnr + deltam[s], _wr[s]        , _wr[nr] + dwnr, multigraph, _bg);
                    }
                }

                if(s == r)
                {
                    Si += eterm_dense(r, r, ers                                  , _wr[r]      , _wr[r]      , multigraph, _bg);
                    Sf += eterm_dense(r, r, ers - deltap[r]  - deltam[r] - deltal, _wr[r] - dwr, _wr[r] - dwr, multigraph, _bg);

                    if (nr != null_group)
                    {
                        Si += eterm_dense(r, nr, esnr                         , _wr[r]      , _wr[nr]       , multigraph, _bg);
                        Sf += eterm_dense(r, nr, esnr - deltap[nr] + deltam[r], _wr[r] - dwr, _wr[nr] + dwnr, multigraph, _bg);
                    }
                }

                if(s == nr)
                {
                    Si += eterm_dense(nr, nr, esnr                                   , _wr[nr]       , _wr[nr]       , multigraph, _bg);
                    Sf += eterm_dense(nr, nr, esnr + deltap[nr] + deltam[nr] + deltal, _wr[nr] + dwnr, _wr[nr] + dwnr, multigraph, _bg);

                    if (r != null_group)
                    {
                        Si += eterm_dense(nr, r, esr                         , _wr[nr]       , _wr[r]      , multigraph, _bg);
                        Sf += eterm_dense(nr, r, esr + deltap[r] - deltam[nr], _wr[nr] + dwnr, _wr[r] - dwr, multigraph, _bg);
                    }
                }
            }
        }

        return Sf - Si + dS;
    }


    template <class MEntries>
    [[gnu::hot]]
    double virtual_move(size_t v, size_t r, size_t nr, const entropy_args_t& ea,
                        MEntries& m_entries)
    {
        assert(size_t(_b[v]) == r || r == null_group);

        if (r != null_group && nr != null_group && !allow_move(r, nr))
            return std::numeric_limits<double>::infinity();

        get_move_entries(v, r, nr, m_entries, [](auto) constexpr { return false; });

        if (r == nr || _vweight[v] == 0)
            return 0;

        double dS = 0;
        if (ea.adjacency)
        {
            if (ea.dense)
            {
                dS = virtual_move_dense(v, r, nr, ea.multigraph);
            }
            else
            {
                if (ea.exact)
                    dS = virtual_move_sparse<true>(v, r, nr, m_entries);
                else
                    dS = virtual_move_sparse<false>(v, r, nr, m_entries);
            }
        }

        double dS_dl = 0;

        dS_dl += get_delta_partition_dl(v, r, nr, ea);

        if (ea.degree_dl || ea.edges_dl)
        {
            enable_partition_stats();
            auto& ps = get_partition_stats(v);
            if (_deg_corr && ea.degree_dl)
                dS_dl += ps.get_delta_deg_dl(v, r, nr, _vweight, _eweight,
                                             _degs, _g, ea.degree_dl_kind);
            if (ea.edges_dl)
            {
                size_t actual_B = 0;
                for (auto& ps : _partition_stats)
                    actual_B += ps.get_actual_B();
                dS_dl += ps.get_delta_edges_dl(v, r, nr, _vweight, actual_B,
                                               _g);
            }
        }

        if (!_Bfield.empty() && ea.Bfield)
        {
            int dB = 0;
            if (virtual_remove_size(v) == 0)
                dB--;
            if (_wr[nr] == 0)
                dB++;
            if (dB != 0)
            {
                size_t actual_B = 0;
                for (auto& ps : _partition_stats)
                    actual_B += ps.get_actual_B();
                dS_dl += (actual_B < _Bfield.size()) ?
                    _Bfield[actual_B] : _Bfield.back();
                actual_B += dB;
                dS_dl -= (actual_B < _Bfield.size()) ?
                    _Bfield[actual_B] : _Bfield.back();
            }
        }

        int dL = 0;
        if (ea.recs && _rt != weight_type::NONE)
        {
            std::fill(_LdBdx.begin(), _LdBdx.end(), 0);
            auto rdS = rec_entries_dS(*this, m_entries, ea, _LdBdx, dL);
            dS += get<0>(rdS);
            dS_dl += get<1>(rdS);
        }

        if (_coupled_state != nullptr && _vweight[v] > 0)
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

            int dr = (_wr[r] == _vweight[v] && _vweight[v] > 0) ? -1 : 0;
            int dnr = (_wr[nr] == 0 && _vweight[v] > 0) ? 1 : 0;
            if (!m_entries._p_entries.empty() || dr != 0 || dnr != 0)
                dS_dl += _coupled_state->propagate_entries_dS(r, nr, dr, dnr,
                                                              m_entries._p_entries,
                                                              _coupled_entropy_args,
                                                              _LdBdx, dL);
        }
        return dS + ea.beta_dl * dS_dl;
    }

    double propagate_entries_dS(size_t u, size_t v, int du, int dv,
                                std::vector<std::tuple<size_t, size_t, GraphInterface::edge_t, int,
                                                       std::vector<double>>>& entries,
                                const entropy_args_t& ea, std::vector<double>& dBdx,
                                int dL)
    {
        openmp_scoped_lock lock(_lock);

        size_t r = _b[u];
        size_t s = _b[v];

        if (u == v)
        {
            if (ea.recs && _rt == weight_type::REAL_NORMAL)
            {
                _m_entries.set_move(r, s, num_vertices(_bg));
                auto rdS = rec_entries_dS(*this, _m_entries, ea, dBdx, dL);
                double dS = get<0>(rdS) + get<1>(rdS);
                entries.clear();
                if (_coupled_state != nullptr)
                    dS += _coupled_state->propagate_entries_dS(r, s, 0, 0,
                                                               entries,
                                                               _coupled_entropy_args,
                                                               dBdx, dL);
                return dS;
            }
            return 0.;
        }

        _m_entries.set_move(r, s, num_vertices(_bg));

        auto comp =
            [&](auto&... dummy)
            {
                if (du != 0)
                {
                    for (auto t : out_neighbors_range(r, _bg))
                        _m_entries.template insert_delta<true>(r, t, 0, dummy...);
                    for (auto t : in_neighbors_range(r, _bg))
                        _m_entries.template insert_delta<true>(t, r, 0, dummy...);
                }

                if (dv != 0)
                {
                    for (auto t : out_neighbors_range(s, _bg))
                        _m_entries.template insert_delta<true>(s, t, 0, dummy...);
                    for (auto t : in_neighbors_range(s, _bg))
                        _m_entries.template insert_delta<true>(t, s, 0, dummy...);
                }
            };

        std::vector<double> dummy;
        if (!ea.recs || _rt == weight_type::NONE)
        {
            for (auto& iter : entries)
                _m_entries.template insert_delta<true>(_b[get<0>(iter)],
                                                       _b[get<1>(iter)],
                                                       get<3>(iter));
            comp();
        }
        else
        {
            for (auto& iter : entries)
                recs_propagate_insert(*this, _b[get<0>(iter)], _b[get<1>(iter)],
                                      get<2>(iter), get<3>(iter), get<4>(iter),
                                      _m_entries);
            dummy.resize(_rec.size(), 0.);
            comp(dummy, dummy);
        }

        double dS = 0;

        entries.clear();

        auto e_diff =
            [&](auto rr, auto ss, auto& me, auto d)
            {
                int ers = 0;
                if (me != _emat.get_null_edge())
                    ers = this->_mrs[me];
                auto wr = this->_wr[rr];
                auto ws = this->_wr[ss];

                dS -= eterm_dense(rr, ss, ers, wr, ws, true, _bg);

                if (rr == r)
                    wr += du;
                if (rr == s)
                    wr += dv;

                if (ss == r)
                    ws += du;
                if (ss == s)
                    ws += dv;

                dS += eterm_dense(rr, ss, ers + d, wr, ws, true, _bg);
            };

        if (!ea.recs || _rt == weight_type::NONE)
        {
            if (ea.adjacency)
            {
                entries_op(_m_entries, _emat,
                           [&](auto rr, auto ss, auto& me, auto d)
                           {
                               e_diff(rr, ss, me, d);
                               if (d == 0)
                                   return;
                               entries.emplace_back(rr, ss, me, d, dummy);
                           });
            }
            else
            {
                entries_op(_m_entries, _emat,
                           [&](auto rr, auto ss, auto& me, auto d)
                           {
                               if (d == 0)
                                   return;
                               entries.emplace_back(rr, ss, me, d, dummy);
                           });
            }
        }
        else
        {
            if (ea.adjacency)
            {
                wentries_op(_m_entries, _emat,
                            [&](auto rr, auto ss, auto& me, auto d, auto& ed)
                            {
                                e_diff(rr, ss, me, d);
                                entries.emplace_back(rr, ss, me, d, get<0>(ed));
                            });
            }
            else
            {
                wentries_op(_m_entries, _emat,
                            [&](auto rr, auto ss, auto& me, auto d, auto& ed)
                            {
                                entries.emplace_back(rr, ss, me, d, get<0>(ed));
                            });
            }

            auto rdS = rec_entries_dS(*this, _m_entries, ea, dBdx, dL);
            dS += get<0>(rdS) + get<1>(rdS);
        }

        int dr = (_wr[r] + du == 0) ? -1 : 0;
        int ds = (_wr[s] == 0) ? 1 : 0;
        if (_coupled_state != nullptr)
        {
            dS += _coupled_state->propagate_entries_dS(r, s, dr, ds,
                                                       entries,
                                                       _coupled_entropy_args,
                                                       dBdx, dL);
        }
        else
        {
            if (r != s && dr + ds != 0 && ea.edges_dl)
            {
                size_t actual_B = 0;
                enable_partition_stats();
                for (auto& ps : _partition_stats)
                    actual_B += ps.get_actual_B();
                size_t E = _partition_stats.front().get_E();
                dS -= get_edges_dl(actual_B, E, _g);
                dS += get_edges_dl(actual_B + dr + ds, E, _g);
            }
        }
        return dS;
    }

    [[gnu::hot]]
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

        auto& f = _bfield[v];
        if (!f.empty())
        {
            if (nr != null_group)
                dS -= (nr < f.size()) ? f[nr] : f.back();
            if (r != null_group)
                dS += (r < f.size()) ? f[r] : f.back();
        }

        if (ea.partition_dl)
        {
            enable_partition_stats();
            auto& ps = get_partition_stats(v);
            dS += ps.get_delta_partition_dl(v, r, nr, _vweight);
        }

        if (_coupled_state != nullptr)
        {
            bool r_vacate = (r != null_group && _wr[r] == _vweight[v]);
            bool nr_occupy = (nr != null_group && _wr[nr] == 0);

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

    // =========================================================================
    // Move proposals
    // =========================================================================

    size_t get_empty_block(size_t v, bool force_add = false)
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

    void sample_branch(size_t v, size_t u, rng_t& rng)
    {
        size_t s;
        auto r = _b[u];

        std::bernoulli_distribution new_r(1./_candidate_blocks.size());
        if (_candidate_blocks.size() <= num_vertices(_g) && new_r(rng))
        {
            get_empty_block(v);
            s = uniform_sample(_empty_blocks, rng);
            if (_coupled_state != nullptr)
            {
                _coupled_state->sample_branch(s, r, rng);
                auto& hpclabel = _coupled_state->get_pclabel();
                hpclabel[s] = _pclabel[u];
            }
            _bclabel[s] = _bclabel[r];
        }
        else
        {
            s = uniform_sample(_candidate_blocks.begin() + 1,
                               _candidate_blocks.end(), rng);
        }
        _b[v] = s;
    }

    // Sample node placement
    size_t sample_block(size_t v, double c, double d, rng_t& rng)
    {
        size_t B = _candidate_blocks.size() - 1;

        // attempt new block
        std::bernoulli_distribution new_r(d);
        if (d > 0 && B < _N && new_r(rng))
        {
            get_empty_block(v);
            auto s = uniform_sample(_empty_blocks, rng);
            auto r = _b[v];
            if (_coupled_state != nullptr)
            {
                _coupled_state->sample_branch(s, r, rng);
                auto& hpclabel = _coupled_state->get_pclabel();
                hpclabel[s] = _pclabel[v];
            }
            _bclabel[s] = _bclabel[r];
            return s;
        }

        size_t s;
        if (!std::isinf(c) && !_neighbor_sampler.empty(v))
        {
            auto u = _neighbor_sampler.sample(v, rng);
            size_t t = _b[u];
            double p_rand = 0;
            if (c > 0)
            {
                if (is_directed_::apply<g_t>::type::value)
                    p_rand = c * B / double(_mrp[t] + _mrm[t] + c * B);
                else
                    p_rand = c * B / double(_mrp[t] + c * B);
            }

            std::uniform_real_distribution<> rdist;
            if (c == 0 || rdist(rng) >= p_rand)
            {
                if (_egroups.empty())
                    _egroups.init(_b, _eweight, _g, _bg);
                const auto& e = _egroups.sample_edge(t, rng);
                s = _b[target(e, _g)];
                if (s == t)
                    s = _b[source(e, _g)];
                else
                    assert(size_t(_b[source(e, _g)]) == t);
            }
            else
            {
                s = uniform_sample(_candidate_blocks.begin() + 1,
                                   _candidate_blocks.end(), rng);
            }
        }
        else
        {
            s = uniform_sample(_candidate_blocks.begin() + 1,
                               _candidate_blocks.end(), rng);
        }

        return s;
    }

    size_t random_neighbor(size_t v, rng_t& rng)
    {
        if (_neighbor_sampler.empty(v))
            return v;
        return _neighbor_sampler.sample(v, rng);
    }


    // Computes the move proposal probability
    template <class MEntries>
    double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                         bool reverse, MEntries& m_entries)
    {
        size_t B = _candidate_blocks.size() - 1;

        if (reverse)
        {
            if (_wr[s] == _vweight[v])
                return d;
            if (_wr[r] == 0)
                B++;
        }
        else
        {
            if (_wr[s] == 0)
                return d;
        }

        if (B == _N)
            d = 0;

        if (std::isinf(c))
            return (1. - d) / B;

        double p = 0;
        size_t w = 0;

        size_t kout = 0, kin = 0;
        degs_op(v, _vweight, _eweight, _degs, _g,
                [&] ([[maybe_unused]] size_t din, size_t dout, auto count)
                {
                    kout += dout * count;
                    if constexpr (is_directed_::apply<g_t>::type::value)
                        kin += din * count;
                });
        if constexpr (!is_directed_::apply<g_t>::type::value)
            kin = kout;

        m_entries.get_mes(_emat);

        auto sum_prob = [&](auto& e, auto u)
            {
                size_t t = _b[u];
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

                if constexpr (is_directed_::apply<g_t>::type::value)
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
                    if constexpr (is_directed_::apply<g_t>::type::value)
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

                if constexpr (is_directed_::apply<g_t>::type::value)
                {
                    p += ew * ((mts + mst + c) / (mtp + mtm + c * B));
                }
                else
                {
                    if (t == s)
                        mts *= 2;
                    p += ew * (mts + c) / (mtp + c * B);
                }
            };

        // self-loops are always ignored when sampling neighbors
        for (auto e : out_edges_range(v, _g))
        {
            if (target(e, _g) == v)
                continue;
            sum_prob(e, target(e, _g));
        }

        if constexpr (is_directed_::apply<g_t>::type::value)
        {
            for (auto e : in_edges_range(v, _g))
            {
                if (source(e, _g) == v)
                    continue;
                sum_prob(e, source(e, _g));
            }
        }

        if (w > 0)
            return (1. - d) * p / w;
        else
            return (1. - d) / B;
    }

    double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                         bool reverse,
                         std::vector<std::tuple<size_t, size_t, int>>& p_entries)
    {
        _m_entries.set_move(r, s, num_vertices(_bg));
        for (auto& rsd : p_entries)
            _m_entries.template insert_delta<true>(get<0>(rsd), get<1>(rsd),
                                                   get<2>(rsd));
        return get_move_prob(v, r, s, c, d, reverse);
    }

    double get_move_prob(size_t v, size_t r, size_t s, double c, double d,
                         bool reverse)
    {
        get_move_entries(v, _b[v], (reverse) ? r : s, _m_entries);
        return get_move_prob(v, r, s, c, d, reverse, _m_entries);
    }

    bool is_last(size_t v)
    {
        return (_vweight[v] > 0) && (_wr[_b[v]] == _vweight[v]);
    }

    size_t node_weight(size_t v)
    {
        return _vweight[v];
    }

    // =========================================================================
    // Entropy computation
    // =========================================================================

    double get_deg_entropy(size_t v, const simple_degs_t&)
    {
        auto kin = in_degreeS()(v, _g, _eweight);
        auto kout = out_degreeS()(v, _g, _eweight);
        double S = -lgamma_fast(kin + 1) - lgamma_fast(kout + 1);
        return S * _vweight[v];
    }

    double get_deg_entropy(size_t v, const typename degs_map_t::unchecked_t& degs)
    {
        double S = 0;
        for (auto& ks : degs[v])
        {
            auto kin = get<0>(ks);
            auto kout = get<1>(ks);
            int n = get<2>(ks);
            S -= n * (lgamma_fast(kin + 1) + lgamma_fast(kout + 1));
        }
        return S;
    }

    double sparse_entropy(bool multigraph, bool deg_entropy, bool exact)
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
            for (auto v : vertices_range(_g))
                S += get_deg_entropy(v, _degs);
        }

        if (multigraph)
            S += get_parallel_entropy();

        return S;
    }

    double dense_entropy(bool multigraph)
    {
        if (_deg_corr)
            throw GraphException("Dense entropy for degree corrected model not implemented!");
        double S = 0;
        for (auto e : edges_range(_bg))
        {
            auto r = source(e, _bg);
            auto s = target(e, _bg);
            S += eterm_dense(r, s, _mrs[e], _wr[r], _wr[s], multigraph, _bg);
        }
        return S;
    }

    double entropy(const entropy_args_t& ea, bool propagate=false)
    {
        double S = 0, S_dl = 0;

        if (ea.adjacency)
        {
            if (!ea.dense)
                S = sparse_entropy(ea.multigraph, ea.deg_entropy, ea.exact);
            else
                S = dense_entropy(ea.multigraph);

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

        for (auto v : vertices_range(_g))
        {
            auto& f = _bfield[v];
            if (f.empty())
                continue;
            size_t r = _b[v];
            S_dl -= (r < f.size()) ? f[r] : f.back();
        }

        if (ea.recs)
        {
            auto rdS = rec_entropy(*this, ea);
            S += get<0>(rdS);
            S_dl += get<1>(rdS);
        }

        if (!_Bfield.empty() && ea.Bfield)
        {
            size_t actual_B = 0;
            for (auto& ps : _partition_stats)
                actual_B += ps.get_actual_B();
            S_dl -= (actual_B < _Bfield.size()) ?
                _Bfield[actual_B] : _Bfield.back();
        }

        if (_coupled_state != nullptr && propagate)
            S_dl += _coupled_state->entropy(_coupled_entropy_args, true);

        return S + S_dl * ea.beta_dl;
    }

    double get_partition_dl()
    {
        enable_partition_stats();
        double S = 0;
        for (auto& ps : _partition_stats)
            S += ps.get_partition_dl();
        return S;
    }

    double get_deg_dl(int kind)
    {
        enable_partition_stats();
        double S = 0;
        for (auto& ps : _partition_stats)
            S += ps.get_deg_dl(kind);
        return S;
    }

    template <class Vs, class Skip>
    double get_parallel_entropy(Vs&& vs, Skip&& skip)
    {
        double S = 0;
        for (auto v : vs)
        {
            gt_hash_map<decltype(v), size_t> us;
            for (auto e : out_edges_range(v, _g))
            {
                auto u = target(e, _g);
                if (skip(v, u))
                    continue;
                us[u] += _eweight[e];
            }

            for (auto& uc : us)
            {
                auto& u = uc.first;
                auto& m = uc.second;
                if (m > 1)
                {
                    if (u == v && !is_directed_::apply<g_t>::type::value)
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
        }
        return S;
    }

    double get_parallel_entropy()
    {
        return get_parallel_entropy(vertices_range(_g),
                                    [](auto u, auto v)
                                    { return (u < v &&
                                              !is_directed_::apply<g_t>::type::value); 
                                    });
    }

    template <bool Add>
    double edge_entropy_term(size_t u, size_t v, const entropy_args_t& ea)
    {
        double S = 0, S_dl = 0;
        size_t r = _b[u];
        size_t s = _b[v];

        if (is_partition_stats_enabled() && ea.degree_dl && _deg_corr)
        {
            if (r != s || u == v)
            {
                std::array<std::pair<size_t, size_t>, 2> degs;

                degs_op(u, _vweight, _eweight, _degs, _g,
                        [&] (size_t kin, size_t kout, auto)
                        {
                            degs[0] = {kin, kout};
                            if (u != v)
                            {
                                if (Add)
                                    degs[1] = {kin, kout + 1};
                                else
                                    degs[1] = {kin, kout - 1};
                            }
                            else
                            {
                                if constexpr (!is_directed_::apply<g_t>::type::value)
                                {
                                    if (Add)
                                        degs[1] = {kin, kout + 2};
                                    else
                                        degs[1] = {kin, kout - 2};
                                }
                                else
                                {
                                    if (Add)
                                        degs[1] = {kin + 1, kout + 1};
                                    else
                                        degs[1] = {kin - 1, kout - 1};
                                }
                            }
                        });

                S_dl += get_partition_stats(u).get_deg_dl(ea.degree_dl_kind,
                                                          std::array<size_t,1>({r}),
                                                          degs);

                if (u != v) // r != s
                {
                    std::array<std::pair<size_t, size_t>, 2> degs;

                    degs_op(v, _vweight, _eweight, _degs, _g,
                            [&] (size_t kin, size_t kout, auto)
                            {
                                degs[0] = {kin, kout};
                                if constexpr (!is_directed_::apply<g_t>::type::value)
                                {
                                    if (Add)
                                        degs[1] = {kin, kout + 1};
                                    else
                                        degs[1] = {kin, kout - 1};
                                }
                                else
                                {
                                    if (Add)
                                        degs[1] = {kin + 1, kout};
                                    else
                                        degs[1] = {kin - 1, kout};
                                }
                            });

                    S_dl += get_partition_stats(v).get_deg_dl(ea.degree_dl_kind,
                                                              std::array<size_t,1>({s}),
                                                              degs);
                }
            }
            else // r == s && u != v
            {
                std::array<std::pair<size_t, size_t>, 4> degs;

                degs_op(u, _vweight, _eweight, _degs, _g,
                        [&] (size_t kin, size_t kout, auto)
                        {
                            degs[0] = {kin, kout};
                            if (Add)
                                degs[1] = {kin, kout + 1};
                            else
                                degs[1] = {kin, kout - 1};
                        });

                degs_op(v, _vweight, _eweight, _degs, _g,
                        [&] (size_t kin, size_t kout, auto)
                        {
                            degs[2] = {kin, kout};

                            if constexpr (!is_directed_::apply<g_t>::type::value)
                            {
                                if (Add)
                                    degs[3] = {kin, kout + 1};
                                else
                                    degs[3] = {kin, kout - 1};
                            }
                            else
                            {
                                if (Add)
                                    degs[3] = {kin + 1, kout};
                                else
                                    degs[3] = {kin - 1, kout};
                            }

                            for (size_t i = 0; i < 2; ++i)
                            {
                                if (degs[2] == degs[i])
                                    degs[2] = {0, numeric_limits<size_t>::max()};
                                if (degs[3] == degs[i])
                                    degs[3] = {0, numeric_limits<size_t>::max()};
                            }
                        });

                S_dl += get_partition_stats(u).get_deg_dl(ea.degree_dl_kind,
                                                          std::array<size_t,1>({r}),
                                                          degs);
            }
        }

        auto& me = _emat.get_me(r, s);
        size_t mrs = 0;
        if (me != _emat.get_null_edge())
            mrs = _mrs[me];

        if (ea.adjacency)
        {
            if (ea.dense)
            {
                S += eterm_dense(r, s, mrs, _wr[r], _wr[s], ea.multigraph, _bg);
            }
            else
            {
                if (ea.exact)
                {
                    S += eterm_exact(r, s, mrs, _bg);
                    S += vterm_exact(_mrp[r], _mrm[r], _wr[r], _deg_corr, _bg);
                    if (s != r)
                        S += vterm_exact(_mrp[s], _mrm[s], _wr[s], _deg_corr, _bg);
                }
                else
                {
                    S += eterm(r, s, mrs, _bg);
                    S += vterm(_mrp[r], _mrm[r], _wr[r], _deg_corr, _bg);
                    if (s != r)
                        S += vterm(_mrp[s], _mrm[s], _wr[s], _deg_corr, _bg);
                }

                if (ea.multigraph)
                {
                    S += get_parallel_entropy(std::array<size_t, 1>({u}),
                                              [&](auto, auto w){ return w != v; });
                }

                if (_deg_corr)
                {
                    S += get_deg_entropy(u, _degs);
                    if (u != v)
                        S += get_deg_entropy(v, _degs);
                }
            }
        }

        if (_coupled_state != nullptr)
        {
            S_dl += _coupled_state->edge_entropy_term(r, s, _coupled_entropy_args);
        }
        else
        {
            if (ea.edges_dl && is_partition_stats_enabled())
            {
                size_t actual_B = 0;
                for (auto& psi : _partition_stats)
                    actual_B += psi.get_actual_B();
                auto& ps = get_partition_stats(u);
                S_dl += ps.get_edges_dl(actual_B, _g);
            }
        }

        return S + S_dl * ea.beta_dl;
    }

    double edge_entropy_term(size_t u, size_t v, const entropy_args_t& ea)
    {
        return edge_entropy_term<true>(u, v, ea);
    }

    template <bool Add>
    double modify_edge_dS(size_t u, size_t v, GraphInterface::edge_t& e,
                          const std::vector<double>& recs, const entropy_args_t& ea)
    {
        double dS = 0;
        dS -= edge_entropy_term<Add>(u, v, ea);
        modify_edge<Add>(u, v, e, recs);
        dS += edge_entropy_term<!Add>(u, v, ea);
        modify_edge<!Add>(u, v, e, recs);
        return dS;
    }

    void enable_partition_stats()
    {
        openmp_scoped_lock lock(_partition_lock);
        if (_partition_stats.empty())
        {
            size_t E = 0;
            for (auto e : edges_range(_g))
                E += _eweight[e];
            size_t B = num_vertices(_bg);

// Clang 8.0 fails to correctly recognize these as ForwardIterators,
// triggering a static_assert in std::max_element(). See #576.
#ifndef __clang__
            auto vi = std::max_element(
#else
            auto vi = boost::first_max_element(
#endif
                vertices(_g).first, vertices(_g).second,
                [&](auto u, auto v)
                { return (this->_pclabel[u] <
                          this->_pclabel[v]); });

            size_t C = _pclabel[*vi] + 1;

            vector<vector<size_t>> vcs(C);
            vector<size_t> rc(num_vertices(_bg));
            for (auto v : vertices_range(_g))
            {
                vcs[_pclabel[v]].push_back(v);
                rc[_b[v]] = _pclabel[v];
            }

            for (size_t c = 0; c < C; ++c)
                _partition_stats.emplace_back(_g, _b, vcs[c], E, B,
                                              _vweight, _eweight, _degs,
                                              _bmap);

            for (auto r : vertices_range(_bg))
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

    partition_stats_t& get_partition_stats(size_t v)
    {
        size_t r = _pclabel[v];
        if (r >= _partition_stats.size())
        {
            disable_partition_stats();
            enable_partition_stats();
        }
        return _partition_stats[r];
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
            _egroups.clear();
            _egroups.init(_b, _eweight, _g, _bg);
            rebuild_neighbor_sampler();
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
        _neighbor_sampler.init(_eweight);
    }

    void sync_emat()
    {
        _emat.sync(_bg);
    }

    size_t get_B_E()
    {
        return _B_E;
    }

    size_t get_B_E_D()
    {
        return _B_E_D;
    }

    size_t get_N()
    {
        size_t N = 0;
        for (auto r : vertices_range(_bg))
            N += _wr[r];
        return N;
    }

    vprop_map_t<int32_t>::type::unchecked_t& get_b()
    {
        return _b;
    }

    vprop_map_t<int32_t>::type::unchecked_t& get_pclabel()
    {
        return _pclabel;
    }

    bool check_edge_counts(bool emat=true)
    {
        gt_hash_map<std::pair<size_t, size_t>, size_t> mrs;
        for (auto e : edges_range(_g))
        {
            assert(std::max(source(e, _g),
                            target(e, _g)) < _b.get_storage().size());
            size_t r = _b[source(e, _g)];
            size_t s = _b[target(e, _g)];
            if (!is_directed_::apply<g_t>::type::value && s < r)
                std::swap(r, s);
            mrs[std::make_pair(r, s)] += _eweight[e];
        }

        for (auto& rs_m : mrs)
        {
            auto r = rs_m.first.first;
            auto s = rs_m.first.second;
            size_t m_rs = 0;
            typename graph_traits<bg_t>::edge_descriptor me;
            if (emat)
            {
                me = _emat.get_me(r, s);
                if (me != _emat.get_null_edge())
                    m_rs = _mrs[me];
            }
            else
            {
                auto ret = boost::edge(r, s, _bg);
                me = ret.first;
                if (ret.second)
                    m_rs = _mrs[me];
            }
            if (m_rs != rs_m.second)
            {
                assert(false);
                return false;
            }
        }

        for (auto me : edges_range(_bg))
        {
            auto r = source(me, _bg);
            auto s = target(me, _bg);
            if (!is_directed_::apply<g_t>::type::value && s < r)
                std::swap(r, s);
            auto m_rs = mrs[std::make_pair(r, s)];
            if (m_rs != size_t(_mrs[me]))
            {
                assert(false);
                return false;
            }
        }

        if (_coupled_state != nullptr)
            if (!_coupled_state->check_edge_counts(false))
            {
                assert(false);
                return false;
            }
        return true;
    }

    void check_node_counts()
    {
        vector<size_t> wr(num_vertices(_bg));
        for (auto v : vertices_range(_g))
            wr[_b[v]] += _vweight[v];

        for (auto r : vertices_range(_bg))
            assert(size_t(_wr[r]) == wr[r]);
    }

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
    std::vector<typename rec_t::value_type::checked_t> _c_rec;
    std::vector<typename drec_t::value_type::checked_t> _c_drec;
    std::vector<typename brec_t::value_type::checked_t> _c_brec;
    std::vector<typename bdrec_t::value_type::checked_t> _c_bdrec;
    std::vector<double> _recsum;
    std::vector<double> _recx2;
    std::vector<double> _dBdx;
    std::vector<double> _LdBdx;
    size_t _B_E = 0;
    size_t _B_E_D = 0;
    int _rt = weight_type::NONE;
    size_t _N;

    typedef typename std::conditional<is_weighted_t::value,
                                      vmap_t::unchecked_t, vcmap_t>::type vweight_t;
    vweight_t _vweight;

    typedef typename std::conditional<is_weighted_t::value,
                                      emap_t::unchecked_t, ecmap_t>::type eweight_t;
    eweight_t _eweight;

    typedef typename std::conditional<is_weighted_t::value,
                                      degs_map_t::unchecked_t,
                                      simple_degs_t>::type degs_t;

    degs_t _degs;

    typedef typename std::conditional<use_hash_t::value,
                                      EHash<bg_t>,
                                      EMat<bg_t>>::type
        emat_t;
    emat_t _emat;

    EGroups<g_t, is_weighted_t> _egroups;
    bool _egroups_update = true;

    typedef NeighborSampler<g_t, is_weighted_t, mpl::false_>
        neighbor_sampler_t;

    neighbor_sampler_t _neighbor_sampler;
    std::vector<partition_stats_t> _partition_stats;
    std::vector<size_t> _bmap;

    typedef EntrySet<g_t, bg_t, std::vector<double>,
                     std::vector<double>> m_entries_t;
    m_entries_t _m_entries;

    std::vector<std::tuple<size_t, size_t, int>>
        _pp_entries;

    typedef entropy_args_t _entropy_args_t;
    BlockStateVirtualBase* _coupled_state = nullptr;
    entropy_args_t _coupled_entropy_args;

    openmp_mutex _lock;
    openmp_mutex _partition_lock;
};

} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_HH
