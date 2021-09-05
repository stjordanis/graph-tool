
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

#ifndef UTIL_HH
#define UTIL_HH

#include "config.h"

#include <cmath>

#include "cache.hh"

namespace graph_tool
{
using namespace boost;

template <class T1, class T2>
[[gnu::const]]
inline double lbinom(T1 N, T2 k)
{
    if (N == 0 || k == 0 || k >= N)
        return 0;
    return ((std::lgamma(N + 1) - std::lgamma(k + 1)) - std::lgamma(N - k + 1));
}

template <bool Init=true, class T1, class T2>
[[gnu::const]]
inline double lbinom_fast(T1 N, T2 k)
{
    if (N == 0 || k == 0 || k > N)
        return 0;
    return ((lgamma_fast<Init>(N + 1) - lgamma_fast<Init>(k + 1)) - lgamma_fast<Init>(N - k + 1));
}

template <class T1, class T2>
[[gnu::const]]
inline double lbinom_careful(T1 N, T2 k)
{
    if (N == 0 || k == 0 || k >= N)
        return 0;
    double lgN = std::lgamma(N + 1);
    double lgk = std::lgamma(k + 1);
    if (lgN - lgk > 1e8)
    {
        // We have N >> k. Use Stirling's approximation: ln N! ~ N ln N - N
        // and reorder
        return - N * log1p(-k / N) - k * log1p(-k / N) - k - lgk + k * log(N);
    }
    else
    {
        return lgN - std::lgamma(N - k + 1) - lgk;
    }
}

template <class T>
[[gnu::const]]
inline auto lbeta(T x, T y)
{
    return (std::lgamma(x) + std::lgamma(y)) - std::lgamma(x + y);
}

template <class T>
[[gnu::const]]
inline T log_sum_exp(T a, T b)
{
    if (a == b)  // handles infinity
        return a + std::log(2);
    else if (a > b)
        return a + std::log1p(exp(b-a));
    else
        return b + std::log1p(exp(a-b));
}

namespace detail {
   template <class F, class Tuple, std::size_t... I>
   constexpr decltype(auto) tuple_apply_impl(F&& f, Tuple&& t,
                                             std::index_sequence<I...>)
   {
       return f(std::get<I>(std::forward<Tuple>(t))...);
   }
} // namespace detail

template <class F, class Tuple>
constexpr decltype(auto) tuple_apply(F&& f, Tuple&& t)
{
    return detail::tuple_apply_impl
        (std::forward<F>(f), std::forward<Tuple>(t),
         std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>{}>{});
}

} // namespace graph_tool

#endif // UTIL_HH
