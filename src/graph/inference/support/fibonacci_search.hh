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

#ifndef FIBONACCI_SEARCH_HH
#define FIBONACCI_SEARCH_HH

#include <algorithm>
#include <array>
#include <random>
#include <cmath>

namespace graph_tool
{

class FibonacciSearch
{
public:
    FibonacciSearch() {}

    template <class F, class... RNG>
    size_t search(size_t x_min, size_t x_max, F&& f, RNG&... rng)
    {
        size_t x_mid;
        return search(x_min, x_mid, x_max, f, rng...);
    }

    template <class F, class... RNG>
    size_t search(size_t& x_min, size_t& x_mid, size_t& x_max, F&& f, RNG&... rng)
    {
        // initial bracketing
        x_mid = get_mid(x_min, x_max, rng...);
        double f_max = f(x_max);
        double f_mid = f(x_mid);
        double f_min = f(x_min);

        while (f_mid > f_min || f_mid > f_max)
        {
            if (f_min < f_max)
            {
                x_max = x_mid;
                f_max = f_mid;
                x_mid = get_mid(x_min, x_mid, rng...);
            }
            else
            {
                x_min = x_mid;
                f_min = f_mid;
                x_mid = get_mid(x_mid, x_max, rng...);
            }

            f_mid = f(x_mid);

            if (x_min == x_mid && (x_max - x_mid) <= 1)
                break;
        }

        // Fibonacci search
        while (x_max - x_mid > 1)
        {
            size_t x;
            if (x_max - x_mid > x_mid - x_min)
                x = get_mid(x_mid, x_max, rng...);
            else
                x = get_mid(x_min, x_mid, rng...);

            double f_x = f(x);

            if (f_x < f_mid)
            {
                if (x_max - x_mid > x_mid - x_min)
                {
                    x_min = x_mid;
                    f_min = f_mid;
                }
                else
                {
                    x_max = x_mid;
                    f_max = f_mid;
                }
                x_mid = x;
                f_mid = f_x;
            }
            else
            {
                if (x_max - x_mid > x_mid - x_min)
                {
                    x_max = x;
                    f_max = f_x;
                }
                else
                {
                    x_min = x;
                    f_min = f_x;
                }
            }
        }

        std::array<size_t,3> xs = {x_min, x_mid, x_max};
        std::array<double,3> fs = {f_min, f_mid, f_max};

        return xs[std::min_element(fs.begin(), fs.end()) - fs.begin()];
    }

    size_t fibo(size_t n)
    {
        return size_t(std::round(std::pow(_phi, n) / std::sqrt(5)));
    }

    size_t fibo_n_floor(size_t x)
    {
        return std::floor(std::log(x * std::sqrt(5) + .5) / std::log(_phi));
    }

    size_t get_mid(size_t a, size_t b)
    {
        if (a == b)
            return a;
        auto n = fibo_n_floor(b - a);
        return b - fibo(n - 1);
    }

    template <class RNG>
    size_t get_mid(size_t a, size_t b, RNG& rng)
    {
        if (a == b)
            return a;
        std::uniform_int_distribution<size_t> sample(a, b - 1);
        return sample(rng);
    }

private:
#ifndef __clang__
    constexpr static
#endif
    double _phi = (1 + std::sqrt(5)) / 2;
};

}
#endif // FIBONACCI_SEARCH_HH
