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

#ifndef MODULE_REGISTRY_HH
#define MODULE_REGISTRY_HH

#include <functional>
#include <vector>
#include <tuple>
#include <algorithm>
#include <limits>

#ifndef __MOD__
#error "__MOD__ needs to be defined"
#endif

#define REGISTER_MOD static __MOD__::RegisterMod __reg

namespace __MOD__
{

typedef std::vector<std::tuple<int,std::function<void()>>> reg_t;

reg_t& get_module_registry()
#ifdef DEF_REGISTRY
{
    static reg_t* reg = new reg_t();
    return *reg;
}
#else
;
#endif


class RegisterMod
{
public:
    RegisterMod(std::function<void()> f, int p = std::numeric_limits<int>::max())
    {
        get_module_registry().emplace_back(p, f);
    }
};

class EvokeRegistry
{
public:
    EvokeRegistry()
    {
        reg_t& reg = get_module_registry();
        std::sort(reg.begin(), reg.end(),
                  [](const auto& a, const auto& b)
                  { return std::get<0>(a) < std::get<0>(b); });
        for (auto& [p, f] : reg)
            f();
        delete &reg;
    }
};

} // namespace __MOD__

#endif // MODULE_REGISTRY_HH
