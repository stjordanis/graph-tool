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
#include <boost/python.hpp>
#include <iostream>

#ifndef __MOD__
#error "__MOD__ needs to be defined"
#endif

#define REGISTER_MOD \
    __attribute__((init_priority(300))) static __MOD__::RegisterMod __reg

namespace __MOD__
{

#ifndef DEF_REGISTRY
extern
#else
__attribute__((init_priority(200)))
#endif
std::vector<std::function<void()>> __module_registry;

class RegisterMod
{
public:
    RegisterMod(std::function<void()> f)
    {
        __module_registry.push_back(f);
    }
};

class EvokeRegistry
{
public:
    EvokeRegistry()
    {
        for (auto& f : __module_registry)
            f();
        __module_registry.clear();
        __module_registry.shrink_to_fit();
    }
};

} // namespace __MOD__

#endif // MODULE_REGISTRY_HH
