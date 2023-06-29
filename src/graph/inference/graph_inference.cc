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

#include "graph_tool.hh"

#include <boost/python.hpp>
#include "numpy_bind.hh"
#include "hash_map_wrap.hh"
#include "random.hh"

#include "support/util.hh"
#include "support/fibonacci_search.hh"

using namespace std;
using namespace boost;
using namespace graph_tool;

typedef mpl::vector<int8_t, int16_t, int32_t, int64_t,
                    uint8_t, uint16_t, uint32_t,
                    uint64_t, float, double, long double> scalars_t;

class stop: public std::exception {};

void vector_map(boost::python::object ovals, boost::python::object omap)
{
    bool found = false;
    try
    {
        mpl::for_each<scalars_t>
            ([&](auto idx)
             {
                 typedef decltype(idx) idx_t;
                 try
                 {
                     auto vals = get_array<idx_t,1>(ovals);
                     mpl::for_each<scalars_t>
                         ([&](auto val)
                          {
                              typedef decltype(val) val_t;
                              try
                              {
                                  auto map = get_array<val_t,1>(omap);
                                  for (size_t i = 0; i < vals.size(); ++i)
                                      vals[i] = map[vals[i]];
                                  found = true;
                                  throw stop();
                              }
                              catch (InvalidNumpyConversion&) {};
                          });
                 }
                 catch (InvalidNumpyConversion&) {};
             });
    }
    catch (stop&) {};
    if (!found)
        throw ValueException("Invalid array types");
}

void vector_contiguous_map(boost::python::object ovals)
{
    bool found = false;
    try
    {
        mpl::for_each<scalars_t>
            ([&](auto idx)
             {
                 typedef decltype(idx) idx_t;
                 try
                 {
                     auto vals = get_array<idx_t,1>(ovals);
                     gt_hash_map<idx_t, size_t> map;

                     for (size_t i = 0; i < vals.size(); ++i)
                     {
                         auto v = vals[i];
                         auto iter = map.find(v);
                         if (iter == map.end())
                             iter = map.insert({v, map.size()}).first;
                         vals[i] = iter->second;
                     }
                     found = true;
                     throw stop();
                 }
                 catch (InvalidNumpyConversion&) {};
             });
    }
    catch (stop&) {};
    if (!found)
        throw ValueException("Invalid array type");
}

void vector_rmap(boost::python::object ovals, boost::python::object omap)
{
    bool found = false;
    try
    {
        mpl::for_each<scalars_t>
            ([&](auto idx)
             {
                 typedef decltype(idx) idx_t;
                 try
                 {
                     auto vals = get_array<idx_t,1>(ovals);
                     mpl::for_each<scalars_t>
                         ([&](auto val)
                          {
                              typedef decltype(val) val_t;
                              try
                              {
                                  auto map = get_array<val_t,1>(omap);
                                  for (size_t i = 0; i < vals.size(); ++i)
                                      map[vals[i]] = i;
                                  found = true;
                                  throw stop();
                              }
                              catch (InvalidNumpyConversion&) {};
                          });
                 }
                 catch (InvalidNumpyConversion&) {};
             });
    }
    catch (stop&) {};
    if (!found)
        throw ValueException("Invalid array types");
}

#define __MOD__ inference
#define DEF_REGISTRY
#include "module_registry.hh"

BOOST_PYTHON_MODULE(libgraph_tool_inference)
{
    using namespace boost::python;
    docstring_options dopt(true, false);

    def("vector_map", vector_map);
    def("vector_rmap", vector_rmap);
    def("vector_contiguous_map", vector_contiguous_map);

    def("lbinom", lbinom<size_t, size_t>);
    def("lbinom_fast", lbinom_fast<true, size_t, size_t>);
    def("lgamma_fast", lgamma_fast<true, size_t>);
    def("safelog_fast", safelog_fast<true, size_t>);
    def("init_cache", init_cache);
    def("log_sum_exp", +[](double x, double y){ return log_sum_exp(x, y); });

    class_<FibonacciSearch>("FibonacciSearch")
        .def("search",
             +[](FibonacciSearch& s, size_t x_min, size_t x_max, python::object f)
              {
                  return s.search(x_min, x_max,
                                  [&](size_t x)
                                  {
                                      return python::extract<double>(f(x));
                                  });
              })
        .def("search_random",
             +[](FibonacciSearch& s, size_t x_min, size_t x_max,
                 python::object f, rng_t& rng)
              {
                  return s.search(x_min, x_max,
                                  [&](size_t x)
                                  {
                                      return python::extract<double>(f(x));
                                  }, rng);
              });

    __MOD__::EvokeRegistry();
}
