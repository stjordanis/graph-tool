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

#include "graph.hh"

#ifdef HAVE_CAIROMM

#include PYCAIRO_HEADER

#if PY_MAJOR_VERSION < 3
static Pycairo_CAPI_t *Pycairo_CAPI = nullptr;
#endif

extern "C"
PyObject* gt_PycairoContext_FromContext(cairo_t *ctx, PyTypeObject *type,
                                        PyObject *base)
{
#if PY_MAJOR_VERSION < 3
    if (Pycairo_CAPI == nullptr)
        Pycairo_IMPORT;
#endif
    return PycairoContext_FromContext(ctx, type, base);
}

#endif // HAVE_CAIROMM
