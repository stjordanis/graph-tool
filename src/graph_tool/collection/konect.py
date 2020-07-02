 #! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2020 Tiago de Paula Peixoto <tiago@skewed.de>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import os.path
import tempfile
from urllib.request import urlopen, URLError
from tempfile import TemporaryDirectory
import tarfile
import warnings
import numpy

from .. import Graph

def loadtxt_buf(f, buf_size=100000, **kwargs):
    rows = [None]
    while len(rows) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows = numpy.loadtxt(f, max_rows=buf_size, **kwargs)
        if len(rows) > 0:
            yield rows

def load_koblenz_dir(dirname):
    g = Graph()
    g.gp.meta = g.new_gp("string")
    g.gp.readme = g.new_gp("string")
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if f.startswith("README"):
                with open(os.path.join(root, f)) as fo:
                    g.gp.readme = fo.read()
            if f.startswith("meta."):
                with open(os.path.join(root, f)) as fo:
                    g.gp.meta = fo.read()
            if f.startswith("out."):
                chunk_size = 100000
                bip = False
                with open(os.path.join(root, f), "r") as fo:
                    line = next(fo)
                    if "asym" not in line:
                        g.set_directed(False)
                    if "bip" in line: # bipartite graphs have non-unique indexing
                        bip = True
                if bip:
                    N1 = 0
                    with open(os.path.join(root, f), "r") as fo:
                        for edges in loadtxt_buf(fo, comments="%"):
                            N1 = max(edges[:,0].max(), N1)
                with open(os.path.join(root, f), "r") as fo:
                    eprops = []
                    for edges in loadtxt_buf(fo, comments="%"):
                        edges[:,:2] -= 1  # we need zero-based indexing
                        if bip:
                            edges[:,1] += N1
                        if edges.shape[1] > 2 and len(eprops) < 1:
                            g.ep.weight = g.new_ep("double")
                            eprops.append(g.ep.weight)
                        if edges.shape[1] > 3 and len(eprops) < 2:
                            g.ep.time = g.new_ep("int64_t")
                            eprops.append(g.ep.time)
                        g.add_edge_list(edges, eprops=eprops)
        for f in files:
            if f.startswith("ent."):
                try:
                    g.vp.meta = g.new_vertex_property("string")
                    meta = g.vp.meta
                    count = 0
                    for line in open(os.path.join(root, f)):
                        vals = line.split()
                        if len(vals) == 0 or (len(vals) == 1 and vals[0] == "%"):
                            continue
                        if vals[0] == "%":
                            g.gp.meta_desc = g.new_gp("string", line)
                            continue
                        v = g.vertex(count)
                        meta[v] = line.strip()
                        count += 1
                except ValueError as e:
                    warnings.warn("error automatically reading node metadata from file '%s': %s" % (f, str(e)))
    return g

def get_koblenz_network_data(name):
    with tempfile.TemporaryFile(mode='w+b') as ftemp:
        try:
            response = urlopen('http://konect.cc/files/download.tsv.%s.tar.bz2' % name)
        except URLError:
            response = urlopen('http://konect.uni-koblenz.de/downloads/tsv/%s.tar.bz2' % name)
        buflen = 1 << 20
        while True:
            buf = response.read(buflen)
            ftemp.write(buf)
            if len(buf) < buflen:
                break
        ftemp.seek(0)
        with TemporaryDirectory(suffix=name) as tempdir:
            with tarfile.open(fileobj=ftemp, mode='r:bz2') as tar:
                tar.extractall(path=tempdir)
            g = load_koblenz_dir(tempdir)
            return g

class LazyKoblenzDataDict(dict):
    def __getitem__(self, k):
        if k not in self:
            g = get_koblenz_network_data(k)
            dict.__setitem__(self, k, g)
            return g
        return dict.__getitem__(self, k)


konect_data = LazyKoblenzDataDict()
