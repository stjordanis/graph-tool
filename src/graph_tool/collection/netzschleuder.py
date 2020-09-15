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

import urllib.request
import base64
import contextlib
import json
import os.path

from .. import load_graph

try:
    import zstandard
except ImportError:
    pass

url_prefix = "https://networks.skewed.de"

username = None
password = None

@contextlib.contextmanager
def open_ns_file(url, token=None):
    global username, password

    if username is not None:
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, url_prefix, username, password)
        handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
        opener = urllib.request.build_opener(handler)
        urllib.request.install_opener(opener)

    if token is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url,
                                     headers={"WWW-Authenticate" :
                                              f"Authorization: Bearer {token}"})

    with urllib.request.urlopen(req) as f:
        yield f

def get_ns_network(k, token=None):

    if isinstance(k, str):
        net = k.split("/")
        if len(net) == 1:
            url = f"{url_prefix}/net/{k}/files/network.gt.zst"
        else:
            url = f"{url_prefix}/net/{net[0]}/files/{net[1]}.gt.zst"
    else:
        url = f"{url_prefix}/net/{k[0]}/files/{k[1]}.gt.zst"

    with open_ns_file(url, token) as f:
        try:
            cctx = zstandard.ZstdDecompressor()
        except NameError:
            raise NotImplementedError("zstandard module not installed, but it's required for zstd de-compression")
        with cctx.stream_reader(f) as fc:
            g = load_graph(fc, fmt="gt")

    return g

def get_ns_info(k):

    url = f"{url_prefix}/api/net/{k}"

    with open_ns_file(url) as f:
        return json.load(f)



class LazyNSDataDict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keys = None
        self.token = None
        path = os.path.expanduser("~/.gt_token")
        if os.path.exists(path):
            self.token = open(path).read().strip()

    def set_token(self, token):
        self.token = token

    def sync_keys(self):
        "Download all keys from upstream."
        with open_ns_file(f"{url_prefix}/api/nets?full=True") as f:
            d = json.load(f)
            self._keys = []
            for k, v in d.items():
                if len(v["nets"]) == 1:
                    self._keys.append(k)
                else:
                    for net in v["nets"]:
                        self._keys.append(f"{k}/{net}")

    def keys(self):
        if self._keys is None:
            self.sync_keys()
        return self._keys

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def __getitem__(self, k):
        if k not in self:
            try:
                g = get_ns_network(k, self.token)
            except urllib.error.URLError as e:
                raise KeyError(str(e))
            dict.__setitem__(self, k, g)
            return g
        return dict.__getitem__(self, k)

class LazyNSInfoDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sync = False
        self._keys = None

    def sync_keys(self):
        "Download all keys from upstream."
        with open_ns_file(f"{url_prefix}/api/nets") as f:
            self._keys = json.load(f)

    def sync(self):
        "Download all keys and values from upstream."
        with open_ns_file(f"{url_prefix}/api/nets?full=True") as f:
            self.update(json.load(f))
            self._sync = True
            self._keys = None

    def keys(self):
        if self._keys is not None:
            return self._keys
        if not self._sync:
            self.sync_keys()
            return self._keys
        return super().keys()

    def values(self):
        if not self._sync:
            self.sync()
        return super().values()

    def items(self):
        if not self._sync:
            self.sync()
        return super().items()

    def __getitem__(self, k):
        if k not in self:
            try:
                d = get_ns_info(k)
            except urllib.error.URLError as e:
                raise KeyError(str(e))
            dict.__setitem__(self, k, d)
            return d
        return dict.__getitem__(self, k)

ns = LazyNSDataDict()
ns_info = LazyNSInfoDict()
