ARG BASE

FROM $BASE as builder

ARG BASE

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get -y dist-upgrade
RUN apt-get -y install git dpkg-dev dh-make autotools-dev autoconf python3-dev python3-scipy libboost-dev libboost-graph-dev libboost-iostreams-dev libboost-python-dev libboost-context-dev libboost-coroutine-dev libboost-regex-dev libcgal-dev python3-cairo-dev libsparsehash-dev libcairomm-1.0-dev libffi-dev libexpat1-dev cdbs devscripts
RUN git clone https://git.skewed.de/count0/graph-tool.git python3-graph-tool
WORKDIR python3-graph-tool

ARG REF

RUN git reset --hard $REF
ADD debian debian
RUN cp LICENSE debian/copyright
ADD git2debchangelog.sh .
RUN /bin/bash git2debchangelog.sh > debian/changelog

ARG DEB_VERSION=1

RUN if [ "$DEB_VERSION" -gt "1" ]; then gt_version=`git describe --tags | grep -o 'release-[^-]*' | sed s/release-//`; dch -v ${gt_version}-${DEB_VERSION} -m 'New package release'; dch -r --no-force-save-on-release -m 'New package release'; fi
RUN head debian/changelog

RUN ./autogen.sh

ARG NJOBS=2

RUN NJOBS=$NJOBS dpkg-buildpackage -us -uc -j$NJOBS
WORKDIR ..
RUN mkdir build
RUN mv python3-graph-tool*.deb build/

FROM $BASE

RUN apt-get update
RUN apt-get -y dist-upgrade
RUN apt-get -y install gdebi-core
RUN apt-get -y install python3-matplotlib gir1.2-gtk-3.0 python3-cairo

RUN mkdir build
COPY --from=builder build/* build/
RUN gdebi -n build/python3-graph-tool_*.deb
RUN python3 -c "from graph_tool.all import *; show_config(); g = random_graph(10, lambda: 5, directed=False); graph_draw(g, output='foo.png')"
