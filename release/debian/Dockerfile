ARG BASE
ARG REF

FROM $BASE as builder

RUN apt-get update
RUN apt-get -y dist-upgrade
RUN apt-get -y install git dpkg-dev dh-make autotools-dev autoconf python3-dev python3-scipy libboost-dev libboost-graph-dev libboost-iostreams-dev libboost-python-dev libboost-regex-dev libcgal-dev python3-cairo-dev libsparsehash-dev libcairomm-1.0-dev libffi-dev libexpat1-dev cdbs
RUN git clone https://git.skewed.de/count0/graph-tool.git python3-graph-tool
WORKDIR python3-graph-tool
RUN git reset --hard $REF
ADD debian debian
RUN cp LICENSE debian/copyright
ADD git2debchangelog.sh .
RUN /bin/bash git2debchangelog.sh > debian/changelog
RUN ./autogen.sh
RUN dpkg-buildpackage -us -uc -j2
WORKDIR ..
RUN mkdir build
RUN mv python3-graph-tool*.deb build/

FROM $BASE

RUN apt-get update
RUN apt-get -y dist-upgrade
RUN apt-get -y install gdebi-core

RUN mkdir build
COPY --from=builder build/* build/
RUN gdebi -n build/python3-graph-tool_*.deb
RUN python3 -c "from graph_tool.all import *; show_config(); g = random_graph(10, lambda: 5, directed=False)"
