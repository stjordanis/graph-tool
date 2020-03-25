ARG GIT_REF

FROM archlinux/base:latest as builder
MAINTAINER Tiago de Paula Peixoto <tiago@skewed.de>

RUN pacman -Sy
RUN pacman -S pacman-contrib --noconfirm --disable-download-timeout
RUN curl -s "https://www.archlinux.org/mirrorlist/?country=US&protocol=https&use_mirror_status=on" | sed -e 's/^#Server/Server/' -e '/^#/d' | rankmirrors -n 5 - > /etc/pacman.d/mirrorlist

RUN pacman -Syu --noconfirm --disable-download-timeout
RUN pacman -S binutils make gcc fakeroot --noconfirm --needed
RUN pacman -S expac yajl git --noconfirm --needed
RUN pacman -S sudo grep file --noconfirm --needed

RUN pacman -S sudo boost python3 python3-scipy python-numpy \
              cgal cairomm python-cairo sparsehash cairomm   \
	      autoconf-archive autoconf automake pkg-config --noconfirm --needed

ENV MAKEPKG_USER=mkpkg \
    MAKEPKG_GROUP=mkpkg \
    MAKEPKG_ROOT=/tmp/build

RUN groupadd "${MAKEPKG_USER}" \
    && useradd -g "${MAKEPKG_GROUP}" "${MAKEPKG_USER}"

RUN mkdir -p ${MAKEPKG_ROOT}; chown mkpkg:mkpkg ${MAKEPKG_ROOT}

WORKDIR ${MAKEPKG_ROOT}

USER ${MAKEPKG_USER}
ADD PKGBUILD PKGBUILD
RUN makepkg PKGBUILD --needed GIT_REF=$GIT_REF CXXFLAGS="-mtune=generic -O3 -pipe -flto=8 -ffunction-sections -fdata-sections" LDFLAGS="-Wl,--gc-sections"

USER root
RUN mv -v "${MAKEPKG_ROOT}/"python-graph-tool-*-x86_64.pkg.tar.xz /tmp/python-graph-tool.pkg.tar.xz

# Non-build

FROM archlinux/base:latest

RUN echo "Server=https://archive.archlinux.org/repos/`date +%Y/%m/%d`/\$repo/os/\$arch" > /etc/pacman.d/mirrorlist

COPY --from=builder /tmp/python-graph-tool.pkg.tar.xz /tmp/python-graph-tool.pkg.tar.xz

RUN pacman -Syu --noconfirm --disable-download-timeout \
 && pacman -U --noconfirm --noprogressbar --needed  /tmp/python-graph-tool.pkg.tar.xz \
 && yes | pacman -Scc --noconfirm

RUN pacman -S ipython gtk3 python-gobject python-matplotlib python-pandas jupyter-notebook mathjax python-cairocffi pandoc --noconfirm --needed

RUN useradd -m -g users user

ENV PYTHONIOENCODING=utf8
