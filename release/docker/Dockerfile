ARG GIT_REF

FROM archlinux:latest as builder
MAINTAINER Tiago de Paula Peixoto <tiago@skewed.de>

RUN pacman -Sy
RUN pacman -S pacman-contrib --noconfirm --disable-download-timeout
RUN curl -s "https://archlinux.org/mirrorlist/?country=DE&protocol=https&use_mirror_status=on" | sed -e 's/^#Server/Server/' -e '/^#/d' | rankmirrors -n 5 - > /etc/pacman.d/mirrorlist

RUN pacman -Syu --noconfirm --disable-download-timeout
RUN pacman -S binutils make gcc fakeroot libtool --noconfirm --needed
RUN pacman -S expac yajl git --noconfirm --needed
RUN pacman -S sudo grep file --noconfirm --needed

RUN pacman -S sudo boost python python-scipy python-numpy python-zstandard \
              cgal cairomm-1.16 python-cairo sparsehash cairomm debugedit \
              autoconf-archive autoconf automake pkg-config gd lib32-gcc-libs bison \
              --noconfirm --needed

ENV MAKEPKG_USER=mkpkg \
    MAKEPKG_GROUP=mkpkg \
    MAKEPKG_ROOT=/tmp/build

RUN groupadd "${MAKEPKG_USER}" \
    && useradd -g "${MAKEPKG_GROUP}" "${MAKEPKG_USER}"

RUN mkdir -p ${MAKEPKG_ROOT}; chown mkpkg:mkpkg ${MAKEPKG_ROOT}

# Install glibc version compatible with linux 4 (for HPC systems)
USER ${MAKEPKG_USER}
WORKDIR ${MAKEPKG_ROOT}
RUN git clone https://aur.archlinux.org/glibc-linux4.git
WORKDIR ${MAKEPKG_ROOT}/glibc-linux4
RUN makepkg PKGBUILD --needed --skipinteg

USER root
RUN mv -v "${MAKEPKG_ROOT}"/glibc-linux4/glibc-linux4-2*.pkg.tar.zst /tmp/glibc-linux4.pkg.tar.zst

USER ${MAKEPKG_USER}
WORKDIR ${MAKEPKG_ROOT}
ADD PKGBUILD PKGBUILD
RUN nice -n 19 makepkg PKGBUILD --needed GIT_REF=$GIT_REF CXXFLAGS="-mtune=generic -O3 -pipe -ffunction-sections -fdata-sections" LDFLAGS="-Wl,--gc-sections"

USER root
RUN mv -v "${MAKEPKG_ROOT}/"python-graph-tool-*-x86_64.pkg.tar.zst /tmp/python-graph-tool.pkg.tar.zst

# Non-build

FROM archlinux:latest

RUN echo "Server=https://archive.archlinux.org/repos/`date -d 'yesterday 13:00' +%Y/%m/%d`/\$repo/os/\$arch" > /etc/pacman.d/mirrorlist

COPY --from=builder /tmp/python-graph-tool.pkg.tar.zst /tmp/python-graph-tool.pkg.tar.zst
COPY --from=builder /tmp/glibc-linux4.pkg.tar.zst /tmp/glibc-linux4.pkg.tar.zst

RUN pacman -Syu --noconfirm --disable-download-timeout \
 && pacman -U --noconfirm --noprogressbar --needed  /tmp/python-graph-tool.pkg.tar.zst \
 && yes | pacman -Scc --noconfirm

RUN yes | pacman -S ipython gtk3 python-gobject python-matplotlib python-pandas jupyter-notebook mathjax python-cairocffi pandoc --noconfirm --needed

RUN yes | pacman -U --noprogressbar --needed  /tmp/glibc-linux4.pkg.tar.zst

RUN useradd -m -g users user

ENV PYTHONIOENCODING=utf8
