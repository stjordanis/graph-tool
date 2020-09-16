## Process this file with automake to produce Makefile.in

SUBDIRS = src

graphtooldocdir = ${PACKAGE_DOC_DIR}
nobase_dist_graphtooldoc_DATA = \
	README.md \
	COPYING \
	COPYING.LESSER \
	AUTHORS \
	INSTALL \
	src/boost-workaround/LICENSE_1_0.txt \
	doc/centrality.rst \
	doc/correlations.rst \
	doc/price.py \
	doc/search_module.rst \
	doc/util.rst \
	doc/clustering.rst \
	doc/draw.rst \
	doc/index.rst \
	doc/spectral.rst \
	doc/inference.rst  \
	doc/flow.rst \
	doc/Makefile \
	doc/quickstart.rst \
	doc/generation.rst \
	doc/stats.rst \
	doc/graph_tool.rst \
	doc/modules.rst \
	doc/topology.rst \
	doc/conf.py \
	doc/pyenv.py \
	doc/search_example.xml \
	doc/demos/index.rst \
	doc/demos/animation/animation.rst \
	doc/demos/inference/inference.rst \
	doc/demos/cppextensions/Makefile \
	doc/demos/cppextensions/cppextensions.rst \
	doc/demos/cppextensions/kcore.cc \
	doc/demos/cppextensions/kcore.hh \
	doc/demos/cppextensions/kcore.py \
	doc/sphinxext/comment_eater.py \
	doc/sphinxext/phantom_import.py \
	doc/sphinxext/compiler_unparse.py \
	doc/sphinxext/traitsdoc.py \
	doc/gt_theme/static/flasky.css \
	doc/gt_theme/static/gt_style.css \
	doc/gt_theme/theme.conf \
	doc/.static/graph-icon.png \
	doc/.static/graph-icon.ico


EXTRA_DIST = $(graphtooldoc_DATA) autogen.sh

pkgconfigdir = @pkgconfigdir@
pkgconfig_DATA = graph-tool-py${PYTHON_VERSION}.pc

# Copy all the spec files. Of course, only one is actually used.
dist-hook:
	cd ${top_srcdir} && git log --stat --name-only --date=short --abbrev-commit > ${distdir}/ChangeLog
	for specfile in *.spec; do \
		if test -f $$specfile; then \
			cp -p $$specfile $(distdir); \
		fi \
	done