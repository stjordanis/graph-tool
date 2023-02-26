## Process this file with automake to produce Makefile.in

AM_CPPFLAGS = $(MOD_CPPFLAGS)
AM_CFLAGS = $(AM_CXXFLAGS)

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
	doc/.static/graph-icon.png \
	doc/.static/graph-icon.ico \
	doc/.static/graph-tool-logo.svg

EXTRA_DIST = autogen.sh

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

##############
# src/graph/ #
##############

libgraph_tool_coredir = $(MOD_DIR)
libgraph_tool_core_LTLIBRARIES = libgraph_tool_core.la
libgraph_tool_core_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_core_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_core_la_SOURCES = \
	src/graph/base64.cc \
	src/graph/demangle.cc \
	src/graph/graph.cc \
	src/graph/graph_exceptions.cc \
	src/graph/graph_bind.cc \
	src/graph/graph_copy.cc \
	src/graph/graph_filtering.cc \
	src/graph/graph_io.cc \
	src/graph/graph_openmp.cc \
	src/graph/graph_properties.cc \
	src/graph/graph_properties_imp1.cc \
	src/graph/graph_properties_imp2.cc \
	src/graph/graph_properties_imp3.cc \
	src/graph/graph_properties_imp4.cc \
	src/graph/graph_properties_copy.cc \
	src/graph/graph_properties_copy_imp1.cc \
	src/graph/graph_properties_group.cc \
	src/graph/graph_properties_ungroup.cc \
	src/graph/graph_properties_map_values.cc \
	src/graph/graph_properties_map_values_imp1.cc \
	src/graph/graph_python_interface.cc \
	src/graph/graph_python_interface_imp1.cc \
	src/graph/graph_python_interface_export.cc \
	src/graph/graph_selectors.cc \
	src/graph/graphml.cpp \
	src/graph/random.cc \
	src/graph/read_graphviz_new.cpp
libgraph_tool_core_la_includedir = $(MOD_DIR)/include
libgraph_tool_core_la_include_HEADERS = \
	config.h \
	src/graph/base64.hh \
	src/graph/coroutine.hh \
	src/graph/demangle.hh \
	src/graph/fast_vector_property_map.hh \
	src/graph/gml.hh \
	src/graph/graph.hh \
	src/graph/graph_adjacency.hh \
	src/graph/graph_adaptor.hh \
	src/graph/graph_exceptions.hh \
	src/graph/graph_filtered.hh \
	src/graph/graph_filtering.hh \
	src/graph/graph_io_binary.hh \
	src/graph/graph_properties.hh \
	src/graph/graph_properties_copy.hh \
	src/graph/graph_properties_group.hh \
	src/graph/graph_properties_map_values.hh \
	src/graph/graph_python_interface.hh \
	src/graph/graph_reverse.hh \
	src/graph/graph_selectors.hh \
	src/graph/graph_tool.hh \
	src/graph/graph_util.hh \
	src/graph/hash_map_wrap.hh \
	src/graph/histogram.hh \
	src/graph/idx_map.hh \
	src/graph/mpl_nested_loop.hh \
	src/graph/numpy_bind.hh \
	src/graph/openmp_lock.hh \
	src/graph/parallel_rng.hh \
	src/graph/random.hh \
	src/graph/str_repr.hh \
	src/graph/shared_map.hh \
	src/graph/transform_iterator.hh
libgraph_tool_core_la_workarounddir = $(MOD_DIR)/include/boost-workaround/boost/graph/
libgraph_tool_core_la_workaround_HEADERS = \
	src/boost-workaround/boost/graph/adjacency_iterator.hpp \
	src/boost-workaround/boost/graph/betweenness_centrality.hpp \
	src/boost-workaround/boost/graph/detail/read_graphviz_new.hpp \
	src/boost-workaround/boost/graph/graphml.hpp \
	src/boost-workaround/boost/graph/graphviz.hpp \
	src/boost-workaround/boost/graph/hawick_circuits.hpp \
	src/boost-workaround/boost/graph/isomorphism.hpp \
	src/boost-workaround/boost/graph/kamada_kawai_spring_layout.hpp \
	src/boost-workaround/boost/graph/maximum_weighted_matching.hpp \
	src/boost-workaround/boost/graph/metric_tsp_approx.hpp \
	src/boost-workaround/boost/graph/named_function_params-alt.hpp \
	src/boost-workaround/boost/graph/overloading.hpp \
	src/boost-workaround/boost/graph/push_relabel_max_flow.hpp \
	src/boost-workaround/boost/graph/copy_alt.hpp \
	src/boost-workaround/boost/graph/stoer_wagner_min_cut.hpp \
	src/boost-workaround/boost/graph/vf2_sub_graph_iso.hpp
libgraph_tool_core_la_pcg_cppdir = $(MOD_DIR)/include/pcg-cpp
libgraph_tool_core_la_pcg_cpp_HEADERS = \
	src/pcg-cpp/include/pcg_extras.hpp \
	src/pcg-cpp/include/pcg_random.hpp \
	src/pcg-cpp/include/pcg_uint128.hpp

#########################
# src/graph/centrality/ #
#########################

libgraph_tool_centralitydir = $(MOD_DIR)/centrality
libgraph_tool_centrality_LTLIBRARIES = libgraph_tool_centrality.la
libgraph_tool_centrality_la_includedir = $(MOD_DIR)/include/centrality
libgraph_tool_centrality_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_centrality_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_centrality_la_SOURCES = \
	src/graph/centrality/graph_betweenness.cc \
	src/graph/centrality/graph_centrality_bind.cc \
	src/graph/centrality/graph_closeness.cc \
	src/graph/centrality/graph_eigentrust.cc \
	src/graph/centrality/graph_eigenvector.cc \
	src/graph/centrality/graph_hits.cc \
	src/graph/centrality/graph_katz.cc \
	src/graph/centrality/graph_pagerank.cc \
	src/graph/centrality/graph_trust_transitivity.cc
libgraph_tool_centrality_la_include_HEADERS = \
	src/graph/centrality/graph_closeness.hh \
	src/graph/centrality/graph_eigentrust.hh \
	src/graph/centrality/graph_eigenvector.hh \
	src/graph/centrality/graph_pagerank.hh \
	src/graph/centrality/graph_hits.hh \
	src/graph/centrality/graph_katz.hh \
	src/graph/centrality/graph_trust_transitivity.hh \
	src/graph/centrality/minmax.hh

#########################
# src/graph/clustering/ #
#########################

libgraph_tool_clusteringdir = $(MOD_DIR)/clustering
libgraph_tool_clustering_LTLIBRARIES = libgraph_tool_clustering.la
libgraph_tool_clustering_la_includedir = $(MOD_DIR)/include/clustering
libgraph_tool_clustering_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_clustering_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_clustering_la_SOURCES = \
	src/graph/clustering/graph_clustering.cc \
	src/graph/clustering/graph_extended_clustering.cc \
	src/graph/clustering/graph_motifs.cc
libgraph_tool_clustering_la_include_HEADERS = \
	src/graph/clustering/graph_clustering.hh \
	src/graph/clustering/graph_extended_clustering.hh \
	src/graph/clustering/graph_motifs.hh

###########################
# src/graph/correlations/ #
###########################

libgraph_tool_correlationsdir = $(MOD_DIR)/correlations
libgraph_tool_correlations_LTLIBRARIES = libgraph_tool_correlations.la
libgraph_tool_correlations_la_includedir = $(MOD_DIR)/include/correlations
libgraph_tool_correlations_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_correlations_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_correlations_la_SOURCES = \
	src/graph/correlations/graph_assortativity.cc \
	src/graph/correlations/graph_correlations.cc \
	src/graph/correlations/graph_correlations_imp1.cc \
	src/graph/correlations/graph_avg_correlations.cc \
	src/graph/correlations/graph_avg_correlations_imp1.cc \
	src/graph/correlations/graph_avg_correlations_combined.cc \
	src/graph/correlations/graph_correlations_combined.cc \
	src/graph/correlations/graph_correlations_bind.cc
libgraph_tool_correlations_la_include_HEADERS = \
	src/graph/correlations/graph_assortativity.hh \
	src/graph/correlations/graph_correlations.hh \
	src/graph/correlations/graph_corr_hist.hh \
	src/graph/correlations/graph_avg_correlations.hh

###################
# src/graph/draw/ #
###################

libgraph_tool_drawdir = $(MOD_DIR)/draw
libgraph_tool_draw_LTLIBRARIES = libgraph_tool_draw.la
libgraph_tool_draw_la_includedir = $(MOD_DIR)/include/draw
libgraph_tool_draw_la_CPPFLAGS = $(AM_CPPFLAGS) $(CAIROMM_CFLAGS)
libgraph_tool_draw_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_draw_la_LDFLAGS = $(MOD_LDFLAGS) $(CAIROMM_LIBS)
libgraph_tool_draw_la_SOURCES = \
	src/graph/draw/graph_cairo_draw.cc \
	src/graph/draw/graph_tree_cts.cc
libgraph_tool_draw_la_include_HEADERS =

libgt_pycairo_auxdir = $(MOD_DIR)/draw
libgt_pycairo_aux_LTLIBRARIES = libgt_pycairo_aux.la
libgt_pycairo_aux_la_includedir = $(MOD_DIR)/include/draw
libgt_pycairo_aux_la_CPPFLAGS = $(AM_CPPFLAGS) $(CAIROMM_CFLAGS)
libgt_pycairo_aux_la_LIBADD = $(MOD_LIBADD)
libgt_pycairo_aux_la_LDFLAGS = $(MOD_LDFLAGS) $(CAIROMM_LIBS)
libgt_pycairo_aux_la_SOURCES = \
	src/graph/draw/pycairo_aux.cc
libgt_pycairo_aux_la_include_HEADERS =

#######################
# src/graph/dynamics/ #
#######################

libgraph_tool_dynamicsdir = $(MOD_DIR)/dynamics
libgraph_tool_dynamics_LTLIBRARIES = libgraph_tool_dynamics.la
libgraph_tool_dynamics_la_includedir = $(MOD_DIR)/include/dynamics
libgraph_tool_dynamics_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_dynamics_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_dynamics_la_SOURCES = \
	src/graph/dynamics/graph_continuous.cc \
	src/graph/dynamics/graph_discrete.cc \
	src/graph/dynamics/graph_dynamics.cc
libgraph_tool_dynamics_la_include_HEADERS = \
	src/graph/dynamics/graph_continuous.hh \
	src/graph/dynamics/graph_discrete.hh

###################
# src/graph/flow/ #
###################

libgraph_tool_flowdir = $(MOD_DIR)/flow
libgraph_tool_flow_LTLIBRARIES = libgraph_tool_flow.la
libgraph_tool_flow_la_includedir = $(MOD_DIR)/include/flow
libgraph_tool_flow_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_flow_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_flow_la_SOURCES = \
	src/graph/flow/graph_edmonds_karp.cc \
	src/graph/flow/graph_push_relabel.cc \
	src/graph/flow/graph_kolmogorov.cc \
	src/graph/flow/graph_minimum_cut.cc \
	src/graph/flow/graph_flow_bind.cc
libgraph_tool_flow_la_include_HEADERS = \
	src/graph/flow/graph_augment.hh

#########################
# src/graph/generation/ #
#########################

libgraph_tool_generationdir = $(MOD_DIR)/generation
libgraph_tool_generation_LTLIBRARIES = libgraph_tool_generation.la
libgraph_tool_generation_la_includedir = $(MOD_DIR)/include/generation
libgraph_tool_generation_la_CPPFLAGS = $(AM_CPPFLAGS) $(CGAL_CPPFLAGS)
libgraph_tool_generation_la_LIBADD = $(MOD_LIBADD) $(CGAL_LIBADD) $(CGAL_LDFLAGS)
libgraph_tool_generation_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_generation_la_SOURCES = \
	src/graph/generation/graph_community_network.cc \
	src/graph/generation/graph_community_network_eavg.cc \
	src/graph/generation/graph_community_network_eavg_imp1.cc \
	src/graph/generation/graph_community_network_edges.cc \
	src/graph/generation/graph_community_network_vavg.cc \
	src/graph/generation/graph_complete.cc \
	src/graph/generation/graph_contract_edges.cc \
	src/graph/generation/graph_generation.cc \
	src/graph/generation/graph_geometric.cc \
	src/graph/generation/graph_knn.cc \
	src/graph/generation/graph_lattice.cc \
	src/graph/generation/graph_line_graph.cc \
	src/graph/generation/graph_maxent_sbm.cc \
	src/graph/generation/graph_predecessor.cc \
	src/graph/generation/graph_price.cc \
	src/graph/generation/graph_random_edges.cc \
	src/graph/generation/graph_rewiring.cc \
	src/graph/generation/graph_sbm.cc \
	src/graph/generation/graph_triadic_closure.cc \
	src/graph/generation/graph_triangulation.cc \
	src/graph/generation/graph_union.cc \
	src/graph/generation/graph_union_eprop.cc \
	src/graph/generation/graph_union_vprop.cc
libgraph_tool_generation_la_include_HEADERS = \
	src/graph/generation/dynamic_sampler.hh \
	src/graph/generation/graph_community_network.hh \
	src/graph/generation/graph_complete.hh \
	src/graph/generation/graph_contract_edges.hh \
	src/graph/generation/graph_generation.hh \
	src/graph/generation/graph_geometric.hh \
	src/graph/generation/graph_knn.hh \
	src/graph/generation/graph_lattice.hh \
	src/graph/generation/graph_maxent_sbm.hh \
	src/graph/generation/graph_predecessor.hh \
	src/graph/generation/graph_price.hh \
	src/graph/generation/graph_random_edges.hh \
	src/graph/generation/graph_rewiring.hh \
	src/graph/generation/graph_sbm.hh \
	src/graph/generation/graph_triadic_closure.hh \
	src/graph/generation/graph_triangulation.hh \
	src/graph/generation/graph_union.hh \
	src/graph/generation/sampler.hh \
	src/graph/generation/urn_sampler.hh

########################
# src/graph/inference/ #
########################

libgraph_tool_inferencedir = $(MOD_DIR)/inference
libgraph_tool_inference_LTLIBRARIES = libgraph_tool_inference.la
libgraph_tool_inference_la_includedir = $(MOD_DIR)/include/inference
libgraph_tool_inference_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_inference_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_inference_la_SOURCES = \
	src/graph/inference/blockmodel/graph_blockmodel.cc \
	src/graph/inference/blockmodel/graph_blockmodel_em.cc \
	src/graph/inference/blockmodel/graph_blockmodel_exhaustive.cc \
	src/graph/inference/blockmodel/graph_blockmodel_gibbs.cc \
	src/graph/inference/blockmodel/graph_blockmodel_imp.cc \
	src/graph/inference/blockmodel/graph_blockmodel_imp2.cc \
	src/graph/inference/blockmodel/graph_blockmodel_imp3.cc \
	src/graph/inference/blockmodel/graph_blockmodel_marginals.cc \
	src/graph/inference/blockmodel/graph_blockmodel_mcmc.cc \
	src/graph/inference/blockmodel/graph_blockmodel_multicanonical.cc \
	src/graph/inference/blockmodel/graph_blockmodel_multicanonical_multiflip.cc \
	src/graph/inference/blockmodel/graph_blockmodel_multiflip_mcmc.cc \
	src/graph/inference/blockmodel/graph_blockmodel_multilevel_mcmc.cc \
	src/graph/inference/cliques/graph_clique_decomposition.cc \
	src/graph/inference/histogram/graph_histogram.cc \
	src/graph/inference/histogram/graph_histogram_mcmc.cc \
	src/graph/inference/modularity/graph_modularity.cc \
	src/graph/inference/modularity/graph_modularity_gibbs.cc \
	src/graph/inference/modularity/graph_modularity_mcmc.cc \
	src/graph/inference/modularity/graph_modularity_multiflip_mcmc.cc \
	src/graph/inference/modularity/graph_modularity_multilevel_mcmc.cc \
	src/graph/inference/norm_cut/graph_norm_cut.cc \
	src/graph/inference/norm_cut/graph_norm_cut_gibbs.cc \
	src/graph/inference/norm_cut/graph_norm_cut_mcmc.cc \
	src/graph/inference/norm_cut/graph_norm_cut_multiflip_mcmc.cc \
	src/graph/inference/norm_cut/graph_norm_cut_multilevel_mcmc.cc \
	src/graph/inference/partition_centroid/graph_partition_centroid.cc \
	src/graph/inference/partition_centroid/graph_partition_centroid_mcmc.cc \
	src/graph/inference/partition_centroid/graph_partition_centroid_multiflip_mcmc.cc \
	src/graph/inference/partition_centroid/graph_partition_centroid_multilevel_mcmc.cc \
	src/graph/inference/partition_centroid/graph_partition_centroid_rmi.cc \
	src/graph/inference/partition_centroid/graph_partition_centroid_rmi_mcmc.cc \
	src/graph/inference/partition_centroid/graph_partition_centroid_rmi_multiflip_mcmc.cc \
	src/graph/inference/partition_centroid/graph_partition_centroid_rmi_multilevel_mcmc.cc \
	src/graph/inference/partition_modes/graph_partition_mode.cc \
	src/graph/inference/partition_modes/graph_partition_mode_clustering.cc \
	src/graph/inference/partition_modes/graph_partition_mode_clustering_mcmc.cc \
	src/graph/inference/partition_modes/graph_partition_mode_clustering_multiflip_mcmc.cc \
	src/graph/inference/partition_modes/graph_partition_mode_clustering_multilevel_mcmc.cc \
	src/graph/inference/planted_partition/graph_planted_partition.cc \
	src/graph/inference/planted_partition/graph_planted_partition_gibbs.cc \
	src/graph/inference/planted_partition/graph_planted_partition_mcmc.cc \
	src/graph/inference/planted_partition/graph_planted_partition_multiflip_mcmc.cc \
	src/graph/inference/planted_partition/graph_planted_partition_multilevel_mcmc.cc \
	src/graph/inference/overlap/graph_blockmodel_overlap.cc \
	src/graph/inference/overlap/graph_blockmodel_overlap_exhaustive.cc \
	src/graph/inference/overlap/graph_blockmodel_overlap_gibbs.cc \
	src/graph/inference/overlap/graph_blockmodel_overlap_mcmc.cc \
	src/graph/inference/overlap/graph_blockmodel_overlap_mcmc_bundled.cc \
	src/graph/inference/overlap/graph_blockmodel_overlap_multicanonical.cc \
	src/graph/inference/overlap/graph_blockmodel_overlap_multicanonical_multiflip.cc \
	src/graph/inference/overlap/graph_blockmodel_overlap_multiflip_mcmc.cc \
	src/graph/inference/overlap/graph_blockmodel_overlap_multilevel_mcmc.cc \
	src/graph/inference/overlap/graph_blockmodel_overlap_vacate.cc \
	src/graph/inference/layers/graph_blockmodel_layers.cc \
	src/graph/inference/layers/graph_blockmodel_layers_exhaustive.cc \
	src/graph/inference/layers/graph_blockmodel_layers_gibbs.cc \
	src/graph/inference/layers/graph_blockmodel_layers_imp.cc \
	src/graph/inference/layers/graph_blockmodel_layers_mcmc.cc \
	src/graph/inference/layers/graph_blockmodel_layers_multicanonical.cc \
	src/graph/inference/layers/graph_blockmodel_layers_multicanonical_multiflip.cc \
	src/graph/inference/layers/graph_blockmodel_layers_multiflip_mcmc.cc \
	src/graph/inference/layers/graph_blockmodel_layers_multilevel_mcmc.cc \
	src/graph/inference/layers/graph_blockmodel_layers_overlap.cc \
	src/graph/inference/layers/graph_blockmodel_layers_overlap_exhaustive.cc \
	src/graph/inference/layers/graph_blockmodel_layers_overlap_gibbs.cc \
	src/graph/inference/layers/graph_blockmodel_layers_overlap_mcmc.cc \
	src/graph/inference/layers/graph_blockmodel_layers_overlap_mcmc_bundled.cc \
	src/graph/inference/layers/graph_blockmodel_layers_overlap_multicanonical.cc \
	src/graph/inference/layers/graph_blockmodel_layers_overlap_multicanonical_multiflip.cc \
	src/graph/inference/layers/graph_blockmodel_layers_overlap_multiflip_mcmc.cc \
	src/graph/inference/layers/graph_blockmodel_layers_overlap_multilevel_mcmc.cc \
	src/graph/inference/layers/graph_blockmodel_layers_overlap_vacate.cc \
	src/graph/inference/ranked/graph_ranked.cc \
	src/graph/inference/ranked/graph_ranked_mcmc.cc \
	src/graph/inference/ranked/graph_ranked_gibbs.cc \
	src/graph/inference/ranked/graph_ranked_multiflip_mcmc.cc \
	src/graph/inference/ranked/graph_ranked_multilevel_mcmc.cc \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_epidemics.cc \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_epidemics_mcmc.cc \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_epidemics_mcmc_r.cc \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_cising_glauber.cc \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_cising_glauber_mcmc.cc \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_ising_glauber.cc \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_ising_glauber_mcmc.cc \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_pseudo_cising.cc \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_pseudo_cising_mcmc.cc \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_pseudo_cising_mcmc_h.cc \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_pseudo_ising.cc \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_pseudo_ising_mcmc.cc \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_pseudo_ising_mcmc_h.cc \
	src/graph/inference/uncertain/graph_blockmodel_latent_closure.cc \
	src/graph/inference/uncertain/graph_blockmodel_latent_closure_mcmc.cc \
	src/graph/inference/uncertain/graph_blockmodel_measured.cc \
	src/graph/inference/uncertain/graph_blockmodel_measured_mcmc.cc \
	src/graph/inference/uncertain/graph_blockmodel_uncertain.cc \
	src/graph/inference/uncertain/graph_blockmodel_uncertain_marginal.cc \
	src/graph/inference/uncertain/graph_blockmodel_uncertain_mcmc.cc \
	src/graph/inference/support/cache.cc \
	src/graph/inference/support/int_part.cc \
	src/graph/inference/support/spence.cc \
	src/graph/inference/graph_inference.cc \
	src/graph/inference/graph_latent_multigraph.cc \
	src/graph/inference/graph_modularity.cc

graph_inference_blockmodeldir = $(MOD_DIR)/include/inference/blockmodel
dist_graph_inference_blockmodel_HEADERS = \
	src/graph/inference/blockmodel/graph_blockmodel.hh \
	src/graph/inference/blockmodel/graph_blockmodel_em.hh \
	src/graph/inference/blockmodel/graph_blockmodel_emat.hh \
	src/graph/inference/blockmodel/graph_blockmodel_elist.hh \
	src/graph/inference/blockmodel/graph_blockmodel_entries.hh \
	src/graph/inference/blockmodel/graph_blockmodel_entropy.hh \
	src/graph/inference/blockmodel/graph_blockmodel_exhaustive.hh \
	src/graph/inference/blockmodel/graph_blockmodel_gibbs.hh \
	src/graph/inference/blockmodel/graph_blockmodel_mcmc.hh \
	src/graph/inference/blockmodel/graph_blockmodel_multicanonical.hh \
	src/graph/inference/blockmodel/graph_blockmodel_multiflip_mcmc.hh \
	src/graph/inference/blockmodel/graph_blockmodel_multilevel_mcmc.hh \
	src/graph/inference/blockmodel/graph_blockmodel_partition.hh \
	src/graph/inference/blockmodel/graph_blockmodel_util.hh \
	src/graph/inference/blockmodel/graph_blockmodel_weights.hh

graph_inference_cliquesdir = $(MOD_DIR)/include/inference/cliques
dist_graph_inference_cliques_HEADERS = \
	src/graph/inference/cliques/graph_clique_decomposition.hh

graph_inference_histogramdir = $(MOD_DIR)/include/inference/histogram
dist_graph_inference_histogram_HEADERS = \
	src/graph/inference/histogram/graph_histogram.hh \
	src/graph/inference/histogram/graph_histogram_mcmc.hh

graph_inference_modularitydir = $(MOD_DIR)/include/inference/modularity
dist_graph_inference_modularity_HEADERS = \
	src/graph/inference/modularity/graph_modularity.hh

graph_inference_norm_cutdir = $(MOD_DIR)/include/inference/norm_cut
dist_graph_inference_norm_cut_HEADERS = \
	src/graph/inference/norm_cut/graph_norm_cut.hh

graph_inference_partition_centroiddir = $(MOD_DIR)/include/inference/partition_centroid
dist_graph_inference_partition_centroid_HEADERS = \
	src/graph/inference/partition_centroid/graph_partition_centroid.hh \
	src/graph/inference/partition_centroid/graph_partition_centroid_rmi.hh

graph_inference_partition_modesdir = $(MOD_DIR)/include/inference/partition_modes
dist_graph_inference_partition_modes_HEADERS = \
	src/graph/inference/partition_modes/graph_partition_mode.hh \
	src/graph/inference/partition_modes/graph_partition_mode_clustering.hh

graph_inference_planted_partitiondir = $(MOD_DIR)/include/inference/planted_partition
dist_graph_inference_planted_partition_HEADERS = \
	src/graph/inference/planted_partition/graph_planted_partition.hh

graph_inference_rankeddir = $(MOD_DIR)/include/inference/ranked
dist_graph_inference_ranked_HEADERS = \
	src/graph/inference/ranked/graph_ranked.hh \
	src/graph/inference/ranked/graph_ranked_mcmc.hh \
	src/graph/inference/ranked/graph_ranked_gibbs.hh \
	src/graph/inference/ranked/graph_ranked_multiflip_mcmc.hh \
	src/graph/inference/ranked/graph_ranked_multilevel_mcmc.hh

graph_inference_overlapdir = $(MOD_DIR)/include/inference/overlap
dist_graph_inference_overlap_HEADERS = \
	src/graph/inference/overlap/graph_blockmodel_overlap.hh \
	src/graph/inference/overlap/graph_blockmodel_overlap_mcmc_bundled.hh \
	src/graph/inference/overlap/graph_blockmodel_overlap_util.hh \
	src/graph/inference/overlap/graph_blockmodel_overlap_vacate.hh \
	src/graph/inference/overlap/graph_blockmodel_overlap_partition.hh

graph_inference_layersdir = $(MOD_DIR)/include/inference/layers
dist_graph_inference_layers_HEADERS = \
	src/graph/inference/layers/graph_blockmodel_layers.hh \
	src/graph/inference/layers/graph_blockmodel_layers_util.hh

graph_inference_uncertaindir = $(MOD_DIR)/include/inference/uncertain
dist_graph_inference_uncertain_HEADERS = \
	src/graph/inference/uncertain/graph_blockmodel_dynamics.hh \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_continuous.hh \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_discrete.hh \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_epidemics_mcmc_h.hh \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_mcmc.hh \
	src/graph/inference/uncertain/graph_blockmodel_dynamics_pseudo_ising_mcmc_h.hh \
	src/graph/inference/uncertain/graph_blockmodel_latent_closure.hh \
	src/graph/inference/uncertain/graph_blockmodel_latent_layers.hh \
	src/graph/inference/uncertain/graph_blockmodel_latent_layers_mcmc.hh \
	src/graph/inference/uncertain/graph_blockmodel_measured.hh \
	src/graph/inference/uncertain/graph_blockmodel_uncertain.hh \
	src/graph/inference/uncertain/graph_blockmodel_uncertain_marginal.hh \
	src/graph/inference/uncertain/graph_blockmodel_uncertain_mcmc.hh \
	src/graph/inference/uncertain/graph_blockmodel_uncertain_util.hh \
	src/graph/inference/uncertain/graph_blockmodel_sample_edge.hh

graph_inference_loopsdir = $(MOD_DIR)/include/inference/loops
dist_graph_inference_loops_HEADERS = \
	src/graph/inference/loops/bundled_vacate_loop.hh \
	src/graph/inference/loops/exhaustive_loop.hh \
	src/graph/inference/loops/gibbs_loop.hh \
	src/graph/inference/loops/mcmc_loop.hh \
	src/graph/inference/loops/merge_split.hh \
	src/graph/inference/loops/multilevel.hh

graph_inference_supportdir = $(MOD_DIR)/include/inference/support
dist_graph_inference_support_HEADERS = \
	src/graph/inference/support/cache.hh \
	src/graph/inference/support/contingency.hh \
	src/graph/inference/support/graph_state.hh \
	src/graph/inference/support/int_part.hh \
	src/graph/inference/support/util.hh

graph_inferencedir = $(MOD_DIR)/include/inference
dist_graph_inference_HEADERS = \
	src/graph/inference/graph_modularity.hh \
	src/graph/inference/graph_latent_multigraph.hh

#####################
# src/graph/layout/ #
#####################

libgraph_tool_layoutdir = $(MOD_DIR)/draw
libgraph_tool_layout_LTLIBRARIES = libgraph_tool_layout.la
libgraph_tool_layout_la_includedir = $(MOD_DIR)/include/layout
libgraph_tool_layout_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_layout_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_layout_la_SOURCES = \
	src/graph/layout/graph_arf.cc \
	src/graph/layout/graph_planar_layout.cc \
	src/graph/layout/graph_fruchterman_reingold.cc \
	src/graph/layout/graph_sfdp.cc \
	src/graph/layout/graph_radial.cc \
	src/graph/layout/graph_bind_layout.cc
libgraph_tool_layout_la_include_HEADERS = \
	src/graph/layout/graph_arf.hh \
	src/graph/layout/graph_sfdp.hh

#####################
# src/graph/search/ #
#####################

libgraph_tool_searchdir = $(MOD_DIR)/search
libgraph_tool_search_LTLIBRARIES = libgraph_tool_search.la
libgraph_tool_search_la_includedir = $(MOD_DIR)/include/search
libgraph_tool_search_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_search_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_search_la_SOURCES = \
	src/graph/search/graph_bfs.cc \
	src/graph/search/graph_dfs.cc \
	src/graph/search/graph_dijkstra.cc \
	src/graph/search/graph_bellman_ford.cc \
	src/graph/search/graph_astar.hh \
	src/graph/search/graph_astar.cc \
	src/graph/search/graph_astar_implicit.cc \
	src/graph/search/graph_search_bind.cc
libgraph_tool_search_la_include_HEADERS =

#######################
# src/graph/spectral/ #
#######################

libgraph_tool_spectraldir = $(MOD_DIR)/spectral
libgraph_tool_spectral_LTLIBRARIES = libgraph_tool_spectral.la
libgraph_tool_spectral_la_includedir = $(MOD_DIR)/include/spectral
libgraph_tool_spectral_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_spectral_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_spectral_la_SOURCES = \
	src/graph/spectral/graph_adjacency.cc \
	src/graph/spectral/graph_incidence.cc \
	src/graph/spectral/graph_laplacian.cc \
	src/graph/spectral/graph_norm_laplacian.cc \
	src/graph/spectral/graph_matrix.cc \
	src/graph/spectral/graph_transition.cc \
	src/graph/spectral/graph_nonbacktracking.cc
libgraph_tool_spectral_la_include_HEADERS = \
	src/graph/spectral/graph_adjacency.hh \
	src/graph/spectral/graph_incidence.hh \
	src/graph/spectral/graph_laplacian.hh \
	src/graph/spectral/graph_transition.hh \
	src/graph/spectral/graph_nonbacktracking.hh

####################
# src/graph/stats/ #
####################

libgraph_tool_statsdir = $(MOD_DIR)/stats
libgraph_tool_stats_LTLIBRARIES = libgraph_tool_stats.la
libgraph_tool_stats_la_includedir = $(MOD_DIR)/include/stats
libgraph_tool_stats_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_stats_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_stats_la_SOURCES = \
	src/graph/stats/graph_histograms.cc \
	src/graph/stats/graph_average.cc \
	src/graph/stats/graph_parallel.cc \
	src/graph/stats/graph_distance.cc \
	src/graph/stats/graph_distance_sampled.cc \
	src/graph/stats/graph_stats_bind.cc
libgraph_tool_stats_la_include_HEADERS = \
	src/graph/stats/graph_parallel.hh \
	src/graph/stats/graph_histograms.hh \
	src/graph/stats/graph_average.hh \
	src/graph/stats/graph_distance_sampled.hh \
	src/graph/stats/graph_distance.hh

#######################
# src/graph/topology/ #
#######################

libgraph_tool_topologydir = $(MOD_DIR)/topology
libgraph_tool_topology_LTLIBRARIES = libgraph_tool_topology.la
libgraph_tool_topology_la_includedir = $(MOD_DIR)/include/topology
libgraph_tool_topology_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_topology_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_topology_la_SOURCES = \
	src/graph/topology/graph_all_circuits.cc \
	src/graph/topology/graph_all_distances.cc \
	src/graph/topology/graph_bipartite.cc \
	src/graph/topology/graph_components.cc \
	src/graph/topology/graph_distance.cc \
	src/graph/topology/graph_diameter.cc \
	src/graph/topology/graph_dominator_tree.cc \
	src/graph/topology/graph_isomorphism.cc \
	src/graph/topology/graph_kcore.cc \
	src/graph/topology/graph_matching.cc \
	src/graph/topology/graph_maximal_cliques.cc \
	src/graph/topology/graph_maximal_planar.cc \
	src/graph/topology/graph_maximal_vertex_set.cc \
	src/graph/topology/graph_minimum_spanning_tree.cc \
	src/graph/topology/graph_percolation.cc \
	src/graph/topology/graph_planar.cc \
	src/graph/topology/graph_random_matching.cc \
	src/graph/topology/graph_random_spanning_tree.cc \
	src/graph/topology/graph_reciprocity.cc \
	src/graph/topology/graph_sequential_color.cc \
	src/graph/topology/graph_similarity.cc \
	src/graph/topology/graph_similarity_imp.cc \
	src/graph/topology/graph_subgraph_isomorphism.cc \
	src/graph/topology/graph_topological_sort.cc \
	src/graph/topology/graph_topology.cc \
	src/graph/topology/graph_tsp.cc \
	src/graph/topology/graph_transitive_closure.cc \
	src/graph/topology/graph_vertex_similarity.cc
libgraph_tool_topology_la_include_HEADERS = \
	src/graph/topology/graph_bipartite_weighted_matching.hh \
	src/graph/topology/graph_components.hh \
	src/graph/topology/graph_kcore.hh \
	src/graph/topology/graph_maximal_cliques.hh \
	src/graph/topology/graph_percolation.hh \
	src/graph/topology/graph_similarity.hh \
	src/graph/topology/graph_vertex_similarity.hh

###################
# src/graph/util/ #
###################

libgraph_tool_utildir = $(MOD_DIR)/util
libgraph_tool_util_LTLIBRARIES = libgraph_tool_util.la
libgraph_tool_util_la_includedir = $(MOD_DIR)/include/util
libgraph_tool_util_la_LIBADD = $(MOD_LIBADD)
libgraph_tool_util_la_LDFLAGS = $(MOD_LDFLAGS)
libgraph_tool_util_la_SOURCES = \
	src/graph/util/graph_search.cc \
	src/graph/util/graph_util_bind.cc
libgraph_tool_util_la_include_HEADERS = \
	src/graph/util/graph_search.hh

###################
# src/graph_tool/ #
###################

graph_tool_PYTHON = \
	src/graph_tool/__init__.py \
	src/graph_tool/dl_import.py \
	src/graph_tool/decorators.py \
	src/graph_tool/gt_io.py \
	src/graph_tool/all.py
graph_tooldir = $(MOD_DIR)

graph_tool_generation_PYTHON = \
	src/graph_tool/generation/__init__.py
graph_tool_generationdir = $(MOD_DIR)/generation

graph_tool_collection_PYTHON = \
	src/graph_tool/collection/__init__.py \
	src/graph_tool/collection/netzschleuder.py \
	src/graph_tool/collection/small.py
dist_graph_tool_collection_DATA = \
	src/graph_tool/collection/atlas.dat.gz \
	src/graph_tool/collection/adjnoun.gt.gz \
	src/graph_tool/collection/as-22july06.gt.gz \
	src/graph_tool/collection/astro-ph.gt.gz \
	src/graph_tool/collection/celegansneural.gt.gz \
	src/graph_tool/collection/cond-mat-2003.gt.gz \
	src/graph_tool/collection/cond-mat-2005.gt.gz \
	src/graph_tool/collection/cond-mat.gt.gz \
	src/graph_tool/collection/dolphins.gt.gz \
	src/graph_tool/collection/football.gt.gz \
	src/graph_tool/collection/hep-th.gt.gz \
	src/graph_tool/collection/karate.gt.gz \
	src/graph_tool/collection/lesmis.gt.gz \
	src/graph_tool/collection/netscience.gt.gz \
	src/graph_tool/collection/polblogs.gt.gz \
	src/graph_tool/collection/polbooks.gt.gz \
	src/graph_tool/collection/power.gt.gz \
	src/graph_tool/collection/pgp-strong-2009.gt.gz \
	src/graph_tool/collection/serengeti-foodweb.gt.gz \
	src/graph_tool/collection/email-Enron.gt.gz
graph_tool_collectiondir = $(MOD_DIR)/collection

graph_tool_correlations_PYTHON = \
	src/graph_tool/correlations/__init__.py
graph_tool_correlationsdir = $(MOD_DIR)/correlations

graph_tool_stats_PYTHON = \
	src/graph_tool/stats/__init__.py
graph_tool_statsdir = $(MOD_DIR)/stats

graph_tool_clustering_PYTHON = \
	src/graph_tool/clustering/__init__.py
graph_tool_clusteringdir = $(MOD_DIR)/clustering

graph_tool_centrality_PYTHON = \
	src/graph_tool/centrality/__init__.py
graph_tool_centralitydir = $(MOD_DIR)/centrality

graph_tool_draw_PYTHON = \
	src/graph_tool/draw/__init__.py \
	src/graph_tool/draw/cairo_draw.py \
	src/graph_tool/draw/gtk_draw.py \
	src/graph_tool/draw/graphviz_draw.py
dist_graph_tool_draw_DATA = \
	src/graph_tool/draw/graph-tool-logo.svg
graph_tool_drawdir = $(MOD_DIR)/draw

graph_tool_inference_PYTHON = \
	src/graph_tool/inference/__init__.py \
	src/graph_tool/inference/base_states.py \
	src/graph_tool/inference/blockmodel.py \
	src/graph_tool/inference/blockmodel_em.py \
	src/graph_tool/inference/clique_decomposition.py \
	src/graph_tool/inference/histogram.py \
	src/graph_tool/inference/layered_blockmodel.py \
	src/graph_tool/inference/nested_blockmodel.py \
	src/graph_tool/inference/overlap_blockmodel.py \
	src/graph_tool/inference/uncertain_blockmodel.py \
	src/graph_tool/inference/mcmc.py \
	src/graph_tool/inference/minimize.py \
	src/graph_tool/inference/modularity.py \
	src/graph_tool/inference/norm_cut.py \
	src/graph_tool/inference/partition_centroid.py \
	src/graph_tool/inference/partition_modes.py \
	src/graph_tool/inference/planted_partition.py \
	src/graph_tool/inference/ranked.py \
	src/graph_tool/inference/latent_multigraph.py \
	src/graph_tool/inference/latent_layers.py \
	src/graph_tool/inference/util.py
graph_tool_inferencedir = $(MOD_DIR)/inference

graph_tool_util_PYTHON = \
	src/graph_tool/util/__init__.py
graph_tool_utildir = $(MOD_DIR)/util

graph_tool_topology_PYTHON = \
	src/graph_tool/topology/__init__.py
graph_tool_topologydir = $(MOD_DIR)/topology

graph_tool_flow_PYTHON = \
	src/graph_tool/flow/__init__.py
graph_tool_flowdir = $(MOD_DIR)/flow

graph_tool_spectral_PYTHON = \
	src/graph_tool/spectral/__init__.py
graph_tool_spectraldir = $(MOD_DIR)/spectral

graph_tool_dynamics_PYTHON = \
	src/graph_tool/dynamics/__init__.py
graph_tool_dynamicsdir = $(MOD_DIR)/dynamics

graph_tool_search_PYTHON = \
	src/graph_tool/search/__init__.py
graph_tool_searchdir = $(MOD_DIR)/search
