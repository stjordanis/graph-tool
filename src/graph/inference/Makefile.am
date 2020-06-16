## Process this file with automake to produce Makefile.in

AM_CPPFLAGS = $(MOD_CPPFLAGS)

AM_CFLAGS = $(AM_CXXFLAGS)

libgraph_tool_inferencedir = $(MOD_DIR)/inference

libgraph_tool_inference_LTLIBRARIES = libgraph_tool_inference.la

libgraph_tool_inference_la_includedir = $(MOD_DIR)/include/inference

libgraph_tool_inference_la_LIBADD = $(MOD_LIBADD)

libgraph_tool_inference_la_LDFLAGS = $(MOD_LDFLAGS)

libgraph_tool_inference_la_SOURCES = \
    blockmodel/graph_blockmodel.cc \
    blockmodel/graph_blockmodel_em.cc \
    blockmodel/graph_blockmodel_exhaustive.cc \
    blockmodel/graph_blockmodel_gibbs.cc \
    blockmodel/graph_blockmodel_imp.cc \
    blockmodel/graph_blockmodel_imp2.cc \
    blockmodel/graph_blockmodel_imp3.cc \
    blockmodel/graph_blockmodel_marginals.cc \
    blockmodel/graph_blockmodel_mcmc.cc \
    blockmodel/graph_blockmodel_merge.cc \
    blockmodel/graph_blockmodel_multicanonical.cc \
    blockmodel/graph_blockmodel_multicanonical_multiflip.cc \
    blockmodel/graph_blockmodel_multiflip_mcmc.cc \
    modularity/graph_modularity.cc \
    modularity/graph_modularity_mcmc.cc \
    modularity/graph_modularity_multiflip_mcmc.cc \
    partition_centroid/graph_partition_centroid.cc \
    partition_centroid/graph_partition_centroid_mcmc.cc \
    partition_centroid/graph_partition_centroid_multiflip_mcmc.cc \
    partition_centroid/graph_partition_centroid_rmi.cc \
    partition_centroid/graph_partition_centroid_rmi_mcmc.cc \
    partition_centroid/graph_partition_centroid_rmi_multiflip_mcmc.cc \
    partition_modes/graph_partition_mode.cc \
    partition_modes/graph_partition_mode_clustering.cc \
    partition_modes/graph_partition_mode_clustering_mcmc.cc \
    partition_modes/graph_partition_mode_clustering_multiflip_mcmc.cc \
    planted_partition/graph_planted_partition.cc \
    planted_partition/graph_planted_partition_mcmc.cc \
    planted_partition/graph_planted_partition_multiflip_mcmc.cc \
    overlap/graph_blockmodel_overlap.cc \
    overlap/graph_blockmodel_overlap_exhaustive.cc \
    overlap/graph_blockmodel_overlap_gibbs.cc \
    overlap/graph_blockmodel_overlap_mcmc.cc \
    overlap/graph_blockmodel_overlap_mcmc_bundled.cc \
    overlap/graph_blockmodel_overlap_multicanonical.cc \
    overlap/graph_blockmodel_overlap_multicanonical_multiflip.cc \
    overlap/graph_blockmodel_overlap_multiflip_mcmc.cc \
    overlap/graph_blockmodel_overlap_vacate.cc \
    layers/graph_blockmodel_layers.cc \
    layers/graph_blockmodel_layers_exhaustive.cc \
    layers/graph_blockmodel_layers_gibbs.cc \
    layers/graph_blockmodel_layers_imp.cc \
    layers/graph_blockmodel_layers_mcmc.cc \
    layers/graph_blockmodel_layers_merge.cc \
    layers/graph_blockmodel_layers_multicanonical.cc \
    layers/graph_blockmodel_layers_multicanonical_multiflip.cc \
    layers/graph_blockmodel_layers_multiflip_mcmc.cc \
    layers/graph_blockmodel_layers_overlap.cc \
    layers/graph_blockmodel_layers_overlap_exhaustive.cc \
    layers/graph_blockmodel_layers_overlap_gibbs.cc \
    layers/graph_blockmodel_layers_overlap_mcmc.cc \
    layers/graph_blockmodel_layers_overlap_mcmc_bundled.cc \
    layers/graph_blockmodel_layers_overlap_multicanonical.cc \
    layers/graph_blockmodel_layers_overlap_multicanonical_multiflip.cc \
    layers/graph_blockmodel_layers_overlap_multiflip_mcmc.cc \
    layers/graph_blockmodel_layers_overlap_vacate.cc \
    uncertain/graph_blockmodel_dynamics_epidemics.cc \
    uncertain/graph_blockmodel_dynamics_epidemics_mcmc.cc \
    uncertain/graph_blockmodel_dynamics_epidemics_mcmc_r.cc \
    uncertain/graph_blockmodel_dynamics_cising_glauber.cc \
    uncertain/graph_blockmodel_dynamics_cising_glauber_mcmc.cc \
    uncertain/graph_blockmodel_dynamics_ising_glauber.cc \
    uncertain/graph_blockmodel_dynamics_ising_glauber_mcmc.cc \
    uncertain/graph_blockmodel_dynamics_pseudo_cising.cc \
    uncertain/graph_blockmodel_dynamics_pseudo_cising_mcmc.cc \
    uncertain/graph_blockmodel_dynamics_pseudo_cising_mcmc_h.cc \
    uncertain/graph_blockmodel_dynamics_pseudo_ising.cc \
    uncertain/graph_blockmodel_dynamics_pseudo_ising_mcmc.cc \
    uncertain/graph_blockmodel_dynamics_pseudo_ising_mcmc_h.cc \
    uncertain/graph_blockmodel_measured.cc \
    uncertain/graph_blockmodel_measured_mcmc.cc \
    uncertain/graph_blockmodel_uncertain.cc \
    uncertain/graph_blockmodel_uncertain_marginal.cc \
    uncertain/graph_blockmodel_uncertain_mcmc.cc \
    support/cache.cc \
    support/int_part.cc \
    support/spence.cc \
    graph_inference.cc \
    graph_latent_multigraph.cc \
    graph_modularity.cc

nobase_libgraph_tool_inference_la_include_HEADERS = \
    blockmodel/graph_blockmodel.hh \
    blockmodel/graph_blockmodel_em.hh \
    blockmodel/graph_blockmodel_emat.hh \
    blockmodel/graph_blockmodel_elist.hh \
    blockmodel/graph_blockmodel_entries.hh \
    blockmodel/graph_blockmodel_entropy.hh \
    blockmodel/graph_blockmodel_exhaustive.hh \
    blockmodel/graph_blockmodel_gibbs.hh \
    blockmodel/graph_blockmodel_mcmc.hh \
    blockmodel/graph_blockmodel_merge.hh \
    blockmodel/graph_blockmodel_multicanonical.hh \
    blockmodel/graph_blockmodel_multiflip_mcmc.hh \
    blockmodel/graph_blockmodel_partition.hh \
    blockmodel/graph_blockmodel_util.hh \
    blockmodel/graph_blockmodel_weights.hh \
    modularity/graph_modularity.hh \
    partition_centroid/graph_partition_centroid.hh \
    partition_centroid/graph_partition_centroid_rmi.hh \
    partition_modes/graph_partition_mode.hh \
    partition_modes/graph_partition_mode_clustering.hh \
    planted_partition/graph_planted_partition.hh \
    overlap/graph_blockmodel_overlap.hh \
    overlap/graph_blockmodel_overlap_mcmc_bundled.hh \
    overlap/graph_blockmodel_overlap_util.hh \
    overlap/graph_blockmodel_overlap_vacate.hh \
    overlap/graph_blockmodel_overlap_partition.hh \
    layers/graph_blockmodel_layers.hh \
    layers/graph_blockmodel_layers_util.hh \
    uncertain/graph_blockmodel_dynamics.hh \
    uncertain/graph_blockmodel_dynamics_continuous.hh \
    uncertain/graph_blockmodel_dynamics_discrete.hh \
    uncertain/graph_blockmodel_dynamics_epidemics_mcmc_h.hh \
    uncertain/graph_blockmodel_dynamics_mcmc.hh \
    uncertain/graph_blockmodel_dynamics_pseudo_ising_mcmc_h.hh \
    uncertain/graph_blockmodel_measured.hh \
    uncertain/graph_blockmodel_uncertain.hh \
    uncertain/graph_blockmodel_uncertain_marginal.hh \
    uncertain/graph_blockmodel_uncertain_mcmc.hh \
    uncertain/graph_blockmodel_uncertain_util.hh \
    uncertain/graph_blockmodel_sample_edge.hh \
    loops/bundled_vacate_loop.hh \
    loops/exhaustive_loop.hh \
    loops/gibbs_loop.hh \
    loops/mcmc_loop.hh \
    loops/merge_loop.hh \
    support/cache.hh \
    support/graph_neighbor_sampler.hh \
    support/graph_state.hh \
    support/int_part.hh \
    support/util.hh \
    graph_modularity.hh \
    graph_latent_multigraph.hh
