.. _inference-howto:

Inferring modular network structure
===================================

``graph-tool`` includes algorithms to identify the large-scale structure
of networks in the :mod:`~graph_tool.inference` submodule. Here we
explain the basic functionality with self-contained examples. For a more
thorough theoretical introduction to the methods described here, the
reader is referred to [peixoto-bayesian-2019]_.


.. include:: _background.rst
.. include:: _minimization.rst
.. include:: _model_selection.rst
.. include:: _sampling.rst
.. include:: _model_class_selection.rst
.. include:: _edge_weights.rst
.. include:: _layers.rst
.. include:: _assortative.rst
.. include:: _reconstruction.rst
.. include:: _prediction.rst


References
----------

.. [peixoto-bayesian-2019] Tiago P. Peixoto, "Bayesian stochastic blockmodeling",
   Advances in Network Clustering and Blockmodeling, edited
   by P. Doreian, V. Batagelj, A. Ferligoj, (Wiley, New York, 2019)
   :doi:`10.1002/9781119483298.ch11`, :arxiv:`1705.10225`

.. [holland-stochastic-1983] Paul W. Holland, Kathryn Blackmond Laskey,
   Samuel Leinhardt, "Stochastic blockmodels: First steps", Social Networks
   Volume 5, Issue 2, Pages 109-137 (1983). :doi:`10.1016/0378-8733(83)90021-7`

.. [karrer-stochastic-2011] Brian Karrer, M. E. J. Newman "Stochastic
   blockmodels and community structure in networks", Phys. Rev. E 83,
   016107 (2011). :doi:`10.1103/PhysRevE.83.016107`, :arxiv:`1008.3926`
   
.. [peixoto-nonparametric-2017] Tiago P. Peixoto, "Nonparametric
   Bayesian inference of the microcanonical stochastic block model",
   Phys. Rev. E 95 012317 (2017). :doi:`10.1103/PhysRevE.95.012317`,
   :arxiv:`1610.02703`

.. [peixoto-parsimonious-2013] Tiago P. Peixoto, "Parsimonious module
   inference in large networks", Phys. Rev. Lett. 110, 148701 (2013).
   :doi:`10.1103/PhysRevLett.110.148701`, :arxiv:`1212.4794`.

.. [peixoto-hierarchical-2014] Tiago P. Peixoto, "Hierarchical block
   structures and high-resolution model selection in large networks",
   Phys. Rev. X 4, 011047 (2014). :doi:`10.1103/PhysRevX.4.011047`,
   :arxiv:`1310.4377`.

.. [peixoto-model-2016] Tiago P. Peixoto, "Model selection and hypothesis
   testing for large-scale network models with overlapping groups",
   Phys. Rev. X 5, 011033 (2016). :doi:`10.1103/PhysRevX.5.011033`,
   :arxiv:`1409.3059`.

.. [peixoto-inferring-2015] Tiago P. Peixoto, "Inferring the mesoscale
   structure of layered, edge-valued and time-varying networks",
   Phys. Rev. E 92, 042807 (2015). :doi:`10.1103/PhysRevE.92.042807`,
   :arxiv:`1504.02381`

.. [aicher-learning-2015] Christopher Aicher, Abigail Z. Jacobs, and
   Aaron Clauset, "Learning Latent Block Structure in Weighted
   Networks", Journal of Complex Networks 3(2). 221-248
   (2015). :doi:`10.1093/comnet/cnu026`, :arxiv:`1404.0431`

.. [peixoto-weighted-2017] Tiago P. Peixoto, "Nonparametric weighted
   stochastic block models", Phys. Rev. E 97, 012306 (2018),
   :doi:`10.1103/PhysRevE.97.012306`, :arxiv:`1708.01432`
          
.. [peixoto-efficient-2014] Tiago P. Peixoto, "Efficient Monte Carlo and
   greedy heuristic for the inference of stochastic block models", Phys.
   Rev. E 89, 012804 (2014). :doi:`10.1103/PhysRevE.89.012804`,
   :arxiv:`1310.4378`

.. [peixoto-merge-split-2020] Tiago P. Peixoto, "Merge-split Markov
   chain Monte Carlo for community detection", Phys. Rev. E 102, 012305
   (2020), :doi:`10.1103/PhysRevE.102.012305`, :arxiv:`2003.07070`

.. [peixoto-revealing-2020] Tiago P. Peixoto, "Revealing consensus and
   dissensus between network partitions", :arxiv:`2005.13977`

.. [lizhi-statistical-2020] Lizhi Zhang, Tiago P. Peixoto, "Statistical
   inference of assortative community structures", :arxiv:`2006.14493`

.. [peixoto-reconstructing-2018] Tiago P. Peixoto, "Reconstructing
   networks with unknown and heterogeneous errors", Phys. Rev. X 8
   041011 (2018). :doi:`10.1103/PhysRevX.8.041011`, :arxiv:`1806.07956`

.. [peixoto-network-2019] Tiago P. Peixoto, "Network reconstruction and
   community detection from dynamics", Phys. Rev. Lett. 123 128301
   (2019), :doi:`10.1103/PhysRevLett.123.128301`, :arxiv:`1903.10833`

.. [peixoto-latent-2020] Tiago P. Peixoto, "Latent Poisson models for
   networks with heterogeneous density", :arxiv:`2002.07803`

.. [martin-structural-2015] Travis Martin, Brian Ball, M. E. J. Newman,
   "Structural inference for uncertain networks", Phys. Rev. E 93,
   012306 (2016). :doi:`10.1103/PhysRevE.93.012306`, :arxiv:`1506.05490`
   
.. [clauset-hierarchical-2008] Aaron Clauset, Cristopher
   Moore, M. E. J. Newman, "Hierarchical structure and the prediction of
   missing links in networks", Nature 453, 98-101 (2008).
   :doi:`10.1038/nature06830`

.. [guimera-missing-2009] Roger Guimerà, Marta Sales-Pardo, "Missing and
   spurious interactions and the reconstruction of complex networks", PNAS
   vol. 106 no. 52 (2009). :doi:`10.1073/pnas.0908366106`
          
.. [valles-catala-consistencies-2018] Toni Vallès-Català,
   Tiago P. Peixoto, Roger Guimerà, Marta Sales-Pardo, "Consistencies
   and inconsistencies between model selection and link prediction in
   networks", Phys. Rev. E 97 062316 (2018),
   :doi:`10.1103/PhysRevE.97.062316`, :arxiv:`1705.07967`

.. [guimera-modularity-2004] Roger Guimerà, Marta Sales-Pardo, and
   Luís A. Nunes Amaral, "Modularity from fluctuations in random graphs
   and complex networks", Phys. Rev. E 70, 025101(R) (2004),
   :doi:`10.1103/PhysRevE.70.025101`

.. [hayes-connecting-2006] Brian Hayes, "Connecting the dots. can the
   tools of graph theory and social-network studies unravel the next big
   plot?", American Scientist, 94(5):400-404, 2006.
   http://www.jstor.org/stable/27858828

.. [ulanowicz-network-2005] Robert E. Ulanowicz, and
   Donald L. DeAngelis. "Network analysis of trophic dynamics in south
   florida ecosystems." US Geological Survey Program on the South
   Florida Ecosystem 114 (2005).
   https://fl.water.usgs.gov/PDF_files/ofr99_181_gerould.pdf#page=125

.. [read-cultures-1954] Kenneth E. Read, "Cultures of the Central
   Highlands, New Guinea", Southwestern J. of Anthropology,
   10(1):1-43 (1954). :doi:`10.1086/soutjanth.10.1.3629074`

.. rubric:: Footnotes

.. [#prediction_posterior] Note that the posterior of Eq. :eq:`posterior-missing`
   cannot be used to sample the reconstruction :math:`\delta \boldsymbol
   G`, as it is not informative of the overall network density
   (i.e. absolute number of missing and spurious edges). It can,
   however, be used to compare different reconstructions with the same
   density.