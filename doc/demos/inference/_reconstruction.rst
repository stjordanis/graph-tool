Network reconstruction
----------------------

An important application of generative models is to be able to
generalize from observations and make predictions that go beyond what is
seen in the data. This is particularly useful when the network we
observe is incomplete, or contains errors, i.e. some of the edges are
either missing or are outcomes of mistakes in measurement, or is not
even observed at all. In this situation, we can use statistical
inference to reconstruct the original network. Following
[peixoto-reconstructing-2018]_, if :math:`\boldsymbol{\mathcal{D}}` is
the observed data, the network can be reconstructed according to the
posterior distribution,

.. math::
   :label: posterior-reconstruction

   P(\boldsymbol A, \boldsymbol b | \boldsymbol{\mathcal{D}}) =
   \frac{P(\boldsymbol{\mathcal{D}} | \boldsymbol A)P(\boldsymbol A, \boldsymbol b)}{P(\boldsymbol{\mathcal{D}})}

where the likelihood :math:`P(\boldsymbol{\mathcal{D}}|\boldsymbol A)`
models the measurement process, and for the prior :math:`P(\boldsymbol
A, \boldsymbol b)` we use the SBM as before. This means that when
performing reconstruction, we sample both the community structure
:math:`\boldsymbol b` and the network :math:`\boldsymbol A` itself from
the posterior distribution. From it, we can obtain the marginal probability
of each edge,

.. math::

   \pi_{ij} = \sum_{\boldsymbol A, \boldsymbol b}A_{ij}P(\boldsymbol A, \boldsymbol b | \boldsymbol{\mathcal{D}}).

Based on the marginal posterior probabilities, the best estimate for the
whole underlying network :math:`\boldsymbol{\hat{A}}` is given by the
maximum of this distribution,

.. math::

   \hat A_{ij} =
       \begin{cases}
           1 & \text{ if } \pi_{ij} > \frac{1}{2},\\
           0 & \text{ if } \pi_{ij} < \frac{1}{2}.\\
       \end{cases}

We can also make estimates :math:`\hat y` of arbitrary scalar network
properties :math:`y(\boldsymbol A)` via posterior averages,
      
.. math::
   \begin{align}
       \hat y &= \sum_{\boldsymbol A, \boldsymbol b}y(\boldsymbol A)P(\boldsymbol A, \boldsymbol b | \boldsymbol{\mathcal{D}}),\\
       \sigma^2_y &= \sum_{\boldsymbol A, \boldsymbol b}(y(\boldsymbol A)-\hat y)^2P(\boldsymbol A, \boldsymbol b | \boldsymbol{\mathcal{D}})
   \end{align}

with uncertainty given by :math:`\sigma_y`. This is gives us a complete
probabilistic reconstruction framework that fully reflects both the
information and the uncertainty in the measurement data. Furthermore,
the use of the SBM means that the reconstruction can take advantage of
the *correlations* observed in the data to further inform it, which
generally can lead to substantial improvements
[peixoto-reconstructing-2018]_ [peixoto-network-2019]_.
       
In graph-tool there is support for reconstruction with the above
framework for three measurement processes: 1. Repeated measurements with
uniform errors (via
:class:`~graph_tool.inference.uncertain_blockmodel.MeasuredBlockState`), 2. Repeated
measurements with heterogeneous errors (via
:class:`~graph_tool.inference.uncertain_blockmodel.MixedMeasuredBlockState`),
and 3. Extraneously obtained edge probabilities (via
:class:`~graph_tool.inference.uncertain_blockmodel.UncertainBlockState`),
which we describe in the following.

In addition, it is also possible to reconstruct networks from observed
dynamical, as described in :ref:`reconstruction_dynamics`.

Measured networks
+++++++++++++++++

This model assumes that the node pairs :math:`(i,j)` were measured
:math:`n_{ij}` times, and an edge has been recorded :math:`x_{ij}`
times, where a missing edge occurs with probability :math:`p` and a
spurious edge occurs with probability :math:`q`, uniformly for all node
pairs, yielding a likelihood

.. math::

   P(\boldsymbol x | \boldsymbol n, \boldsymbol A, p, q) =
   \prod_{i<j}{n_{ij}\choose x_{ij}}\left[(1-p)^{x_{ij}}p^{n_{ij}-x_{ij}}\right]^{A_{ij}}
   \left[q^{x_{ij}}(1-q)^{n_{ij}-x_{ij}}\right]^{1-A_{ij}}.

In general, :math:`p` and :math:`q` are not precisely known *a priori*,
so we consider the integrated likelihood

.. math::

   P(\boldsymbol x | \boldsymbol n, \boldsymbol A, \alpha,\beta,\mu,\nu) =
   \int P(\boldsymbol x | \boldsymbol n, \boldsymbol A, p, q) P(p|\alpha,\beta) P(q|\mu,\nu)\;\mathrm{d}p\,\mathrm{d}q

where :math:`P(p|\alpha,\beta)` and :math:`P(q|\mu,\nu)` are `Beta
distributions <https://en.wikipedia.org/wiki/Beta_distribution>`__, which
specify the amount of prior knowledge we have on the noise
parameters. An important special case, which is the default unless
otherwise specified, is when we are completely agnostic *a priori* about
the noise magnitudes, and all hyperparameters are unity,

.. math::

   P(\boldsymbol x | \boldsymbol n, \boldsymbol A) \equiv
   P(\boldsymbol x | \boldsymbol n, \boldsymbol A, \alpha=1,\beta=1,\mu=1,\nu=1).

In this situation the priors :math:`P(p|\alpha=1,\beta=1)` and
:math:`P(q|\mu=1,\nu=1)` are uniform distribution in the interval :math:`[0,1]`.

.. note::

   Since this approach also makes use of the *correlations* between
   edges to inform the reconstruction, as described by the inferred SBM,
   this means it can also be used when only single measurements have
   been performed, :math:`n_{ij}=1`, and the error magnitudes :math:`p`
   and :math:`q` are unknown. Since every arbitrary adjacency matrix can
   be cast in this setting, this method can be used to reconstruct
   networks for which no error assessments of any kind have been
   provided.

Below, we illustrate how the reconstruction can be performed with a
simple example, using
:class:`~graph_tool.inference.uncertain_blockmodel.MeasuredBlockState`:
      
.. testsetup:: measured

   import os
   try:
      os.chdir("demos/inference")
   except FileNotFoundError:
       pass
   np.random.seed(43)
   gt.seed_rng(45)

.. testcode:: measured

   g = gt.collection.data["lesmis"].copy()

   # pretend we have measured and observed each edge twice

   n = g.new_ep("int", 2)   # number of measurements
   x = g.new_ep("int", 2)   # number of observations

   e = g.edge(11, 36)
   x[e] = 1                 # pretend we have observed edge (11, 36) only once

   e = g.add_edge(15, 73)
   n[e] = 2                 # pretend we have measured non-edge (15, 73) twice,
   x[e] = 1                 # but observed it as an edge once.

   # We inititialize MeasuredBlockState, assuming that each non-edge has
   # been measured only once (as opposed to twice for the observed
   # edges), as specified by the 'n_default' and 'x_default' parameters.

   state = gt.MeasuredBlockState(g, n=n, n_default=1, x=x, x_default=0)

   # We will first equilibrate the Markov chain
   gt.mcmc_equilibrate(state, wait=1000, mcmc_args=dict(niter=10))

   # Now we collect the marginals for exactly 100,000 sweeps, at
   # intervals of 10 sweeps:

   u = None              # marginal posterior edge probabilities
   bs = []               # partitions
   cs = []               # average local clustering coefficient

   def collect_marginals(s):
      global u, bs, cs
      u = s.collect_marginal(u)
      bstate = s.get_block_state()
      bs.append(bstate.levels[0].b.a.copy())
      cs.append(gt.local_clustering(s.get_graph()).fa.mean())

   gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                       callback=collect_marginals)

   eprob = u.ep.eprob
   print("Posterior probability of edge (11, 36):", eprob[u.edge(11, 36)])
   print("Posterior probability of non-edge (15, 73):", eprob[u.edge(15, 73)])
   print("Estimated average local clustering: %g ± %g" % (np.mean(cs), np.std(cs)))


Which yields the following output:
   
.. testoutput:: measured

   Posterior probability of edge (11, 36): 0.768976...
   Posterior probability of non-edge (15, 73): 0.039203...
   Estimated average local clustering: 0.571939 ± 0.003534...

We have a successful reconstruction, where both ambiguous adjacency
matrix entries are correctly recovered. The value for the average
clustering coefficient is also correctly estimated, and is compatible
with the true value :math:`0.57313675`, within the estimated error.

Below we visualize the maximum marginal posterior estimate of the
reconstructed network:

.. testcode:: measured

   # The maximum marginal posterior estimator can be obtained by
   # filtering the edges with probability larger than .5

   u = gt.GraphView(u, efilt=u.ep.eprob.fa > .5)

   # Mark the recovered true edges as red, and the removed spurious edges as green
   ecolor = u.new_ep("vector<double>", val=[0, 0, 0, .6])
   for e in u.edges():
       if g.edge(e.source(), e.target()) is None or (e.source(), e.target()) == (11, 36):
           ecolor[e] = [1, 0, 0, .6]
   for e in g.edges():
       if u.edge(e.source(), e.target()) is None:
           ne = u.add_edge(e.source(), e.target())
           ecolor[ne] = [0, 1, 0, .6]

   # Duplicate the internal block state with the reconstructed network
   # u, for visualization purposes.

   bstate = state.get_block_state()
   bstate = bstate.levels[0].copy(g=u)

   # Disambiguate partitions and obtain marginals
   pmode = gt.PartitionModeState(bs, converge=True)
   pv = pmode.get_marginal(u)
   
   edash = u.new_ep("vector<double>")
   edash[u.edge(15, 73)] = [.1, .1, 0]
   bstate.draw(pos=u.own_property(g.vp.pos), vertex_shape="pie", vertex_pie_fractions=pv,
               edge_color=ecolor, edge_dash_style=edash, edge_gradient=None,
               output="lesmis-reconstruction-marginals.svg")

.. figure:: lesmis-reconstruction-marginals.*
   :align: center
   :width: 450px

   Reconstructed network of characters in the novel Les Misérables,
   assuming that each edge has been measured and recorded twice, and
   each non-edge has been measured only once, with the exception of edge
   (11, 36), shown in red, and non-edge (15, 73), shown in green, which
   have been measured twice and recorded as an edge once. Despite the
   ambiguity, both errors are successfully corrected by the
   reconstruction. The pie fractions on the nodes correspond to the
   probability of being in group associated with the respective color.

Heterogeneous errors
^^^^^^^^^^^^^^^^^^^^

In a more general scenario the measurement errors can be different for
each node pair, i.e. :math:`p_{ij}` and :math:`q_{ij}` are the missing
and spurious edge probabilities for node pair :math:`(i,j)`. The
measurement likelihood then becomes

.. math::

   P(\boldsymbol x | \boldsymbol n, \boldsymbol A, \boldsymbol p, \boldsymbol q) =
   \prod_{i<j}{n_{ij}\choose x_{ij}}\left[(1-p_{ij})^{x_{ij}}p_{ij}^{n_{ij}-x_{ij}}\right]^{A_{ij}}
   \left[q_{ij}^{x_{ij}}(1-q_{ij})^{n_{ij}-x_{ij}}\right]^{1-A_{ij}}.


Since the noise magnitudes are *a priori* unknown, we consider the
integrated likelihood

.. math::

   P(\boldsymbol x | \boldsymbol n, \boldsymbol A, \alpha,\beta,\mu,\nu) =
   \prod_{i<j}\int P(x_{ij} | n_{ij}, A_{ij}, p_{ij}, q_{ij}) P(p_{ij}|\alpha,\beta) P(q_{ij}|\mu,\nu)\;\mathrm{d}p_{ij}\,\mathrm{d}q_{ij}

where :math:`P(p_{ij}|\alpha,\beta)` and :math:`P(q_{ij}|\mu,\nu)` are
`Beta prior distributions
<https://en.wikipedia.org/wiki/Beta_distribution>`__, like
before. Instead of pre-specifying the hyperparameters, we include them
from the posterior distribution

.. math::

   P(\boldsymbol A, \boldsymbol b, \alpha,\beta,\mu,\nu | \boldsymbol x, \boldsymbol n) =
   \frac{P(\boldsymbol x | \boldsymbol n, \boldsymbol A, \alpha,\beta,\mu,\nu)P(\boldsymbol A, \boldsymbol b)P(\alpha,\beta,\mu,\nu)}{P(\boldsymbol x| \boldsymbol n)},

where :math:`P(\alpha,\beta,\mu,\nu)\propto 1` is a uniform hyperprior.

Operationally, the inference with this model works similarly to the one
with uniform error rates, as we see with the same example:

.. testcode:: measured

   state = gt.MixedMeasuredBlockState(g, n=n, n_default=1, x=x, x_default=0)

   # We will first equilibrate the Markov chain
   gt.mcmc_equilibrate(state, wait=1000, mcmc_args=dict(niter=10))

   # Now we collect the marginals for exactly 100,000 sweeps, at
   # intervals of 10 sweeps:

   u = None              # marginal posterior edge probabilities
   bs = []               # partitions
   cs = []               # average local clustering coefficient

   gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                       callback=collect_marginals)

   eprob = u.ep.eprob
   print("Posterior probability of edge (11, 36):", eprob[u.edge(11, 36)])
   print("Posterior probability of non-edge (15, 73):", eprob[u.edge(15, 73)])
   print("Estimated average local clustering: %g ± %g" % (np.mean(cs), np.std(cs)))

Which yields:
   
.. testoutput:: measured

   Posterior probability of edge (11, 36): 0.631563...
   Posterior probability of non-edge (15, 73): 0.022402...
   Estimated average local clustering: 0.570065 ± 0.007145...

The results are very similar to the ones obtained with the uniform model
in this case, but can be quite different in situations where a large
number of measurements has been performed (see
[peixoto-reconstructing-2018]_ for details).
    
Extraneous error estimates
++++++++++++++++++++++++++

In some situations the edge uncertainties are estimated by means other
than repeated measurements, using domain-specific models. Here we
consider the general case where the error estimates are extraneously
provided as independent edge probabilities :math:`\boldsymbol Q`,

.. math::

   P_Q(\boldsymbol A | \boldsymbol Q) = \prod_{i<j}Q_{ij}^{A_{ij}}(1-Q_{ij})^{1-A_{ij}},

where :math:`Q_{ij}` is the estimated probability of edge
:math:`(i,j)`. Although in principle we could reconstruct networks
directly from the above distribution, we can also incorporate it with
SBM inference to take advantage of large-scale structures present in the
data. We do so by employing Bayes' rule to extract the noise model from
the provided values [martin-structural-2015]_
[peixoto-reconstructing-2018]_,

.. math::

   \begin{align}
       P_Q(\boldsymbol Q | \boldsymbol A) &= \frac{P_Q(\boldsymbol A | \boldsymbol Q)P_Q(\boldsymbol Q)}{P_Q(\boldsymbol A)},\\
       & = P_Q(\boldsymbol Q) \prod_{i<j} \left(\frac{Q_{ij}}{\bar Q}\right)^{A_{ij}}\left(\frac{1-Q_{ij}}{1-\bar Q}\right)^{1-A_{ij}},
   \end{align}

where :math:`\bar Q = \sum_{i<j}Q_{ij}/{N\choose 2}` is the estimated
network density, and :math:`P_Q(\boldsymbol Q)` is an unknown prior for
:math:`\boldsymbol Q`, which can remain unspecified as it has no effect
on the posterior distribution. With the above, we can reconstruct the
network based on the posterior distribution,

.. math::

   P(\boldsymbol A, \boldsymbol b | \boldsymbol Q) = \frac{P_Q(\boldsymbol Q | \boldsymbol A)P(\boldsymbol A, \boldsymbol b)}{P(\boldsymbol Q)}

where :math:`P(\boldsymbol A, \boldsymbol b)` is the joint SBM
distribution used before. Note that this reconstruction will be
different from the one obtained directly from the original estimation, i.e.

.. math::

   P(\boldsymbol A | \boldsymbol Q) = \sum_{\boldsymbol b}P(\boldsymbol A, \boldsymbol b | \boldsymbol Q) \neq P_Q(\boldsymbol A | \boldsymbol Q).

This is because the posterior :math:`P(\boldsymbol A | \boldsymbol Q)`
will take into consideration the correlations found in the data, as
captured by the inferred SBM structure, as further evidence for the
existence and non-existence of edges. We illustrate this with an example
similar to the one considered previously, where two adjacency matrix
entries with the same ambiguous edge probability :math:`Q_{ij}=1/2` are
correctly reconstructed as edge and non-edge, due to the joint SBM
inference:
   
.. testsetup:: uncertain

   import os
   try:
      os.chdir("demos/inference")
   except FileNotFoundError:
       pass
   np.random.seed(48)
   gt.seed_rng(48)


.. testcode:: uncertain

   g = gt.collection.data["lesmis"].copy()

   N = g.num_vertices()
   E = g.num_edges()

   q = g.new_ep("double", .98)   # edge uncertainties

   e = g.edge(11, 36)
   q[e] = .5                     # ambiguous true edge

   e = g.add_edge(15, 73)
   q[e] = .5                     # ambiguous spurious edge
   
   # We inititialize UncertainBlockState, assuming that each non-edge
   # has an uncertainty of q_default, chosen to preserve the expected
   # density of the original network:

   q_default = (E - q.a.sum()) / ((N * (N - 1))/2 - E)
   
   state = gt.UncertainBlockState(g, q=q, q_default=q_default)

   # We will first equilibrate the Markov chain
   gt.mcmc_equilibrate(state, wait=2000, mcmc_args=dict(niter=10))

   # Now we collect the marginals for exactly 100,000 sweeps, at
   # intervals of 10 sweeps:

   u = None              # marginal posterior edge probabilities
   bs = []               # partitions
   cs = []               # average local clustering coefficient
   
   def collect_marginals(s):
      global bs, u, cs
      u = s.collect_marginal(u)
      bstate = s.get_block_state()
      bs.append(bstate.levels[0].b.a.copy())
      cs.append(gt.local_clustering(s.get_graph()).fa.mean())

   gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                       callback=collect_marginals)

   eprob = u.ep.eprob
   print("Posterior probability of edge (11, 36):", eprob[u.edge(11, 36)])
   print("Posterior probability of non-edge (15, 73):", eprob[u.edge(15, 73)])
   print("Estimated average local clustering: %g ± %g" % (np.mean(cs), np.std(cs)))

The above yields the output:
   
.. testoutput:: uncertain

   Posterior probability of edge (11, 36): 0.775777...
   Posterior probability of non-edge (15, 73): 0.010001...
   Estimated average local clustering: 0.523013 ± 0.017268...

The reconstruction is accurate, despite the two ambiguous entries having
the same measurement probability. The reconstructed network is visualized below.
    
.. testcode:: uncertain

   # The maximum marginal posterior estimator can be obtained by
   # filtering the edges with probability larger than .5

   u = gt.GraphView(u, efilt=u.ep.eprob.fa > .5)
                       
   # Mark the recovered true edges as red, and the removed spurious edges as green
   ecolor = u.new_ep("vector<double>", val=[0, 0, 0, .6])
   edash = u.new_ep("vector<double>")
   for e in u.edges():
       if g.edge(e.source(), e.target()) is None or (e.source(), e.target()) == (11, 36):
           ecolor[e] = [1, 0, 0, .6]
       
   for e in g.edges():
       if u.edge(e.source(), e.target()) is None:
           ne = u.add_edge(e.source(), e.target())
           ecolor[ne] = [0, 1, 0, .6]
           if (e.source(), e.target()) == (15, 73):
               edash[ne] = [.1, .1, 0]

   bstate = state.get_block_state()
   bstate = bstate.levels[0].copy(g=u)

   # Disambiguate partitions and obtain marginals
   pmode = gt.PartitionModeState(bs, converge=True)
   pv = pmode.get_marginal(u)

   bstate.draw(pos=u.own_property(g.vp.pos), vertex_shape="pie", vertex_pie_fractions=pv,
               edge_color=ecolor, edge_dash_style=edash, edge_gradient=None,
               output="lesmis-uncertain-reconstruction-marginals.svg")

.. figure:: lesmis-uncertain-reconstruction-marginals.*
   :align: center
   :width: 450px

   Reconstructed network of characters in the novel Les Misérables,
   assuming that each edge as a measurement probability of
   :math:`.98`. Edge (11, 36), shown in red, and non-edge (15, 73),
   shown in green, both have probability :math:`0.5`. Despite the
   ambiguity, both errors are successfully corrected by the
   reconstruction. The pie fractions on the nodes correspond to the
   probability of being in group associated with the respective color.


Latent Poisson multigraphs
++++++++++++++++++++++++++

Even in situations where measurement errors can be neglected, it can
still be useful to assume a given network is the outcome of a "hidden"
multigraph model, i.e. more than one edge between nodes is allowed, but
then its multiedges are "erased" by transforming them into simple
edges. In this way, it is possible to construct generative models that
can better handle situations where the underlying network possesses
heterogeneous density, such as strong community structure and broad
degree distributions [peixoto-latent-2020]_. This can be incorporated
into the scheme of Eq. :eq:`posterior-reconstruction` by considering
the data to be the observed simple graph,
:math:`\boldsymbol{\mathcal{D}} = \boldsymbol G`.  We proceed in same
way as in the previous reconstruction scenarios, but using instead
:class:`~graph_tool.inference.uncertain_blockmodel.LatentMultigraphBlockState`.

For example, in the following we will obtain the community structure and
latent multiedges of a network of political books:

.. testcode:: erased-poisson

   g = gt.collection.data["polbooks"]

   state = gt.LatentMultigraphBlockState(g)

   # We will first equilibrate the Markov chain
   gt.mcmc_equilibrate(state, wait=100, mcmc_args=dict(niter=10))

   # Now we collect the marginals for exactly 100,000 sweeps, at
   # intervals of 10 sweeps:

   u = None              # marginal posterior multigraph
   bs = []               # partitions
   
   def collect_marginals(s):
      global bs, u
      u = s.collect_marginal_multigraph(u)
      bstate = state.get_block_state()
      bs.append(bstate.levels[0].b.a.copy())

   gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                       callback=collect_marginals)

   # compute average multiplicities

   ew = u.new_ep("double")
   w = u.ep.w
   wcount = u.ep.wcount
   for e in u.edges():
       ew[e] = (wcount[e].a * w[e].a).sum() / wcount[e].a.sum()
   
   bstate = state.get_block_state()
   bstate = bstate.levels[0].copy(g=u)

   # Disambiguate partitions and obtain marginals
   pmode = gt.PartitionModeState(bs, converge=True)
   pv = pmode.get_marginal(u)

   bstate.draw(pos=u.own_property(g.vp.pos), vertex_shape="pie", vertex_pie_fractions=pv,
               edge_pen_width=gt.prop_to_size(ew, .1, 8, power=1), edge_gradient=None,
               output="polbooks-erased-poisson.svg")
   
.. figure:: polbooks-erased-poisson.*
   :align: center
   :width: 450px

   Reconstructed latent Poisson degree-corrected SBM for a network of
   political books, showing the marginal mean edge multiplicities as
   line thickness.  The pie fractions on the nodes correspond to the
   probability of being in group associated with the respective color.

.. include:: _reconstruction_dynamics.rst
