Model class selection
+++++++++++++++++++++

When averaging over partitions, we may be interested in evaluating which
**model class** provides a better fit of the data, considering all
possible parameter choices. This is done by evaluating the model
evidence summed over all possible partitions [peixoto-nonparametric-2017]_:

.. math::

   P(\boldsymbol A) = \sum_{\boldsymbol\theta,\boldsymbol b}P(\boldsymbol A,\boldsymbol\theta, \boldsymbol b) =  \sum_{\boldsymbol b}P(\boldsymbol A,\boldsymbol b).

This quantity is analogous to a `partition function
<https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)>`_
in statistical physics, which we can write more conveniently as a
negative `free energy
<https://en.wikipedia.org/wiki/Thermodynamic_free_energy>`_ by taking
its logarithm

.. math::
   :label: free-energy

   \ln P(\boldsymbol A) = \underbrace{\sum_{\boldsymbol b}q(\boldsymbol b)\ln P(\boldsymbol A,\boldsymbol b)}_{-\left<\Sigma\right>}\;
              \underbrace{- \sum_{\boldsymbol b}q(\boldsymbol b)\ln q(\boldsymbol b)}_{\mathcal{S}}

where

.. math::

   q(\boldsymbol b) = \frac{P(\boldsymbol A,\boldsymbol b)}{\sum_{\boldsymbol b'}P(\boldsymbol A,\boldsymbol b')}

is the posterior probability of partition :math:`\boldsymbol b`. The
first term of Eq. :eq:`free-energy` (the "negative energy") is minus the
average of description length :math:`\left<\Sigma\right>`, weighted
according to the posterior distribution. The second term
:math:`\mathcal{S}` is the `entropy
<https://en.wikipedia.org/wiki/Entropy_(information_theory)>`_ of the
posterior distribution, and measures, in a sense, the "quality of fit"
of the model: If the posterior is very "peaked", i.e. dominated by a
single partition with a very large probability, the entropy will tend to
zero. However, if there are many partitions with similar probabilities
--- meaning that there is no single partition that describes the network
uniquely well --- it will take a large value instead.

Since the MCMC algorithm samples partitions from the distribution
:math:`q(\boldsymbol b)`, it can be used to compute
:math:`\left<\Sigma\right>` easily, simply by averaging the description
length values encountered by sampling from the posterior distribution
many times.

The computation of the posterior entropy :math:`\mathcal{S}`, however,
is significantly more difficult, since it involves measuring the precise
value of :math:`q(\boldsymbol b)`. A direct "brute force" computation of
:math:`\mathcal{S}` is implemented via
:meth:`~graph_tool.inference.blockmodel.BlockState.collect_partition_histogram`
and :func:`~graph_tool.inference.blockmodel.microstate_entropy`, however
this is only feasible for very small networks. For larger networks, we
are forced to perform approximations. One possibility is to employ the
method described in [peixoto-revealing-2021]_, based on fitting a
mixture "random label" model to the posterior distribution, which allows
us to compute its entropy. In graph-tool this is done by using
:class:`~graph_tool.inference.partition_modes.ModeClusterState`, as we
show in the example below.

.. testsetup:: model-evidence

   np.random.seed(43)            
   gt.seed_rng(43)

.. testcode:: model-evidence

   from scipy.special import gammaln

   g = gt.collection.data["lesmis"]

   for deg_corr in [True, False]:
       state = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=deg_corr))  # Initialize the Markov
                                                                                 # chain from the "ground
                                                                                 # state"
       dls = []         # description length history
       bs = []          # partitions

       def collect_partitions(s):
           global bs, dls
           bs.append(s.get_state().a.copy())
           dls.append(s.entropy())

       # Now we collect 2000 partitions; but the larger this is, the
       # more accurate will be the calculation
       
       gt.mcmc_equilibrate(state, force_niter=2000, mcmc_args=dict(niter=10),
                           callback=collect_partitions)

       # Infer partition modes
       pmode = gt.ModeClusterState(bs)

       # Minimize the mode state itself
       gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))

       # Posterior entropy
       H = pmode.posterior_entropy()

       # log(B!) term
       logB = mean(gammaln(np.array([len(np.unique(b)) for b in bs]) + 1))

       # Evidence
       L = -mean(dls) + logB + H
                           
       print(f"Model log-evidence for deg_corr = {deg_corr}: {L}")

.. testoutput:: model-evidence

   Model log-evidence for deg_corr = True: -678.633834...
   Model log-evidence for deg_corr = False: -670.190510...

The outcome shows a preference for the non-degree-corrected model.

When using the nested model, the approach is entirely analogous. We show below the
approach for the same network, using the nested model.

.. testsetup:: nested-model-evidence

   np.random.seed(43)            
   gt.seed_rng(42)

.. testcode:: nested-model-evidence

   from scipy.special import gammaln

   g = gt.collection.data["lesmis"]

   for deg_corr in [True, False]:
       state = gt.NestedBlockState(g, state_args=dict(deg_corr=deg_corr))

       # Equilibrate
       gt.mcmc_equilibrate(state, force_niter=1000, mcmc_args=dict(niter=10))
       
       dls = []         # description length history
       bs = []          # partitions

       def collect_partitions(s):
           global bs, dls
           bs.append(s.get_state())
           dls.append(s.entropy())

       # Now we collect 2000 partitions; but the larger this is, the
       # more accurate will be the calculation
       
       gt.mcmc_equilibrate(state, force_niter=2000, mcmc_args=dict(niter=10),
                           callback=collect_partitions)

       # Infer partition modes
       pmode = gt.ModeClusterState(bs, nested=True)

       # Minimize the mode state itself
       gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))

       # Posterior entropy
       H = pmode.posterior_entropy()

       # log(B!) term
       logB = mean([sum(gammaln(len(np.unique(bl))+1) for bl in b) for b in bs])

       # Evidence
       L = -mean(dls) + logB + H
                           
       print(f"Model log-evidence for deg_corr = {deg_corr}: {L}")

.. testoutput:: nested-model-evidence

   Model log-evidence for deg_corr = True: -666.147684...
   Model log-evidence for deg_corr = False: -657.426243...

The results are similar: The non-degree-corrected model possesses the
largest evidence. Note also that we observe a better evidence for the
nested models themselves, when comparing to the evidences for the
non-nested model --- which is not quite surprising, since the non-nested
model is a special case of the nested one.
