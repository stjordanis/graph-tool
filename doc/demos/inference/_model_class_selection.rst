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
:meth:`~graph_tool.inference.blockmodel.BlockState.collect_partition_histogram` and
:func:`~graph_tool.inference.blockmodel.microstate_entropy`, however this is only
feasible for very small networks. For larger networks, we are forced to
perform approximations. The simplest is a "mean field" one, where we
assume the posterior factorizes as

.. math::

   q(\boldsymbol b) \approx \prod_i{q_i(b_i)}

where

.. math::

   q_i(r) = P(b_i = r | \boldsymbol A)

is the marginal group membership distribution of node :math:`i`. This
yields an entropy value given by

.. math::

   S \approx -\sum_i\sum_rq_i(r)\ln q_i(r).

This approximation should be seen as an upper bound, since any existing
correlation between the nodes (which are ignored here) will yield
smaller entropy values.

A more accurate assumption is called the `Bethe approximation`
[mezard-information-2009]_, and takes into account the correlation
between adjacent nodes in the network,

.. math::

   q(\boldsymbol b) \approx \prod_{i<j}q_{ij}(b_i,b_j)^{A_{ij}}\prod_iq_i(b_i)^{1-k_i}

where :math:`A_{ij}` is the `adjacency matrix
<https://en.wikipedia.org/wiki/Adjacency_matrix>`_, :math:`k_i` is the
degree of node :math:`i`, and

.. math::

   q_{ij}(r, s) = P(b_i = r, b_j = s|\boldsymbol A)

is the joint group membership distribution of nodes :math:`i` and
:math:`j` (a.k.a. the `edge marginals`). This yields an entropy value
given by

.. math::

   S \approx -\sum_{i<j}A_{ij}\sum_{rs}q_{ij}(r,s)\ln q_{ij}(r,s) - \sum_i(1-k_i)\sum_rq_i(r)\ln q_i(r).

Typically, this approximation yields smaller values than the mean field
one, and is generally considered to be superior. However, formally, it
depends on the graph being sufficiently locally "tree-like", and the
posterior being indeed strongly correlated with the adjacency matrix
itself --- two characteristics which do not hold in general. Although
the approximation often gives reasonable results even when these
conditions do not strictly hold, in some situations when they are
strongly violated this approach can yield meaningless values, such as a
negative entropy. Therefore, it is useful to compare both approaches
whenever possible.

With these approximations, it possible to estimate the full model
evidence efficiently, as we show below, using
:meth:`~graph_tool.inference.blockmodel.BlockState.collect_vertex_marginals`,
:meth:`~graph_tool.inference.blockmodel.BlockState.collect_edge_marginals`,
:meth:`~graph_tool.inference.blockmodel.mf_entropy` and
:meth:`~graph_tool.inference.blockmodel.bethe_entropy`.

.. testcode:: model-evidence

   g = gt.collection.data["lesmis"]

   for deg_corr in [True, False]:
       state = gt.minimize_blockmodel_dl(g, deg_corr=deg_corr)     # Initialize the Markov
                                                                   # chain from the "ground
                                                                   # state"
       state = state.copy(B=g.num_vertices())

       dls = []         # description length history
       vm = None        # vertex marginals
       em = None        # edge marginals

       def collect_marginals(s):
           global vm, em
           vm = s.collect_vertex_marginals(vm)
           em = s.collect_edge_marginals(em)
           dls.append(s.entropy())

       # Now we collect the marginal distributions for exactly 200,000 sweeps
       gt.mcmc_equilibrate(state, force_niter=20000, mcmc_args=dict(niter=10),
                           callback=collect_marginals)

       S_mf = gt.mf_entropy(g, vm)
       S_bethe = gt.bethe_entropy(g, em)[0]
       L = -mean(dls)

       print("Model evidence for deg_corr = %s:" % deg_corr,
             L + S_mf, "(mean field),", L + S_bethe, "(Bethe)")

.. testoutput:: model-evidence

   Model evidence for deg_corr = True: -383.787042... (mean field), -1053.691486... (Bethe)
   Model evidence for deg_corr = False: -363.454006... (mean field), -948.144381... (Bethe)

If we consider the more accurate approximation, the outcome shows a
preference for the non-degree-corrected model.

When using the nested model, the approach is entirely analogous. The
only difference now is that we have a hierarchical partition
:math:`\{\boldsymbol b_l\}` in the equations above, instead of simply
:math:`\boldsymbol b`. In order to make the approach tractable, we
assume the factorization

.. math::

   q(\{\boldsymbol b_l\}) \approx \prod_lq_l(\boldsymbol b_l)

where :math:`q_l(\boldsymbol b_l)` is the marginal posterior for the
partition at level :math:`l`. For :math:`q_0(\boldsymbol b_0)` we may
use again either the mean-field or Bethe approximations, however for
:math:`l>0` only the mean-field approximation is applicable, since the
adjacency matrix of the higher layers is not constant. We show below the
approach for the same network, using the nested model.


.. testcode:: model-evidence

   g = gt.collection.data["lesmis"]

   nL = 10

   for deg_corr in [True, False]:
       state = gt.minimize_nested_blockmodel_dl(g, deg_corr=deg_corr)     # Initialize the Markov
                                                                          # chain from the "ground
                                                                          # state"
       bs = state.get_bs()                     # Get hierarchical partition.
       bs += [np.zeros(1)] * (nL - len(bs))    # Augment it to L = 10 with
                                               # single-group levels.

       state = state.copy(bs=bs, sampling=True)

       dls = []                               # description length history
       vm = [None] * len(state.get_levels())  # vertex marginals
       em = None                              # edge marginals

       def collect_marginals(s):
           global vm, em
           levels = s.get_levels()
           vm = [sl.collect_vertex_marginals(vm[l]) for l, sl in enumerate(levels)]
           em = levels[0].collect_edge_marginals(em)
           dls.append(s.entropy())

       # Now we collect the marginal distributions for exactly 200,000 sweeps
       gt.mcmc_equilibrate(state, force_niter=20000, mcmc_args=dict(niter=10),
                           callback=collect_marginals)

       S_mf = [gt.mf_entropy(sl.g, vm[l]) for l, sl in enumerate(state.get_levels())]
       S_bethe = gt.bethe_entropy(g, em)[0]
       L = -mean(dls)

       print("Model evidence for deg_corr = %s:" % deg_corr,
             L + sum(S_mf), "(mean field),", L + S_bethe + sum(S_mf[1:]), "(Bethe)")


.. testoutput:: model-evidence

   Model evidence for deg_corr = True: -183.999497... (mean field), -840.108164... (Bethe)
   Model evidence for deg_corr = False: -187.344952... (mean field), -738.797625... (Bethe)

The results are similar: If we consider the most accurate approximation,
the non-degree-corrected model possesses the largest evidence. Note also
that we observe a better evidence for the nested models themselves, when
comparing to the evidences for the non-nested model --- which is not
quite surprising, since the non-nested model is a special case of the
nested one.
