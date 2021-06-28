.. _sampling:

Sampling from the posterior distribution
----------------------------------------

When analyzing empirical networks, one should be open to the possibility
that there will be more than one fit of the SBM with similar posterior
probabilities. In such situations, one should instead `sample`
partitions from the posterior distribution, instead of simply finding
its maximum. One can then compute quantities that are averaged over the
different model fits, weighted according to their posterior
probabilities.

Full support for model averaging is implemented in ``graph-tool`` via an
efficient `Markov chain Monte Carlo (MCMC)
<https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo>`_ algorithm
[peixoto-efficient-2014]_. It works by attempting to move nodes into
different groups with specific probabilities, and `accepting or
rejecting
<https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm>`_
such moves so that, after a sufficiently long time, the partitions will
be observed with the desired posterior probability. The algorithm is
designed so that its run-time (i.e. each sweep of the MCMC) is linear on
the number of edges in the network, and independent on the number of
groups being used in the model, and hence is suitable for use on very
large networks.

In order to perform such moves, one needs again to operate with
:class:`~graph_tool.inference.blockmodel.BlockState` or
:class:`~graph_tool.inference.nested_blockmodel.NestedBlockState`
instances, and calling either their
:meth:`~graph_tool.inference.blockmodel.BlockState.mcmc_sweep` or
:meth:`~graph_tool.inference.blockmodel.BlockState.multiflip_mcmc_sweep`
methods. The former implements a simpler MCMC where a single node is
moved at a time, where the latter is a more efficient version that
performs merges and splits [peixoto-merge-split-2020]_, which should be
in general preferred.

For example, the following will perform 1000 sweeps of the algorithm
with the network of characters in the novel Les Misérables, starting
from a random partition into 20 groups

.. testcode:: model-averaging

   g = gt.collection.data["lesmis"]

   state = gt.BlockState(g)   # This automatically initializes the state with a partition
                              # into one group. The user could also pass a higher number
                              # to start with a random partition of a given size, or pass a
                              # specific initial partition using the 'b' parameter.

   # Now we run 1,000 sweeps of the MCMC. Note that the number of groups
   # is allowed to change, so it will eventually move from the initial
   # value of B=1 to whatever is most appropriate for the data.

   dS, nattempts, nmoves = state.multiflip_mcmc_sweep(niter=1000)

   print("Change in description length:", dS)
   print("Number of accepted vertex moves:", nmoves)

.. testoutput:: model-averaging

   Change in description length: -77.095673...
   Number of accepted vertex moves: 128045

Although the above is sufficient to implement sampling from the
posterior, there is a convenience function called
:func:`~graph_tool.inference.mcmc.mcmc_equilibrate` that is intend to
simplify the detection of equilibration, by keeping track of the maximum
and minimum values of description length encountered and how many sweeps
have been made without a "record breaking" event. For example,

.. testcode:: model-averaging

   # We will accept equilibration if 10 sweeps are completed without a
   # record breaking event, 2 consecutive times.

   gt.mcmc_equilibrate(state, wait=10, nbreaks=2, mcmc_args=dict(niter=10))

Note that the value of ``wait`` above was made purposefully low so that
the output would not be overly long. The most appropriate value requires
experimentation, but a typically good value is ``wait=1000``.

The function :func:`~graph_tool.inference.mcmc.mcmc_equilibrate` accepts
a ``callback`` argument that takes an optional function to be invoked
after each call to
:meth:`~graph_tool.inference.blockmodel.BlockState.multiflip_mcmc_sweep`. This
function should accept a single parameter which will contain the actual
:class:`~graph_tool.inference.blockmodel.BlockState` instance. We will
use this in the example below to collect the posterior vertex marginals
(via :class:`~graph_tool.inference.partition_modes.PartitionModeState`,
which disambiguates group labels [peixoto-revealing-2021]_), i.e. the
posterior probability that a node belongs to a given group:

.. testcode:: model-averaging

   # We will first equilibrate the Markov chain
   gt.mcmc_equilibrate(state, wait=1000, mcmc_args=dict(niter=10))

   bs = [] # collect some partitions

   def collect_partitions(s):
      global bs
      bs.append(s.b.a.copy())

   # Now we collect partitions for exactly 100,000 sweeps, at intervals
   # of 10 sweeps:
   gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                       callback=collect_partitions)

   # Disambiguate partitions and obtain marginals
   pmode = gt.PartitionModeState(bs, converge=True)
   pv = pmode.get_marginal(g)
                       
   # Now the node marginals are stored in property map pv. We can
   # visualize them as pie charts on the nodes:
   state.draw(pos=g.vp.pos, vertex_shape="pie", vertex_pie_fractions=pv,
              output="lesmis-sbm-marginals.svg")

.. figure:: lesmis-sbm-marginals.*
   :align: center
   :width: 450px

   Marginal probabilities of group memberships of the network of
   characters in the novel Les Misérables, according to the
   degree-corrected SBM. The `pie fractions
   <https://en.wikipedia.org/wiki/Pie_chart>`_ on the nodes correspond
   to the probability of being in group associated with the respective
   color.

We can also obtain a marginal probability on the number of groups
itself, as follows.

.. testcode:: model-averaging

   h = np.zeros(g.num_vertices() + 1)

   def collect_num_groups(s):
       B = s.get_nonempty_B()
       h[B] += 1

   # Now we collect partitions for exactly 100,000 sweeps, at intervals
   # of 10 sweeps:
   gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                       callback=collect_num_groups)

.. testcode:: model-averaging
   :hide:

   figure()
   Bs = np.arange(len(h))
   idx = h > 0
   bar(Bs[idx], h[idx] / h.sum(), width=1, color="#ccb974")
   gca().set_xticks([6,7,8,9])
   xlabel("$B$")
   ylabel(r"$P(B|\boldsymbol A)$")
   savefig("lesmis-B-posterior.svg")

.. figure:: lesmis-B-posterior.*
   :align: center

   Marginal posterior probability of the number of nonempty groups for
   the network of characters in the novel Les Misérables, according to
   the degree-corrected SBM.


Hierarchical partitions
+++++++++++++++++++++++

We can also perform model averaging using the nested SBM, which will
give us a distribution over hierarchies. The whole procedure is fairly
analogous, but now we make use of
:class:`~graph_tool.inference.nested_blockmodel.NestedBlockState` instances.

Here we perform the sampling of hierarchical partitions using the same
network as above.

.. testcode:: nested-model-averaging

   g = gt.collection.data["lesmis"]

   state = gt.NestedBlockState(g)   # By default this creates a state with an initial single-group
                                    # hierarchy of depth ceil(log2(g.num_vertices()).

   # Now we run 1000 sweeps of the MCMC

   dS, nmoves = 0, 0
   for i in range(100):
       ret = state.multiflip_mcmc_sweep(niter=10)
       dS += ret[0]
       nmoves += ret[1]

   print("Change in description length:", dS)
   print("Number of accepted vertex moves:", nmoves)

.. testoutput:: nested-model-averaging

   Change in description length: -73.911066...
   Number of accepted vertex moves: 435339

.. warning::

   When using
   :class:`~graph_tool.inference.nested_blockmodel.NestedBlockState`, a
   single call to
   :meth:`~graph_tool.inference.nested_blockmodel.NestedBlockState.multiflip_mcmc_sweep`
   or
   :meth:`~graph_tool.inference.nested_blockmodel.NestedBlockState.mcmc_sweep`
   performs ``niter`` sweeps at each hierarchical level once. This means
   that in order for the chain to equilibrate, we need to call these
   functions several times, i.e. it is not enough to call it once with a
   large value of ``niter``.
   
Similarly to the the non-nested case, we can use
:func:`~graph_tool.inference.mcmc.mcmc_equilibrate` to do most of the boring
work, and we can now obtain vertex marginals on all hierarchical levels:

.. testcode:: nested-model-averaging

   # We will first equilibrate the Markov chain
   gt.mcmc_equilibrate(state, wait=1000, mcmc_args=dict(niter=10))

   # collect nested partitions
   bs = []

   def collect_partitions(s):
      global bs
      bs.append(s.get_bs())

   # Now we collect the marginals for exactly 100,000 sweeps
   gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                       callback=collect_partitions)

   # Disambiguate partitions and obtain marginals
   pmode = gt.PartitionModeState(bs, nested=True, converge=True)
   pv = pmode.get_marginal(g)

   # Get consensus estimate
   bs = pmode.get_max_nested()

   state = state.copy(bs=bs)

   # We can visualize the marginals as pie charts on the nodes:
   state.draw(vertex_shape="pie", vertex_pie_fractions=pv,
              output="lesmis-nested-sbm-marginals.svg")

.. figure:: lesmis-nested-sbm-marginals.*
   :align: center
   :width: 450px

   Marginal probabilities of group memberships of the network of
   characters in the novel Les Misérables, according to the nested
   degree-corrected SBM. The pie fractions on the nodes correspond to
   the probability of being in group associated with the respective
   color.

We can also obtain a marginal probability of the number of groups
itself, as follows.

.. testcode:: nested-model-averaging

   h = [np.zeros(g.num_vertices() + 1) for s in state.get_levels()]

   def collect_num_groups(s):
       for l, sl in enumerate(s.get_levels()):
          B = sl.get_nonempty_B()
          h[l][B] += 1

   # Now we collect the marginal distribution for exactly 100,000 sweeps
   gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                       callback=collect_num_groups)

.. testcode:: nested-model-averaging
   :hide:

   figure()
   f, ax = plt.subplots(1, 5, figsize=(10, 3))
   for i, h_ in enumerate(h[:5]):
       Bs = np.arange(len(h_))
       idx = h_ > 0
       ax[i].bar(Bs[idx], h_[idx] / h_.sum(), width=1, color="#ccb974")
       ax[i].set_xticks(Bs[idx])
       ax[i].set_xlabel("$B_{%d}$" % i)
       ax[i].set_ylabel(r"$P(B_{%d}|\boldsymbol A)$" % i)
       locator = MaxNLocator(prune='both', nbins=5)
       ax[i].yaxis.set_major_locator(locator)
   tight_layout()
   savefig("lesmis-nested-B-posterior.svg")

.. figure:: lesmis-nested-B-posterior.*
   :align: center

   Marginal posterior probability of the number of nonempty groups
   :math:`B_l` at each hierarchy level :math:`l` for the network of
   characters in the novel Les Misérables, according to the nested
   degree-corrected SBM.

Below we obtain some hierarchical partitions sampled from the posterior
distribution.

.. testcode:: nested-model-averaging

   for i in range(10):
       for j in range(100):
           state.multiflip_mcmc_sweep(niter=10)
       state.draw(output="lesmis-partition-sample-%i.svg" % i, empty_branches=False)

.. image:: lesmis-partition-sample-0.svg
   :width: 19%
.. image:: lesmis-partition-sample-1.svg
   :width: 19%
.. image:: lesmis-partition-sample-2.svg
   :width: 19%
.. image:: lesmis-partition-sample-3.svg
   :width: 19%
.. image:: lesmis-partition-sample-4.svg
   :width: 19%
.. image:: lesmis-partition-sample-5.svg
   :width: 19%
.. image:: lesmis-partition-sample-6.svg
   :width: 19%
.. image:: lesmis-partition-sample-7.svg
   :width: 19%
.. image:: lesmis-partition-sample-8.svg
   :width: 19%
.. image:: lesmis-partition-sample-9.svg
   :width: 19%

Characterizing the posterior distribution
+++++++++++++++++++++++++++++++++++++++++

The posterior distribution of partitions can have an elaborate
structure, containing multiple possible explanations for the data. In
order to summarize it, we can infer the modes of the distribution using
:class:`~graph_tool.inference.partition_modes.ModeClusterState`, as
described in [peixoto-revealing-2021]_. This amounts to identifying
clusters of partitions that are very similar to each other, but
sufficiently different from those that belong to other
clusters. Collective, such "modes" represent the different stories that
the data is telling us through the model. Here is an example using again
the Les Misérables network:

.. testcode:: partition-modes

   g = gt.collection.data["lesmis"]

   state = gt.NestedBlockState(g)

   # Equilibration
   gt.mcmc_equilibrate(state, force_niter=1000, mcmc_args=dict(niter=10))

   bs = []
   
   def collect_partitions(s):
      global bs
      bs.append(s.get_bs())

   # We will collect only partitions 1000 partitions. For more accurate
   # results, this number should be increased.
   gt.mcmc_equilibrate(state, force_niter=1000, mcmc_args=dict(niter=10),
                       callback=collect_partitions)

   # Infer partition modes
   pmode = gt.ModeClusterState(bs, nested=True)

   # Minimize the mode state itself
   gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))

   # Get inferred modes
   modes = pmode.get_modes()

   for i, mode in enumerate(modes):
       b = mode.get_max_nested()    # mode's maximum
       pv = mode.get_marginal(g)    # mode's marginal distribution

       print(f"Mode {i} with size {mode.get_M()/len(bs)}")
       state = state.copy(bs=b)
       state.draw(vertex_shape="pie", vertex_pie_fractions=pv,
                  output="lesmis-partition-mode-%i.svg" % i)

Running the above code gives us the relative size of each mode,
corresponding to their collective posterior probability.

.. testoutput:: partition-modes

    Mode 0 with size 0.493493...
    Mode 1 with size 0.352352...
    Mode 2 with size 0.121121...
    Mode 3 with size 0.033033...
                  
Below are the marginal node distributions representing the partitions that belong to each inferred mode:
       
.. image:: lesmis-partition-mode-0.svg
   :width: 19%
.. image:: lesmis-partition-mode-1.svg
   :width: 19%
.. image:: lesmis-partition-mode-2.svg
   :width: 19%
.. image:: lesmis-partition-mode-3.svg
   :width: 19%
.. image:: lesmis-partition-mode-4.svg
   :width: 19%
