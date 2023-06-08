Edge prediction as binary classification
++++++++++++++++++++++++++++++++++++++++

A more traditional approach to the prediction of missing and spurious
edges formulates it as a supervised `binary classification task
<https://en.wikipedia.org/wiki/Binary_classification>`__, where the
edge/non-edge scores are computed by fitting a generative model to the
observed data, and computing their probabilities under that model
[clauset-hierarchical-2008]_ [guimera-missing-2009]_. In this setting,
one typically omits any explicit model of the measurement process (hence
intrinsically assuming it to be uniform), and as a consequence of the
overall setup, only *relative probabilities* between individual missing
and spurious edges can be produced, instead of the full posterior
distribution considered in the last section. Since this limits the
overall network reconstruction, and does not yield confidence
intervals, it is a less powerful approach. Nevertheless, it is a popular
procedure, which can also be performed with graph-tool, as we describe
in the following.

We set up the classification task by dividing the edges/non-edges into
two sets :math:`\boldsymbol A` and :math:`\delta \boldsymbol A`, where
the former corresponds to the observed network and the latter either to
the missing or spurious edges. We may compute the posterior of
:math:`\delta \boldsymbol A` as [valles-catala-consistencies-2018]_

.. math::
   :label: posterior-missing

   P(\delta \boldsymbol A | \boldsymbol A) \propto
   \sum_{\boldsymbol b}\frac{P(\boldsymbol A \cup \delta\boldsymbol A| \boldsymbol b)}{P(\boldsymbol A| \boldsymbol b)}P(\boldsymbol b | \boldsymbol A)

up to a normalization constant [#prediction_posterior]_. Although the
normalization constant is difficult to obtain in general (since we need
to perform a sum over all possible spurious/missing edges), the
numerator of Eq. :eq:`posterior-missing` can be computed by sampling
partitions from the posterior, and then inserting or deleting edges from
the graph and computing the new likelihood. This means that we can
easily compare alternative predictive hypotheses :math:`\{\delta
\boldsymbol A_i\}` via their likelihood ratios

.. math::

   \lambda_i = \frac{P(\delta \boldsymbol A_i | \boldsymbol A)}{\sum_j P(\delta \boldsymbol A_j | \boldsymbol A)}

which do not depend on the normalization constant.

The values :math:`P(\delta \boldsymbol A | \boldsymbol A, \boldsymbol b)`
can be computed with
:meth:`~graph_tool.inference.BlockState.get_edges_prob`. Hence, we can
compute spurious/missing edge probabilities just as if we were
collecting marginal distributions when doing model averaging.

Below is an example for predicting the two following edges in the
football network, using the nested model (for which we need to replace
:math:`\boldsymbol b` by :math:`\{\boldsymbol b_l\}` in the equations
above).

.. testcode:: missing-edges
   :hide:

   import os
   try:
      os.chdir("demos/inference")
   except FileNotFoundError:
       pass

   g = gt.collection.data["football"].copy()
   color = g.new_vp("string", val="#cccccc")
   ecolor = g.new_ep("string", val="#cccccc")
   ewidth = g.new_ep("double", 1)
   e = g.add_edge(101, 102)
   ecolor[e] = "#a40000"
   ewidth[e] = 5
   e = g.add_edge(17, 56)
   ecolor[e] = "#a40000"
   ewidth[e] = 5
   eorder = g.edge_index.copy("int")

   gt.graph_draw(g, pos=g.vp.pos, vertex_color=color,
                 vertex_fill_color=color, edge_color=ecolor,
                 eorder=eorder, edge_pen_width=ewidth,
                 output="football_missing.svg")

.. figure:: football_missing.*
   :align: center
   :width: 350px

   Two non-existing edges in the football network (in red):
   :math:`(101,102)` in the middle, and :math:`(17,56)` in the upper
   right region of the figure.

.. testsetup:: missing-edges

   gt.seed_rng(7)

.. testcode:: missing-edges

   g = gt.collection.data["football"]

   missing_edges = [(101, 102), (17, 56)]
   
   L = 10

   state = gt.minimize_nested_blockmodel_dl(g)

   probs = ([], [])

   def collect_edge_probs(s):
       p1 = s.get_edges_prob([missing_edges[0]], entropy_args=dict(partition_dl=False))
       p2 = s.get_edges_prob([missing_edges[1]], entropy_args=dict(partition_dl=False))
       probs[0].append(p1)
       probs[1].append(p2)

   # Now we collect the probabilities for exactly 100,000 sweeps
   gt.mcmc_equilibrate(state, force_niter=10000, mcmc_args=dict(niter=10),
                       callback=collect_edge_probs)


   def get_avg(p):
      p = np.array(p)
      pmax = p.max()
      p -= pmax
      return pmax + log(exp(p).mean())

   p1 = get_avg(probs[0])
   p2 = get_avg(probs[1])

   p_sum = get_avg([p1, p2]) + log(2)
   
   l1 = p1 - p_sum
   l2 = p2 - p_sum

   print("likelihood-ratio for %s: %g" % (missing_edges[0], exp(l1)))
   print("likelihood-ratio for %s: %g" % (missing_edges[1], exp(l2)))


.. testoutput:: missing-edges

   likelihood-ratio for (101, 102): 0.0203792
   likelihood-ratio for (17, 56): 0.979621

From which we can conclude that edge :math:`(17, 56)` is more likely
than :math:`(101, 102)` to be a missing edge.

The prediction using the non-nested model can be performed in an
entirely analogous fashion.
