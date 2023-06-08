.. _weights:

Edge weights and covariates
---------------------------

Very often networks cannot be completely represented by simple graphs,
but instead have arbitrary "weights" :math:`x_{ij}` on the edges. Edge
weights can be continuous or discrete numbers, and either strictly
positive or positive or negative, depending on context. The SBM can be
extended to cover these cases by treating edge weights as covariates
that are sampled from some distribution conditioned on the node
partition [aicher-learning-2015]_ [peixoto-weighted-2017]_, i.e.

.. math::

   P(\boldsymbol x,\boldsymbol A|\boldsymbol b) =
   P(\boldsymbol x|\boldsymbol A,\boldsymbol b) P(\boldsymbol A|\boldsymbol b),

where :math:`P(\boldsymbol A|\boldsymbol b)` is the likelihood of the
unweighted SBM described previously, and :math:`P(\boldsymbol
x|\boldsymbol A,\boldsymbol b)` is the integrated likelihood of the edge
weights

.. math::

   P(\boldsymbol x|\boldsymbol A,\boldsymbol b) =
   \prod_{r\le s}\int P({\boldsymbol x}_{rs}|\gamma)P(\gamma)\,\mathrm{d}\gamma,

where :math:`P({\boldsymbol x}_{rs}|\gamma)` is some model for the weights
:math:`{\boldsymbol x}_{rs}` between groups :math:`(r,s)`, conditioned on
some parameter :math:`\gamma`, sampled from its prior
:math:`P(\gamma)`. A hierarchical version of the model can also be
implemented by replacing this prior by a nested sequence of priors and
hyperpriors, as described in [peixoto-weighted-2017]_. The posterior
partition distribution is then simply

.. math::

   P(\boldsymbol b | \boldsymbol A,\boldsymbol x) =
   \frac{P(\boldsymbol x|\boldsymbol A,\boldsymbol b) P(\boldsymbol A|\boldsymbol b)
         P(\boldsymbol b)}{P(\boldsymbol A,\boldsymbol x)},

which can be sampled from, or maximized, just like with the unweighted
case, but will use the information on the weights to guide the partitions.

A variety of weight models is supported, reflecting different kinds of
edge covariates:

.. csv-table::
   :header: "Name", "Domain", "Bounds", "Shape"
   :widths: 10, 5, 5, 5
   :delim: |
   :align: center

   ``"real-exponential"``   | Real    :math:`(\mathbb{R})` | :math:`[0,\infty]`       | `Exponential <https://en.wikipedia.org/wiki/Exponential_distribution>`_
   ``"real-normal"``        | Real    :math:`(\mathbb{R})` | :math:`[-\infty,\infty]` | `Normal <https://en.wikipedia.org/wiki/Normal_distribution>`_
   ``"discrete-geometric"`` | Natural :math:`(\mathbb{N})` | :math:`[0,\infty]`       | `Geometric <https://en.wikipedia.org/wiki/Geometric_distribution>`_
   ``"discrete-binomial"``  | Natural :math:`(\mathbb{N})` | :math:`[0,M]`            | `Binomial <https://en.wikipedia.org/wiki/Binomial_distribution>`_
   ``"discrete-poisson"``   | Natural :math:`(\mathbb{N})` | :math:`[0,\infty]`       | `Poisson <https://en.wikipedia.org/wiki/Poisson_distribution>`_

In fact, the actual model implements `microcanonical
<https://en.wikipedia.org/wiki/Microcanonical_ensemble>`_ versions of
these distributions that are asymptotically equivalent, as described in
[peixoto-weighted-2017]_. These can be combined with arbitrary weight
transformations to achieve a large family of associated
distributions. For example, to use a `log-normal
<https://en.wikipedia.org/wiki/Log-normal_distribution>`_ weight model
for positive real weights :math:`\boldsymbol x`, we can use the
transformation :math:`y_{ij} = \ln x_{ij}` together with the
``"real-normal"`` model for :math:`\boldsymbol y`. To model weights that
are positive or negative integers in :math:`\mathbb{Z}`, we could either
subtract the minimum value, :math:`y_{ij} = x_{ij} - x^*`, with
:math:`x^*=\operatorname{min}_{ij}x_{ij}`, and use any of the above
models for non-negative integers in :math:`\mathbb{N}`, or
alternatively, consider the sign as an additional covariate,
i.e. :math:`s_{ij} = [\operatorname{sign}(x_{ij})+1]/2 \in \{0,1\}`,
using the Binomial distribution with :math:`M=1` (a.k.a. the `Bernoulli
distribution <https://en.wikipedia.org/wiki/Bernoulli_distribution>`_),
and any of the other discrete distributions for the magnitude,
:math:`y_{ij} = \operatorname{abs}(x_{ij})`.
   
The support for weighted networks is activated by passing the parameters
``recs`` and ``rec_types`` to
:class:`~graph_tool.inference.BlockState` (or
:class:`~graph_tool.inference.OverlapBlockState`),
that specify the edge covariates (an edge
:class:`~graph_tool.PropertyMap`) and their types (a string from the
table above), respectively. Note that these parameters expect *lists*,
so that multiple edge weights can be used simultaneously.

For example, let us consider a network of suspected terrorists involved
in the train bombing of Madrid on March 11, 2004
[hayes-connecting-2006]_. An edge indicates that a connection between
the two persons have been identified, and the weight of the edge (an
integer in the range :math:`[0,3]`) indicates the "strength" of the
connection. We can apply the weighted SBM, using a Binomial model for
the weights, as follows:


.. testsetup:: weighted-model

   import os
   try:
       os.chdir("demos/inference")
   except FileNotFoundError:
       pass
   gt.seed_rng(42)
         
.. testcode:: weighted-model

   g = gt.collection.ns["train_terrorists"]

   # This network contains an internal edge property map with name
   # "weight" that contains the strength of interactions. The values
   # integers in the range [0, 3].
   
   state = gt.minimize_nested_blockmodel_dl(g, state_args=dict(recs=[g.ep.weight],
                                                               rec_types=["discrete-binomial"]))

   # improve solution with merge-split

   for i in range(100):
       ret = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)

   state.draw(edge_color=g.ep.weight, ecmap=(matplotlib.cm.inferno, .6),
              eorder=g.ep.weight, edge_pen_width=gt.prop_to_size(g.ep.weight, 2, 8, power=1),
              edge_gradient=[], output="moreno-train-wsbm.pdf")

.. testcleanup:: weighted-model

   conv_png("moreno-train-wsbm.pdf")

.. figure:: moreno-train-wsbm.png
   :align: center
   :width: 450px

   Best fit of the Binomial-weighted degree-corrected SBM for a network
   of terror suspects, using the strength of connection as edge
   covariates. The edge colors and widths correspond to the strengths.

Model selection
+++++++++++++++

In order to select the best weighted model, we proceed in the same
manner as described in Sec. :ref:`model_selection`. However, when using
transformations on continuous weights, we must include the associated
scaling of the probability density, as described in
[peixoto-weighted-2017]_.

For example, consider a `food web
<https://en.wikipedia.org/wiki/Food_web>`_ between species in south
Florida [ulanowicz-network-2005]_. A directed link exists from species
:math:`i` to :math:`j` if a energy flow exists between them, and a
weight :math:`x_{ij}` on this edge indicates the magnitude of the energy
flow (a positive real value, i.e. :math:`x_{ij}\in [0,\infty]`). One
possibility, therefore, is to use the ``"real-exponential"`` model, as
follows:

.. testsetup:: food-web

   import os
   try:
       os.chdir("demos/inference")
   except FileNotFoundError:
       pass
   gt.seed_rng(44)
         
.. testcode:: food-web

   g = gt.collection.ns["foodweb_baywet"]

   # This network contains an internal edge property map with name
   # "weight" that contains the energy flow between species. The values
   # are continuous in the range [0, infinity].
   
   state = gt.minimize_nested_blockmodel_dl(g, state_args=dict(recs=[g.ep.weight],
                                                               rec_types=["real-exponential"]))

   # improve solution with merge-split

   for i in range(100):
       ret = state.multiflip_mcmc_sweep(niter=10, beta=np.inf)


   state.draw(edge_color=gt.prop_to_size(g.ep.weight, power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
              eorder=g.ep.weight, edge_pen_width=gt.prop_to_size(g.ep.weight, 1, 4, power=1, log=True),
              edge_gradient=[], output="foodweb-wsbm.pdf")

.. testcleanup:: food-web

   conv_png("foodweb-wsbm.pdf")
              
.. figure:: foodweb-wsbm.png
   :align: center
   :width: 450px

   Best fit of the exponential-weighted degree-corrected SBM for a food
   web, using the energy flow as edge covariates (indicated by the edge
   colors and widths).

Alternatively, we may consider a transformation of the type

.. math::
   :label: log_transform

   y_{ij} = \ln x_{ij}

so that :math:`y_{ij} \in [-\infty,\infty]`. If we use a model
``"real-normal"`` for :math:`\boldsymbol y`, it amounts to a `log-normal
<https://en.wikipedia.org/wiki/Log-normal_distribution>`_ model for
:math:`\boldsymbol x`. This can be a better choice if the weights are
distributed across many orders of magnitude, or show multi-modality. We
can fit this alternative model simply by using the transformed weights:

.. testcode:: food-web

   # Apply the weight transformation
   y = g.ep.weight.copy()
   y.a = log(y.a)
   
   state_ln = gt.minimize_nested_blockmodel_dl(g, state_args=dict(recs=[y],
                                                                  rec_types=["real-normal"]))

   # improve solution with merge-split

   for i in range(100):
       ret = state_ln.multiflip_mcmc_sweep(niter=10, beta=np.inf)

   state_ln.draw(edge_color=gt.prop_to_size(g.ep.weight, power=1, log=True), ecmap=(matplotlib.cm.inferno, .6),
                 eorder=g.ep.weight, edge_pen_width=gt.prop_to_size(g.ep.weight, 1, 4, power=1, log=True),
                 edge_gradient=[], output="foodweb-wsbm-lognormal.pdf")

.. testcleanup:: food-web

   conv_png("foodweb-wsbm-lognormal.pdf")
                 
.. figure:: foodweb-wsbm-lognormal.png
   :align: center
   :width: 450px

   Best fit of the log-normal-weighted degree-corrected SBM for a food
   web, using the energy flow as edge covariates (indicated by the edge
   colors and widths).

At this point, we ask ourselves which of the above models yields the
best fit of the data. This is answered by performing model selection via
posterior odds ratios just like in Sec. :ref:`model_selection`. However,
here we need to take into account the scaling of the probability density
incurred by the variable transformation, i.e.

.. math::

    P(\boldsymbol x | \boldsymbol A, \boldsymbol b) =
    P(\boldsymbol y(\boldsymbol x) | \boldsymbol A, \boldsymbol b)
    \prod_{ij}\left[\frac{\mathrm{d}y_{ij}}{\mathrm{d}x_{ij}}(x_{ij})\right]^{A_{ij}}.

In the particular case of Eq. :eq:`log_transform`, we have

.. math::

    \prod_{ij}\left[\frac{\mathrm{d}y_{ij}}{\mathrm{d}x_{ij}}(x_{ij})\right]^{A_{ij}}
    = \prod_{ij}\frac{1}{x_{ij}^{A_{ij}}}.

Therefore, we can compute the posterior odds ratio between both models as:

.. testcode:: food-web

   L1 = -state.entropy()
   L2 = -state_ln.entropy() - log(g.ep.weight.a).sum()
              
   print(u"ln \u039b: ", L2 - L1)

.. testoutput:: food-web
   :options: +NORMALIZE_WHITESPACE

   ln Λ:  -20.635724...

A value of :math:`\Lambda \approx \mathrm{e}^{83} \approx 10^{36}` in
favor the exponential model indicates that the log-normal model does not
provide a better fit for this particular data.
   
   
Posterior sampling
++++++++++++++++++
   
The procedure to sample from the posterior distribution is identical to
what is described in Sec. :ref:`sampling`, but with the appropriate
initialization, e.g..

.. testcode:: weighted-model

   g = gt.collection.ns["foodweb_baywet"]

   state = gt.NestedBlockState(g, state_args=dict(recs=[g.ep.weight], rec_types=["real-exponential"]))

   gt.mcmc_equilibrate(state, force_niter=100, mcmc_args=dict(niter=10))
