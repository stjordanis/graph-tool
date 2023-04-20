Welcome to graph-tool's documentation!
======================================

``graph-tool`` is an efficient Python module for manipulation and statistical
analysis of `graphs <https://en.wikipedia.org/wiki/Graph#Mathematics>`__ (a.k.a.
`networks <https://en.wikipedia.org/wiki/Network_theory>`__).

The :mod:`graph_tool` module provides a :class:`~graph_tool.Graph` class and
several algorithms that operate on it. The internals of this class, and of most
algorithms, are written in C++ for performance, using `template metaprogramming
<https://en.wikipedia.org/wiki/Template_metaprogramming>`_ for code
specialization, and the `Boost Graph Library <http://www.boost.org>`_.

``graph-tool`` can be `orders of magnitude faster
<https://graph-tool.skewed.de/performance>`_ than Python-only alternatives, and
therefore it is specially suited for large-scale network analysis.

Besides superior performance, ``graph-tool`` contains the following set of
functionalities which are currently not available in most other comparable
packages:

1. Comprehensive framework for :ref:`inferential community detection
   <inference-howto>`, build upon statistically principled approaches that avoid
   overfitting and are interpretable. (See `here
   <https://skewed.de/tiago/blog/modularity-harmful>`__ as well as
   [descriptive-vs-inferential]_ for why you should avoid off-the-shelf methods
   available in other software packages.)

2. Support for `OpenMP <https://en.wikipedia.org/wiki/OpenMP>`_ shared memory
   parallelism for several algorithms.

3. High-quality :ref:`network visualization <draw>`, both static and
   interactive, supporting :ref:`animations <animation>` and :ref:`matplotlib
   integration <matplotlib_sec>`.

4. :ref:`Filtered graphs <sec_graph_filtering>`, i.e. graphs where nodes and
   edges are temporarily masked. These are first class citizens in the library,
   and are accepted by every function. Due to the use C++ template
   metaprogramming, this functionality comes at no performance cost when
   filtering is not being used.

5. Efficient and fully documented :ref:`binary format <sec_gt_format>` for network files.

6. Integration with the `Netzschleuder <https://networks.skewed.de>`_ network
   data repository, enabling :data:`easy loading <graph_tool.collection.ns>` of network data.

7. Support for writing custom :ref:`C++ extensions <cppextensions>`.

Installing graph-tool
---------------------

Detailed installation instructions for various platforms are available `here
<https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions>`__.

The easiest option is to use `conda <https://docs.conda.io>`__:

.. code-block:: bash

   conda create --name gt -c conda-forge graph-tool
   conda activate gt

For HPC systems it is also straightforward to use :mod:`graph_tool` with
`Apptainer/Singularity
<https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#installing-using-apptainersingularity-useful-for-hpc-systems>`_.

Getting started
---------------

Yous should read first the :ref:`quick start guide <quickstart>`, followed by the
various :ref:`cookbooks <demos>` and explore all the examples in various :ref:`submodules
<submodules>`. For commonly asked questions, read the :ref:`FAQ <sec_faq>`.

Asking questions and reporting bugs
-----------------------------------

If you have questions about using ``graph-tool``, you are welcome to visit the
`discussion forum <https://forum.skewed.de/c/graph-tool/5>`_.

If you encounter a problem, open an issue in the `git repository
<https://git.skewed.de/count0/graph-tool/-/issues>`_.

Please don't forget to check if your question has been asked before, or if a
similar issue is open. When asking questions or reporting problems, it is
important to include:

1. Your exact ``graph-tool`` version.
2. Your operating system.
3. A **minimal working example** that shows the problem.

Item **3** above is **very important**! If you provide us only the part of the
code that you believe causes the problem, then it is not possible to understand
the context that may have contributed to it.

How to use the documentation
----------------------------

Documentation is available in two forms: docstrings provided
with the code, and the full documentation available in
`the graph-tool homepage <http://graph-tool.skewed.de/doc>`_.

We recommend exploring the docstrings using `IPython
<http://ipython.scipy.org>`_, an advanced Python shell with TAB-completion and
introspection capabilities.

The docstring examples assume that ``graph_tool.all`` has been imported as
``gt``::

   >>> import graph_tool.all as gt

Code snippets are indicated by three greater-than signs::

   >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

   >>> help(gt.Graph)

References
----------

.. [descriptive-vs-inferential] Tiago P. Peixoto, “Descriptive vs. inferential
   community detection in networks: pitfalls, myths and half-truths”,
   :arxiv:`2112.00183`

Contents
--------
.. toctree::
   :maxdepth: 3

   quickstart
   demos/index
   modules
   gt_format
   faq
