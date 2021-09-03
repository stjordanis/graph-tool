#!/bin/env python

from __future__ import print_function

verbose = __name__ == "__main__"

import os
import sys
if not verbose:
    out = open(os.devnull, 'w')
else:
    out = sys.stdout
import itertools
from graph_tool.all import *
import numpy.random
from numpy.random import randint, normal, random

numpy.random.seed(42)
seed_rng(42)

graph_tool.inference.set_test(True)

g = collection.data["football"]

# # add self-loops
# for i in range(10):
#     v = numpy.random.randint(g.num_vertices())
#     g.add_edge(v, v)

# # add parallel edges
# for e in list(g.edges())[:10]:
#     g.add_edge(e.source(), e.target())

ec = g.new_ep("int", randint(0, 10, g.num_edges()))

rec_p = g.new_ep("double", random(g.num_edges()))
rec_s = g.new_ep("double", normal(0, 10, g.num_edges()))


def _gen_state(directed, deg_corr, layers, overlap, rec, rec_type, dense_bg):
    u = GraphView(g, directed=directed)
    if layers != False:
        base_type = graph_tool.inference.LayeredBlockState
        state_args = dict(B=u.num_vertices(),
                          deg_corr=deg_corr,
                          ec=ec.copy(),
                          recs=[rec] if rec is not None else [],
                          rec_types=[rec_type] if rec is not None else [],
                          overlap=overlap,
                          layers=layers == True, dense_bg=dense_bg)
    elif overlap:
        base_type = graph_tool.inference.OverlapBlockState
        state_args = dict(B=2 * u.num_edges(),
                          recs=[rec] if rec is not None else [],
                          rec_types=[rec_type] if rec is not None else [],
                          deg_corr=deg_corr, dense_bg=dense_bg)
    else:
        base_type = graph_tool.inference.BlockState
        state_args = dict(B=u.num_vertices(),
                          recs=[rec] if rec is not None else [],
                          rec_types=[rec_type] if rec is not None else [],
                          deg_corr=deg_corr, dense_bg=dense_bg)
    return u, base_type, state_args


def gen_state(*args):
    u, base_type, state_args = _gen_state(*args)
    return base_type(u, **state_args)

def gen_nested_state(*args):
    u, base_type, state_args = _gen_state(*args)
    B = state_args.pop("B")
    return NestedBlockState(u,
                            bs=[numpy.arange(B)] + [numpy.zeros(1)] * 6,
                            base_type=base_type, state_args=state_args)


pranges = {"directed": [False, True],
           "overlap": [False, True],
           "layered": [False, "covariates", True],
           "rec": [None, "real-exponential", "real-normal"],
           "deg_corr": [False, True],
           "dl": [False, True],
           "degree_dl_kind": ["uniform", "distributed", "entropy"],
           "dense_bg": [True, False]}

def iter_ranges(ranges):
    for vals in itertools.product(*[v for k, v in ranges.items()]):
        yield zip(ranges.keys(), vals)


for pvals in iter_ranges(pranges):
    params = dict(pvals)

    locals().update(params)

    if not deg_corr and degree_dl_kind != "uniform":
        continue

    if overlap and deg_corr and degree_dl_kind != "distributed":      # FIXME
        continue

    print(params, file=out)

    rec_ = None
    if rec == "real-exponential":
        rec_ = rec_p
    elif rec == "real-normal":
        rec_ = rec_s

    print("\t mcmc (unweighted)", file=out)
    state = gen_state(directed, deg_corr, layered, overlap, rec_, rec, dense_bg)

    eargs = dict(dl=dl, degree_dl_kind=degree_dl_kind, beta_dl=0.95)

    print("\t\t",
          state.mcmc_sweep(beta=0, entropy_args=eargs),
          state.get_nonempty_B(), file=out)

    if overlap:
        print("\t\t",
              state.mcmc_sweep(beta=0, bundled=True,
                               entropy_args=dict(dl=dl,
                                                 degree_dl_kind=degree_dl_kind,
                                                 beta_dl=0.95)),
              state.get_nonempty_B(), file=out)

    print("\t mcmc (unweighted, multiflip)", file=out)
    state = gen_state(directed, deg_corr, layered, overlap, rec_, rec, dense_bg)
    print("\t\t",
          state.multiflip_mcmc_sweep(beta=0, entropy_args=eargs),
          state.get_nonempty_B(), file=out)

    print("\t mcmc (unweighted, multilevel)", file=out)
    state = gen_state(directed, deg_corr, layered, overlap, rec_, rec, dense_bg)
    print("\t\t",
          state.multilevel_mcmc_sweep(beta=0, M=3, entropy_args=eargs),
          state.get_nonempty_B(), file=out)

    print("\t gibbs (unweighted)", file=out)
    state = gen_state(directed, deg_corr, layered, overlap, rec_, rec, dense_bg)

    print("\t\t",
          state.gibbs_sweep(beta=0, entropy_args=eargs),
          state.get_nonempty_B(), file=out)

    if not overlap:
        state = gen_state(directed, deg_corr, layered, overlap, rec_, rec, dense_bg)

        print("\t mcmc", file=out)
        bstate = state.get_block_state(vweight=True,  deg_corr=deg_corr)

        print("\t\t",
              bstate.mcmc_sweep(beta=0, entropy_args=eargs),
              bstate.get_nonempty_B(), file=out)

        print("\t\t",
              bstate.mcmc_sweep(beta=0, entropy_args=eargs),
              bstate.get_nonempty_B(), file=out)

        print("\t\t",
              bstate.gibbs_sweep(beta=0, entropy_args=eargs),
              bstate.get_nonempty_B(), file=out)


pranges = {"directed": [False, True],
           "overlap": [False],
           "layered": [False, "covariates", True],
           "rec": [None, "real-exponential", "real-normal"],
           "deg_corr": [True, False],
           "degree_dl_kind": ["distributed"],
           "dense_bg": [True, False]}

def iter_ranges(ranges):
    for vals in itertools.product(*[v for k, v in ranges.items()]):
        yield zip(ranges.keys(), vals)

for pvals in iter_ranges(pranges):
    params = dict(pvals)

    locals().update(params)

    if overlap and deg_corr and degree_dl_kind != "distributed":      # FIXME
        continue

    print(params, file=out)

    rec_ = None
    if rec == "real-exponential":
        rec_ = rec_p
    elif rec == "real-normal":
        rec_ = rec_s

    print("\t mcmc (single flip)", file=out)
    state = gen_nested_state(directed, deg_corr, layered, overlap, rec_, rec, dense_bg)

    eargs = dict(dl=dl, degree_dl_kind=degree_dl_kind, beta_dl=0.95)

    for i in range(5):
        print("\t\t", state.mcmc_sweep(beta=0, d=0.5, entropy_args=eargs),
              [s.get_nonempty_B() for s in state.levels], file=out)

    print("\n\t mcmc (multiple flip)", file=out)
    state = gen_nested_state(directed, deg_corr, layered, overlap, rec_, rec, dense_bg)

    for i in range(5):
        print("\t\t",
              state.multiflip_mcmc_sweep(beta=0, d=0.5, entropy_args=eargs),
              [s.get_nonempty_B() for s in state.levels], file=out)

pranges = {"directed": [False, True],
           "overlap": [False, True],
           "layered": [False, "covariates", True],
           "rec": [None, "real-exponential", "real-normal"],
           "deg_corr": [False, True],
           "degree_dl_kind": ["uniform", "distributed", "entropy"],
           "dense_bg": [True, False]}

for pvals in iter_ranges(pranges):
    params = dict(pvals)

    locals().update(params)

    if not deg_corr and degree_dl_kind != "uniform":
        continue

    if overlap and deg_corr and degree_dl_kind != "distributed":    # FIXME
        continue

    print(params, file=out)

    rec_ = []
    if rec == "real-exponential":
        rec_ = [rec_p]
        rec = [rec]
    elif rec == "real-normal":
        rec_ = [rec_s]
        rec = [rec]
    else:
        rec = []

    if layered != False:
        state_t = LayeredBlockState
        state_args = dict(ec=ec, layers=(layered == True), recs=rec_,
                          rec_types=rec, deg_corr=deg_corr, overlap=overlap)
    else:
        if overlap:
            state_t = OverlapBlockState
        else:
            state_t = BlockState
        state_args = dict(recs=rec_, rec_types=rec, deg_corr=deg_corr)

    entropy_args = dict(beta_dl=0.95)

    state = minimize_blockmodel_dl(GraphView(g, directed=directed),
                                   state=state_t,
                                   state_args=state_args,
                                   multilevel_mcmc_args=dict(entropy_args=entropy_args))

    print(state.get_B(), state.entropy(), file=out)

    state_args = dict(state_args=state_args)
    if layered != False:
        state_args["base_state"] = LayeredBlockState
        state_args["overlap"] = overlap
    elif overlap:
            state_args["base_state"] = OverlapBlockState


    state = minimize_nested_blockmodel_dl(GraphView(g, directed=directed),
                                          state_args=state_args,
                                          multilevel_mcmc_args=dict(entropy_args=entropy_args))
    if verbose:
        state.print_summary()

    print(state.entropy(), "\n", file=out)

print("OK")
