#!/bin/env python

from __future__ import print_function

from pylab import *
from graph_tool.all import *
import numpy.random
from numpy.random import randint
import scipy.stats

numpy.random.seed(43)
seed_rng(43)

verbose = __name__ == "__main__"

g = collection.data["football"]
B = 10

for directed in [True, False]:
    clf()
    g.set_directed(directed)

    state = minimize_blockmodel_dl(g, deg_corr=False, B_min=B, B_max=B)
    state = state.copy(B=B+1)

    c = 0.01
    for v in g.vertices():
        r = state.b[v]
        res = zeros(state.B)
        for s in range(state.B):
            pf = state.get_move_prob(v, s, c=c, reverse=True)
            state.move_vertex(v, s)
            pb = state.get_move_prob(v, r, c=c)
            res[s] = pf - pb
            if abs(res[s]) > 1e-8:
                print("Warning, wrong reverse probability: ", v, r, s, pf, pb,
                      res[s], directed)
            state.move_vertex(v, r)
        plot(res)
    gca().set_ylim(-.1, .1)

    savefig("test_mcmc_reverse_prob_directed%s.pdf" % directed)

for directed in [True, False]:
    clf()
    g.set_directed(directed)

    state = minimize_blockmodel_dl(g, deg_corr=False, B_min=B, B_max=B)
    state = state.copy(B=B+1)

    c = 0.1
    for v in g.vertices():

        # computed probabilities
        mp = zeros(state.B)
        for s in range(state.B):
            mp[s] = state.get_move_prob(v, s, c)

        n_samples = min(int(200 / mp.min()), 1000000)

        # actual samples
        samples = [state.sample_vertex_move(v, c) for i in range(n_samples)]

        # samples from computed distribution
        true_hist = numpy.random.multinomial(n_samples, mp)
        true_samples = []
        for r, count in enumerate(true_hist):
            true_samples.extend([r] * count)

        mp_h = bincount(samples)
        if len(mp_h) < B + 1:
            mp_h = list(mp_h) + [0] * (B + 1 - len(mp_h))
        mp_h = array(mp_h, dtype="float")
        mp_h /= mp_h.sum()
        res = mp - mp_h

        samples = array(samples, dtype="float")
        true_samples = array(true_samples, dtype="float")

        p = scipy.stats.ks_2samp(samples, true_samples)[1]

        if verbose:
            print("directed:", directed, "vertex:", v, "p-value:", p)

        if p < 0.001:
            print(("Warning, move probability for node %d does not " +
                   "match the computed distribution, with p-value: %g") %
                  (v, p))
            clf()
            plot(res)
            savefig("test_mcmc_move_prob_directed%s_v%d.pdf" % (directed, int(v)))

        plot(res)
    gca().set_ylim(-.1, .1)

    savefig("test_mcmc_move_prob_directed%s.pdf" % directed)

g = graph_union(complete_graph(4), complete_graph(4))
g.add_edge(3, 4)
vs = list(g.add_vertex(8))
for i in range(3 * 8):
    s = vs[randint(4)]
    t = vs[randint(4) + 4]
    g.add_edge(s, t)


for directed in [False, True]:
    g.set_directed(directed)
    for nested in [False, True]:

        if nested:
            state = minimize_nested_blockmodel_dl(g, deg_corr=True)
        else:
            state = minimize_blockmodel_dl(g, deg_corr=True)

        hists = {}

        cs = list(reversed([numpy.inf, 1, 0.1, 0.01, "gibbs", -numpy.inf, -1]))
        inits = [1, "N"]

        for init in inits:
            if nested:
                bs = [g.get_vertices()] if init == "N" else [zeros(1)]
                state = state.copy(bs=bs + [zeros(1)] * 6, sampling=True)
            else:
                state = state.copy(B=g.num_vertices() if init == "N" else 1)

            for i, c in enumerate(cs):
                if c != "gibbs":
                    mcmc_args=dict(beta=.3, c=abs(c), niter=40)
                else:
                    mcmc_args=dict(beta=.3, niter=40)

                if nested:
                    get_B = lambda s: [sl.get_nonempty_B() for sl in s.levels]
                else:
                    get_B = lambda s: [s.get_nonempty_B()]

                hists[(c, init)] = mcmc_equilibrate(state,
                                                    mcmc_args=mcmc_args,
                                                    gibbs=c=="gibbs",
                                                    multiflip = c != "gibbs" and c < 0,
                                                    wait=6000,
                                                    nbreaks=6,
                                                    callback=get_B,
                                                    verbose=(1, "c = %s " % str(c)) if verbose else False,
                                                    history=True)

        output = open("test_mcmc_nested%s_directed%s-output" % (nested, directed), "w")
        for c1 in cs:
            for init1 in inits:
                for c2 in cs:
                    for init2 in inits:
                        try:
                            if (c2, init2) < (c1, init1):
                                continue
                        except TypeError:
                            pass
                        Ss1 = array(list(zip(*hists[(c1, init1)]))[2])
                        Ss2 = array(list(zip(*hists[(c2, init2)]))[2])
                        # add very small normal noise, to solve discreteness issue
                        Ss1 += numpy.random.normal(0, 1e-2, len(Ss1))
                        Ss2 += numpy.random.normal(0, 1e-2, len(Ss2))
                        D, p = scipy.stats.ks_2samp(Ss1, Ss2)
                        D_c = 1.63 * sqrt((len(Ss1) + len(Ss2)) / (len(Ss1) * len(Ss2)))
                        print("nested:", nested, "directed:", directed, "c1:", c1,
                              "init1:", init1, "c2:", c2, "init2:", init2, "D:",
                              D, "D_c:", D_c, "p-value:", p,
                              file=output)
                        if p < .001:
                            print(("Warning, distributions for directed=%s (c1, c2) = " +
                                   "(%s, %s) are not the same, with a p-value: %g (D=%g, D_c=%g)") %
                                  (str(directed), str((c1, init1)),
                                   str((c2, init2)), p, D, D_c),
                                  file=output)
                        output.flush()

        for cum in [True, False]:
            clf()
            bins = None
            for c in cs:
                for init in inits:
                    hist = hists[(c,init)]
                    if cum:
                        h = histogram(list(zip(*hist))[2], 1000000,
                                      density=True)
                        y = numpy.cumsum(h[0])
                        y /= y[-1]
                        plot(h[-1][:-1], y, "-", label="c=%s, init=%s" % (str(c), init))

                        if c != numpy.inf:
                            hist = hists[(numpy.inf, 1)]
                            h2 = histogram(list(zip(*hist))[2], bins=h[-1],
                                           density=True)
                            y2 = numpy.cumsum(h2[0])
                            y2 /= y2[-1]
                            res = abs(y - y2)
                            i = res.argmax()
                            axvline(h[-1][i], color="grey")
                    else:
                        h = histogram(list(zip(*hist))[2], 40, density=True)
                        plot(h[-1][:-1], h[0], "s-", label="c=%s, init=%s" % (str(c), init))
                        gca().set_yscale("log")
            if cum:
                legend(loc="lower right")
            else:
                legend(loc="best")
            savefig("test_mcmc_nested%s_directed%s-cum%s.pdf" % (nested, directed, str(cum)))

print("OK")