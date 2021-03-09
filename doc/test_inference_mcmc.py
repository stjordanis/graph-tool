#!/bin/env python

from __future__ import print_function

from pylab import *
from graph_tool.all import *
import numpy.random
from numpy.random import randint
from collections import defaultdict
import scipy.stats
import os.path

try:
    from data_cache import FSDataCache
    cache = FSDataCache()
    def get_lock(name, params):
        return cache.get_lock(name, params, block=False)
    def get_cache(name, params):
        return cache.get(name, params)
    def put_cache(name, params, val):
        return cache.put(name, params, val)
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def get_lock(name, params):
        return None
    def get_cache(name, params):
        raise KeyError()
    def put_cache(name, params, val):
        pass

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

    savefig("test_mcmc/test_mcmc_reverse_prob_directed%s.pdf" % directed)

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
            savefig("test_mcmc/test_mcmc_move_prob_directed%s_v%d.pdf" % (directed, int(v)))

        plot(res)
    gca().set_ylim(-.1, .1)

    savefig("test_mcmc/test_mcmc_move_prob_directed%s.pdf" % directed)

for directed in [True, False]:
    for deg_corr in [True, False]:
        print(f"test edge sampling directed={directed}, deg_corr={deg_corr}...")

        g = collection.data["football"]
        g.set_directed(directed)
        state = LatentMultigraphBlockState(g, state_args=dict(deg_corr=deg_corr))
        for i in range(1000):
            state.multiflip_mcmc_sweep(niter=10)
        print(state)
        esampler = state.bstate._state.get_edge_sampler(False)

        ecount = defaultdict(int)
        for i in range(10000000):
            u, v = esampler.sample(graph_tool._get_rng())
            if v < u and not directed:
                u, v = v, u
            ecount[(u,v)] += 1

        N = sum([x for x in ecount.values()])

        Lp = []
        Ls = []
        for e, c in ecount.items():
            p = log(c)-log(N)
            u, v = e
            e = state.bstate.g.edge(u, v)
            if e is None:
                m = 0
            else:
                m = state.bstate.eweight[e]
            Le = esampler.log_prob(u, v, m)
            Lp.append(Le)
            Ls.append(p)

        clf()
        plot(Ls, Lp, "o")
        plot(Ls, Ls, "-")
        savefig(f"test_mcmc/test_mcmc_edge_sample_directed{directed}_deg_corr{deg_corr}.pdf")

if os.path.exists("g_small.gt"):
    g_small = load_graph("g_small.gt")
else:
    g = graph_union(complete_graph(4), complete_graph(4))
    g.add_edge(3, 4)
    vs = list(g.add_vertex(8))
    for i in range(3 * 8):
        s = vs[randint(4)]
        t = vs[randint(4) + 4]
        g.add_edge(s, t)
    g_small = g
    g.save("g_small.gt")

for wait in [50000, 100000, 250000, 500000, 1000000]:
    for g, name in [(g_small, "small"),
                    (collection.data["lesmis"], "lesmis"),
                    (collection.data["karate"], "karate")]:
        for directed in [False, True]:
            g.set_directed(directed)
            for nested in [False, True]:

                cs = list(reversed([numpy.inf, 0.01, "gibbs", -numpy.inf, -0.1]))
                inits = ["N"]
                state = None
                hists = {}

                for init in inits:

                    istep = 50

                    for i, c in enumerate(cs):

                        params = dict(name=name, directed=directed, nested=nested,
                                      wait=wait, init=init, c=c)


                        with get_lock("mcmc_test", params) as lock:
                            if lock is None:
                                continue

                            try:
                                h = get_cache("hists", params)
                            except KeyError:
                                if state is None:
                                    if nested:
                                        state = minimize_nested_blockmodel_dl(g, deg_corr=True)
                                    else:
                                        state = minimize_blockmodel_dl(g, deg_corr=True)
                                if nested:
                                    bs = [g.get_vertices()] if init == "N" else [zeros(1)]
                                    state = state.copy(bs=bs + [zeros(1)] * 6, sampling=True)
                                else:
                                    b = g.get_vertices() if init == "N" else zeros(g.num_vertices())
                                    state = state.copy(b=b)

                                state = state.copy(dense_bg=True)

                                if c != "gibbs":
                                    mcmc_args=dict(beta=.3, c=abs(c), niter=istep)
                                    if c < 0:
                                        mcmc_args = dict(mcmc_args, gibbs_sweeps=3)
                                else:
                                    mcmc_args=dict(beta=.3, niter=istep)

                                if name in ["lesmis", "karate", "football", "polbooks"]:
                                    mcmc_args["beta"] = .9

                                if nested:
                                    get_B = lambda s: [s.levels[0].get_Be()]
                                else:
                                    get_B = lambda s: [s.get_Be()]

                                h = mcmc_equilibrate(state,
                                                     mcmc_args=mcmc_args,
                                                     gibbs=c=="gibbs",
                                                     multiflip = c != "gibbs" and c < 0,
                                                     force_niter=wait,
                                                     callback=get_B,
                                                     verbose=(1, "c = %s " % str(c)) if verbose else False,
                                                     history=True)
                                put_cache("hists", params, h)


                    skip = False
                    for init in inits:
                        for i, c in enumerate(cs):
                            params = dict(name=name, directed=directed,
                                          nested=nested, wait=wait,
                                          init=init, c=c)
                            try:
                                hists[(c, init)] = get_cache("hists", params)
                            except KeyError:
                                skip = True
                    if skip:
                        continue

                    output = open(f"test_mcmc/test_mcmc_{name}_nested{nested}_directed{directed}-wait{wait}-output", "w")
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
                        for Be in [True, False]:
                            output = f"test_mcmc/test_mcmc_{name}_nested{nested}_directed{directed}-Be{Be}-cum{cum}-wait{wait}.pdf"
                            if not os.path.exists(output) and False:
                                figure(figsize=(10 * 4/3, 10))
                                hc = []
                                for c in cs:
                                    for init in inits:
                                        hist_i = hists[(c,init)]
                                        vals = list(zip(*hist_i))
                                        if Be:
                                            hc += vals[-1]
                                        else:
                                            hc += vals[2]
                                bins = linspace(min(hc), max(hc), 60)
                                for c in cs:
                                    for init in inits:
                                        hist_i = hists[(c,init)]
                                        vals = list(zip(*hist_i))
                                        if Be:
                                            vals = vals[-1]
                                        else:
                                            vals = vals[2]

                                        if cum:
                                            h = histogram(vals, 1000000, density=True)
                                            y = numpy.cumsum(h[0])
                                            y /= y[-1]
                                            plot(h[-1][:-1], y, "-", lw=.5,
                                                 label="c=%s, init=%s" % (str(c), init))

                                            if c != numpy.inf:
                                                hist_i = hists[(numpy.inf, 1)]
                                                h2 = histogram(list(zip(*hist_i))[2],
                                                               bins=h[-1],
                                                               density=True)
                                                y2 = numpy.cumsum(h2[0])
                                                y2 /= y2[-1]
                                                res = abs(y - y2)
                                                i = res.argmax()
                                                axvline(h[-1][i], color="grey")
                                        else:
                                            hist(vals, bins=bins, density=True, #log=True,
                                                 histtype="step", label="c=%s, init=%s" % (str(c), init))
                                if cum:
                                    legend(loc="lower right")
                                else:
                                    legend(loc="best")
                                ylabel("Prob. density")
                                xlabel("Entropy" if not Be else r"$B_e$")
                                savefig(output)

                            if not cum:
                                output = f"test_mcmc/test_mcmc_{name}_nested{nested}_directed{directed}-Be{Be}-res-wait{wait}.pdf"
                                if not os.path.exists(output):
                                    figure(figsize=(10 * 4/3, 10))
                                    hc = []
                                    for c in cs:
                                        for init in inits:
                                            hist_i = hists[(c,init)]
                                            vals = list(zip(*hist_i))
                                            if Be:
                                                hc += vals[-1]
                                            else:
                                                hc += vals[2]
                                    bins = linspace(min(hc), max(hc), 60)
                                    h_mean = zeros(len(bins) - 1)
                                    count = 0
                                    ymax = 0
                                    for c in cs:
                                        for init in inits:
                                            hist_i = hists[(c,init)]
                                            vals = list(zip(*hist_i))
                                            if Be:
                                                vals = vals[-1]
                                            else:
                                                vals = vals[2]

                                            h = histogram(vals, bins=bins, density=True)
                                            ymax = max(ymax, h[0].max())
                                            h_mean += h[0]
                                            count += 1
                                    h_mean /= count

                                    for c in cs:
                                        for init in inits:
                                            hist_i = hists[(c,init)]
                                            vals = list(zip(*hist_i))
                                            if Be:
                                                vals = vals[-1]
                                            else:
                                                vals = vals[2]

                                            h = histogram(vals, bins=bins, density=True)
                                            h = list(h)
                                            h[0] -= h_mean

                                            step(bins[:-1], h[0], label="c=%s, init=%s" % (str(c), init))
                                    if cum:
                                        legend(loc="lower right")
                                    else:
                                        legend(loc="best")
                                    ylim(-ymax/10, ymax/10)
                                    ylabel("Prob. density residue")
                                    xlabel("Entropy" if not Be else r"$B_e$")
                                    savefig(output)

                                output = f"test_mcmc/test_mcmc_{name}_nested{nested}_directed{directed}-Be{Be}-kde-wait{wait}.pdf"
                                if not os.path.exists(output):
                                    figure(figsize=(10 * 4/3, 10))

                                    vals = []
                                    for c in cs:
                                        for init in inits:
                                            hist_i = hists[(c,init)]
                                            vals_i = list(zip(*hist_i))
                                            if Be:
                                                vals += vals_i[-1]
                                            else:
                                                vals += vals_i[2]
                                    x = linspace(min(vals), max(vals), 1000)
                                    for c in cs:
                                        for init in inits:
                                            hist_i = hists[(c,init)]
                                            vals = list(zip(*hist_i))
                                            if Be:
                                                vals = vals[-1]
                                            else:
                                                vals = vals[2]

                                            kernel = scipy.stats.gaussian_kde(vals)
                                            y = kernel(x)
                                            plot(x, y, "-", linewidth=.8,
                                                 label="c=%s, init=%s" % (str(c), init))
                                    legend(loc="best")
                                    ylabel("Prob. density")
                                    xlabel("Entropy" if not Be else r"$B_e$")
                                    savefig(output)

                                output = f"test_mcmc/test_mcmc_{name}_nested{nested}_directed{directed}-Be{Be}-kde-res-wait{wait}.pdf"
                                if not os.path.exists(output):
                                    figure(figsize=(10 * 4/3, 10))

                                    vals = []
                                    for c in cs:
                                        for init in inits:
                                            hist_i = hists[(c,init)]
                                            vals_i = list(zip(*hist_i))
                                            if Be:
                                                vals += vals_i[-1]
                                            else:
                                                vals += vals_i[2]
                                    x = linspace(min(vals), max(vals), 1000)

                                    ymean = zeros(len(x))
                                    count = 0
                                    for c in cs:
                                        for init in inits:
                                            hist_i = hists[(c,init)]
                                            vals = list(zip(*hist_i))
                                            if Be:
                                                vals = vals[-1]
                                            else:
                                                vals = vals[2]

                                            kernel = scipy.stats.gaussian_kde(vals)
                                            ymean += kernel(x)
                                            count += 1
                                    ymean /= count

                                    ymax = 0
                                    for c in cs:
                                        for init in inits:
                                            hist_i = hists[(c,init)]
                                            vals = list(zip(*hist_i))
                                            if Be:
                                                vals = vals[-1]
                                            else:
                                                vals = vals[2]

                                            kernel = scipy.stats.gaussian_kde(vals)
                                            y = kernel(x)
                                            ymax = max(y.max(), ymax)
                                            y -= ymean
                                            plot(x, y, "-", linewidth=1.5,
                                                 label="c=%s, init=%s" % (str(c), init))
                                    ylim(-ymax/10, ymax/10)
                                    legend(loc="best")
                                    ylabel("Prob. density residue")
                                    xlabel("Entropy" if not Be else r"$B_e$")
                                    savefig(output)
print("OK")