#!/bin/env python

from __future__ import print_function

from pylab import *
from graph_tool.all import *
import numpy.random
from numpy.random import randint
from collections import defaultdict
import scipy.stats
import os.path
from multiprocessing import Pool
from tqdm import tqdm

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


#graph_tool.inference.blockmodel.set_test(True)
openmp_set_num_threads(1)
numpy.random.seed(43)
seed_rng(43)

verbose = __name__ == "__main__"

g = collection.data["football"]
B = 10

for directed in [True, False]:
    clf()
    g.set_directed(directed)

    state = minimize_blockmodel_dl(g, state_args=dict(deg_corr=False),
                                   multilevel_mcmc_args=dict(B_min=B, B_max=B))
    state = state.copy(b=order_partition_labels(state.b.a))
    state = state.copy(B=B+1)

    c = 0.01
    for v in g.vertices():
        r = state.b[v]
        res = zeros(state.get_B())
        for s in range(state.get_B()):
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

for directed in [False, True]:
    clf()
    g.set_directed(directed)

    state = minimize_blockmodel_dl(g, state_args=dict(deg_corr=False),
                                   multilevel_mcmc_args=dict(B_min=B, B_max=B))
    state = state.copy(b=order_partition_labels(state.b.a))
    state = state.copy(B=state.get_B()+1)

    c = 0.1
    for v in g.vertices():

        # computed probabilities
        mp = zeros(state.get_B())
        n_empty = sum(state.wr.a == 0)
        for s in range(state.get_B()):
            mp[s] = exp(state.get_move_prob(v, s, c))
            if state.wr[s] == 0:
                mp[s] /= n_empty

        n_samples = min(int(200 / mp[mp > 0].min()), 1000000)

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


def run_mcmc(state, algo, params={}):
    if algo == "single-flip-metropolis":
        return state.mcmc_sweep(**params)
    elif algo == "single-flip-gibbs":
        return state.gibbs_sweep(**params)
    elif algo == "multiflip":
        return state.multiflip_mcmc_sweep(**params)
    elif algo == "multilevel":
        return state.multilevel_mcmc_sweep(**params)

alist = [("single-flip-metropolis", [dict(c=0.1),
                                     dict(c=numpy.inf)]),
         ("single-flip-gibbs", [dict()]),
         ("multiflip", [dict(c=0.1, gibbs_sweeps=3),
                        dict(c=numpy.inf, gibbs_sweeps=3)]),
         #("multilevel", [dict(c=0.1, mh_sweeps=3, M=2, gibbs=False)])
         ]


def get_hist(params, pos=0, check_only=False):

    params = params.copy()
    g = params.get("g", None)

    name = params["name"]
    directed = params["directed"]
    model = params["model"]
    wait = params["wait"]
    init = params["init"]
    algo = params["algo"]
    aparams = params["aparams"]

    params = dict(params, **aparams)
    if g is not None:
        del params["g"]

    try:
        h = get_cache("hists", params)
        if check_only:
            return True
    except KeyError:
        if check_only:
            return False

        g = g.copy()
        g.set_directed(directed)

        if model == "nsbm":
            state = minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=True))
        if model == "sbm":
            state = minimize_blockmodel_dl(g, state_args=dict(deg_corr=True))
        if model == "pp":
            state = minimize_blockmodel_dl(g, state=PPBlockState)
        if model == "modularity":
            state = minimize_blockmodel_dl(g, state=ModularityState)

        if model == "nsbm":
            bs = [g.get_vertices()] if init == "N" else [zeros(1)]
            state = state.copy(bs=bs + [zeros(1)] * 6)
        else:
            b = g.get_vertices() if init == "N" else zeros(g.num_vertices())
            state = state.copy(b=b)

        if model in ["nsbm", "sbm"]:
            state = state.copy(dense_bg=True)

        aparams = dict(**aparams)
        aparams["beta"] = .3

        if name in ["lesmis", "karate", "football", "polbooks"]:
            aparams["beta"] = .9

        if model == "nsbm":
            get_B = lambda s: s.levels[0].get_Be()
        else:
            get_B = lambda s: s.get_Be()

        istep = 5
        aparams["niter"] = istep

        if algo == "multilevel":
            aparams["niter"] = istep * g.num_vertices()

        h = [[],[]]

        desc=f"{pos}: {name} {algo}"
        for t in tqdm(range(wait), position=pos, desc=desc + " " * max(35 - len(desc), 0)):
            ret = run_mcmc(state, algo, aparams)
            S, B = state.entropy(), get_B(state)
            h[0].append(S)
            h[1].append(B)
            #print(name, directed, model, algo, aparams, wait, t, S, B)

        put_cache("hists", params, h)
    return h

all_params = []
done = []

for wait in [50000, 500000,
             #5000000
             ]:
    for g, name in [(g_small, "small"),
                    (collection.data["lesmis"], "lesmis"),
                    (collection.data["karate"], "karate")
                    ]:

        for directed in [False, True]:
            for model in ["sbm", "nsbm",
                          "pp", "modularity"
                          ]:
                for algo, aparam_list in alist:
                    for aparams in aparam_list:
                        inits = ["N"]
                        state = None
                        hists = {}

                        for init in inits:
                            params = dict(g=g, name=name, directed=directed,
                                          model=model, wait=wait, init=init,
                                          algo=algo, aparams=aparams)
                            all_params.append(params)
                            done.append(get_hist(params, check_only=True))

# pos = 0
# for params, done in zip(all_params, done):
#     print(params)
#     get_hist(params, pos)
#     pos += 1

with Pool(12) as p:
    rets = []
    pos = 0
    for params, done in zip(all_params, done):
        if done:
            continue
        ret = p.apply_async(get_hist, (params, pos))
        rets.append(ret)
        pos += 1
    for ret in rets:
        ret.get()

for params in all_params:

    g = params["g"]
    del params["g"]

    name = params["name"]
    directed = params["directed"]
    model = params["model"]
    wait = params["wait"]
    init = params["init"]
    algo = params["algo"]
    aparams = params["aparams"]
    for Be in [True, False]:
        output = f"test_mcmc/test_mcmc_{name}_model{model}_directed{directed}_Be{Be}_wait{wait}-KS.pdf"
        if not os.path.exists(output):
            figure(figsize=(10 * 4/3, 10))

            for m_algo, m_aparam_list in alist:
                for m_aparams in m_aparam_list:
                    for m_init in inits:

                        hist_m = get_hist(dict(name=name,
                                               directed=directed,
                                               model=model, wait=wait,
                                               init=m_init, algo=m_algo,
                                               aparams=m_aparams))

                        if Be:
                            S1 = hist_m[1]
                        else:
                            S1 = hist_m[0]

                        hc = []
                        for algo, aparam_list in alist:
                            for aparams in aparam_list:
                                if algo == m_algo and aparams == m_aparams:
                                    continue
                                for init in inits:
                                    hist_i = get_hist(dict(name=name,
                                                           directed=directed,
                                                           model=model, wait=wait,
                                                           init=init, algo=algo,
                                                           aparams=aparams))

                                    if Be:
                                        hc.append(hist_i[1])
                                    else:
                                        hc.append(hist_i[0])

                        ts = logspace(1, log10(len(S1)), 100, dtype="int")

                        Ds = []
                        Dcs = []

                        for t in ts:
                            S = S1[:t]
                            Sa = []
                            for Sm in hc:
                                Sa += Sm[:t]
                            D, p = scipy.stats.ks_2samp(S, Sa)
                            Ds.append(D)
                            D_c = 1.63 * sqrt((len(S) + len(Sa)) / (len(S) * len(Sa)))
                            Dcs.append(D_c)
                        loglog(ts, Ds, label="%s, p=%s, init=%s" % (str(m_algo), str(m_aparams), m_init))

            plot(ts, Dcs, "--", label="D_c")
            legend(loc="best")
            ylabel("KS statistics")
            xlabel("Time")
            savefig(output)


        output = f"test_mcmc/test_mcmc_{name}_model{model}_directed{directed}-Be{Be}-wait{wait}-res.pdf"
        if not os.path.exists(output):
            figure(figsize=(10 * 4/3, 10))
            hc = []
            for algo, aparam_list in alist:
                for aparams in aparam_list:
                    for init in inits:
                        hist_i = get_hist(dict(name=name,
                                               directed=directed,
                                               model=model, wait=wait,
                                               init=init, algo=algo,
                                               aparams=aparams))

                        if Be:
                            hc += hist_i[1]
                        else:
                            hc += hist_i[0]

            bins = linspace(min(hc), max(hc), 60)
            h_mean = zeros(len(bins) - 1)
            count = 0
            ymax = 0

            for algo, aparam_list in alist:
                for aparams in aparam_list:
                    for init in inits:
                        hist_i = get_hist(dict(name=name,
                                               directed=directed,
                                               model=model, wait=wait,
                                               init=init, algo=algo,
                                               aparams=aparams))

                        if Be:
                            vals = hist_i[1]
                        else:
                            vals = hist_i[0]

                        h = histogram(vals, bins=bins, density=True)
                        ymax = max(ymax, h[0].max())
                        h_mean += h[0]
                        count += 1
            h_mean /= count

            for algo, aparam_list in alist:
                for aparams in aparam_list:
                    for init in inits:
                        hist_i = get_hist(dict(name=name,
                                               directed=directed,
                                               model=model, wait=wait,
                                               init=init, algo=algo,
                                               aparams=aparams))

                        if Be:
                            vals = hist_i[1]
                        else:
                            vals = hist_i[0]

                        h = histogram(vals, bins=bins, density=True)
                        h = list(h)
                        h[0] -= h_mean

                        step(bins[:-1], h[0], label="%s, p=%s, init=%s" % (str(algo), str(aparams), init))
            legend(loc="best")
            ylim(-ymax/10, ymax/10)
            ylabel("Prob. density residue")
            xlabel("Entropy" if not Be else r"$B_e$")
            savefig(output)

        output = f"test_mcmc/test_mcmc_{name}_model{model}_directed{directed}-Be{Be}-wait{wait}-kde.pdf"
        if not os.path.exists(output):
            figure(figsize=(10 * 4/3, 10))

            vals = []
            for algo, aparam_list in alist:
                for aparams in aparam_list:
                    for init in inits:
                        hist_i = get_hist(dict(name=name,
                                               directed=directed,
                                               model=model, wait=wait,
                                               init=init, algo=algo,
                                               aparams=aparams))
                        if Be:
                            vals += hist_i[1]
                        else:
                            vals += hist_i[0]

            x = linspace(min(vals), max(vals), 1000)

            for algo, aparam_list in alist:
                for aparams in aparam_list:
                    for init in inits:
                        hist_i = get_hist(dict(name=name,
                                               directed=directed,
                                               model=model, wait=wait,
                                               init=init, algo=algo,
                                               aparams=aparams))
                        if Be:
                            vals = hist_i[1]
                        else:
                            vals = hist_i[0]

                        kernel = scipy.stats.gaussian_kde(vals)
                        y = kernel(x)
                        plot(x, y, "-", linewidth=.8,
                             label="%s, p=%s, init=%s" % (str(algo), str(aparams), init))
            legend(loc="best")
            ylabel("Prob. density")
            xlabel("Entropy" if not Be else r"$B_e$")
            savefig(output)

        output = f"test_mcmc/test_mcmc_{name}_model{model}_directed{directed}-Be{Be}-wait{wait}-kde-res.pdf"
        if not os.path.exists(output):
            figure(figsize=(10 * 4/3, 10))

            vals = []
            for algo, aparam_list in alist:
                for aparams in aparam_list:
                    for init in inits:
                        hist_i = get_hist(dict(name=name,
                                               directed=directed,
                                               model=model, wait=wait,
                                               init=init, algo=algo,
                                               aparams=aparams))
                        if Be:
                            vals += hist_i[1]
                        else:
                            vals += hist_i[0]

            x = linspace(min(vals), max(vals), 1000)

            ymean = zeros(len(x))
            count = 0

            for algo, aparam_list in alist:
                for aparams in aparam_list:
                    for init in inits:
                        hist_i = get_hist(dict(name=name,
                                               directed=directed,
                                               model=model, wait=wait,
                                               init=init, algo=algo,
                                               aparams=aparams))
                        if Be:
                            vals = hist_i[1]
                        else:
                            vals = hist_i[0]

                        kernel = scipy.stats.gaussian_kde(vals)
                        ymean += kernel(x)
                        count += 1
            ymean /= count

            ymax = 0
            for algo, aparam_list in alist:
                for aparams in aparam_list:
                    for init in inits:
                        hist_i = get_hist(dict(name=name,
                                               directed=directed,
                                               model=model, wait=wait,
                                               init=init, algo=algo,
                                               aparams=aparams))
                        if Be:
                            vals = hist_i[1]
                        else:
                            vals = hist_i[0]

                        kernel = scipy.stats.gaussian_kde(vals)
                        y = kernel(x)
                        ymax = max(y.max(), ymax)
                        y -= ymean
                        plot(x, y, "-", linewidth=1.5,
                             label="%s, p=%s, init=%s" % (str(algo), str(aparams), init))
            ylim(-ymax/10, ymax/10)
            legend(loc="best")
            ylabel("Prob. density residue")
            xlabel("Entropy" if not Be else r"$B_e$")
            savefig(output)
