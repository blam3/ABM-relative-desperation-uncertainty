import sys, os, csv, itertools, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from model import rel_DMAP_model
import cognitive_functions as cf

GRID, STEPS = 50, 25
SPREAD = {1: (175, 50), 2: (100, 200), 3: (25, 350)}  # mean-preserving spreads, initial Gini ~.08/.30/.52
SETTINGS = {
    'legacy': dict(weighting='prelec_raw', alpha_dist='t_clip', alpha_scale=0.5, lambd=0.2),
    'betafix': dict(weighting='prelec_log_elevation', alpha_dist='t_clip', alpha_scale=0.5, lambd=0.2),
    'fullfix': dict(weighting='prelec_log_elevation', alpha_dist='lognormal', alpha_scale=0.35, lambd=0.2),
}

def pct_gap(w): return float(np.percentile(w, 90) - np.percentile(w, 10))

def worker(task):
    study, rep, beta, alpha, thresh, shape, rule = task
    kw = dict(SETTINGS['fullfix' if study == 'ineq' else study])
    seed = 20260915 + rep * 1000 + int((beta + 2) * 100) + int(alpha * 10) + int(thresh * 1000) + shape * 7
    np.random.seed(seed)
    m = rel_DMAP_model(width=GRID, height=GRID, beta_loc=beta, alpha_loc=alpha,
                       income_rank_threshold=thresh, pareto_shape=3, pareto_scale=SPREAD[shape][1], min_start_wealth=SPREAD[shape][0],
                       desperation_rule=rule, kappa=0.5, seed=seed, **kw)
    agents = list(m.agents)
    w0 = np.array([a.wealth for a in agents])
    wp = np.array([cf.prelec(a.beta, a.alpha, 0.8, a.weighting) for a in agents])
    crime, fdesp = [], []
    for s in range(STEPS):
        m.step()
        d = np.array([a.decision for a in agents]); ds = np.array([a.desperate_state for a in agents])
        crime.append(d.mean()); fdesp.append(ds.mean())
    wstart = np.array([a.wealth_start for a in agents]); wend = np.array([a.wealth for a in agents])
    X = np.column_stack([np.ones(len(d)), ds, (wstart - wstart.mean()) / wstart.std()])
    coef = np.linalg.lstsq(X, d, rcond=None)[0]
    return dict(study=study, rep=rep, beta=beta, alpha=alpha, thresh=thresh, shape=shape, rule=rule,
                crime_final=crime[-1], crime_step1=crime[0], crime_mean=float(np.mean(crime)),
                desp_rate=float(d[ds == 1].mean()) if ds.sum() else np.nan,
                nondesp_rate=float(d[ds == 0].mean()) if (ds == 0).sum() else np.nan,
                frac_desp_step1=fdesp[0], frac_desp_final=fdesp[-1],
                b_desp=coef[1], b_wealth=coef[2],
                mean_w0=float(w0.mean()), gap_w0=pct_gap(w0),
                mean_wealth=float(wend.mean()), median_wealth=float(np.median(wend)),
                p90_wealth=float(np.percentile(wend, 90)), gap_final=pct_gap(wend),
                frac_debt=float((wend < 0).mean()),
                mean_wp=float(wp.mean()), frac_wp_gt1=float((wp > 1).mean()),
                frac_wp_eq1=float(np.isclose(wp, 1).mean()),
                traj=';'.join(f'{c:.4f}' for c in crime))

if __name__ == '__main__':
    study, R = sys.argv[1], int(sys.argv[2])
    if study == 'ineq':
        tasks = [('ineq', r, 0, 1, th, sh, 'rank') for r in range(R) for sh in (1, 2, 3) for th in (0.2, 0.5)]
        tasks += [('ineq', r, 0, 1, 0.0, sh, 'median_ratio') for r in range(R) for sh in (1, 2, 3)]
    else:
        tasks = [(study, r, b, a, th, 3, 'rank') for r in range(R)
                 for b, a, th in itertools.product((-1, 0, 1), (0.5, 1, 2), (0.05, 0.2, 0.35, 0.5, 0.65))]
        tasks = [(st, r, b, a, th, 2, ru) for st, r, b, a, th, _, ru in tasks]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results_{study}.csv')
    t0 = time.time()
    with Pool(2, maxtasksperchild=4) as pool, open(out, 'w', newline='') as f:
        wr = None
        for i, row in enumerate(pool.imap_unordered(worker, tasks), 1):
            if wr is None:
                wr = csv.DictWriter(f, fieldnames=list(row)); wr.writeheader()
            wr.writerow(row); f.flush()
            if i % 20 == 0: print(study, i, len(tasks), round(time.time() - t0), flush=True)
    print('DONE', study, round(time.time() - t0), flush=True)
