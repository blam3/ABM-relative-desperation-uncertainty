import os
import sys, csv, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from model import rel_DMAP_model
SET = {'legacy': dict(weighting='prelec_raw', alpha_dist='t_clip', alpha_scale=0.5, lambd=0.2),
       'fullfix': dict(weighting='prelec_log_elevation', alpha_dist='lognormal', alpha_scale=0.35, lambd=0.2)}
def gap(w): return float(np.percentile(w, 90) - np.percentile(w, 10))
def worker(t):
    ver, beta, thresh, rep = t
    seed = 777000 + rep * 17 + int(thresh * 100) + (beta + 2) * 3
    np.random.seed(seed)
    m = rel_DMAP_model(width=50, height=50, beta_loc=beta, alpha_loc=1, income_rank_threshold=thresh, seed=seed, **SET[ver])
    ag = list(m.agents); rows = []
    for s in range(50):
        m.step()
        d = np.array([a.decision for a in ag]); ds = np.array([a.desperate_state for a in ag])
        D = np.array([a.rel_deprivation for a in ag]); w = np.array([a.wealth for a in ag])
        rows.append(dict(ver=ver, beta=beta, thresh=thresh, rep=rep, step=s + 1, crime=d.mean(), fdesp=ds.mean(),
                         desp_rate=d[ds == 1].mean(), nond_rate=d[ds == 0].mean(), D_desp=D[ds == 1].mean(),
                         gap=gap(w), median=float(np.median(w)), debt=float((w < 0).mean())))
    return rows
if __name__ == '__main__':
    tasks = [(v, b, th, r) for v, b in (('fullfix', 0), ('legacy', 1)) for th in (0.2, 0.5) for r in range(6)]
    with Pool(2, maxtasksperchild=2) as p, open('results_traj.csv', 'w', newline='') as f:
        wr = None
        for rows in p.imap_unordered(worker, tasks):
            if wr is None: wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader()
            wr.writerows(rows)
    print('DONE traj')
