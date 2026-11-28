import os
"""Compute every pilot number used in the recommendations document."""
import json, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
R = os.path.dirname(os.path.abspath(__file__)) + '/'
MS = dict(beta={-1: .302, 0: .186, 1: .123}, alpha={0.5: .228, 1.0: .201, 2.0: .181},
          thresh_lo=.165, thresh_hi=.242, neutral=.159, gap=.115, step1=.257, step25=.204, r=.979,
          b=dict(zb=-.073, za=-.019, zt=.028, **{'zb:za': .046, 'zb:zt': -.011, 'za:zt': .002, 'zb:za:zt': .006}),
          peta=dict(zb=.946, za=.548, zt=.724, **{'zb:za': .873, 'zb:zt': .280, 'za:zt': .008, 'zb:za:zt': .103}), R2=.967)

def summarize(study):
    d = pd.read_csv(R + f'results_{study}.csv')
    x = d.copy()
    x['zb'] = (x.beta - x.beta.mean()) / x.beta.std()
    x['la'] = np.log2(x.alpha); x['za'] = (x.la - x.la.mean()) / x.la.std()
    x['zt'] = (x.thresh - x.thresh.mean()) / x.thresh.std()
    fit = smf.ols('crime_final ~ zb * za * zt', data=x).fit()
    a = anova_lm(fit, typ=3); ssr = a.loc['Residual', 'sum_sq']; sst = ((x.crime_final - x.crime_final.mean())**2).sum()
    terms = ['zb', 'za', 'zt', 'zb:za', 'zb:zt', 'za:zt', 'zb:za:zt']
    tr = d.traj.str.split(';', expand=True).astype(float)
    d['gap'] = d.desp_rate - d.nondesp_rate
    cm = d.groupby(['beta', 'alpha', 'thresh'])[['gap_final', 'crime_final']].mean()
    ba = d.pivot_table('crime_final', 'beta', 'alpha')
    bt = d.pivot_table('crime_final', 'beta', 'thresh')
    return dict(
        n=len(d), reps=int(d.rep.nunique()),
        wp=d.groupby('beta').mean_wp.mean().to_dict(), wp_gt1=d.groupby('beta').frac_wp_gt1.mean().to_dict(),
        beta=d.groupby('beta').crime_final.mean().to_dict(), alpha=d.groupby('alpha').crime_final.mean().to_dict(),
        thresh=d.groupby('thresh').crime_final.mean().to_dict(),
        ba={f'{b}|{a}': ba.loc[b, a] for b in ba.index for a in ba.columns},
        bt={f'{b}|{t}': bt.loc[b, t] for b in bt.index for t in bt.columns},
        neutral=d.query('beta==0 and alpha==1 and thresh==0.2').crime_final.mean(),
        undistorted_legacy=d.query('beta==1 and alpha==1 and thresh==0.2').crime_final.mean(),
        gap=d.gap.mean(), gap_by_beta=d.groupby('beta').gap.mean().to_dict(),
        b_desp=d.b_desp.mean(), b_wealth=d.b_wealth.mean(),
        wealth={int(k): v for k, v in d.groupby('beta')[['mean_wealth', 'median_wealth', 'p90_wealth', 'frac_debt']].mean().to_dict('index').items()},
        debt=d.frac_debt.mean(), step1=tr[0].mean(), step25=tr[24].mean(),
        traj_by_beta={int(b): [g[0].mean(), g[24].mean()] for b, g in tr.assign(beta=d.beta).groupby('beta')},
        frac_desp_by_thresh=d.groupby('thresh').frac_desp_final.mean().to_dict(),
        r=cm.corr().iloc[0, 1], R2=fit.rsquared,
        b={t: fit.params[t] for t in terms},
        peta={t: a.loc[t, 'sum_sq'] / (a.loc[t, 'sum_sq'] + ssr) for t in terms},
        eta={t: a.loc[t, 'sum_sq'] / sst for t in terms},
    )

out = dict(ms=MS)
for s in ['legacy', 'fullfix', 'betafix']:
    try: out[s] = summarize(s)
    except FileNotFoundError: pass
q = pd.read_csv(R + 'results_ineq.csv')
q['level'] = q['shape'].map({1: 'low', 2: 'mid', 3: 'high'})
g = q.groupby(['rule', 'thresh', 'level']).agg(gap0=('gap_w0', 'mean'), gapF=('gap_final', 'mean'), c1=('crime_step1', 'mean'),
     cF=('crime_final', 'mean'), fd1=('frac_desp_step1', 'mean'), fdF=('frac_desp_final', 'mean')).reset_index()
out['ineq'] = g.to_dict('records'); out['ineq_reps'] = int(q.rep.nunique())
try:
    t = pd.read_csv(R + 'results_traj.csv')
    tt = t.groupby(['ver', 'thresh', 'step'])[['crime', 'fdesp', 'desp_rate', 'nond_rate', 'D_desp', 'gap', 'median', 'debt']].mean().reset_index()
    out['traj'] = tt[tt.step.isin([1, 5, 10, 25, 50])].to_dict('records')
    out['traj_corr'] = {f'{v}|{th}': float(np.corrcoef(g_.gap, g_.crime)[0, 1]) for (v, th), g_ in tt.groupby(['ver', 'thresh'])}
except FileNotFoundError: pass
json.dump(out, open(R + 'stats.json', 'w'), indent=1, default=float)
print(json.dumps({k: (v if k in ('ms',) else '...') for k, v in out.items()})[:200])
