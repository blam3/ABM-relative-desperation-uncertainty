import sys, pandas as pd, numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
for study in sys.argv[1:]:
    d = pd.read_csv(f'results_{study}.csv')
    print(f'\n===== {study}  n={len(d)} runs =====')
    print('mean w(p) by beta:', d.groupby('beta').mean_wp.mean().round(3).to_dict(),
          ' share w>1:', d.groupby('beta').frac_wp_gt1.mean().round(3).to_dict(),
          ' share w=1:', d.groupby('beta').frac_wp_eq1.mean().round(3).to_dict())
    for v in ['beta', 'alpha', 'thresh']:
        print(v, d.groupby(v).crime_final.mean().round(3).to_dict())
    print('beta x alpha (crime_final):'); print(d.pivot_table('crime_final', 'beta', 'alpha').round(3))
    print('neutral cell beta=0 alpha=1 thresh=.2:', round(d.query('beta==0 and alpha==1 and thresh==0.2').crime_final.mean(), 3))
    d['gap'] = d.desp_rate - d.nondesp_rate
    print('desperation gap by beta:', d.groupby('beta').gap.mean().round(3).to_dict(), ' overall', round(d.gap.mean(), 3))
    print('LPM b_desp mean', round(d.b_desp.mean(), 3), ' b_wealth mean', round(d.b_wealth.mean(), 3), ' ratio', round(abs(d.b_desp.mean() / d.b_wealth.mean()), 1))
    print('wealth by beta (mean/median/p90):', d.groupby('beta')[['mean_wealth', 'median_wealth', 'p90_wealth', 'frac_debt']].mean().round(1).to_dict('index'))
    print('crime step1 vs final (all):', round(d.crime_step1.mean(), 3), round(d.crime_final.mean(), 3))
    cm = d.groupby(['beta', 'alpha', 'thresh'])[['gap_final', 'crime_final']].mean()
    print('r(P90-P10 final, crime final) across 45 cells:', round(cm.corr().iloc[0, 1], 3))
    x = d.copy()
    x['zb'] = (x.beta - x.beta.mean()) / x.beta.std()
    x['la'] = np.log2(x.alpha); x['za'] = (x.la - x.la.mean()) / x.la.std()
    x['zt'] = (x.thresh - x.thresh.mean()) / x.thresh.std()
    fit = smf.ols('crime_final ~ zb * za * zt', data=x).fit()
    a = anova_lm(fit, typ=3)
    ss_res = a.loc['Residual', 'sum_sq']
    sst = ((x.crime_final - x.crime_final.mean())**2).sum()
    tab = pd.DataFrame({'b': fit.params, 'partial_eta2': a['sum_sq'] / (a['sum_sq'] + ss_res), 'eta2': a['sum_sq'] / sst}).drop(['Intercept', 'Residual'], errors='ignore').round(3)
    print('R2', round(fit.rsquared, 3)); print(tab.to_string())
    print('sum of partial eta2:', round(tab.partial_eta2.sum(), 2), ' sum of eta2:', round(tab.eta2.sum(), 3))
    print('mean crime by beta x thresh:'); print(d.pivot_table('crime_final','beta','thresh').round(3))
    print('traj mean step1/5/10/25 by beta:'); tr = d.traj.str.split(';', expand=True).astype(float); tr['beta']=d.beta; print(tr.groupby('beta')[[0,4,9,24]].mean().round(3))
