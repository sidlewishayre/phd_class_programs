"""Drechsler, Savov and Schnabl (2017), Table IX, Panel A, column 1.

Run prepare_data.py first. Each first-stage OLS has an intercept and the
current change in the Fed funds target plus four quarterly lags.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent
OUTCOMES = ['totdep', 'assets', 'securities', 'loans', 'reloans', 'ciloans']


def winsor(s, p=.01):
    """Stata winsor: replace the h extreme observations with adjacent values."""
    v = np.sort(s.dropna().to_numpy())
    h = int(p * len(v))
    return s.clip(v[h], v[-h - 1]) if h else s.copy()


def main():
    d = pd.read_pickle(ROOT / 'analysis_panel.pkl').sort_values(['cert', 'q'])
    f = pd.read_csv(ROOT / 'fedfunds.csv', parse_dates=['observation_date'])
    f['target'] = f.DFEDTAR.fillna((f.DFEDTARU + f.DFEDTARL) / 2)
    f['q'] = f.observation_date.dt.to_period('Q').astype('int64')
    rates = f.groupby('q').target.agg(ff_end='last', ff_avg='mean')
    rates['dff'] = rates.ff_end.diff()
    d = d.merge(rates, on='q', how='left', validate='many_to_one')
    g = d.groupby('cert', sort=False)
    consecutive = d.q - g.q.shift() == 1

    # Annualized quarterly interest expense divided by previous quarter deposits.
    rate = 400 * d.intexpalldep / g.totdep.shift().where(consecutive)
    d['spread'] = d.ff_avg - winsor(rate.replace([np.inf, -np.inf], np.nan))
    d['dy_spread'] = d.groupby('cert').spread.diff().where(consecutive)
    for name in OUTCOMES:
        loglevel = np.log(d[name].where(d[name] > 0))
        growth = 100 * loglevel.groupby(d.cert).diff().where(consecutive)
        d['dy_' + name] = winsor(growth)

    # Reproduce time-series lags within the retained 1994–2013 bank panel.
    xs = ['dff']
    for lag in range(1, 5):
        col = f'dff_lag{lag}'
        d[col] = g.dff.shift(lag).where(d.q - g.q.shift(lag) == lag)
        xs.append(col)

    names = ['spread'] + OUTCOMES
    beta_frames = []
    for name in names:
        rows = []
        for cert, bank in d.groupby('cert', sort=False):
            b = bank[xs + ['dy_' + name]].dropna()
            if len(b) < 7:
                continue
            x = np.column_stack([np.ones(len(b)), b[xs].to_numpy()])
            coef, _, rank, _ = np.linalg.lstsq(x, b['dy_' + name], rcond=None)
            # Authors drop banks with missing/omitted/zero policy coefficients.
            if rank < 6 or np.any(np.abs(coef[1:]) < 1e-12):
                continue
            rows.append((cert, coef[1:].sum(), len(b)))
        beta = pd.DataFrame(rows, columns=['cert', name + '_raw', name + '_n']).set_index('cert')
        beta[name + '_beta'] = winsor(beta[name + '_raw'])
        beta_frames.append(beta)
        print(f'{name}: {len(beta):,} bank betas', flush=True)

    # The published table uses a common bank sample across all six outcomes.
    betas = pd.concat(beta_frames, axis=1, join='inner')
    fit = sm.OLS(betas.totdep_beta, sm.add_constant(betas.spread_beta)).fit(cov_type='HC1')
    result = {
        'coefficient': float(fit.params.spread_beta),
        'robust_se_HC1': float(fit.bse.spread_beta),
        'intercept': float(fit.params['const']),
        'n_banks': int(fit.nobs), 'r_squared': float(fit.rsquared),
        'paper_coefficient': -3.610, 'paper_se': .203,
        'paper_n_banks': 11091, 'paper_r_squared': .132,
        'relative_error': float(abs(fit.params.spread_beta / -3.610 - 1)),
        'spread_beta_raw_mean': float(betas.spread_raw.mean()),
        'spread_beta_raw_p10': float(betas.spread_raw.quantile(.1)),
        'spread_beta_raw_p90': float(betas.spread_raw.quantile(.9)),
    }
    betas.to_csv(ROOT / 'bank_betas.csv')
    (ROOT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
