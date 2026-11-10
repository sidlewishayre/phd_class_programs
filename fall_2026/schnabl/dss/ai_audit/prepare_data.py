"""Read only the supplied Call Reports and downloaded Chicago Fed raw files."""
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def historical_quarter(path):
    with zipfile.ZipFile(path) as z:
        member = next(n for n in z.namelist() if n.lower().endswith('.xpt'))
        with z.open(member) as f:
            reader = pd.read_sas(f, format='xport', chunksize=500)
            return pd.concat([select_historical(chunk) for chunk in reader], ignore_index=True)


def select_historical(raw):
    raw.columns = raw.columns.str.upper()

    def get(*codes):
        out = pd.Series(np.nan, index=raw.index)
        for code in codes:
            if code in raw:
                out = out.fillna(raw[code])
        return out.mask(out.abs() < 1e-30, 0)

    return pd.DataFrame({
        'rssdid': get('RSSD9001'), 'cert': get('RSSD9050'),
        'dateq': pd.to_datetime(get('RSSD9999').astype('int64').astype(str)),
        'chartertype': get('RSSD9048'),
        'domdep': get('RCON2200'),
        'totdep': get('RCFD2200', 'RCON2200'),
        'intexpalldep': get('RIAD4170'),
        'assets': get('RCFD2170', 'RCON2170'),
        'loans': get('RCFD2122', 'RCON2122'),
        'reloans': get('RCFD1410', 'RCON1410'),
        'ciloans': get('RCFD1766', 'RCON1766'),
        'securities': get('RCFD1754', 'RCON1754') + get('RCFD1773', 'RCON1773'),
    })


def main():
    cache = ROOT / 'historical_selected.pkl'
    if cache.exists():
        old = pd.read_pickle(cache)
    else:
        frames = []
        for year in range(1994, 2001):
            for month in (3, 6, 9, 12):
                p = ROOT / f'call{year % 100:02}{month:02}.zip'
                frames.append(historical_quarter(p))
                print(f'Read {p.name}: {len(frames[-1]):,} banks', flush=True)
        old = pd.concat(frames, ignore_index=True).sort_values(['rssdid', 'dateq'])
        # Raw Schedule RI is year-to-date. Do not difference across years/gaps.
        q = old.dateq.dt.to_period('Q').astype('int64')
        lagq = q.groupby(old.rssdid).shift()
        lag_exp = old.groupby('rssdid').intexpalldep.shift()
        old['intexpalldep'] = old.intexpalldep.where(
            old.dateq.dt.quarter == 1,
            (old.intexpalldep - lag_exp).where(q - lagq == 1))
        old.to_pickle(cache)

    cache = ROOT / 'provided_analysis.pkl'
    if cache.exists():
        new = pd.read_pickle(cache)
    else:
        cols = ['rssdid', 'cert', 'dateq', 'chartertype', 'domdep', 'foreigndep',
                'intexpalldep', 'assets', 'loans', 'reloans', 'ciloans', 'securities']
        reader = pd.read_stata(ROOT / 'call_reports.dta', columns=cols,
                               convert_categoricals=False, chunksize=10000)
        new = pd.concat([chunk.loc[chunk.dateq.dt.year <= 2013].copy()
                         for chunk in reader], ignore_index=True)
        # Supplied income variables are already quarterly, unlike raw RI fields.
        new['totdep'] = new.domdep + new.foreigndep.fillna(0)
        new.to_pickle(cache)

    panel = pd.concat([old, new], ignore_index=True)
    panel = panel.loc[(panel.cert > 0) &
                      (panel.chartertype.isna() | (panel.chartertype == 200))].copy()
    panel['q'] = panel.dateq.dt.to_period('Q').astype('int64')
    panel = panel.sort_values(['cert', 'q'])
    if panel.duplicated(['cert', 'q']).any():
        raise ValueError('Multiple reports for a certificate-quarter: investigate before estimating.')
    panel.to_pickle(ROOT / 'analysis_panel.pkl')
    print(f'Panel: {len(panel):,} observations, {panel.cert.nunique():,} banks', flush=True)
    print(panel.groupby(panel.dateq.dt.year).size().to_string())


if __name__ == '__main__':
    main()
