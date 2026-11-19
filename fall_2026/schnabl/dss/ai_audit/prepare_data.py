"""Prepare the audit panel from the New York Fed Call Reports."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))

from get_data import load_call_reports


def main():
    panel = load_call_reports(
        columns=["chartertype", "totdep", "reloans", "ciloans"],
    ).rename(columns={"yq": "dateq"})
    panel = panel.sort_values(["rssdid", "dateq"])
    # Raw Schedule RI is year-to-date. Do not difference across years/gaps.
    q = panel.dateq.dt.to_period("Q").astype("int64")
    lagq = q.groupby(panel.rssdid).shift()
    lag_exp = panel.groupby("rssdid").intexpalldep.shift()
    panel["intexpalldep"] = panel.intexpalldep.where(
        panel.dateq.dt.quarter == 1,
        (panel.intexpalldep - lag_exp).where(q - lagq == 1),
    )

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
