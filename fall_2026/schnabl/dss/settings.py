"""Project input, cache, and output locations."""

import os
import sys

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CWD)

from pathlib import Path

ROOT = Path(CWD)
REUSABLE_CODE_DIR = ROOT.parents[3] / "reusable_code"
DATA_DIR = ROOT / "data"
NY_FED_CALL_REPORTS_DIR = DATA_DIR / "nyfed"
NY_FED_CALL_REPORTS_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/banking_research/"
    "balance-sheets-income-statements"
)
CALL_REPORTS_CACHE_DIR = DATA_DIR / "call_reports_online"
FDIC_FILE = DATA_DIR / "fdic_historical.csv"
FRED_CACHE_DIR = DATA_DIR / "fred"
RESULTS_DIR = ROOT / "results"
BETA_COMPARISON_FILE = RESULTS_DIR / "beta_comparison.tex"
ROBUSTNESS_DIR = RESULTS_DIR / "beta_regs_robustness"
ROBUSTNESS_DATA_FILE = ROBUSTNESS_DIR / "regression_inputs.dta"
ROBUSTNESS_COEFFICIENTS_FILE = ROBUSTNESS_DIR / "coefficients.csv"
MMF_DIR = RESULTS_DIR / "mmf_betas"
MMF_FRED_SERIES = "MMMFFAQ027S"
MMF_AUM_CACHE_FILE = FRED_CACHE_DIR / f"{MMF_FRED_SERIES}.csv"
MMF_DEPOSIT_SERIES = "DPSACBW027NBOG"
MMF_DEPOSIT_CACHE_FILE = FRED_CACHE_DIR / f"{MMF_DEPOSIT_SERIES}.csv"
MMF_SHARE_INPUT_FILE = MMF_DIR / "share_inputs.csv"
MMF_BETAS_FILE = MMF_DIR / "bank_betas.csv"
MMF_CONTROL_DIR = RESULTS_DIR / "mmf_control_regs"
MMF_CONTROL_REGRESSION_FILE = MMF_CONTROL_DIR / "regression_inputs.dta"
MMF_SPREAD_REG_DIR = MMF_CONTROL_DIR / "spread_on_mmf"
MMF_DECILE_DIR = MMF_CONTROL_DIR / "mmf_deciles"
OUT_OF_SAMPLE_FDIC_FILE = DATA_DIR / "fdic_out_of_sample.csv"
OUT_OF_SAMPLE_RATES_FILE = FRED_CACHE_DIR / "out_of_sample_targets.csv"
OUT_OF_SAMPLE_TABLE_FILE = ROOT / "out_of_sample.tex"
OUT_OF_SAMPLE_DIR = RESULTS_DIR / "out_of_sample"
OUT_OF_SAMPLE_REGRESSION_FILE = OUT_OF_SAMPLE_DIR / "regression_inputs.dta"
OUT_OF_SAMPLE_TABLE9_DIR = OUT_OF_SAMPLE_DIR / "table_ix"
AI_AUDIT_DIR = ROOT / "ai_audit"
AI_AUDIT_PANEL_FILE = AI_AUDIT_DIR / "analysis_panel.pkl"
AI_AUDIT_RATES_FILE = AI_AUDIT_DIR / "fedfunds.csv"
AI_COMPARISONS_DIR = RESULTS_DIR / "ai_audit_comparisons"

DKN_RATES_FILE = FRED_CACHE_DIR / "dkn_rates.csv"
DKN_RESULTS_DIR = RESULTS_DIR / "dkn_comparison"
DKN_REGRESSION_FILE = DKN_RESULTS_DIR / "regression_inputs.dta"
DSS_DKN_COMPARE_DIR = RESULTS_DIR / "dss_dkn_compare"
DSS_DKN_PANEL_FILE = DSS_DKN_COMPARE_DIR / "bank_inputs.dta"
DSS_DKN_AGGREGATE_FILE = DSS_DKN_COMPARE_DIR / "aggregate_inputs.dta"
