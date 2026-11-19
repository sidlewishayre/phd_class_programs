#!/usr/bin/env bash
set -e

echo "Computing bank betas..."
/home/sidlh/Documents/python_envs/data_env/bin/python compute_betas.py
echo "Generating quarterly plots..."
/home/sidlh/Documents/python_envs/data_env/bin/python raw_data_plots.py
echo "Running beta regressions and generating plots..."
/home/sidlh/Documents/python_envs/data_env/bin/python beta_regs.py
echo "Running robustness regressions..."
/home/sidlh/Documents/python_envs/data_env/bin/python beta_regs_robustness.py
echo "Computing bank MMF betas..."
/home/sidlh/Documents/python_envs/data_env/bin/python mmf_betas.py
echo "Running regressions with the MMF beta control..."
/home/sidlh/Documents/python_envs/data_env/bin/python mmf_control_regs.py
echo "Running out-of-sample regressions..."
/home/sidlh/Documents/python_envs/data_env/bin/python out_of_sample.py
echo "Running the audit replication..."
bash ai_audit/download_data.sh
/home/sidlh/Documents/python_envs/data_env/bin/python ai_audit/prepare_data.py
/home/sidlh/Documents/python_envs/data_env/bin/python ai_audit/replicate_table9.py
/home/sidlh/Documents/python_envs/data_env/bin/python ai_audit/comparisons.py
echo "Running the DKN replication..."
/home/sidlh/Documents/python_envs/data_env/bin/python dkn_comparison/dkn_computation.py
/home/sidlh/Documents/python_envs/data_env/bin/python dkn_comparison/dkn_regs.py
echo "Compiling the PDFs..."
pdflatex -interaction=nonstopmode -halt-on-error pset1_writeup.tex
pdflatex -interaction=nonstopmode -halt-on-error pset1_figures.tex
