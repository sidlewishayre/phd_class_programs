Run `comparisons.py` with `/home/sidlh/Documents/python_envs/data_env/bin/python`.
It writes the four-column regression table and supporting files to
`results/ai_audit_comparisons/`.

The columns are cumulative:

1. **Original replication:** baseline data preparation, domestic deposits and
   domestic deposit interest expense, no trimming of quarterly inputs, 5%
   winsorization of estimated betas, and at least six complete observations.
2. **1% quarterly winsorization:** use the audit's
   Stata order-statistic 1% trimming of deposit rates (before spread differences)
   and all six stock growth series (before first-stage regressions), together
   with its 1% trimming of estimated betas instead of the baseline's 5%.
3. **Plus total deposits:** retain column 2 but use total deposits for growth
   and lagged total deposits for the interest-rate denominator, together with
   total deposit interest expense. Missing foreign deposits are filled with
   zero as in the audit. Other baseline preparation remains unchanged.
4. **Full AI audit:** use the audit's prepared panel and policy-rate inputs,
   within-panel lags, seven-observation minimum, near-zero coefficient exclusion,
   and 1% winsorization of estimated betas as well as quarterly inputs.
   The baseline bank-beta estimator is reused with these options.

All columns retain the original common-bank-sample requirement across the seven
first-stage betas. Only deposit beta on spread beta is run in the cross section,
with an intercept and HC1 robust standard errors. The bank sample can change
across columns; the last step combines the remaining audit differences and does
not isolate any one of them. `methodologies.csv` records the main choices,
`bank_betas.csv` preserves first-stage estimates and observation counts, and
`coefficients.csv` records coefficients, standard errors, and confidence limits.

The writeup describes only the audit's quarterly trimming; the audit code also
trims final betas at 1%, which is included in columns 2--4 here. Omitting that
second trimming stage leaves extreme betas from short bank histories and does
not reproduce the audit's winsorization method.

Diagnostic: quarterly trimming without final-beta trimming produced a slope
of +12.875. Certificate 57121 had six usable observations, spread beta 87.232,
and deposit beta 3421.693. Excluding this bank alone gave -3.664. Six observations
fit an intercept and five policy slopes with no residual degrees of freedom;
capping quarterly inputs does not prevent extreme fitted slopes. Columns 2--3
therefore include the audit's final-beta trimming as well as input trimming.
