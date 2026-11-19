# Robustness

Plot coefficients across following specifications:
* Sample period: run with 5-year windows and plot coefficient overtime
* Set of banks:
    * Community, regional banks vs 250B+
    * quintiles of leverage
    * quintiles of uninsured deposits
* Winsorizing level: try 0%, 1%, 3%, 5%, 10%, 15%
* FFR lags
* minimum quarters per bank

# Audit the AI

Replicate the regresion of deposit-beta on sprad-beta in table 9, Panel A of Drechsler, Savov, and Schnabl (QJE, 2017) from raw call report data. Use the PDF of the paper, dss.pdf, and the data call_reports.dta, but do not look at other files in other directories (you can use the internet). The dataset only goes back to 2001, so you will have to pull data going back to 1994 to get the dataset to align with the paper. Make the code as simple as possible and try to match their methodology and coefficients closely. Run until numbers are within 20% of the coefficient given in the paper (−3.610) or until you use the 5 hour token limit. Make sure to have some results before you hit 5 hour limit. Remember the call report files are large, so avoid crashing the computer by using too much RAM when running the code.