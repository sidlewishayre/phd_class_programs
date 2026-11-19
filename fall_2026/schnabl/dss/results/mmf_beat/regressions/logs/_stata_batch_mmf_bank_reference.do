
        
    cd /home/sidlh/Documents/class_programs/fall_2026/schnabl/dss/results/mmf_beat
    use regression_inputs
    

        
program define make_rtable, rclass

    matrix b = e(b)
    matrix V = e(V)

    local k = colsof(b)

    matrix rtable = J(9, `k', .)

    local use_t = !missing(e(df_r))

    if `use_t' {
        scalar df = e(df_r)
        scalar crit = invttail(df, .025)
    }
    else {
        scalar df = .
        scalar crit = invnormal(.975)
    }

    forvalues i = 1/`k' {

        scalar beta = b[1,`i']
        scalar se = sqrt(V[`i',`i'])
        scalar stat = beta/se

        if `use_t' {
            scalar p = 2*ttail(df, abs(stat))
        }
        else {
            scalar p = 2*normal(-abs(stat))
        }

        matrix rtable[1,`i'] = beta
        matrix rtable[2,`i'] = se
        matrix rtable[3,`i'] = stat
        matrix rtable[4,`i'] = p
        matrix rtable[5,`i'] = beta - crit*se
        matrix rtable[6,`i'] = beta + crit*se
        matrix rtable[7,`i'] = df
        matrix rtable[8,`i'] = crit
        matrix rtable[9,`i'] = 0
    }

    matrix colnames rtable = `: colnames e(b)'
    matrix rownames rtable = b se t pvalue ll ul df crit eform

    return matrix table = rtable

end
regress depositflow mmfflow if cert == 8, vce(robust)


// save values if not accessible
capture confirm matrix r(table)
if _rc {
    make_rtable
}

// get list of stored items
return list

// show coefficients matrix
matlist r(table)

// save matrix as results
matrix results=r(table)

// save other coeficients
local N = e(N)

local r2 = e(r2)


// preserve data
preserve

// drop extra rows
keep in 1/15

// save table
svmat double results, name(coef)

// make table
keep coef*

local row_names : rownames results
local col_names : colnames results
forvalues i=1/`: word count `col_names'' {
    local newname = "`: word `i' of `col_names''"
    local newname = subinstr("`newname'", ".", "", .)
    local newname = subinstr("`newname'", "_", "", .)
    local newname = subinstr("`newname'", "#", "", .)
    local newname = subinstr("`newname'", "/", "", .)
    local newname = subinstr("`newname'", "(", "", .)
    local newname = subinstr("`newname'", ")", "", .)
    local newname = substr("`newname'", 1, 24)
    rename coef`i' coef`i'_`newname'
}
gen row_names = ""
forvalues i=1/`: word count `row_names'' {
  replace row_names=`"`: word `i' of `row_names''"' in `i'
}
keep if row_names != ""

// add save vars
 

// save table
cd /home/sidlh/Documents/class_programs/fall_2026/schnabl/dss/results/mmf_beat/regressions/regs/mmf_bank_reference
save coefs, replace

// add save vars

            ds
            local col_names_new = r(varlist)
            local col_num = `: word count `col_names_new'' - 1
            
            insobs 1      
            gen var_row = _N == _n
            forvalues i=1/`col_num' {
                local name = "`: word `i' of `col_names_new''"
                replace `name' = `N' if var_row == 1
            }
            replace row_names = "N" if var_row == 1
            drop var_row
            

            insobs 1      
            gen var_row = _N == _n
            forvalues i=1/`col_num' {
                local name = "`: word `i' of `col_names_new''"
                replace `name' = `r2' if var_row == 1
            }
            replace row_names = "r2" if var_row == 1
            drop var_row
             

// save save vars

            gen save_stats_row_names = 0
            foreach var in N r2{
                    replace save_stats_row_names = 1 if row_names == "`var'"
            }
            keep if save_stats_row_names == 1
            keep row_names coef1_*
            
            save /home/sidlh/Documents/class_programs/fall_2026/schnabl/dss/results/mmf_beat/regressions/regs/mmf_bank_reference/save_stats_info.dta, replace
            


// restore data
restore

preserve
statsby mmf_beta=_b[mmfflow] se=_se[mmfflow] intercept=_b[_cons] nobs=e(N) r2=e(r2), by(cert) saving("/home/sidlh/Documents/class_programs/fall_2026/schnabl/dss/results/mmf_beat/bank_betas.dta", replace): regress depositflow mmfflow, vce(robust)
restore



        cd /home/sidlh/Documents/class_programs/fall_2026/schnabl/dss/results/mmf_beat
        
                
        

        cd /home/sidlh/Documents/class_programs/fall_2026/schnabl/dss/results/mmf_beat
        

        clear
        exit, STATA
        