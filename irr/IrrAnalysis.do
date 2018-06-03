clear all
import delimited "C:\Users\DavidsSurface4\Documents\GitHub\cs224u-master\cs224u-project-master\cs224u-project-master\data\irr\Sample Ratings With Lief - DL.csv"
gen rater1=rate_heresml 
drop rate_heresml
save "C:\Users\DavidsSurface4\Documents\GitHub\cs224u-master\cs224u-project-master\cs224u-project-master\data\irr\Rater1irr.dta" ,replace
import delimited "C:\Users\DavidsSurface4\Documents\GitHub\cs224u-master\cs224u-project-master\cs224u-project-master\data\irr\Sample Ratings With Lief - Lief.csv", clear 
gen rater2=rate_heresml


joinby session_id sequence using "C:\Users\DavidsSurface4\Documents\GitHub\cs224u-master\cs224u-project-master\cs224u-project-master\data\irr\Rater1irr.dta", unmatched(none)

foreach var in rater1 rater2{

replace `var'="1" if `var'=="s"
replace `var'="2" if `var'=="m"
replace `var'="3" if `var'=="l"
destring `var',replace force
}
drop if rater2==.

kap rater1  rater2 ,wgt(w)
gen true_rate=1 if response_time_sec<15
replace true_rate=2 if response_time_sec>=15
replace true_rate=3 if response_time_sec>=45
