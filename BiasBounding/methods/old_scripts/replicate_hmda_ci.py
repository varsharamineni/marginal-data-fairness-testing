"""Reproduce the paper's HMDA Figure 4 intervals (repo HMDA/CI_*.csv) with debiased_bounds."""
import numpy as np, pandas as pd, sys
from kallus_bounds import debiased_bounds
repo = sys.argv[1] if len(sys.argv) > 1 else "repo/HMDA"
for proxy, ci in [("county", "geolocation"), ("income", "income"), ("county_income", "income_geolocation")]:
    df = pd.read_csv(f"{repo}/small_proxy_{proxy}.csv")
    ref = pd.read_csv(f"{repo}/CI_{ci}.csv").set_index("race")
    p_raw = df.race.value_counts(normalize=True)
    for race, name in [("White", "White-vs-Rest"), ("Black", "Black-vs-Rest"), ("API", "API-vs-Rest")]:
        # Appendix A.5 variant: race probabilities known from the population auxiliary data,
        # primary nuisance P(Yhat=1|Z) pre-computed (column yhat1), as in computeCI.R
        res = debiased_bounds(np.zeros(len(df)), df.outcome == 1, np.ones(len(df), bool),
                              eta_a_pri=df[race], p_a=p_raw[race], eta_target_pri=df.yhat1)
        ours = np.array([res["lower"], res["upper"], res["ci_lower"], res["ci_upper"]])
        theirs = ref.loc[name, ["PI_lower", "PI_upper", "CI_lower", "CI_upper"]].to_numpy(float)
        print(f"{ci:<19}{name:<15} ours {np.round(ours,4)}  repo {np.round(theirs,4)}  max diff {np.abs(ours-theirs).max():.1e}")