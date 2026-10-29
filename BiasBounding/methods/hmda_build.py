"""
Build one master HMDA dataset (2007-2017 historic loan-level files) with every variable you
might want later, cleaned and derived, without deciding yet which are internal, common or
external. Experiments then pick roles, filters, bins and splits in their YAML configs
(run_config.py), so changing the design never means re-processing the raw files.

Edit the CONFIG block, then run:   python hmda_build.py
Output: OUT_FILE (parquet if pyarrow is installed, else .csv.gz) plus a variables.csv
data dictionary (name, type, description, values or range).

Only population-level filters are applied here (years, actions). Everything else
(races to compare, income cap, minimum cell sizes, ...) belongs in the experiment config.
"""
import os
import urllib.request

import numpy as np
import pandas as pd

# ============================== CONFIG ==============================
YEARS = [2017]
DATA_DIR = "data/hmda_raw"
OUT_FILE = "data/hmda_processed/hmda_master_2017.parquet"
URL = ("https://files.consumerfinance.gov/hmda-historic-loan-data/"
       "hmda_{year}_nationwide_all-records_labels.zip")
ACTIONS = ["Loan originated", "Application approved but not accepted",
           "Application denied by financial institution"]
INCOME_BINS = [0, 25, 50, 75, 100, 150, 250, np.inf]      # $000s, for income_bin
LOAN_TO_INCOME_BINS = [0, 1, 2, 3, 4, 5, np.inf]
TRACT_MINORITY_BINS = 5                                    # quantile bins of tract minority %
CHUNK = 1_000_000
# ====================================================================

# raw columns read when present (the 2007-2017 'labels' files); missing ones are skipped
RAW = ["as_of_year", "respondent_id", "agency_abbr", "loan_type_name", "property_type_name",
       "loan_purpose_name", "owner_occupancy_name", "loan_amount_000s", "preapproval_name",
       "action_taken_name", "msamd", "state_abbr", "state_code", "county_code",
       "census_tract_number", "applicant_ethnicity_name", "co_applicant_ethnicity_name",
       "applicant_race_name_1", "applicant_race_name_2", "co_applicant_race_name_1",
       "applicant_sex_name", "co_applicant_sex_name", "applicant_income_000s",
       "purchaser_type_name", "denial_reason_name_1", "lien_status_name", "population",
       "minority_population", "hud_median_family_income", "tract_to_msamd_income",
       "number_of_owner_occupied_units"]
REQUIRED = ["as_of_year", "action_taken_name", "applicant_ethnicity_name",
            "applicant_race_name_1", "applicant_sex_name", "applicant_income_000s"]

# output variables: name -> description (the order of the output columns)
VARIABLES = {
    "year": "HMDA activity year",
    "outcome": "1 = approved (originated or approved but not accepted), 0 = denied",
    "action": "originated / approved_not_accepted / denied",
    "denial_reason": "first denial reason (denied applications only)",
    "race": "KMZ definition: non-Hispanic single-race applicant: White, Black, API; else Other",
    "race_any_ethnicity": "applicant's first race regardless of ethnicity (White, Black, "
                          "Asian, NHPI, AIAN, NotProvided, NA)",
    "hispanic": "applicant ethnicity: Hispanic, NotHispanic, NotProvided, NA",
    "race_ethnicity": "Hispanic, else race_any_ethnicity for non-Hispanic applicants",
    "single_race": "1 if the applicant reported exactly one race",
    "sex": "Male, Female, Other (not provided / not applicable)",
    "co_applicant": "1 if there is a co-applicant",
    "co_applicant_sex": "Male, Female, None, Other",
    "income": "applicant income, $000s",
    "income_bin": f"income binned at {INCOME_BINS[:-1]} ($000s)",
    "loan_amount": "loan amount, $000s",
    "loan_to_income": "loan amount / income",
    "loan_to_income_bin": f"loan_to_income binned at {LOAN_TO_INCOME_BINS[:-1]}",
    "loan_purpose": "Home purchase, Refinancing, Home improvement",
    "loan_type": "Conventional, FHA, VA, FSA/RHS",
    "property_type": "one-to-four family, manufactured, multifamily",
    "occupancy": "owner-occupied or not",
    "lien": "first / subordinate / not secured / not applicable",
    "preapproval": "preapproval requested or not",
    "purchaser_type": "type of purchaser of the loan",
    "agency": "supervisory agency of the lender",
    "lender_id": "respondent id (agency + respondent, unique lender key)",
    "state": "state abbreviation",
    "state_code": "2-digit state FIPS",
    "county": "5-digit county FIPS (state + county)",
    "msa": "MSA/MD code",
    "tract": "11-digit census tract id (state + county + tract)",
    "tract_population": "tract population",
    "tract_minority_pct": "minority share of the tract population, %",
    "tract_minority_bin": f"tract_minority_pct in {TRACT_MINORITY_BINS} quantile bins (0 = lowest)",
    "tract_income_ratio": "tract median family income / MSA median, %",
    "msa_median_income": "HUD median family income of the MSA, $",
    "tract_owner_occupied_units": "owner-occupied units in the tract",
}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def derive(r):
    """Raw chunk (strings) -> derived variables. Add your own here and in VARIABLES."""
    g = lambda c: r[c] if c in r else pd.Series(np.nan, index=r.index, dtype=object)
    out = pd.DataFrame(index=r.index)
    out["year"] = num(g("as_of_year")).astype("Int64")
    act = g("action_taken_name")
    out["outcome"] = (act != "Application denied by financial institution").astype(int)
    out["action"] = act.map({"Loan originated": "originated",
                             "Application approved but not accepted": "approved_not_accepted",
                             "Application denied by financial institution": "denied"})
    out["denial_reason"] = g("denial_reason_name_1")

    eth, r1, r2 = g("applicant_ethnicity_name"), g("applicant_race_name_1"), g("applicant_race_name_2")
    single = r2.isna() & r1.notna()
    nonhisp = eth.eq("Not Hispanic or Latino")
    out["race"] = np.select(
        [nonhisp & single & r1.eq("White"), nonhisp & single & r1.eq("Black or African American"),
         nonhisp & single & r1.isin(["Asian", "Native Hawaiian or Other Pacific Islander"])],
        ["White", "Black", "API"], default="Other")
    race_map = {"White": "White", "Black or African American": "Black", "Asian": "Asian",
                "Native Hawaiian or Other Pacific Islander": "NHPI",
                "American Indian or Alaska Native": "AIAN",
                "Information not provided by applicant in mail, Internet, or telephone application":
                    "NotProvided", "Not applicable": "NA"}
    out["race_any_ethnicity"] = r1.map(race_map).fillna("NA")
    out["hispanic"] = eth.map({"Hispanic or Latino": "Hispanic",
                               "Not Hispanic or Latino": "NotHispanic",
                               "Information not provided by applicant in mail, Internet, or "
                               "telephone application": "NotProvided",
                               "Not applicable": "NA"}).fillna("NA")
    out["race_ethnicity"] = np.where(out["hispanic"].eq("Hispanic"), "Hispanic",
                                     out["race_any_ethnicity"])
    out["single_race"] = single.astype(int)
    sexmap = lambda s: np.select([s.eq("Male"), s.eq("Female")], ["Male", "Female"], "Other")
    out["sex"] = sexmap(g("applicant_sex_name"))
    co = g("co_applicant_sex_name")
    out["co_applicant"] = (~co.isin(["No co-applicant"]) & co.notna()).astype(int)
    out["co_applicant_sex"] = np.where(co.eq("No co-applicant"), "None", sexmap(co))

    inc, amt = num(g("applicant_income_000s")), num(g("loan_amount_000s"))
    out["income"], out["loan_amount"] = inc, amt
    lab = lambda b: [f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+" for lo, hi in zip(b[:-1], b[1:])]
    out["income_bin"] = pd.cut(inc, INCOME_BINS, labels=lab(INCOME_BINS), right=True,
                               include_lowest=True).astype(str).replace("nan", np.nan)
    out["loan_to_income"] = amt / inc.where(inc > 0)
    out["loan_to_income_bin"] = pd.cut(out["loan_to_income"], LOAN_TO_INCOME_BINS,
                                       labels=lab(LOAN_TO_INCOME_BINS)).astype(str) \
        .replace("nan", np.nan)

    for new, raw in [("loan_purpose", "loan_purpose_name"), ("loan_type", "loan_type_name"),
                     ("property_type", "property_type_name"), ("occupancy", "owner_occupancy_name"),
                     ("lien", "lien_status_name"), ("preapproval", "preapproval_name"),
                     ("purchaser_type", "purchaser_type_name"), ("agency", "agency_abbr"),
                     ("state", "state_abbr")]:
        out[new] = g(raw)
    out["lender_id"] = g("agency_abbr").fillna("") + "_" + g("respondent_id").fillna("")
    st = num(g("state_code"))
    cty = num(g("county_code"))
    out["state_code"] = st.map(lambda v: f"{int(v):02d}" if pd.notna(v) else np.nan)
    out["county"] = [f"{int(a):02d}{int(b):03d}" if pd.notna(a) and pd.notna(b) else np.nan
                     for a, b in zip(st, cty)]
    out["msa"] = g("msamd")
    tr = g("census_tract_number").str.replace(".", "", regex=False).str.zfill(6)
    out["tract"] = np.where(out["county"].notna() & tr.notna(), out["county"].astype(str) + tr, np.nan)
    out["tract_population"] = num(g("population"))
    out["tract_minority_pct"] = num(g("minority_population"))
    out["tract_income_ratio"] = num(g("tract_to_msamd_income"))
    out["msa_median_income"] = num(g("hud_median_family_income"))
    out["tract_owner_occupied_units"] = num(g("number_of_owner_occupied_units"))
    return out


def download(year):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, os.path.basename(URL.format(year=year)))
    if not os.path.exists(path):
        print(f"downloading {year} ...")
        try:
            urllib.request.urlretrieve(URL.format(year=year), path)
        except Exception as e:
            raise SystemExit(f"Download failed ({e}). Get the {year} nationwide 'All records' "
                             f"labels file from https://www.consumerfinance.gov/data-research/"
                             f"hmda/historic-data/ and save it as {path}")
    return path


def load(path):
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise SystemExit(f"Required columns not in {path}: {missing}")
    skipped = [c for c in RAW if c not in header]
    if skipped:
        print(f"  not in {os.path.basename(path)} (variables built from them will be empty): {skipped}")
    parts = []
    for ch in pd.read_csv(path, usecols=[c for c in RAW if c in header], dtype=str,
                          chunksize=CHUNK):
        ch = ch[ch["action_taken_name"].isin(ACTIONS)
                & num(ch["as_of_year"]).isin(YEARS)]
        d = derive(ch)
        for c in d.columns:                                  # save memory
            if d[c].dtype == object:
                d[c] = d[c].astype("category")
        parts.append(d)
        print(f"  {os.path.basename(path)}: kept {sum(len(p) for p in parts):,} rows", end="\r")
    print()
    return pd.concat(parts, ignore_index=True)


def dictionary(df):
    rows = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 20:
            vals = f"{s.min():g} .. {s.max():g} (median {s.median():g})"
        else:
            vc = s.astype(str).value_counts()
            vals = "; ".join(f"{k} ({v:,})" for k, v in vc.head(12).items()) + \
                (f"; ... {len(vc) - 12} more" if len(vc) > 12 else "")
        rows.append(dict(variable=c, description=VARIABLES.get(c, ""),
                         missing=f"{s.isna().mean():.1%}", values=vals))
    return pd.DataFrame(rows)


def main():
    df = pd.concat([load(download(y)) for y in YEARS], ignore_index=True)
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(str).replace("nan", np.nan).astype("category")
    m = df["tract_minority_pct"]
    df["tract_minority_bin"] = pd.qcut(m, TRACT_MINORITY_BINS, labels=False, duplicates="drop") \
        .astype("Int64")
    df = df[[c for c in VARIABLES if c in df.columns]]

    os.makedirs(os.path.dirname(OUT_FILE) or ".", exist_ok=True)
    out = OUT_FILE
    try:
        df.to_parquet(out, index=False)
    except ImportError:
        out = os.path.splitext(OUT_FILE)[0] + ".csv.gz"
        print("pyarrow not installed: writing csv.gz instead (pip install pyarrow for parquet)")
        df.to_csv(out, index=False)
    dic = dictionary(df)
    dic.to_csv(os.path.join(os.path.dirname(out) or ".", "variables.csv"), index=False)
    print(f"{len(df):,} rows x {df.shape[1]} variables -> {out}")
    show = dic.assign(values=dic["values"].str.slice(0, 90))
    print(show[["variable", "missing", "values"]].to_string(index=False))


if __name__ == "__main__":
    main()