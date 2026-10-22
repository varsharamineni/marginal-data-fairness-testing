"""Split a dataset that contains the protected class A into
   primary.csv   : every column except A, plus a hidden copy of A (for computing the truth)
   auxiliary.csv : proxy columns Z and A only
The two halves are disjoint random rows, mimicking two separate data sources.

Usage:
  python scripts/split_dataset.py data/german.csv --a sex --z housing
"""
import argparse
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("--a", required=True, help="protected-class column")
ap.add_argument("--z", nargs="+", required=True, help="proxy columns")
ap.add_argument("--primary-fraction", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out-dir", default="data")
args = ap.parse_args()

df = pd.read_csv(args.input)
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])   # stray index column
df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
n = int(round(args.primary_fraction * len(df)))

primary = df.iloc[:n].rename(columns={args.a: "a_hidden"})
auxiliary = df.iloc[n:][args.z + [args.a]]

os.makedirs(args.out_dir, exist_ok=True)
primary.to_csv(os.path.join(args.out_dir, "primary.csv"), index=False)
auxiliary.to_csv(os.path.join(args.out_dir, "auxiliary.csv"), index=False)
print(f"primary: {len(primary)} rows (A hidden as 'a_hidden'), auxiliary: {len(auxiliary)} rows")
