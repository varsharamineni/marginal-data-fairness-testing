# experiments/hmda

This folder holds the full HMDA experiment. It uses the 1% HMDA 2011–12 sample from the Kallus et al. replication repository (`--repo`, default `~/kallus`). Each repetition splits the sample at random into three parts:

- train (20%), used to fit the classifier;
- internal (40%): `D`, `Y`, `Z`;
- external (40%): `Z`, `A`.

A *unit* is one combination of decision (`model`/`lender`), cut source, repetition, setting (`Black-White`, `API-White`, `three`), proxy (`county`, `income`, `both`), internal variable (`D`, `DY`) and version (`split`/`exact`). See `hmda_common.py` for details.

Every method script appends rows to its own CSV and skips units it has already done, so it can be stopped and rerun.

## Pipeline

Run the steps in order. Outputs go to `experiment_results/hmda_full/` by default.

| Step | Script | Output |
|---|---|---|
| 0 | `hmda_prepare.py`: builds every unit's tables and true values | `prepared/units.pkl` (git-ignored) |
| 1 | `hmda_grid.py`: feasible-set bounds for each `num` | `bounds_grid.csv` |
| 2 | `hmda_ci.py`: conditional-independence point values | `bounds_ci.csv` |
| 3 | `hmda_kallus.py`: Kallus et al. bounds (DD, TPRD, PPVD) | `bounds_kallus.csv` |
| 4 | `hmda_hit_and_run.py`: hit-and-run bounds | `bounds_hit_and_run.csv` |
| 5 | `hmda_vertex.py`: vertex bounds | `bounds_vertex.csv` |
| 6 | `hmda_scip.py`: SCIP bounds (each solve runs in its own process) | `bounds_scip.csv` |
| 6b | `hmda_mi.py`: bounds on `I(D;A)` and `I(D;A\|Y)` that exploit convexity | `bounds_mi.csv` |
| 7 | `hmda_collect.py`: combines all of the above | `bounds_long.csv`, `bounds_wide.csv`, `coverage.csv` |

## Analysis and figures

These scripts run after step 7, or after step 0 for the ones that work from the grid's joints.

| Script | What it does | Output |
|---|---|---|
| `hmda_compare.py` | Gaps between each method and a reference (the envelope of all methods, or `--reference <method>`) | `compare/` |
| `hmda_figure.py`, `hmda_figure_new.py` | Bound intervals for each metric and method in one setting (`_new` draws error bars and has a `--separate` option) | `figures/bounds_*.png/.pdf` |
| `hmda_questions.py` | Sensitivity to conditional independence (breakdown points) and a comparison of classifiers A and B | `questions/` |
| `hmda_extenstions_figure.py` | Two-panel figure of the `hmda_questions.py` analyses | `figures/extensions_*` |
| `hmda_regions.py` | Joint feasible regions for pairs of metrics | `figures/regions_*` |
| `hmda_binning.py` | Sensitivity of Kallus' bounds to the proxy binning | `experiments/experiment_results/hmda_binning/` |

`hmda_common.py` holds the shared argument parser, unit loading and paths.
