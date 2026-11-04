# experiment_results

This folder holds the outputs of the experiment scripts in `experiments/`: CSVs and quick-look PNGs. The final paper figures are in `results/`.

| Folder | Produced by | Contents |
|---|---|---|
| `hmda_full/` | the `experiments/hmda/` pipeline | `bounds_<method>.csv` for each method; `bounds_long.csv`, `bounds_wide.csv` and `coverage.csv` (from `hmda_collect.py`); `compare/` (from `hmda_compare.py`); `questions/` (from `hmda_questions.py`); `figures/`. `prepared/` is git-ignored. |
| `exp0/` | `exp0_synth.py` | Grid vs Kallus on synthetic populations: `results.csv`, `summary.csv`, convergence and runtime plots |
| `exp0_synth_results/` | `exp0_synth.py` (an earlier run) | Same files as `exp0/` |
| `exp0_hmda/` | `exp0_hmda.py` | Grid vs Kallus on HMDA: results, summary, convergence plots for each setting |
| `exp0_hmda_results/` | `exp0_hmda.py` (an earlier run) | Same files as `exp0_hmda/` |
| `exp0_binning_tables/` | `exp0_binning.py` | `income_binning.csv` and `county_binning.csv` |
| `exp0_binning/` | an earlier version of `exp0_binning.py` | `bounds.csv` and `binning.png` |
| `exp1_hmda/` | `exp1_hmda.py` | Three-group HMDA results: `results.csv` and `summary.csv` |
| `baselines/` | `exp_baselines.py` | All methods across synthetic scenarios: results, summaries by class and factor, CI error, runtime and gap plots |
| `synthetic/` | `synth.py` | Results, summary and figures 1–4 (gap vs num, width vs strength, gap vs groups/share, feasibility) |
| `compas_ranking/` | an earlier COMPAS ranking run | `compas_bounds.csv` and `figures/` |
