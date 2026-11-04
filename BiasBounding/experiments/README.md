# experiments

This folder holds the experiment scripts. Each script's docstring describes its design in full; `--help` prints it. Run them from `BiasBounding/`.

## Synthetic

The populations are exact (no sampling), so the true values are known.

| Script | What it does | Output |
|---|---|---|
| `synthetic_limit.py` | Main synthetic experiments (`--exp compare joint sensitivity cost ...`): classifier comparison, joint bounds on two metrics, sensitivity to conditional independence, and the cost of the grid | `results/synthetic/` |
| `synthetic_figure.py` | Summary figure built from the CSVs of `synthetic_limit.py` | `results/synthetic/figures/` |
| `synth.py` | Strengths and limits of the grid as strength, share, groups and levels vary | `experiment_results/synthetic/` |
| `exp0_synth.py` | Experiment 0: validates the grid against Kallus' sharp bounds and their convergence in `num` | `experiment_results/exp0/` |
| `exp_baselines.py` | All methods (CI, Kallus, grid, hit-and-run, vertex, SCIP) across the synthetic scenarios | `experiment_results/baselines/` |
| `compare_baselines_test.py` | All methods on one synthetic population, printed as a table | stdout |
| `synthetic_limit_nonlinear.py` | Empty placeholder | none |

## HMDA (stand-alone)

| Script | What it does | Output |
|---|---|---|
| `exp0_hmda.py` | Grid vs Kallus on the HMDA sample, used as the population | `experiment_results/exp0_hmda/` |
| `exp0_binning.py` | Sensitivity of the sharp bounds to the income and county binning | `experiment_results/exp0_binning_tables/` |
| `exp1_hmda.py` | Three groups with a realistic train/internal/external split | `experiment_results/exp1_hmda/` |

The full HMDA pipeline is in [`hmda/`](hmda/README.md).

## COMPAS

The COMPAS ranking case study is in [`compas/`](compas/README.md).

## Shared code

- `fs_common.py`: table helpers and the matplotlib paper style, used by the COMPAS and synthetic scripts.
- `experiment_results/hmda_binning/`: output of `hmda/hmda_binning.py`. The script's default `--out` resolves to this folder rather than to the top-level `experiment_results/`.
