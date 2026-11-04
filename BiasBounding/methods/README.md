# methods

This folder holds the code that computes bounds or point estimates from the two tables:

- `r[z, i] = p(z, internal)`, from the internal data;
- `m[z, a] = p(z, a)`, from the external data.

The experiment scripts add this folder to `sys.path` and import from it.

| File | Contents |
|---|---|
| `grid_joint.py` | **Feasible-set (grid) method.** For each proxy level, it enumerates joint tables consistent with both margins on a grid of `num` values. The `sweep` mode varies each free entry over its Fréchet–Hoeffding range. The `sequential` mode fills entries one at a time, so every grid point is valid. It can also be run from the command line on two CSVs. |
| `ci_baseline.py` | Conditional-independence point estimate, `q(z,i,a) = p(z,i) p(a\|z)`. |
| `kallus_bounds.py` | Python 3/scipy re-implementation of Kallus, Mao & Zhou (2022). It includes the closed-form bounds (binary `A`), LP bounds for more than two groups, and debiased estimators with confidence intervals. |
| `hit_and_run.py` | Uniform random joints from the feasible polytope, drawn by hit-and-run. |
| `vertex_sampling.py` | Vertices of each level's transportation polytope, found by basis enumeration. Combinations of vertices are fully enumerated up to a cap and sampled above it. |
| `global_opt.py` | SCIP global optimisation of each metric. Returns the best value found and SCIP's proven dual bound. |
| `hmda_build.py` | Builds a master HMDA dataset from the 2007–2017 loan-level files (edit its CONFIG block, then run it). Writes to `data/hmda_processed/`, which is git-ignored. |

`ci_baseline.py`, `hit_and_run.py` and `vertex_sampling.py` share one interface: `(r, m) → Q[N, Z, I, G]`.

Several modules run a self-test when executed directly, for example `python ci_baseline.py`.

## Subfolders

- `configs/`: YAML experiment configs (`hmda_simple.yaml`, `hmda_master.yaml`) for the config runner `old_scripts/run_config.py`.
- `old_scripts/`: earlier pipelines and replication scripts, kept for reference. The experiments in `experiments/` replace them.
