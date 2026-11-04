# Marginal-data fairness testing

Bounds on fairness metrics when the protected attribute is not in the data being audited.

The setting has two datasets:

- **internal** (primary) data, such as a lender's records, which hold the decision `D`, the label `Y` and a common proxy `Z`, but not the protected attribute `A`;
- **external** (auxiliary) data, such as census data, which hold `Z` and `A` but not `D` or `Y`.

Many joint distributions `p(z, internal, a)` are consistent with both datasets. Each one gives a value for every fairness metric, and the range of those values bounds the true metric. The main method here is the **feasible-set (grid) approach**. It enumerates the consistent joints and evaluates any metric on them. The other methods are baselines:

| Method | What it gives |
|---|---|
| Grid (feasible set) | Min and max over enumerated joints. An inner bound for any metric. |
| Conditional independence (CI) | A point estimate, assuming `A ⫫ internal \| Z`. |
| Kallus, Mao & Zhou (2022) | Sharp LP or closed-form bounds for DD, TPRD and PPVD. |
| Hit-and-run | Min and max over uniform random joints. |
| Vertex enumeration | Min and max over the vertices of the feasible polytope. |
| SCIP | A certified global optimum for each bound. |

## Layout

```
BiasBounding/
  methods/             bound-computation code (grid, CI, Kallus, hit-and-run, vertex, SCIP)
  experiments/         experiment scripts: synthetic, HMDA, COMPAS
    hmda/              the full HMDA pipeline (prepare → methods → collect → figures)
    compas/            COMPAS ranking case study
  results/             final paper figures and tables (synthetic, COMPAS)
  experiment_results/  outputs of the experiment scripts (CSVs, PNGs)
```

Each of these folders has its own README. The following are git-ignored: `BiasBounding/data/` (raw and processed HMDA), `BiasBounding/old_results/`, and `experiment_results/hmda_full/prepared/`.

## Setup

Python 3 with `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn` and `pyyaml`. The SCIP baseline also needs `pyscipopt`.

The HMDA experiments read the 1% HMDA 2011–12 sample from a clone of the Kallus et al. replication repository, [CausalML/FairnessWithUnobservedProtectedClass](https://github.com/CausalML/FairnessWithUnobservedProtectedClass). The scripts look for it at `~/kallus` by default; use `--repo` to point elsewhere.

Run the scripts from `BiasBounding/`. Some default output paths, such as `results/synthetic`, are relative to that folder.
