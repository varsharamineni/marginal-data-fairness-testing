# fairness-bounds

Estimate or bound the fairness disparities DD, TPRD and TNRD when the protected class A is
not observed together with predictions Ŷ and outcomes Y. A is available only in a separate
auxiliary dataset that shares discrete proxies Z. The bounds implement Kallus, Mao & Zhou,
*Assessing Algorithmic Fairness with Unobserved Protected Class Using Data Combination*,
Section 5 (Propositions 7 and 8).

## Pipeline

1. **Data.** Either generate synthetic data or load `primary.csv` (Z, Y, and Ŷ or features)
   and `auxiliary.csv` (Z, A).
2. **Classifier.** If the primary data has no Ŷ column, fit a classifier on part of it and
   predict Ŷ on the held-out rows.
3. **Methods.**
   * `ci_estimate`: a point estimate under conditional independence of A and the outcomes
     given Z. This uses the weights w_α = P(A = α | Z).
   * `kallus_bounds`: the closed-form partial identification set.
   * `feasible_joints`: a list of joint distributions P(A, Z, Ŷ, Y) that reproduce both
     marginals, from a grid or random samples. You can evaluate any function of a joint on
     every one of them with `joint_metrics(joints, func)`. For example, `joint_disparity`
     computes DD, TPRD or TNRD.

## Run

```bash
conda env create -f environment.yml
conda activate fairness-bounds
pytest                  # 14 tests
python run.py           # uses config.yaml
```

To use your own data, set `data.source: csv` in `config.yaml`, give the two file paths, and
set the column names and `a_levels`. Results go to `outputs/results.csv`. For synthetic data,
the table also has a `truth` column computed with the hidden A. If `config.yaml` has a `joints`
section, the table also has `joint_min` and `joint_max`. These are the range of each metric
over the feasible joint distributions. The per-joint values are saved to `joint_metrics.csv`.

## Mapping to the paper (`fairness/methods.py`)

| code | paper |
|---|---|
| `frechet_weights` | w^L, w^U (Prop. 7) and w̃^L, w̃^U (Prop. 8) |
| `mu` | μ(α; w), Eq. (7) |
| `mu_prime` | μ′_{ŷy}(α; w̃, w̃′) (Prop. 8); equals μ_{ŷy}(α; w̃), Eq. (8), when w̃′ = w̃ |
| `kallus_bounds(..., "DD")` | Eq. (11) |
| `kallus_bounds(..., "TPRD")` | Eq. (14), μ′₁₁ |
| `kallus_bounds(..., "TNRD")` | Eq. (15), μ′₀₀ |

Estimation is plug-in. Each E[·] is a sample mean over the primary data. P(Ŷ=ŷ | Z) and
P(Ŷ=ŷ, Y=y | Z) are empirical frequencies in the primary data. P(A=α | Z) comes from the
auxiliary data. P(A=α) = E[P(A=α | Z)], averaged over the primary data.

## Files

* `config.yaml`: all settings, covering the data source, synthetic parameters, column names,
  classifier and metrics.
* `run.py`: runs the three steps, prints the results and saves them.
* `fairness/data.py`: synthetic generator, CSV loading, validation. It checks that Ŷ and Y
  are binary, that A has exactly two levels, and that every primary Z level appears in the
  auxiliary data.
* `fairness/classifier.py`: logistic regression or random forest. It trains on
  `train_fraction` of the primary rows and predicts Ŷ on the rest.
* `fairness/methods.py`: the CI estimate, the Kallus bounds, and the true disparity for
  synthetic data.
* `tests/test_methods.py`: tests for a hand-computed example, truth and CI estimate lying
  inside the bounds, bounds collapsing to the truth when Z determines A, and rejection of
  Z levels missing from the auxiliary data.
