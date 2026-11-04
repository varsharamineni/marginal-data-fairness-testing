# experiments/compas

This folder holds the COMPAS ranking case study. It bounds ranking-fairness metrics when race is withheld, in the setting of Zehlike et al. (2022), "Fair Top-k Ranking with Multiple Protected Groups".

Each defendant gets the score `0.25·decile + 0.5·v_decile + 0.25·priors`. Each term is scaled to [0, 1], and by default the score is flipped so the least likely to recidivate rank first.

`grid_joint` needs a discrete internal variable, so the ranking is cut into position blocks (default 1–500, 501–1000, 1001–1500, 1501–n). The data are set up as follows:

- internal data: the block and the proxy cell;
- external data: the proxy cell and race;
- proxies: charge degree and age category by default.

| Script | What it does | Default output |
|---|---|---|
| `compas_ranking1.py` | Main version: true individual-position and block-level NDKL, grid bounds, and exact optimisation bounds | `results/compas_zehlike/` |
| `compas_ranking.py` | Earlier, shorter version (Black vs White) | `results/compas_zehlike/` |
| `compas_replication.py` | People of colour vs White, with exposure-ratio and FA*IR ranked-group-fairness metrics (top-k, k = 1500) | `results/compas_fair/` |

Data: `compas_replication.py` downloads ProPublica's `compas-scores-two-years.csv`. The two ranking scripts read a cleaned local copy instead; set its path with `--data`.
