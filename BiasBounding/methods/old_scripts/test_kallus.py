import numpy as np, time
from kallus_bounds import *

rng = np.random.default_rng(1)
d = np.array([0,0,1,1], bool); y = np.array([0,1,0,1], bool)   # internal (d,y): 00 01 10 11
RATES = {k: f(d, y) for k, f in PAPER_RATES.items()}

def random_problem(Z, G=2):
    pz = rng.dirichlet(np.ones(Z))
    r = rng.dirichlet(np.ones(4) * rng.uniform(.3, 3), size=Z) * pz[:, None]
    pa = rng.dirichlet(np.ones(G) * rng.uniform(.3, 3), size=Z)
    return r, pa * pz[:, None]

# 1. LP + grid (Sec. 6/7.2 machinery) reproduces the closed forms (Props 7-8)
worst, t0 = 0, time.time()
for _ in range(10):
    r, m = random_problem(rng.integers(2, 25))
    for name, (tg, cd, cells) in RATES.items():
        cf = np.array(closed_form_bounds(r, m, tg, cd))
        lp = lp_bounds(r, m, tg, cd, cells=cells)[["lower", "upper"]].to_numpy()[0]
        worst = max(worst, np.abs(cf - lp).max())
print(f"1. max |LP - closed form| over 50 cases: {worst:.2e}  ({time.time()-t0:.0f}s)")

# 2. the joint attaining each bound: brute-force random couplings never escape the interval
def random_coupling(r, m):
    """Sample a feasible joint q[g,i,z] by iterative proportional fitting from random start."""
    Z, I = r.shape; G = m.shape[1]
    q = rng.uniform(.01, 1, (G, I, Z))
    for _ in range(500):
        q *= (r.T / q.sum(0).clip(1e-300))[None]
        q *= (m.T / q.sum(1).clip(1e-300))[:, None]
    return q
viol = 0
for _ in range(30):
    r, m = random_problem(8)
    for name, (tg, cd, _) in RATES.items():
        lo, hi = closed_form_bounds(r, m, tg, cd)
        for _ in range(20):
            q = random_coupling(r, m)
            mu = [q[g][tg].sum() / q[g][cd].sum() for g in range(2)]
            viol += not (lo - 1e-9 <= mu[1] - mu[0] <= hi + 1e-9)
print("2. random feasible joints outside the bounds:", viol, "of 3000")

# 3. multi-group LP: the true disparity of random joints lies in the projections;
#    with 2 groups one_vs_rest LP equals the closed form
viol = 0; worst = 0
for _ in range(5):
    r, m = random_problem(6, G=3)
    for name in ["demographic_parity", "equal_opportunity"]:
        tg, cd, cells = RATES[name]
        df = lp_bounds(r, m, tg, cd, cells=cells, n_grid=31)
        for _ in range(20):
            q = random_coupling(r, m)
            mu = np.array([q[g][tg].sum() / q[g][cd].sum() for g in range(3)])
            for _, row in df.iterrows():
                v = mu[int(row.group)] - mu[0]
                viol += not (row.lower - 1e-6 <= v <= row.upper + 1e-6)
        m2 = one_vs_rest(m, 2)
        worst = max(worst, np.abs(np.array(closed_form_bounds(r, m2, tg, cd))
                     - lp_bounds(r, m2, tg, cd, cells=cells)[["lower","upper"]].to_numpy()[0]).max())
print("3. three groups: joints outside projections:", viol, "of 400; one-vs-rest LP vs closed form:", f"{worst:.1e}")

# 4. Lipschitz: L -> infinity recovers the unrestricted set; smaller L nests; truth kept when smooth
Z = 15; zv = np.sort(rng.uniform(0, 10, Z)); pz = rng.dirichlet(np.ones(Z) * 3)
pdy = rng.dirichlet(np.ones(4) * 2, Z)                                   # P(d,y | z)
base = rng.uniform(.2, .8, 4)                                            # P(A=1 | d,y,z) smooth in z
slope = rng.uniform(-.03, .03, 4)
wA = np.clip(base[None] + slope[None] * zv[:, None], 0, 1)              # [Z, 4], Lipschitz 0.03
r = pdy * pz[:, None]
m = np.stack([(r * (1 - wA)).sum(1), (r * wA).sum(1)], 1)
tg, cd, cells = RATES["equal_opportunity"]
truth = (r * wA)[:, tg].sum() / (r * wA)[:, cd].sum() - (r * (1 - wA))[:, tg].sum() / (r * (1 - wA))[:, cd].sum()
print(f"4. truth {truth:+.4f}; closed form [{closed_form_bounds(r, m, tg, cd)[0]:+.4f}, {closed_form_bounds(r, m, tg, cd)[1]:+.4f}]")
for L in [100, 0.1, 0.05, 0.03, 0.02]:
    df = lp_bounds(r, m, tg, cd, cells=cells, lipschitz=L, z_values=zv)
    print(f"   L={L:<5} [{df.lower[0]:+.4f}, {df.upper[0]:+.4f}]  contains truth: {df.lower[0]-1e-9 <= truth <= df.upper[0]+1e-9}")