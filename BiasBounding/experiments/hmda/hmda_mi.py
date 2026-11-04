"""
Step 6b of the HMDA full experiment: bounds on the two information metrics, I(D;A) and
I(D;A|Y), computed with their structure instead of generic global optimisation.

Both metrics depend on the joint q[z, i, a] only through theta[i, a] = sum_z q[z, i, a], and
both are CONVEX in theta (with the margins of theta fixed by the data):
  I(D;A)   = [sum theta log theta + const] / ln 2
  I(D;A|Y) = [sum_{y,a} sum_d theta log(theta / s_ay) + const] / ln 2,  s_ay = sum_d theta
(the second is a relative entropy, hence jointly convex). global_opt.py writes I(D;A|Y) as a
difference of x log x sums, which SCIP must treat as non-convex on both sides.

lower  convex minimisation by away-step Frank-Wolfe. Every iterate is a convex combination of
       exact vertices of the feasible set (so its value is attained), and every iteration
       certifies  min f >= f(t) - max_s grad f(t).(t - s)  by convexity; mutual information is
       also never negative. Stops when value and certificate agree to --gap.
upper  maximum of a convex function over a polytope, attained at a vertex of the polytope of
       feasible theta, which has dimension (I-1)(G-1): 1 or 2 for I(D;A), 3 or 6 for
       I(D;A|Y). Its vertices are enumerated exactly (convex hull grown with a linear oracle
       until no facet can be pushed out), up to dimension --max-dim (default 3). Beyond that
       (I(D;A|Y) with three groups) or past --max-vertices, a multi-start ascent over
       vertices gives the value, and the proven bound is the tighter of the information
       bound min(H(D|.), H(A)) and SCIP's (global_opt.global_bound, --scip-time seconds,
       own process).
The linear oracle is exact and fast: per proxy level z the table is a 2 x n transportation
problem (a greedy knapsack) when I = 2 or G = 2, otherwise one LP (HiGHS) over all z.

Columns as in bounds_scip.csv: value (attained by a feasible joint),
proven_bound (no feasible joint does better), status. Output: <out>/bounds_mi.csv,
method "MI", param = --scip-time.

Usage
  python hmda_mi.py
  python hmda_mi.py --scip-time 600 --runs DY
  python hmda_mi.py --self-test
"""
import math
import os
import time

import numpy as np

LN2 = math.log(2)
TP, FP, FN, TN = 0, 1, 2, 3
CELLS = {(1, 1): TP, (1, 0): FP, (0, 1): FN, (0, 0): TN}          # (d, y) -> internal index
METRIC = {"D": "I(D;A)", "DY": "I(D;A|Y)"}


def _xlogx(p):
    p = np.asarray(p, float)
    return float(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum())


class MIProblem:
    """Feasible set {q >= 0: sum_a q[z,i,a] = r[z,i], sum_i q[z,i,a] = m[z,a]} and the metric."""

    def __init__(self, r, m, run):
        self.r, self.m, self.run = np.asarray(r, float), np.asarray(m, float), run
        self.Z, self.I = self.r.shape
        self.G = self.m.shape[1]
        if run == "D":
            self.const = -_xlogx(self.r.sum(0)) - _xlogx(self.m.sum(0))
        else:
            pdy = self.r.sum(0)
            py = [pdy[CELLS[1, y]] + pdy[CELLS[0, y]] for y in (0, 1)]
            self.const = -_xlogx(pdy) + _xlogx(py)
        self._lp = None

    # ---------------------------------------------------------------- metric and gradient
    def _pairs(self):
        return [(CELLS[1, y], CELLS[0, y]) for y in (0, 1)]

    def f(self, th):
        """Metric in bits."""
        th = np.maximum(th, 0)
        if self.run == "D":
            return (_xlogx(th) + self.const) / LN2
        s = sum(_xlogx(th[i] + th[j]) for i, j in self._pairs())
        return (_xlogx(th) - s + self.const) / LN2

    def grad(self, th, floor=1e-300):
        th = np.maximum(th, floor)
        if self.run == "D":
            return (np.log(th) + 1) / LN2
        g = np.empty_like(th)
        for i, j in self._pairs():
            s = th[i] + th[j]
            g[i], g[j] = np.log(th[i] / s), np.log(th[j] / s)
        return g / LN2

    # ---------------------------------------------------------------- linear optimisation
    def lmo(self, w):
        """theta maximising sum(w * theta) over the feasible set (exact)."""
        r, m = self.r, self.m
        if self.I == 2 or self.G == 2:
            # 2 x n transportation table per z: a continuous knapsack, solved greedily, with
            # the same item order for every z because w does not depend on z
            if self.I == 2:
                c, cap, need = w[1] - w[0], m, r[:, 1]
            else:
                c, cap, need = w[:, 1] - w[:, 0], r, m[:, 1]
            order = np.argsort(-c, kind="stable")
            capo = cap[:, order]
            x = np.clip(need[:, None] - (np.cumsum(capo, 1) - capo), 0, capo)
            fill = np.zeros_like(c)
            fill[order] = x.sum(0)
            if self.I == 2:
                return np.vstack([m.sum(0) - fill, fill])
            return np.column_stack([r.sum(0) - fill, fill])
        return self._lmo_lp(w)

    def _lmo_lp(self, w):
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix, vstack
        Z, I, G = self.Z, self.I, self.G
        if self._lp is None:
            z, i, a = (x.ravel() for x in np.indices((Z, I, G)))
            col, one = np.arange(Z * I * G), np.ones(Z * I * G)
            rows_r = coo_matrix((one, (z * I + i, col)), shape=(Z * I, Z * I * G))
            # column sums for groups 0..G-2 only: the last follows from the row sums, and
            # leaving it out keeps rounding in the margins from making the LP infeasible
            keep = a < G - 1
            rows_m = coo_matrix((one[keep], (z[keep] * (G - 1) + a[keep], col[keep])),
                                shape=(Z * (G - 1), Z * I * G))
            self._lp = (vstack([rows_r, rows_m]).tocsr(),
                        np.concatenate([self.r.ravel(), self.m[:, :-1].ravel()]))
        A, b = self._lp
        res = linprog(-np.tile(w.ravel(), Z), A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        if res.status != 0:
            raise RuntimeError(f"LP failed: {res.message}")
        return res.x.reshape(Z, I, G).sum(0)

    # ---------------------------------------------------------------- lower bound
    def _line_min(self, th, d, gmax, iters=60):
        """argmin over [0, gmax] of the convex f(th + g d), by bisection on the slope."""
        if self.grad(th + gmax * d).ravel() @ d.ravel() <= 0:
            return gmax
        lo, hi = 0.0, gmax
        for _ in range(iters):
            mid = (lo + hi) / 2
            if self.grad(th + mid * d).ravel() @ d.ravel() > 0:
                hi = mid
            else:
                lo = mid
        return lo

    def lower(self, gap=1e-6, max_iter=20000, seed=0):
        """Away-step Frank-Wolfe over vertices of the feasible set. Every iterate is a convex
        combination of exact vertices (feasible), and every iteration certifies
        min f >= f(t) - max_s grad f(t).(t - s)."""
        rng = np.random.default_rng(seed)
        act = {}                                            # key -> [vertex, weight]
        for _ in range(4 * self.I * self.G):
            v = self.lmo(rng.normal(size=(self.I, self.G)))
            act.setdefault(v.round(15).tobytes(), [v, 0.0])[1] += 1.0
        tot = sum(w for _, w in act.values())
        for e in act.values():
            e[1] /= tot
        th = sum(w * v for v, w in act.values())
        best_cert, value = 0.0, self.f(th)                # MI is never negative
        it = 0
        for it in range(1, max_iter + 1):
            fth, g = self.f(th), self.grad(th)
            s = self.lmo(-g)
            fw_gap = float((g * (th - s)).sum())
            best_cert = max(best_cert, fth - fw_gap)
            value = fth
            if value - best_cert <= gap:
                break
            ka = max(act, key=lambda k: (g * act[k][0]).sum())
            va, wa = act[ka]
            away_gap = float((g * (va - th)).sum())
            if fw_gap >= away_gap:
                d, gmax = s - th, 1.0
            else:
                d, gmax = th - va, wa / (1 - wa) if wa < 1 else 1e6
            gam = self._line_min(th, d, gmax)
            if fw_gap >= away_gap:
                for e in act.values():
                    e[1] *= 1 - gam
                act.setdefault(s.round(15).tobytes(), [s, 0.0])[1] += gam
            else:
                for e in act.values():
                    e[1] *= 1 + gam
                act[ka][1] -= gam
                if act[ka][1] <= 1e-15 or gam == gmax:
                    del act[ka]
            th = sum(w * v for v, w in act.values())
        bound = min(best_cert, value)
        status = "optimal" if value - bound <= gap else f"gap {value - bound:.2e}"
        return dict(value=value, bound=bound, status=f"{status} (Frank-Wolfe, {it} it.)")

    # ---------------------------------------------------------------- upper bound
    def dim(self):
        return (self.I - 1) * (self.G - 1)

    def _reduced(self, w2):
        w = np.zeros((self.I, self.G))
        w[1:, 1:] = np.asarray(w2, float).reshape(self.I - 1, self.G - 1)
        return w

    def vertices(self, max_vertices=20000, tol=1e-12):
        """All vertices of the polytope of feasible theta, in the coordinates theta[1:, 1:]
        (the other entries follow from the fixed margins). Grows the convex hull of
        known vertices and asks the linear oracle beyond every facet until no facet moves.
        Returns None if the polytope has more than max_vertices vertices."""
        from scipy.spatial import ConvexHull
        d = self.dim()
        if d == 1:
            return [self.lmo(self._reduced([1.0])), self.lmo(self._reduced([-1.0]))]
        pts, keys = [], set()

        def add(th):
            u = th[1:, 1:].ravel()
            k = tuple(np.round(u, 13))
            if k not in keys:
                keys.add(k)
                pts.append((u, th))
                return True
            return False

        rng = np.random.default_rng(0)
        for k in range(d):
            for sgn in (1.0, -1.0):
                e = np.zeros(d)
                e[k] = sgn
                add(self.lmo(self._reduced(e)))
        for _ in range(4 * d):
            add(self.lmo(self._reduced(rng.normal(size=d))))
        asked = set()
        while True:
            U = np.array([u for u, _ in pts])
            hull = ConvexHull(U)            # raises if the polytope is lower-dimensional
            new = False
            for eq in hull.equations:
                n, c = eq[:-1], eq[-1]
                k = tuple(np.round(eq, 10))
                if k in asked:
                    continue
                asked.add(k)
                th = self.lmo(self._reduced(n))
                if n @ th[1:, 1:].ravel() + c > tol:
                    new = add(th) or new
            if not new:
                return [pts[i][1] for i in hull.vertices]
            if len(pts) > max_vertices:
                return None

    def ascent(self, starts=64, seed=0, max_steps=200):
        """Best vertex found by successive linearisation from random vertices (inner bound)."""
        rng = np.random.default_rng(seed)
        best_v = -np.inf
        for _ in range(starts):
            th = self.lmo(rng.normal(size=(self.I, self.G)))
            v = self.f(th)
            for _ in range(max_steps):
                th2 = self.lmo(self.grad(th))
                v2 = self.f(th2)
                if v2 <= v + 1e-13:
                    break
                th, v = th2, v2
            best_v = max(best_v, v)
        return best_v

    def trivial_upper(self):
        """I(D;A) <= min(H(D), H(A)) and I(D;A|Y) <= min(H(D|Y), H(A)), all fixed by the data."""
        h_a = -_xlogx(self.m.sum(0)) / LN2
        pi = self.r.sum(0)
        if self.run == "D":
            h_d = -_xlogx(pi) / LN2
        else:
            py = [pi[CELLS[1, y]] + pi[CELLS[0, y]] for y in (0, 1)]
            h_d = (-_xlogx(pi) + _xlogx(py)) / LN2
        return min(h_d, h_a)

    def upper(self, gap=1e-6, starts=64, max_vertices=20000, max_dim=3):
        verts, why = None, f"dimension {self.dim()} > {max_dim}"
        if self.dim() <= max_dim:
            try:
                verts = self.vertices(max_vertices)
                why = f"more than {max_vertices} vertices"
            except Exception as e:                   # e.g. a lower-dimensional polytope
                why = f"vertex enumeration failed: {type(e).__name__}"
        if verts is not None:
            vals = [self.f(th) for th in verts]
            return dict(value=max(vals), bound=max(vals), exact=True,
                        status=f"optimal (all {len(vals)} vertices)")
        v = self.ascent(starts=starts)
        return dict(value=v, bound=max(self.trivial_upper(), v), exact=False,
                    status=f"ascent, {starts} starts ({why}); bound min(H(D|.), H(A))")


def mi_bound(r, m, run, side, gap=1e-6, starts=64, max_vertices=20000, max_dim=3,
             harmonize_margins=True):
    if harmonize_margins:
        from ci_baseline import harmonize
        r, m = harmonize(r, m)
    t0 = time.perf_counter()
    p = MIProblem(r, m, run)
    res = p.lower(gap) if side == "lower" else p.upper(gap, starts, max_vertices, max_dim)
    res["seconds"] = time.perf_counter() - t0
    return res


# ------------------------------------------------------------------------- runner
def main():
    from hmda_common import append_rows, base_parser, done_keys, key_tuple, load_units
    ap = base_parser(__doc__)
    ap.add_argument("--gap", type=float, default=1e-6)
    ap.add_argument("--starts", type=int, default=64, help="ascent starts (upper, fallback)")
    ap.add_argument("--max-vertices", type=int, default=20000)
    ap.add_argument("--max-dim", type=int, default=3,
                    help="enumerate vertices up to this dimension (I(D;A): 1-2; "
                         "I(D;A|Y): 3 for two groups, 6 for three)")
    ap.add_argument("--scip-time", type=float, default=300.0,
                    help="SCIP time for the proven upper bound when there are too many vertices "
                         "to enumerate (0: skip; the bound is then left empty)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "bounds_mi.csv")
    done = done_keys(path, ["param", "metric", "side"])
    for k, u in load_units(a):
        metric = METRIC.get(k["run"])
        if metric is None:
            continue
        for side in ("lower", "upper"):
            if key_tuple(k, a.scip_time, metric, side) in done:
                continue
            t0 = time.perf_counter()
            try:
                res = mi_bound(u["r"], u["m"], k["run"], side, a.gap, a.starts, a.max_vertices,
                               a.max_dim)
            except Exception as e:
                res = dict(value=np.nan, bound=np.nan, status=f"error: {e}")
            if side == "upper" and res.get("exact") is False and a.scip_time > 0:
                from hmda_scip import isolated_bound
                s = isolated_bound(u["r"], u["m"], k["run"], metric, side, a.scip_time, a.gap,
                                   10 * a.scip_time + 60)
                if np.isfinite(s["value"]) and s["value"] > res["value"]:
                    res["value"] = s["value"]
                if np.isfinite(s["bound"]):              # both are valid: keep the tighter
                    res["bound"] = min(res["bound"], s["bound"])
                res["bound"] = max(res["bound"], res["value"])
                res["status"] += (f"; SCIP {s['status']}, "
                                  f"gap {res['bound'] - res['value']:.2e}")
            append_rows(path, [dict(**k, method="MI", param=a.scip_time, metric=metric,
                                    side=side, value=res["value"], proven_bound=res["bound"],
                                    status=res["status"],
                                    seconds=time.perf_counter() - t0)])
            print(f"  {key_tuple(k)} {metric} {side}: {res['value']:.6f} "
                  f"(proven {res['bound']:.6f}, {res['status']}, "
                  f"{time.perf_counter() - t0:.1f}s)", flush=True)
    print(f"saved {path}")


# ------------------------------------------------------------------------- self-test
def _random(rng, Z, I, G):
    pz = rng.dirichlet(np.ones(Z) * 3)
    r = rng.dirichlet(np.ones(I), size=Z) * pz[:, None]
    m = rng.dirichlet(np.ones(G), size=Z) * pz[:, None]
    return r, m


def _self_test():
    from scipy.optimize import linprog
    rng = np.random.default_rng(1)
    # 1. greedy linear optimisation = LP, for every table shape used
    for I, G in [(2, 2), (2, 3), (4, 2), (4, 3)]:
        r, m = _random(rng, 6, I, G)
        p = MIProblem(r, m, "D" if I == 2 else "DY")
        for _ in range(5):
            w = rng.normal(size=(I, G))
            assert abs((w * p.lmo(w)).sum() - (w * p._lmo_lp(w)).sum()) < 1e-10
    print("linear optimisation (greedy) = LP: OK")
    # 2. bounds sandwich random feasible points, and lower <= upper
    for I, G, run in [(2, 2, "D"), (2, 3, "D"), (4, 2, "DY"), (4, 3, "DY")]:
        r, m = _random(rng, 8, I, G)
        p = MIProblem(r, m, run)
        lo, up = p.lower(), p.upper()
        assert up["bound"] >= up["value"] - 1e-12
        vals = []
        for _ in range(300):                       # random feasible joints
            ws = rng.dirichlet(np.ones(4))
            vals.append(p.f(sum(wk * p.lmo(rng.normal(size=(I, G)) * 5) for wk in ws)))
        assert lo["bound"] <= lo["value"] + 1e-12 <= min(vals) + 1e-7, (lo, min(vals))
        assert max(vals) <= up["value"] + 1e-9, (up, max(vals))
        print(f"{METRIC[run]} {I}x{G}: lower {lo['value']:.6f} (proven {lo['bound']:.6f}), "
              f"upper {up['value']:.6f} [{up['status']}]; random points in "
              f"[{min(vals):.6f}, {max(vals):.6f}]: OK")
    # 3. agreement with SCIP (global_opt) on small instances, if available
    try:
        from global_opt import global_bound
    except Exception as e:
        print(f"SCIP comparison skipped ({e})")
        return
    for I, G, run in [(2, 2, "D"), (2, 3, "D"), (4, 2, "DY")]:
        r, m = _random(rng, 4, I, G)
        p = MIProblem(r, m, run)
        for side, mine in (("lower", p.lower()), ("upper", p.upper())):
            s = global_bound(r, m, run, METRIC[run], side, time_limit=60, harmonize_margins=False)
            print(f"  {METRIC[run]} {side}: ours {mine['value']:.6f}  SCIP value "
                  f"{s['value']:.6f} bound {s['bound']:.6f} [{s['status']}]")


if __name__ == "__main__":
    main()