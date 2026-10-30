import argparse, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import physion_stats as P
import physion_nonlinear as N
SETS = {"motion": ["d_end", "closing", "extrap_min", "target_speed", "target_vy", "n_moving"],
        "motion+contact": ["d_end", "closing", "extrap_min", "target_speed", "target_vy", "n_moving", "contact_obs"],
        "hybrid": list(P.PHYS)}
MODELS = dict(N.MODELS)
MODELS["lr"] = (lambda g: make_pipeline(StandardScaler(), LogisticRegression(C=g, max_iter=2000)), [0.01, 0.1, 1.0, 10.0])
def auc(y, s): return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
def score_split(root, h, cols):
    d = os.path.join(root, f"holdout_{h}"); base = P.holdout_scores(d, {}); X, Y = N.feats(d)
    sc = {"symbolic_lr": base["symbolic_lr"]}
    for fs, c in cols.items():
        Xs = {sp: X[sp][:, c] for sp in X}
        for mn in ("lr", "physics_mlp", "physics_gbt"):
            mk, grid = MODELS[mn]; sc[f"{fs} | {mn.replace('physics_', '')}"] = N.fit_best(mk, grid, Xs, Y)[0]
    print(f"  done {h}", flush=True)
    return h, Y["test"], sc
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="tkg"); ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args()
    cols = {k: [P.PHYS.index(f) for f in v] for k, v in SETS.items()}; per, indist = {}, {}
    from multiprocessing import Pool
    hs = [h for h in P.SCEN + ["none"] if os.path.isdir(os.path.join(a.root, f"holdout_{h}"))]
    with Pool(len(hs)) as pool:
        res = pool.starmap(score_split, [(a.root, h, cols) for h in hs])
    for h, Y, sc in res:
        if h == "none": indist = {m: auc(Y, s) for m, s in sc.items()}
        else: per[h] = {"y": Y, "S": sc}
    methods = list(per[next(iter(per))]["S"])
    hm = lambda idx, m: float(np.mean([auc(Q["y"][idx[h]], Q["S"][m][idx[h]]) for h, Q in per.items()]))
    full = {h: np.arange(len(Q["y"])) for h, Q in per.items()}; rng = np.random.default_rng(0); boots = []
    for _ in range(a.boot):
        idx = {h: rng.integers(0, len(Q["y"]), len(Q["y"])) for h, Q in per.items()}; boots.append({m: hm(idx, m) for m in methods})
    print(f"\n{'model':24s} {'held-out':>8s} {'95% CI':>16s} {'sym_lr - model':>26s} {'Dominoes':>9s} {'in-dist':>8s}"); out = {}
    for m in methods:
        pt = hm(full, m); b = [x[m] for x in boots]; dd = [x["symbolic_lr"] - x[m] for x in boots]
        lo, hi = np.percentile(b, [2.5, 97.5]); dlo, dhi = np.percentile(dd, [2.5, 97.5]); dom = auc(per["Dominoes"]["y"], per["Dominoes"]["S"][m])
        out[m] = {"heldout": [pt, lo, hi], "diff": [hm(full, "symbolic_lr") - pt, dlo, dhi], "dominoes": dom, "indist": indist.get(m)}
        star = "*" if (dlo > 0 or dhi < 0) and m != "symbolic_lr" else " "
        print(f"{m:24s} {pt:8.3f}  [{lo:.3f}, {hi:.3f}]  {hm(full, 'symbolic_lr') - pt:+.3f} [{dlo:+.3f}, {dhi:+.3f}]{star} {dom:9.3f} {indist.get(m, np.nan):8.3f}")
    print("\nPaired feature-set differences (same resamples), 95% CI")
    for mn in ("lr", "mlp", "gbt"):
        for f1, f2 in (("motion+contact", "motion"), ("hybrid", "motion+contact")):
            m1, m2 = f"{f1} | {mn}", f"{f2} | {mn}"
            dd = [x[m1] - x[m2] for x in boots]; lo, hi = np.percentile(dd, [2.5, 97.5]); pt = hm(full, m1) - hm(full, m2)
            out[f"{m1} - {m2}"] = [pt, lo, hi]
            print(f"  {m1:24s} - {m2:22s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if lo > 0 or hi < 0 else ''}")
    json.dump(out, open(os.path.join(a.root, "physion_featsets_par.json"), "w"), indent=2)
if __name__ == "__main__":
    main()
