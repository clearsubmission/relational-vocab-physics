import argparse, json, os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import physion_stats as P
SCEN = P.SCEN + ["none"]
def feats(d):
    meta = json.load(open(os.path.join(d, "meta.json"))); X, Y = {}, {}
    for sp in ("train", "valid", "test"):
        rows = [l.rstrip("\n").split("\t") for l in open(os.path.join(d, f"queries_{sp}.tsv"))]
        X[sp] = np.array([[float(meta["stimuli"][r[5]][k]) for k in P.PHYS] for r in rows]); Y[sp] = np.array([int(r[4]) for r in rows])
    return X, Y
def fit_best(make, grid, X, Y):
    best = None
    for g in grid:
        m = make(g).fit(X["train"], Y["train"]); v = P.auroc(Y["valid"], m.predict_proba(X["valid"])[:, 1])
        if best is None or v > best[0]: best = (v, g, m)
    return best[2].predict_proba(X["test"])[:, 1], best[1]
MODELS = {
 "physics_mlp": (lambda g: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), alpha=g, max_iter=500, early_stopping=True, random_state=0)), [1e-4, 1e-3, 1e-2, 1e-1]),
 "physics_gbt": (lambda g: HistGradientBoostingClassifier(max_depth=g[0], learning_rate=g[1], max_iter=300, early_stopping=True, random_state=0), [(3, 0.05), (5, 0.05), (None, 0.05), (3, 0.1)])}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="tkg"); ap.add_argument("--boot", type=int, default=2000); a = ap.parse_args()
    per, table = {}, {}
    for h in SCEN:
        d = os.path.join(a.root, f"holdout_{h}")
        if not os.path.isdir(d): continue
        X, Y = feats(d); base = P.holdout_scores(d, {})
        row = {"symbolic_lr": P.auroc(Y["test"], base["symbolic_lr"]), "physics_lr": P.auroc(Y["test"], base["physics_lr"])}
        sc = {"y": Y["test"], "symbolic_lr": base["symbolic_lr"], "physics_lr": base["physics_lr"]}
        for nm, (mk, grid) in MODELS.items():
            s, g = fit_best(mk, grid, X, Y); row[nm] = P.auroc(Y["test"], s); sc[nm] = s
        table[h] = row
        if h != "none": per[h] = sc
        print(f"{h:9s} " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()), flush=True)
    ms = ["symbolic_lr", "physics_lr", "physics_mlp", "physics_gbt"]
    print("held-out mean " + "  ".join(f"{m} {np.mean([table[h][m] for h in per]):.3f}" for m in ms))
    rng = np.random.default_rng(0); boots = []
    for _ in range(a.boot):
        idx = {h: rng.integers(0, len(S["y"]), len(S["y"])) for h, S in per.items()}
        boots.append({m: np.mean([P.auroc(S["y"][idx[h]], S[m][idx[h]]) for h, S in per.items()]) for m in ms})
    print("\nheld-out mean AUROC, 95% CI; paired vs symbolic_lr"); out = {"table": table, "ci": {}}
    for m in ms:
        b = np.array([x[m] for x in boots]); dd = np.array([x["symbolic_lr"] - x[m] for x in boots])
        out["ci"][m] = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
        print(f"  {m:12s} [{out['ci'][m][0]:.3f}, {out['ci'][m][1]:.3f}]   symbolic_lr - {m}: {dd.mean():+.3f} [{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]")
    json.dump(out, open(os.path.join(a.root, "physion_nonlinear.json"), "w"), indent=2)
if __name__ == "__main__":
    main()
