import argparse, glob, json, os
import numpy as np
from sklearn.metrics import roc_auc_score
import physion_stats as P
import physion_nonlinear as N

def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tkg"); ap.add_argument("--enc", nargs="*", default=[]); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args(); enc = dict(x.split("=", 1) for x in a.enc); per = {}
    for h in P.SCEN:
        d = os.path.join(a.root, f"holdout_{h}")
        if not os.path.isdir(d): continue
        base = P.holdout_scores(d, {}); X, Y = N.feats(d)
        S = {"symbolic_lr": [base["symbolic_lr"]], "physics_lr": [base["physics_lr"]]}
        for nm, (mk, grid) in N.MODELS.items(): S[nm] = [N.fit_best(mk, grid, X, Y)[0]]
        for nm, pat in enc.items():
            sc = [np.array(json.load(open(f))["scores"]) for f in sorted(glob.glob(os.path.join(d, pat)))]
            if sc: S[nm] = sc
        per[h] = {"y": base["y"], "fut": ~base["contact_obs"], "S": S}
        print(f"  {h}: " + ", ".join(f"{k}({len(v)})" for k, v in S.items()), flush=True)
    methods = [m for m in per[next(iter(per))]["S"] if all(m in per[h]["S"] for h in per)]
    def stat(idx, fut):
        out = {}
        for m in methods:
            vals = []
            for h, Q in per.items():
                ii = idx[h]
                if fut: ii = ii[Q["fut"][ii]]
                vals.append(np.nanmean([auc(Q["y"][ii], s[ii]) for s in Q["S"][m]]))
            out[m] = float(np.mean(vals))
        return out
    full = {h: np.arange(len(Q["y"])) for h, Q in per.items()}; rng = np.random.default_rng(0); res = {}
    for fut in (False, True):
        pt = stat(full, fut)
        bs = [stat({h: rng.integers(0, len(Q["y"]), len(Q["y"])) for h, Q in per.items()}, fut) for _ in range(a.boot)]
        print(f"\n=== {'FUTURE-ONLY' if fut else 'ALL CLIPS'}: held-out mean AUROC (per-seed AUROC averaged, as Table 3), 95% CI")
        R = {"methods": {}, "paired": {}}
        for m in methods:
            b = [x[m] for x in bs]; R["methods"][m] = [pt[m], float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
            print(f"  {m:12s} {pt[m]:.3f} [{R['methods'][m][1]:.3f}, {R['methods'][m][2]:.3f}]")
        print("  paired (row - col); * = CI excludes 0")
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                dd = [x[m1] - x[m2] for x in bs]; lo, hi = np.percentile(dd, [2.5, 97.5]); R["paired"][f"{m1} - {m2}"] = [pt[m1] - pt[m2], float(lo), float(hi)]
                print(f"    {m1:12s} - {m2:12s} {pt[m1] - pt[m2]:+.3f} [{lo:+.3f}, {hi:+.3f}] {'*' if lo > 0 or hi < 0 else ''}")
        res["future" if fut else "all"] = R
    print("\nfuture-only subset: retained/total, positive rate all -> retained"); cnt = {}
    for h, Q in per.items():
        f = Q["fut"]; cnt[h] = [int(f.sum()), len(f), float(Q["y"].mean()), float(Q["y"][f].mean())]
        print(f"  {h:9s} {f.sum():3d}/{len(f)}   {Q['y'].mean():.2f} -> {Q['y'][f].mean():.2f}")
    print(f"  total {sum(c[0] for c in cnt.values())}/{sum(c[1] for c in cnt.values())}"); res["counts"] = cnt
    json.dump(res, open(os.path.join(a.root, "physion_stats2.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
