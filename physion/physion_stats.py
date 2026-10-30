#!/usr/bin/env python3
"""Bootstrap CIs and no-contact-yet subset for Physion held-out AUROC, from existing outputs."""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

SCEN = ["Collide", "Contain", "Dominoes", "Drape", "Drop", "Link", "Roll", "Support"]
SYM_PAIR = ["contact", "near", "within_reach", "approaching", "heading_toward", "on_top_of"]
SYM_SELF = ["stationary", "speed_slow", "speed_medium", "speed_fast", "falling", "touches_env"]
PHYS = ["d_end", "closing", "extrap_min", "contact_obs", "target_speed", "target_vy", "chain_hops", "n_moving"]


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def logreg(X, y, l2=1e-2, iters=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * w)
        b -= lr * np.mean(p - y)
    return lambda X2: ((X2 - mu) / sd) @ w + b


def holdout_scores(d, enc_groups):
    meta = json.load(open(os.path.join(d, "meta.json")))
    rel2id = {l.split("\t")[0]: int(l.split("\t")[1]) for l in open(os.path.join(d, "relation2id.txt"))}
    id2rel = {v: k for k, v in rel2id.items()}
    rows = {sp: [l.rstrip("\n").split("\t") for l in open(os.path.join(d, f"queries_{sp}.tsv"))] for sp in ("train", "valid", "test")}
    Y = {sp: np.array([int(r[4]) for r in rows[sp]]) for sp in rows}

    def sym(sp):
        facts = defaultdict(set)
        for l in open(os.path.join(d, f"facts_{sp}.txt")):
            s, r, o, t = (int(x) for x in l.split("\t"))
            facts[s].add((id2rel[r], o, t))
        X = []
        for r in rows[sp]:
            s, o, t = int(r[0]), int(r[2]), int(r[3]); last = t - 1
            f = []
            for rel in SYM_PAIR:
                f.append(float((rel, o, last) in facts[s]))
                f.append(float(any(x[0] == rel and x[1] == o for x in facts[s])))
            for rel in SYM_SELF:
                f.append(float(any(x[0] == rel and x[2] == last for x in facts[s])))
            X.append(f)
        return np.array(X)

    phys = {sp: np.array([[float(meta["stimuli"][r[5]][k]) for k in PHYS] for r in rows[sp]]) for sp in rows}
    S = {sp: sym(sp) for sp in rows}
    out = {"y": Y["test"], "contact_obs": phys["test"][:, 3].astype(bool),
           "symbolic_lr": logreg(S["train"], Y["train"])(S["test"]),
           "physics_lr": logreg(phys["train"], Y["train"])(phys["test"])}
    for name, pat in enc_groups.items():
        sc = []
        for f in sorted(glob.glob(os.path.join(d, pat))):
            j = json.load(open(f))
            if "scores" in j:
                x = np.array(j["scores"]); sc.append((x - x.mean()) / (x.std() + 1e-8))
        if sc:
            out[name] = np.mean(sc, axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tkg")
    ap.add_argument("--enc", nargs="*", default=["AAE=aae_results_s*.json"])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    enc_groups = dict(x.split("=", 1) for x in a.enc)
    per = {}
    for h in SCEN:
        d = os.path.join(a.root, f"holdout_{h}")
        if os.path.isdir(d):
            per[h] = holdout_scores(d, enc_groups)
            print(f"  scored {h}: n={len(per[h]['y'])}, methods {[k for k in per[h] if k not in ('y', 'contact_obs')]}", flush=True)
    methods = [m for m in ["symbolic_lr", "physics_lr"] + list(enc_groups) if all(m in per[h] for h in per)]

    def held_mean(m, idx=None, subset=False):
        vals = []
        for h, P in per.items():
            ii = idx[h] if idx is not None else np.arange(len(P["y"]))
            if subset:
                ii = ii[~P["contact_obs"][ii]]
            vals.append(auroc(P["y"][ii], P[m][ii]))
        return float(np.nanmean(vals))

    rng = np.random.default_rng(a.seed)
    boots = []
    for _ in range(a.boot):
        idx = {h: rng.integers(0, len(P["y"]), len(P["y"])) for h, P in per.items()}
        boots.append({m: held_mean(m, idx) for m in methods})
    res = {}
    print(f"\nHeld-out mean AUROC, bootstrap 95% CI ({a.boot} resamples of test clips within scenario)")
    for m in methods:
        b = np.array([x[m] for x in boots]); pt = held_mean(m)
        res[m] = {"auroc": pt, "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                  "no_contact_yet": held_mean(m, subset=True)}
        print(f"  {m:12s} {pt:.3f}  [{res[m]['ci'][0]:.3f}, {res[m]['ci'][1]:.3f}]   no-contact-yet: {res[m]['no_contact_yet']:.3f}")
    print("\nPaired differences (row - col), 95% CI; * = CI excludes 0")
    diffs = {}
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            b = np.array([x[m1] - x[m2] for x in boots]); lo, hi = np.percentile(b, [2.5, 97.5])
            dpt = res[m1]["auroc"] - res[m2]["auroc"]
            diffs[f"{m1} - {m2}"] = {"diff": dpt, "ci": [float(lo), float(hi)]}
            print(f"  {m1:12s} - {m2:12s} {dpt:+.3f}  [{lo:+.3f}, {hi:+.3f}] {'*' if lo > 0 or hi < 0 else ''}")
    print("\nPer-scenario: " + ", ".join(f"{h} n={len(P['y'])} (contact in window {P['contact_obs'].mean():.0%})" for h, P in per.items()))
    json.dump({"methods": res, "paired": diffs}, open(os.path.join(a.root, "physion_stats.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
