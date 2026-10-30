#!/usr/bin/env python3
"""
Non-learned / shallow baselines for Physion OCP on a physion_to_tkg.py output dir.
All fit on train (+ threshold picked on valid), evaluated on test (held-out scenario).

  majority        train majority class
  contact_obs     contact already seen in the observed window
  gap_end         1-feature threshold on target-zone gap at end of window
  extrapolate     1-feature threshold on min gap under constant-velocity extrapolation
  physics_lr      logistic regression on [gap_end, closing, extrap_min, contact_obs, speed, vy]
  symbolic_lr     logistic regression on target-zone / target facts in the TKG (same facts AAE sees)
Usage: python physion_baselines.py --data-dir tkg/holdout_Dominoes
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

SYM_PAIR = ["contact", "near", "within_reach", "approaching", "heading_toward", "on_top_of"]
SYM_SELF = ["stationary", "speed_slow", "speed_medium", "speed_fast", "falling", "touches_env"]


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):                         # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_threshold(y, s):
    cand = np.unique(s)
    accs = [np.mean((s >= c) == y) for c in cand]
    return float(cand[int(np.argmax(accs))]) if len(cand) else 0.0


def logreg(X, y, l2=1e-2, iters=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * w)
        b -= lr * np.mean(p - y)
    return lambda X2: ((X2 - mu) / sd) @ w + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.data_dir, "meta.json")))
    rel2id = {l.split("\t")[0]: int(l.split("\t")[1]) for l in open(os.path.join(a.data_dir, "relation2id.txt"))}
    id2rel = {v: k for k, v in rel2id.items()}

    rows = defaultdict(list)   # split -> list of (key, s, o, t, label)
    for split in ("train", "valid", "test"):
        for l in open(os.path.join(a.data_dir, f"queries_{split}.tsv")):
            s, r, o, t, lab, key, scen = l.rstrip("\n").split("\t")
            rows[split].append((key, int(s), int(o), int(t), int(lab)))

    def sym_features(split):
        facts = defaultdict(set)   # entity -> {(rel, other, t)}
        for l in open(os.path.join(a.data_dir, f"facts_{split}.txt")):
            s, r, o, t = (int(x) for x in l.split("\t"))
            facts[s].add((id2rel[r], o, t))
        X = []
        for key, s, o, t, lab in rows[split]:
            last = t - 1
            f = []
            for rel in SYM_PAIR:
                f.append(float((rel, o, last) in facts[s]))                          # at last observed step
                f.append(float(any(x[0] == rel and x[1] == o for x in facts[s])))     # anywhere in window
            for rel in SYM_SELF:
                f.append(float(any(x[0] == rel and x[2] == last for x in facts[s])))
            X.append(f)
        return np.array(X)

    phys_keys = ["d_end", "closing", "extrap_min", "contact_obs", "target_speed", "target_vy", "chain_hops", "n_moving"]
    def phys(split):
        return np.array([[float(meta["stimuli"][k][f]) for f in phys_keys] for k, *_ in rows[split]])
    Y = {sp: np.array([r[-1] for r in rows[sp]]) for sp in rows}
    P = {sp: phys(sp) for sp in rows}
    S = {sp: sym_features(sp) for sp in rows}

    res = {}
    def report(name, s_tr, s_va, s_te):
        thr = best_threshold(Y["valid"], s_va) if len(Y["valid"]) else best_threshold(Y["train"], s_tr)
        res[name] = {"acc": float(np.mean((s_te >= thr) == Y["test"])), "auroc": auroc(Y["test"], s_te)}

    maj = int(Y["train"].mean() >= 0.5)
    res["majority"] = {"acc": float(np.mean(Y["test"] == maj)), "auroc": 0.5}
    co = [P[sp][:, 3] for sp in ("train", "valid", "test")]
    report("contact_obs", *co)
    report("gap_end", *[-P[sp][:, 0] for sp in ("train", "valid", "test")])
    report("extrapolate", *[-P[sp][:, 2] for sp in ("train", "valid", "test")])
    report("chain_reach", *[-P[sp][:, 6] for sp in ("train", "valid", "test")])
    f = logreg(P["train"], Y["train"]); report("physics_lr", *[f(P[sp]) for sp in ("train", "valid", "test")])
    f = logreg(S["train"], Y["train"]); report("symbolic_lr", *[f(S[sp]) for sp in ("train", "valid", "test")])

    nc = P["test"][:, 3] == 0
    print(f"holdout={meta['holdout']}  test n={len(Y['test'])}  pos={Y['test'].mean():.2f}  "
          f"no-contact-yet n={int(nc.sum())} pos={Y['test'][nc].mean() if nc.any() else float('nan'):.2f}")
    print(f"{'baseline':14s} {'acc':>7s} {'AUROC':>7s}")
    for k, v in res.items():
        print(f"{k:14s} {v['acc']:7.4f} {v['auroc']:7.4f}")
    json.dump(res, open(os.path.join(a.data_dir, "baselines.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
