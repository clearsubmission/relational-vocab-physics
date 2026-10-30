#!/usr/bin/env python3
"""Paired bootstrap CIs for CLEVRER test MRR, resampling collisions (both mirrored queries together)."""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--enc", nargs="*", default=[])
    ap.add_argument("--linear", nargs="*", default=["symbolic_lr", "physics_lr", "physics_mlp"])
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dd = a.data_dir
    meta = json.load(open(os.path.join(dd, "meta.json")))
    name_of = {}
    for l in open(os.path.join(dd, "entity2id.txt")):
        n, i = l.rstrip("\n").split("\t")
        if n.startswith("v") and "_o" in n:
            vid, o = n[1:].split("_o"); name_of[int(i)] = (vid, int(o))
    methods = {}
    B = json.load(open(os.path.join(dd, "baselines_test.json")))
    for m in a.linear:
        if m in B and "rr" in B[m]:
            methods[m] = {(k[0], k[1], k[2], k[3]): r for k, r in zip(B[m]["keys"], B[m]["rr"])}
    for spec in a.enc:
        nm, pre = spec.split("=", 1)
        acc = defaultdict(list)
        files = sorted(glob.glob(os.path.join(dd, f"aae_results_{pre}*.json")))
        for f in files:
            R = json.load(open(f))
            if "rr" not in R:
                continue
            for (s, o, t), r in zip(R["keys"], R["rr"]):
                (vid, so), (_, oo) = name_of[s], name_of[o]
                acc[(vid, so, oo, t)].append(r)
        if acc:
            methods[nm] = {k: float(np.mean(v)) for k, v in acc.items()}
            print(f"  {nm}: {len(files)} seed file(s), {len(acc)} queries")
    keys = sorted(set.intersection(*[set(v) for v in methods.values()]))
    print(f"common queries: {len(keys)}  methods: {list(methods)}")
    groups = defaultdict(list)
    for i, (vid, s, o, t) in enumerate(keys):
        groups[(vid, min(s, o), max(s, o), t)].append(i)
    gl = list(groups.values())
    hz = np.array([t - meta["videos"][vid]["cutoff_bin"] for vid, s, o, t in keys])
    X = {m: np.array([methods[m][k] for k in keys]) for m in methods}
    print(f"collisions: {len(gl)}")
    rng = np.random.default_rng(a.seed)
    idx_boot = [np.concatenate([gl[p] for p in rng.integers(0, len(gl), len(gl))]) for _ in range(a.boot)]

    def ci(vec, mask=None):
        pt = vec[mask].mean() if mask is not None else vec.mean()
        bs = [vec[ib[mask[ib]] if mask is not None else ib].mean() for ib in idx_boot]
        return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    out = {"methods": {}, "paired": {}}
    print(f"\nTest MRR with 95% CI ({a.boot} collision resamples)")
    for m, v in X.items():
        pt, lo, hi = ci(v); out["methods"][m] = {"MRR": pt, "ci": [lo, hi]}
        print(f"  {m:14s} {pt:.4f} [{lo:.4f}, {hi:.4f}]")
    print("\nPaired differences (overall | 6+ steps); * = CI excludes 0")
    names = list(X); far = hz >= 6
    for i, m1 in enumerate(names):
        for m2 in names[i + 1:]:
            d = X[m1] - X[m2]; p, lo, hi = ci(d); pf, lof, hif = ci(d, far)
            out["paired"][f"{m1} - {m2}"] = {"overall": [p, lo, hi], "6+": [pf, lof, hif]}
            print(f"  {m1:14s} - {m2:14s} {p:+.4f} [{lo:+.4f}, {hi:+.4f}]{'*' if lo > 0 or hi < 0 else ' '}   |  "
                  f"{pf:+.4f} [{lof:+.4f}, {hif:+.4f}]{'*' if lof > 0 or hif < 0 else ''}")
    json.dump(out, open(os.path.join(dd, "bootstrap.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
