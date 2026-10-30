#!/usr/bin/env python3
"""
Non-learned baselines for PhysT-KG collision forecasting.
Candidates = other objects in the same video; filtered; ties broken uniformly at random (exact expectation).
  random | activity | pair_history | nearest | extrapolate
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


def load(path):
    with open(path) as fh:
        return [tuple(int(x) for x in line.split("\t")) for line in fh if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--query-rel", default="collides_with")
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.data_dir, "meta.json")))
    rel2id = dict(l.rstrip("\n").split("\t") for l in open(os.path.join(a.data_dir, "relation2id.txt")))
    rel2id = {k: int(v) for k, v in rel2id.items()}
    qr = rel2id[a.query_rel]
    static = {rel2id[r] for r in ("has_shape", "has_material", "has_color") if r in rel2id}
    interact = {rel2id[r] for r in ("collides_with", "near") if r in rel2id}
    bs = meta["bin_size"]

    ent2vid, vobj = {}, {}
    for vid, info in meta["videos"].items():
        if info["split"] != a.split:
            continue
        vobj[vid] = {int(e): o for e, o in info["objects"].items()}
        for e in vobj[vid]:
            ent2vid[e] = vid

    ctx = load(os.path.join(a.data_dir, f"{a.split}_context.txt"))
    queries = [q for q in load(os.path.join(a.data_dir, f"{a.split}.txt")) if q[1] == qr]
    activity, pair, coll = defaultdict(int), defaultdict(int), set()
    for s, r, o, _ in ctx:
        if r in static:
            continue
        activity[s] += 1
        if o in ent2vid:
            activity[o] += 1
        if r == qr:
            coll.update({(s, o), (o, s)})
        if r in interact:
            pair[(s, o)] += 1
            pair[(o, s)] += 1
    last = defaultdict(set)   # relation name -> {(s, o)} at the final observed step
    id2rel = {v: k for k, v in rel2id.items()}
    for s, r, o, t in ctx:
        vid = ent2vid.get(s)
        if vid is not None and t == meta["videos"][vid]["cutoff_bin"] - 1:
            last[id2rel[r]].add((s, o))
    truth = defaultdict(set)
    for s, r, o, t in queries:
        truth[(s, t)].add(o)

    def dist_now(vid, s, c):
        ps, pc = vobj[vid][s]["pos"], vobj[vid][c]["pos"]
        if ps is None or pc is None:
            return np.inf
        return float(np.linalg.norm(np.subtract(ps, pc)))

    def dist_extrap(vid, s, c, t):
        A, B = vobj[vid][s], vobj[vid][c]
        if A["pos"] is None or B["pos"] is None:
            return np.inf
        f0 = meta["videos"][vid]["cutoff_frame"]
        f1 = t * bs + bs // 2
        steps = np.arange(0, max(1, f1 - f0) + 1)[:, None]
        pa = np.array(A["pos"]) + steps * np.array(A["vel"])
        pb = np.array(B["pos"]) + steps * np.array(B["vel"])
        return float(np.min(np.linalg.norm(pa - pb, axis=1)))

    scorers = {
        "random": lambda vid, s, c, t: 0.0,
        "activity": lambda vid, s, c, t: activity[c],
        "pair_history": lambda vid, s, c, t: pair[(s, c)],
        "active_not_partner": lambda vid, s, c, t: activity[c] - 1000.0 * ((s, c) in coll),
        "no_repeat": lambda vid, s, c, t: -1.0 if (s, c) in coll else 0.0,
        "nearest": lambda vid, s, c, t: -dist_now(vid, s, c),
        "extrapolate": lambda vid, s, c, t: -dist_extrap(vid, s, c, t),
    }

    if "approaching" in rel2id:   # symbolic-only references: what the TKG vocabulary alone supports
        scorers["last_approach"] = lambda vid, s, c, t: float((s, c) in last["approaching"])
        scorers["last_heading"] = lambda vid, s, c, t: float((s, c) in last["heading_toward"])
        scorers["sym_rule"] = lambda vid, s, c, t: (2.0 * ((s, c) in last["heading_toward"])
                                                    + 1.0 * ((s, c) in last["approaching"])
                                                    - 4.0 * ((s, c) in coll))

    if not queries:
        raise SystemExit(f"No {a.query_rel} queries in {a.split}; try a lower --cutoff-frac or more videos.")
    results, ncands, hz, allranks = {}, [], [], {}
    for name, fn in scorers.items():
        ranks = []
        for s, _, o, t in queries:
            vid = ent2vid.get(s)
            if vid is None:
                continue
            cands = [c for c in vobj[vid] if c != s and (c == o or c not in truth[(s, t)])]
            sc = {c: fn(vid, s, c, t) for c in cands}
            so = sc[o]
            others = [sc[c] for c in cands if c != o]
            lo = 1 + sum(x > so for x in others)
            ties = sum(x == so for x in others)
            ranks.append(np.arange(lo, lo + ties + 1))
            if name == "random":
                ncands.append(len(cands))
                hz.append(t - meta["videos"][vid]["cutoff_bin"])
        results[name] = {"MRR": float(np.mean([np.mean(1 / rr) for rr in ranks])),
                         "H@1": float(np.mean([np.mean(rr <= 1) for rr in ranks])),
                         "H@3": float(np.mean([np.mean(rr <= 3) for rr in ranks])), "n": len(ranks)}
        allranks[name] = ranks

    print(f"{a.split}: {len(queries)} queries, mean candidates/query {np.mean(ncands):.2f}")
    print(f"{'baseline':14s} {'MRR':>7s} {'H@1':>7s} {'H@3':>7s}")
    for k, v in results.items():
        print(f"{k:14s} {v['MRR']:7.4f} {v['H@1']:7.4f} {v['H@3']:7.4f}")
    # --- MRR by forecast horizon (time steps after cutoff)
    hz = np.array(hz)
    buckets = [(0, 1), (2, 3), (4, 5), (6, 99)]
    labels = [f"{lo}-{hi}" if hi < 99 else f"{lo}+" for lo, hi in buckets]
    print(f"\nMRR by horizon (steps after cutoff; 1 step = {bs} frames)")
    print(f"{'baseline':14s} " + " ".join(f"{l:>8s}" for l in labels))
    print(f"{'n queries':14s} " + " ".join(f"{int(((hz >= lo) & (hz <= hi)).sum()):8d}" for lo, hi in buckets))
    by_h = {}
    for name, ranks in allranks.items():
        rr = np.array([np.mean(1 / x) for x in ranks])
        row = [float(rr[(hz >= lo) & (hz <= hi)].mean()) if ((hz >= lo) & (hz <= hi)).any() else float("nan")
               for lo, hi in buckets]
        by_h[name] = dict(zip(labels, row))
        print(f"{name:14s} " + " ".join(f"{x:8.4f}" for x in row))
    results["_by_horizon"] = by_h
    json.dump(results, open(os.path.join(a.data_dir, f"baselines_{a.split}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
