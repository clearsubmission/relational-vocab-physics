#!/usr/bin/env python3
"""
Physion states -> fully inductive TKG for Object Contact Prediction (OCP).

Each stimulus's objects are new entities. Facts come only from the first --obs-frames
frames (Physion shows models/humans 45 frames = 1.5 s). Query: (target, will_contact,
zone, t=n_steps) with binary label = target contacts zone at any point of the trial.

Leave-one-scenario-out: --holdout Dominoes -> train/valid from readout sets of the other
7 scenarios (thresholds fit on those only); test = Dominoes testing set.
--holdout none -> train on all 8 readout sets, test on all 8 testing sets.

Output dir: entity2id.txt relation2id.txt facts_{train,valid,test}.txt
            queries_{train,valid,test}.tsv (s r o t label stim scenario)  meta.json
"""
import argparse
import glob
import gzip
import json
import os
import pickle
import random
from collections import Counter

import numpy as np

SCEN = ["Collide", "Contain", "Dominoes", "Drape", "Drop", "Link", "Roll", "Support"]
STATIC_RELS = ["has_role", "has_shape"]
DYN_RELS = ["contact", "touches_env", "near", "within_reach", "approaching", "heading_toward", "on_top_of",
            "stationary", "speed_slow", "speed_medium", "speed_fast", "falling"]
SYMMETRIC = {"contact", "near", "approaching", "heading_toward"}
QUERY_REL = "will_contact"
VOCAB = {
    "events": {"has_role", "has_shape", "contact", "touches_env"},
    "kin":    {"has_role", "has_shape", "contact", "touches_env", "near", "on_top_of",
               "stationary", "speed_slow", "speed_medium", "speed_fast", "falling"},
    "rel":    {"has_role", "has_shape", "contact", "touches_env", "near", "on_top_of",
               "stationary", "speed_slow", "speed_medium", "speed_fast", "falling",
               "approaching", "heading_toward"},
}
VOCAB["full"] = VOCAB["rel"] | {"within_reach"}
BBOX = ["top", "bottom", "left", "right", "front", "back"]


# ----------------------------------------------------------------------------- loading
def load_states(states_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(states_dir, "*.pkl.gz"))):
        stem = os.path.basename(f)[:-7]
        scen = stem.split("_")[0]
        split = "pool" if "readout" in stem else ("test" if "testing" in stem else None)
        if scen not in SCEN or split is None:
            continue
        recs = pickle.load(gzip.open(f))
        for r in recs:
            r["scenario"], r["pool"] = scen, split
        out += recs
        print(f"  loaded {len(recs):5d}  {stem}")
    return out


def prep(r, obs, k=3):
    """Centers, AABBs (from bbox points, else scale), per-frame velocities over the observed window."""
    st, ob = r["static"], r["objects"]
    P = ob.get("positions")
    if not isinstance(P, np.ndarray) or "target_id" not in st or "zone_id" not in st:
        return None
    ids = [int(x) for x in st["object_ids"]]
    tid, zid = int(st["target_id"]), int(st["zone_id"])
    if tid not in ids or zid not in ids:
        return None
    T = min(obs, P.shape[0])
    P = P[:T].astype(np.float64)
    if all(isinstance(ob.get(b), np.ndarray) for b in BBOX):
        pts = np.stack([ob[b][:T] for b in BBOX] + [P], axis=0)          # 7,T,n,3
        lo, hi = np.nanmin(pts, axis=0), np.nanmax(pts, axis=0)
    else:
        sc = np.stack([np.asarray(st.get(f"scale_{a}", np.ones(len(ids))), float) for a in "xyz"], axis=1)
        half = 0.5 * sc[None]                                             # 1,n,3
        lo, hi = P - half, P + half
    V = np.zeros_like(P)
    V[k:] = (P[k:] - P[:-k]) / k
    V[1:k] = (P[1:k] - P[0:1]) / np.arange(1, k)[:, None, None]
    names = as_list(st.get("model_names")) or ["obj"] * len(ids)
    return dict(ids=ids, idx={o: i for i, o in enumerate(ids)}, tid=tid, zid=zid, T=T, P=P, lo=lo, hi=hi,
                V=np.nan_to_num(V), names=names)


def gapmat(L1, H1, L2, H2):
    """Pairwise AABB separation: rows = boxes (L1,H1), cols = boxes (L2,H2)."""
    d = np.maximum(0.0, np.maximum(L1[:, None] - H2[None], L2[None] - H1[:, None]))
    return np.sqrt((d ** 2).sum(-1))


def gap(loA, hiA, loB, hiB):
    d = np.maximum(0.0, np.maximum(loA - hiB, loB - hiA))
    return float(np.sqrt((d ** 2).sum()))


def as_list(v):
    """Static fields may be lists, numpy arrays (possibly empty), bytes, or scalars."""
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (bytes, str)):
        v = [v]
    if not isinstance(v, (list, tuple)):
        v = [v]
    return [x.decode(errors="replace") if isinstance(x, bytes) else str(x) for x in v]


def roles(r, g):
    st = r["static"]
    dis = set(as_list(st.get("distractors")))
    occ = set(as_list(st.get("occluders")))
    probe = int(st["probe_id"]) if "probe_id" in st else None
    out = {}
    for i, o in enumerate(g["ids"]):
        if o == g["tid"]:
            out[o] = "target"
        elif o == g["zid"]:
            out[o] = "zone"
        elif probe is not None and o == probe:
            out[o] = "probe"
        elif g["names"][i] in dis:
            out[o] = "distractor"
        elif g["names"][i] in occ:
            out[o] = "occluder"
        else:
            out[o] = "other"
    return out


def bin_ends(T, b):
    nb = (T + b - 1) // b
    return [min((i + 1) * b - 1, T - 1) for i in range(nb)]


# ----------------------------------------------------------------------------- thresholds
def fit_thresholds(gs, recs, bin_size, near_q, eps):
    speeds, gaps, cgaps = [], [], []
    for g, r in zip(gs, recs):
        n = len(g["ids"])
        iu = np.triu_indices(n, 1)
        for f in bin_ends(g["T"], bin_size):
            speeds += np.linalg.norm(g["V"][f], axis=1).tolist()
            gaps += gapmat(g["lo"][f], g["hi"][f], g["lo"][f], g["hi"][f])[iu].tolist()
        for (ti, a, b, s) in r["collisions"]:
            if ti < g["T"] and a in g["idx"] and b in g["idx"]:
                i, j = g["idx"][a], g["idx"][b]
                cgaps.append(gap(g["lo"][ti, i], g["hi"][ti, i], g["lo"][ti, j], g["hi"][ti, j]))
    speeds = np.array(speeds)
    mov = speeds[speeds > eps]
    q1, q2 = (np.quantile(mov, [1 / 3, 2 / 3]) if len(mov) else (eps, eps))
    pg = np.array(gaps); pg = pg[pg > 1e-6]
    return dict(eps=eps, q1=float(q1), q2=float(q2),
                near_thr=float(max(np.quantile(pg, near_q), 0.01)) if len(pg) else 0.05,
                miss_thr=float(max(np.quantile(cgaps, 0.9) if cgaps else 0.0, 0.02)))


# ----------------------------------------------------------------------------- facts
def stim_facts(key, r, g, th, bin_size, horizon):
    ent = {o: f"{key}_o{o}" for o in g["ids"]}
    scene = f"{key}_scene"
    facts = []

    def add(s, rel, o, t):
        facts.append((s, rel, o, t))
        if rel in SYMMETRIC:
            facts.append((o, rel, s, t))

    rl = roles(r, g)
    for i, o in enumerate(g["ids"]):
        add(ent[o], "has_role", f"role_{rl[o]}", 0)
        add(ent[o], "has_shape", f"shape_{g['names'][i]}", 0)

    ends = bin_ends(g["T"], bin_size)
    for (ti, a, b, s) in r["collisions"]:
        if ti < g["T"] and a in ent and b in ent and str(s).lower() in ("enter", "stay", "e", "s", ""):
            add(ent[a], "contact", ent[b], ti // bin_size)
    lab = np.asarray(r["labels"]["target_contacting_zone"])[:g["T"]]
    for ti in np.where(lab)[0]:
        add(ent[g["tid"]], "contact", ent[g["zid"]], int(ti) // bin_size)
    for (ti, a) in r["env_collisions"]:
        if ti < g["T"] and a in ent:
            add(ent[a], "touches_env", scene, ti // bin_size)

    n = len(g["ids"])
    lo, hi, V = g["lo"], g["hi"], g["V"]
    E = [ent[o] for o in g["ids"]]
    steps = np.arange(3, horizon + 1, 3)
    iu = np.triu_indices(n, 1)
    for b, f in enumerate(ends):
        fp = max(0, f - bin_size)
        sp = np.linalg.norm(V[f], axis=1)
        for i in range(n):
            rel = ("stationary" if sp[i] <= th["eps"] else "speed_slow" if sp[i] <= th["q1"] else
                   "speed_medium" if sp[i] <= th["q2"] else "speed_fast")
            add(E[i], rel, scene, b)
            if V[f, i, 1] < -th["q1"]:
                add(E[i], "falling", scene, b)
        L, Hh = lo[f], hi[f]
        G = gapmat(L, Hh, L, Hh)
        Gp = gapmat(lo[fp], hi[fp], lo[fp], hi[fp])
        dV = V[f][:, None] - V[f][None]                                    # n,n,3 relative velocity of row vs col
        Gmin = np.full((n, n), np.inf)
        for st in steps:                                                    # row box translated by st*dV
            d = np.maximum(0.0, np.maximum(L[:, None] + st * dV - Hh[None], L[None] - (Hh[:, None] + st * dV)))
            Gmin = np.minimum(Gmin, np.sqrt((d ** 2).sum(-1)))
        hgt = Hh[:, 1] - L[:, 1]
        for i in range(n):
            for j in range(n):
                if i != j and G[i, j] < hgt[i] and hgt[i] > th["near_thr"]:
                    add(E[i], "within_reach", E[j], b)
        for i, j in zip(*iu):
            if G[i, j] <= th["near_thr"]:
                add(E[i], "near", E[j], b)
            if G[i, j] < Gp[i, j] - th["eps"]:
                add(E[i], "approaching", E[j], b)
            if Gmin[i, j] < th["miss_thr"] and G[i, j] >= th["miss_thr"]:
                add(E[i], "heading_toward", E[j], b)
        ovx = (L[:, None, 0] < Hh[None, :, 0]) & (L[None, :, 0] < Hh[:, None, 0])
        ovz = (L[:, None, 2] < Hh[None, :, 2]) & (L[None, :, 2] < Hh[:, None, 2])
        top = ovx & ovz & (np.abs(L[:, None, 1] - Hh[None, :, 1]) < 0.02)   # row resting on col
        np.fill_diagonal(top, False)
        for i, j in zip(*np.where(top)):
            add(E[i], "on_top_of", E[j], b)
    return list(dict.fromkeys(facts)), len(ends)


def stim_features(r, g, th, horizon_total):
    """Scalar features for non-learned baselines (observed window only)."""
    i, j, f = g["idx"][g["tid"]], g["idx"][g["zid"]], g["T"] - 1
    lo, hi, V = g["lo"], g["hi"], g["V"]
    d_end = gap(lo[f, i], hi[f, i], lo[f, j], hi[f, j])
    d_prev = gap(lo[max(0, f - 5), i], hi[max(0, f - 5), i], lo[max(0, f - 5), j], hi[max(0, f - 5), j])
    rel = V[f, i] - V[f, j]
    ext = min(gap(lo[f, i] + rel * s, hi[f, i] + rel * s, lo[f, j], hi[f, j])
              for s in range(0, horizon_total + 1, 3))
    contact_obs = bool(np.asarray(r["labels"]["target_contacting_zone"])[:g["T"]].any()) or any(
        ti < g["T"] and {a, b} == {g["tid"], g["zid"]} for (ti, a, b, _) in r["collisions"])
    n = len(g["ids"]); L, Hh = lo[f], hi[f]
    G = gapmat(L, Hh, L, Hh); hgt = Hh[:, 1] - L[:, 1]
    adj = (G < np.maximum(hgt[:, None], th["near_thr"])) | (G <= th["near_thr"])
    np.fill_diagonal(adj, False)
    moving = set(np.where(np.linalg.norm(V[f], axis=1) > th["q1"])[0].tolist())
    moving |= {g["idx"][a] for (ti, a, b, s) in r["collisions"] if ti < g["T"] and a in g["idx"] and a != g["zid"]}
    moving |= {g["idx"][b] for (ti, a, b, s) in r["collisions"] if ti < g["T"] and b in g["idx"] and b != g["zid"]}
    moving.discard(j)
    hops, frontier, seen = None, set(moving), set(moving)
    for h in range(n + 1):
        if i in frontier:
            hops = h; break
        nxt = {v for u in frontier for v in np.where(adj[u])[0].tolist() if v not in seen and v != j}
        if not nxt:
            break
        seen |= nxt; frontier = nxt
    return dict(d_end=d_end, closing=d_prev - d_end, extrap_min=ext, contact_obs=bool(contact_obs),
                target_speed=float(np.linalg.norm(V[f, i])), target_vy=float(V[f, i, 1]),
                chain_hops=float(hops) if hops is not None else 99.0, n_moving=float(len(moving)))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="states")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", default="none", help="scenario name or 'none'")
    ap.add_argument("--obs-frames", type=int, default=45)
    ap.add_argument("--bin-size", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=45, help="frames ahead for heading_toward")
    ap.add_argument("--extrap-frames", type=int, default=150, help="frames ahead for extrapolation baseline")
    ap.add_argument("--near-quantile", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab", default="full", choices=["events", "kin", "rel", "full"])
    a = ap.parse_args()

    recs = load_states(a.states)
    keep = []
    for r in recs:
        g = prep(r, a.obs_frames)
        if g is not None:
            keep.append((r, g))
    print(f"usable stimuli {len(keep)} / {len(recs)}")
    ho = None if a.holdout.lower() == "none" else a.holdout
    pool = [(r, g) for r, g in keep if r["pool"] == "pool" and r["scenario"] != ho]
    test = [(r, g) for r, g in keep if r["pool"] == "test" and (ho is None or r["scenario"] == ho)]
    random.Random(a.seed).shuffle(pool)
    nv = int(len(pool) * a.valid_frac)
    splits = {"train": pool[nv:], "valid": pool[:nv], "test": test}
    print({k: len(v) for k, v in splits.items()}, "holdout:", ho)

    th = fit_thresholds([g for _, g in splits["train"]], [r for r, _ in splits["train"]],
                        a.bin_size, a.near_quantile, a.eps)
    print("thresholds (train only):", th)

    rels = STATIC_RELS + DYN_RELS + [QUERY_REL]
    rel2id = {x: i for i, x in enumerate(rels)}
    ent2id = {}
    eid = lambda x: ent2id.setdefault(x, len(ent2id))
    os.makedirs(a.out, exist_ok=True)
    meta = {"thresholds": th, "holdout": ho, "vocab": a.vocab, "args": vars(a), "stimuli": {}}
    for split, items in splits.items():
        ff = open(os.path.join(a.out, f"facts_{split}.txt"), "w")
        qf = open(os.path.join(a.out, f"queries_{split}.tsv"), "w")
        cnt = Counter()
        for n_, (r, g) in enumerate(items):
            key = f"{split[:2]}{n_}"
            facts, nb = stim_facts(key, r, g, th, a.bin_size, a.horizon)
            for s, rel, o, t in facts:
                if rel not in VOCAB[a.vocab]:
                    continue
                ff.write(f"{eid(s)}\t{rel2id[rel]}\t{eid(o)}\t{t}\n")
                cnt[rel] += 1
            s_ent, o_ent = eid(f"{key}_o{g['tid']}"), eid(f"{key}_o{g['zid']}")
            lab = int(r["ocp_any"])
            qf.write(f"{s_ent}\t{rel2id[QUERY_REL]}\t{o_ent}\t{nb}\t{lab}\t{key}\t{r['scenario']}\n")
            feat = stim_features(r, g, th, a.extrap_frames)
            feat.update(split=split, scenario=r["scenario"], label=lab, stim=str(r["static"].get("stimulus_name", "")))
            meta["stimuli"][key] = feat
        ff.close(); qf.close()
        labs = [meta["stimuli"][f"{split[:2]}{i}"]["label"] for i in range(len(items))]
        print(f"  {split:5s} {len(items):5d} stimuli  pos {np.mean(labs) if labs else 0:.2f}  facts {sum(cnt.values())}  {dict(cnt)}")
    with open(os.path.join(a.out, "entity2id.txt"), "w") as fh:
        for e, i in ent2id.items():
            fh.write(f"{e}\t{i}\n")
    with open(os.path.join(a.out, "relation2id.txt"), "w") as fh:
        for x, i in rel2id.items():
            fh.write(f"{x}\t{i}\n")
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"))
    tst = [v for v in meta["stimuli"].values() if v["split"] == "test"]
    if tst:
        print(f"test: contact already observed in window: {np.mean([v['contact_obs'] for v in tst]):.1%}")


if __name__ == "__main__":
    main()
