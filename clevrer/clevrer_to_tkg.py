#!/usr/bin/env python3
"""
CLEVRER annotations -> fully inductive temporal knowledge graph (PhysT-KG).
Each video's objects are new entities (v{vid}_o{oid}); splits are entity-disjoint
(optionally sharing attribute-value entities).

Output (tab-separated integer ids: s r o t):
  entity2id.txt, relation2id.txt, stat.txt, train.txt,
  valid_context.txt / valid.txt, test_context.txt / test.txt, meta.json, named/*.txt
"""
import argparse
import glob
import json
import os
import random
from collections import Counter

import numpy as np

STATIC_RELS = ["has_shape", "has_material", "has_color"]
EVENT_RELS = ["collides_with", "enters_scene", "exits_scene"]
KIN_RELS = ["near", "stationary", "speed_slow", "speed_medium", "speed_fast"]
MOTION_RELS = ["approaching", "heading_toward"]
SYMMETRIC = {"collides_with", "near", "approaching", "heading_toward"}


def iter_annotations(ann_dir):
    files = sorted(glob.glob(os.path.join(ann_dir, "**", "*.json"), recursive=True))
    if not files:
        raise SystemExit(f"No .json annotation files found under {ann_dir}")
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if isinstance(d, dict) and "object_property" in d and "motion_trajectory" in d:
            yield d


def parse_video(d):
    vid = int(d["scene_index"])
    props = {int(o["object_id"]): o for o in d["object_property"]}
    oids = sorted(props)
    idx = {o: i for i, o in enumerate(oids)}
    frames = d["motion_trajectory"]
    T = max(int(fr["frame_id"]) for fr in frames) + 1
    n = len(oids)
    P = np.full((T, n, 2), np.nan, np.float32)
    IN = np.zeros((T, n), bool)
    for fr in frames:
        f = int(fr["frame_id"])
        for o in fr["objects"]:
            i = idx.get(int(o["object_id"]))
            if i is None:
                continue
            P[f, i] = o["location"][:2]
            IN[f, i] = bool(o.get("inside_camera_view", True))
    cols = []
    for c in d.get("collision", []):
        a, b = (int(x) for x in c["object_ids"][:2])
        if a in idx and b in idx:
            cols.append((int(c["frame_id"]), a, b))
    return dict(vid=vid, oids=oids, idx=idx, props=props, P=P, IN=IN, cols=cols, T=T)


def frame_velocity(P, f, k=4):
    f0 = max(0, f - k)
    if f == f0:
        return np.zeros(P.shape[1:], np.float32)
    v = (P[f] - P[f0]) / (f - f0)
    return np.nan_to_num(v, nan=0.0)


def bin_end_frames(T, bin_size):
    nb = (T + bin_size - 1) // bin_size
    return [min((b + 1) * bin_size - 1, T - 1) for b in range(nb)]


def fit_thresholds(videos, bin_size, near_q, eps):
    speeds, dists, cdists = [], [], []
    for v in videos:
        for f in bin_end_frames(v["T"], bin_size):
            vis = np.where(v["IN"][f])[0]
            if len(vis) == 0:
                continue
            vel = frame_velocity(v["P"], f)
            speeds.extend(np.linalg.norm(vel[vis], axis=1).tolist())
            for a in range(len(vis)):
                for b in range(a + 1, len(vis)):
                    dd = np.linalg.norm(v["P"][f, vis[a]] - v["P"][f, vis[b]])
                    if np.isfinite(dd):
                        dists.append(dd)
        for f, a, b in v["cols"]:
            dd = np.linalg.norm(v["P"][f, v["idx"][a]] - v["P"][f, v["idx"][b]])
            if np.isfinite(dd):
                cdists.append(dd)
    speeds = np.array(speeds)
    moving = speeds[speeds > eps]
    q1, q2 = (np.quantile(moving, [1 / 3, 2 / 3]) if len(moving) else (eps, eps))
    near_thr = float(np.quantile(dists, near_q)) if dists else 0.0
    miss_thr = float(np.quantile(cdists, 0.9)) if cdists else near_thr
    return dict(eps=eps, q1=float(q1), q2=float(q2), near_thr=near_thr, miss_thr=miss_thr)


def video_facts(v, bin_size, th, use_attr, use_kin, use_rel=False, horizon=32):
    vid, T, IN, P = v["vid"], v["T"], v["IN"], v["P"]
    ent = {o: f"v{vid}_o{o}" for o in v["oids"]}
    scene = f"v{vid}_scene"
    facts = []

    def add(s, r, o, t):
        facts.append((s, r, o, t))
        if r in SYMMETRIC:
            facts.append((o, r, s, t))

    if use_attr:
        for o in v["oids"]:
            p = v["props"][o]
            add(ent[o], "has_shape", f"shape_{p['shape']}", 0)
            add(ent[o], "has_material", f"material_{p['material']}", 0)
            add(ent[o], "has_color", f"color_{p['color']}", 0)

    for f, a, b in v["cols"]:
        add(ent[a], "collides_with", ent[b], f // bin_size)

    for i, o in enumerate(v["oids"]):
        for f in range(1, T):
            if IN[f, i] and not IN[f - 1, i]:
                add(ent[o], "enters_scene", scene, f // bin_size)
            elif IN[f - 1, i] and not IN[f, i]:
                add(ent[o], "exits_scene", scene, f // bin_size)

    if use_kin:
        for b, f in enumerate(bin_end_frames(T, bin_size)):
            vis = np.where(IN[f])[0]
            vel = frame_velocity(P, f)
            for i in vis:
                s = float(np.linalg.norm(vel[i]))
                r = ("stationary" if s <= th["eps"] else
                     "speed_slow" if s <= th["q1"] else
                     "speed_medium" if s <= th["q2"] else "speed_fast")
                add(ent[v["oids"][i]], r, scene, b)
            for x in range(len(vis)):
                for y in range(x + 1, len(vis)):
                    i, j = vis[x], vis[y]
                    dd = np.linalg.norm(P[f, i] - P[f, j])
                    if np.isfinite(dd) and dd < th["near_thr"]:
                        add(ent[v["oids"][i]], "near", ent[v["oids"][j]], b)

    if use_rel:
        for b, f in enumerate(bin_end_frames(T, bin_size)):
            vis = np.where(IN[f])[0]
            vel = frame_velocity(P, f)
            for x in range(len(vis)):
                for y in range(x + 1, len(vis)):
                    i, j = vis[x], vis[y]
                    d, w = P[f, j] - P[f, i], vel[j] - vel[i]
                    dist, w2 = float(np.linalg.norm(d)), float(np.dot(w, w))
                    if not np.isfinite(dist) or w2 <= th["eps"] ** 2:
                        continue
                    if -float(np.dot(d, w)) / max(dist, 1e-9) <= th["eps"]:
                        continue                      # not closing
                    add(ent[v["oids"][i]], "approaching", ent[v["oids"][j]], b)
                    tca = -float(np.dot(d, w)) / w2    # frames to closest approach
                    miss = float(np.linalg.norm(d + tca * w))
                    if 0 < tca <= horizon and miss < th["miss_thr"]:
                        add(ent[v["oids"][i]], "heading_toward", ent[v["oids"][j]], b)

    return list(dict.fromkeys(facts))


def last_seen(P, f):
    out = np.full(P.shape[1:], np.nan, np.float32)
    for i in range(P.shape[1]):
        ok = np.where(np.isfinite(P[: f + 1, i, 0]))[0]
        if len(ok):
            out[i] = P[ok[-1], i]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--bin-size", type=int, default=8)
    ap.add_argument("--cutoff-frac", type=float, default=0.5)
    ap.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--query-rels", nargs="+", default=["collides_with"])
    ap.add_argument("--near-quantile", type=float, default=0.1)
    ap.add_argument("--stationary-eps", type=float, default=1e-3)
    ap.add_argument("--no-kinematics", action="store_true")
    ap.add_argument("--no-attributes", action="store_true")
    ap.add_argument("--relative-motion", action="store_true", help="add approaching / heading_toward")
    ap.add_argument("--horizon", type=int, default=32, help="frames ahead for heading_toward")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    videos = [parse_video(d) for d in iter_annotations(a.ann_dir)]
    random.Random(a.seed).shuffle(videos)
    if a.max_videos:
        videos = videos[: a.max_videos]
    n = len(videos)
    n_tr = int(a.split[0] * n)
    n_va = int(a.split[1] * n)
    splits = {"train": videos[:n_tr], "valid": videos[n_tr:n_tr + n_va], "test": videos[n_tr + n_va:]}
    print(f"videos: {n}  train {len(splits['train'])}  valid {len(splits['valid'])}  test {len(splits['test'])}")

    th = fit_thresholds(splits["train"], a.bin_size, a.near_quantile, a.stationary_eps)
    print("thresholds (fit on train only):", th)

    use_kin, use_attr = not a.no_kinematics, not a.no_attributes
    rels = (STATIC_RELS if use_attr else []) + EVENT_RELS + (KIN_RELS if use_kin else []) + (MOTION_RELS if a.relative_motion else [])
    rel2id = {r: i for i, r in enumerate(rels)}
    ent2id, meta = {}, {"thresholds": th, "bin_size": a.bin_size, "cutoff_frac": a.cutoff_frac,
                        "query_rels": a.query_rels, "videos": {}}
    out = {k: [] for k in ["train", "valid_context", "valid", "test_context", "test"]}

    def eid(name):
        if name not in ent2id:
            ent2id[name] = len(ent2id)
        return ent2id[name]

    for split, vids in splits.items():
        for v in vids:
            facts = video_facts(v, a.bin_size, th, use_attr, use_kin, a.relative_motion, a.horizon)
            nb = len(bin_end_frames(v["T"], a.bin_size))
            cut_bin = max(1, int(nb * a.cutoff_frac))
            cut_frame = min(cut_bin * a.bin_size - 1, v["T"] - 1)
            obj_ents = {o: eid(f"v{v['vid']}_o{o}") for o in v["oids"]}
            for f in facts:
                if split == "train":
                    out["train"].append(f)
                elif f[3] < cut_bin:
                    out[f"{split}_context"].append(f)
                elif f[1] in a.query_rels:
                    out[split].append(f)
            pos = last_seen(v["P"], cut_frame)
            vel = frame_velocity(v["P"], cut_frame)
            meta["videos"][str(v["vid"])] = {
                "split": split, "cutoff_bin": cut_bin, "cutoff_frame": cut_frame,
                "objects": {str(e): {"oid": o,
                                     "pos": None if not np.isfinite(pos[i]).all() else pos[i].tolist(),
                                     "vel": vel[i].tolist(),
                                     "in_view": bool(v["IN"][cut_frame, i])}
                            for i, (o, e) in enumerate(obj_ents.items())},
            }

    for k in out:
        for s, _, o, _ in out[k]:
            eid(s), eid(o)

    os.makedirs(os.path.join(a.out_dir, "named"), exist_ok=True)
    for k, facts in out.items():
        with open(os.path.join(a.out_dir, f"{k}.txt"), "w") as fh:
            for s, r, o, t in facts:
                fh.write(f"{ent2id[s]}\t{rel2id[r]}\t{ent2id[o]}\t{t}\n")
        with open(os.path.join(a.out_dir, "named", f"{k}.txt"), "w") as fh:
            for s, r, o, t in facts:
                fh.write(f"{s}\t{r}\t{o}\t{t}\n")
    with open(os.path.join(a.out_dir, "entity2id.txt"), "w") as fh:
        for e, i in ent2id.items():
            fh.write(f"{e}\t{i}\n")
    with open(os.path.join(a.out_dir, "relation2id.txt"), "w") as fh:
        for r, i in rel2id.items():
            fh.write(f"{r}\t{i}\n")
    with open(os.path.join(a.out_dir, "stat.txt"), "w") as fh:
        fh.write(f"{len(ent2id)}\t{len(rel2id)}\t0\n")
    with open(os.path.join(a.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh)

    print(f"entities {len(ent2id)}  relations {len(rel2id)}")
    for k, facts in out.items():
        print(f"  {k:14s} {len(facts):8d} facts   {dict(Counter(f[1] for f in facts))}")
    ctx = {(s, o) for s, r, o, _ in out["test_context"] if r in ("collides_with", "near")}
    q = out["test"]
    if q:
        seen_pair = sum((s, o) in ctx for s, _, o, _ in q) / len(q)
        print(f"test queries whose pair interacted (collide/near) in history: {seen_pair:.1%}")
    train_ents = {x for s, _, o, _ in out["train"] for x in (s, o)}
    test_ents = {x for k in ["test_context", "test"] for s, _, o, _ in out[k] for x in (s, o)}
    shared = train_ents & test_ents
    print(f"entities shared train/test: {len(shared)} "
          f"({'attribute values only' if all(not e.startswith('v') for e in shared) else 'CHECK: object leak'})")


if __name__ == "__main__":
    main()
