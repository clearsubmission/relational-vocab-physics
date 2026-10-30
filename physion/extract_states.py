#!/usr/bin/env python3
"""Stream a Physion HDF5 tar.gz from S3; keep only object states + labels. -> states/<stem>.pkl.gz"""
import argparse, gzip, io, os, pickle, sys, tarfile, time, urllib.request
import h5py
import numpy as np

BASE = "https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/"
STATIC_KEYS = ["object_ids", "model_names", "mass", "dynamic_friction", "static_friction", "bounciness",
               "scale_x", "scale_y", "scale_z", "initial_position", "target_id", "zone_id", "probe_id",
               "probe_type", "target_type", "distractors", "occluders", "push_force", "push_position",
               "push_time", "stimulus_name", "room", "trial_num"]
OBJ_KEYS = ["positions", "velocities", "angular_velocities", "rotations", "top", "bottom", "left", "right", "front", "back"]
LABEL_KEYS = ["target_contacting_zone", "target_has_moved", "target_on_ground", "trial_end"]


def dec(v):
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    if isinstance(v, np.ndarray) and v.dtype == object:
        return [dec(x) for x in v.tolist()]
    return v


def parse(data, name):
    f = h5py.File(io.BytesIO(data), "r")
    st = f["static"]
    rec = {"file": name, "static": {k: dec(st[k][()]) for k in STATIC_KEYS if k in st}}
    frames = sorted(f["frames"].keys())
    T = len(frames)
    obj = {k: [] for k in OBJ_KEYS}
    lab = {k: np.zeros(T, bool) for k in LABEL_KEYS}
    coll, env = [], []
    for ti, fk in enumerate(frames):
        fr = f["frames"][fk]
        for k in OBJ_KEYS:
            obj[k].append(fr["objects"][k][()] if k in fr["objects"] else None)
        if "labels" in fr:
            for k in LABEL_KEYS:
                if k in fr["labels"]:
                    lab[k][ti] = bool(fr["labels"][k][()])
        if "collisions" in fr and "object_ids" in fr["collisions"]:
            ids = fr["collisions"]["object_ids"][()]
            sts = fr["collisions"]["states"][()] if "states" in fr["collisions"] else [b""] * len(ids)
            for (a, b), s in zip(ids.reshape(-1, 2), np.asarray(sts).reshape(-1)):
                coll.append((ti, int(a), int(b), dec(s)))
        if "env_collisions" in fr and "object_ids" in fr["env_collisions"]:
            for a in np.asarray(fr["env_collisions"]["object_ids"][()]).reshape(-1):
                env.append((ti, int(a)))
    for k in OBJ_KEYS:
        if all(x is not None for x in obj[k]):
            obj[k] = np.stack(obj[k]).astype(np.float32)
    rec.update(T=T, objects=obj, labels=lab, collisions=coll, env_collisions=env)
    rec["ocp_any"] = bool(lab["target_contacting_zone"].any())
    rec["ocp_final"] = bool(lab["target_contacting_zone"][-1])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--out", default="states")
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    url = a.archive if "://" in a.archive else BASE + a.archive
    stem = os.path.basename(url).replace(".tar.gz", "")
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, stem + ".pkl.gz")
    recs, bad, t0 = [], 0, time.time()
    try:
        tf = tarfile.open(fileobj=urllib.request.urlopen(url, timeout=120), mode="r|gz")
        for m in tf:
            if not (m.isfile() and m.name.endswith(".hdf5")):
                continue
            try:
                recs.append(parse(tf.extractfile(m).read(), m.name))
            except Exception as e:
                bad += 1
                print(f"[{stem}] skip {m.name}: {e}", file=sys.stderr)
            if len(recs) % 100 == 0 and recs:
                print(f"[{stem}] {len(recs)} files  {time.time() - t0:.0f}s", flush=True)
            if a.max and len(recs) >= a.max:
                break
    except Exception as e:
        print(f"[{stem}] STREAM ERROR after {len(recs)} files: {e}", file=sys.stderr)
    with gzip.open(out, "wb") as fh:
        pickle.dump(recs, fh)
    pos = sum(r["ocp_any"] for r in recs)
    print(f"[{stem}] DONE {len(recs)} stimuli ({bad} skipped), contact-any {pos}/{len(recs)}, "
          f"{time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
