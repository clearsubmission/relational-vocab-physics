#!/usr/bin/env python3
"""Draw a REAL CLEVRER test video as a stroboscopic top-down scene for Figure 1, and dump its real facts.
Usage: python make_scene_panel.py --ann-dir clevrer_ann --data-dir data/physt_clevrer_rel_h128 [--vid 12345]"""
import argparse, glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"gray": "#8a8a85", "red": "#d43c3c", "blue": "#2a78d6", "green": "#3f9b4a", "brown": "#9c6b3f",
          "cyan": "#2bb3c0", "purple": "#8a55c9", "yellow": "#e3b505"}
MARKERS = {"cube": "s", "sphere": "o", "cylinder": "h"}
RUNG = {"collides_with": "i", "enters_scene": "i", "exits_scene": "i",
        "stationary": "ii", "speed_slow": "ii", "speed_medium": "ii", "speed_fast": "ii", "near": "ii",
        "approaching": "iii", "heading_toward": "iii"}


def load_ann(ann_dir, vid):
    for f in glob.glob(os.path.join(ann_dir, "**", f"annotation_{int(vid):05d}.json"), recursive=True):
        return json.load(open(f))
    for f in glob.glob(os.path.join(ann_dir, "**", "*.json"), recursive=True):
        d = json.load(open(f))
        if int(d.get("scene_index", -1)) == int(vid):
            return d
    raise SystemExit(f"annotation for video {vid} not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--vid", default=None)
    ap.add_argument("--out", default="scene_panel")
    ap.add_argument("--max-objects", type=int, default=5)
    a = ap.parse_args()

    meta = json.load(open(os.path.join(a.data_dir, "meta.json")))
    named = lambda f: [l.rstrip("\n").split("\t") for l in open(os.path.join(a.data_dir, "named", f)) if l.strip()]
    ctx = named("test_context.txt"); qs = named("test.txt")
    ctx_by_vid = {}
    for s, r, o, t in ctx:
        ctx_by_vid.setdefault(s.split("_")[0][1:], []).append((s, r, o, int(t)))

    cands = []
    for s, r, o, t in qs:
        vid = s.split("_")[0][1:]
        if a.vid and vid != str(a.vid):
            continue
        info = meta["videos"].get(vid)
        if not info or info["split"] != "test" or len(info["objects"]) > a.max_objects:
            continue
        pair = {(x[1], x[3]) for x in ctx_by_vid.get(vid, []) if x[0] == s and x[2] == o}
        cue = any(rr == "heading_toward" for rr, _ in pair) or any(rr == "approaching" for rr, _ in pair)
        if cue or a.vid:
            cands.append((vid, s, o, int(t)))
    if not cands:
        raise SystemExit("no suitable video found (try --max-objects 6 or a specific --vid)")
    vid, subj, partner, tq = cands[0]
    info = meta["videos"][vid]
    cut_f = info["cutoff_frame"]
    d = load_ann(a.ann_dir, vid)
    props = {int(p["object_id"]): p for p in d["object_property"]}
    oids = sorted(props)
    T = max(int(fr["frame_id"]) for fr in d["motion_trajectory"]) + 1
    P = np.full((T, len(oids), 2), np.nan); IN = np.zeros((T, len(oids)), bool)
    for fr in d["motion_trajectory"]:
        f = int(fr["frame_id"])
        for ob in fr["objects"]:
            i = oids.index(int(ob["object_id"]))
            P[f, i] = ob["location"][:2]; IN[f, i] = bool(ob.get("inside_camera_view", True))
    s_oid, p_oid = int(subj.split("_o")[1]), int(partner.split("_o")[1])
    coll = [(int(c["frame_id"]), *[int(x) for x in c["object_ids"][:2]]) for c in d.get("collision", [])]
    hit = [c for c in coll if c[0] >= cut_f and {c[1], c[2]} == {s_oid, p_oid}]
    order = [s_oid, p_oid] + [o for o in oids if o not in (s_oid, p_oid)]
    letter = {o: chr(65 + k) for k, o in enumerate(order)}

    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.set_facecolor("#f3f3f0")
    for i, o in enumerate(oids):
        p = props[o]; col = COLORS.get(p["color"], "#777"); mk = MARKERS.get(p["shape"], "o")
        obs = [f for f in range(0, cut_f + 1, 6) if IN[f, i] and np.isfinite(P[f, i]).all()]
        for k, f in enumerate(obs):
            ax.scatter(*P[f, i], s=26, marker=mk, color=col, alpha=0.18 + 0.82 * (k + 1) / len(obs),
                       edgecolor="white", linewidth=0.4, zorder=3)
        fut = [f for f in range(cut_f, T) if IN[f, i] and np.isfinite(P[f, i]).all()]
        if fut:
            ax.plot(P[fut, i, 0], P[fut, i, 1], color=col, lw=0.8, ls=(0, (1.5, 1.5)), alpha=0.55, zorder=2)
        if obs:
            last = P[obs[-1], i]
            ax.text(last[0], last[1] + 0.32, letter[o], fontsize=7, fontweight="bold", color="#222",
                    ha="center", va="bottom", zorder=5)
    if hit:
        f0 = hit[0][0]
        pt = np.nanmean(P[f0, [oids.index(s_oid), oids.index(p_oid)]], axis=0)
        ax.scatter(*pt, s=170, marker="o", facecolor="none", edgecolor="#222", linewidth=1.0, linestyle="--", zorder=6)
    vis = P[IN & np.isfinite(P[..., 0])]
    if len(vis):
        lo, hi = vis.min(0), vis.max(0)
        ax.set_xlim(lo[0] - 0.6, hi[0] + 0.6); ax.set_ylim(lo[1] - 0.6, hi[1] + 1.0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#d6d6d1")
    fig.savefig(a.out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(a.out + ".pdf", bbox_inches="tight", pad_inches=0.02)

    def nm(e):
        if e.endswith("_scene"):
            return "scene"
        if "_o" in e and e.split("_")[0][1:] == vid:
            return letter.get(int(e.split("_o")[1]), e)
        return e.split("_", 1)[-1] if "_" in e else e
    facts = sorted({(nm(s), r, nm(o), t) for s, r, o, t in ctx_by_vid[vid] if r in RUNG}, key=lambda x: (RUNG[x[1]], -x[3]))
    by_rung = {}
    for s, r, o, t in facts:
        by_rung.setdefault(RUNG[r], [])
        txt = f"{s} {r} {o}" if o != "scene" else f"{s} {r}"
        if txt not in by_rung[RUNG[r]]:
            by_rung[RUNG[r]].append(txt)
    out = {"video": vid, "cutoff_frame": cut_f, "query": "(A, collides_with, ?)", "answer": "B",
           "collision_frame": hit[0][0] if hit else None,
           "objects": {letter[o]: f"{props[o]['color']} {props[o]['material']} {props[o]['shape']}" for o in order},
           "facts_by_rung_most_recent_first": {k: v[:8] for k, v in by_rung.items()}}
    json.dump(out, open(a.out + ".json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"wrote {a.out}.png / .pdf / .json  (video {vid}; {len(cands)} candidate videos)")


if __name__ == "__main__":
    main()
