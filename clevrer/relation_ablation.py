#!/usr/bin/env python3
"""Leave-one-relation-out for symbolic_lr on a CLEVRER build. Usage: python relation_ablation.py --data-dir DIR"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clevrer_learned_baselines as L  # noqa: E402

GROUPS = {
    "none (full)": [],
    "heading_toward": ["heading_toward"],
    "approaching": ["approaching"],
    "near": ["near"],
    "speed buckets": ["stationary", "speed_slow", "speed_medium", "speed_fast"],
    "collides_with (history)": ["collides_with"],
    "enters/exits": ["enters_scene", "exits_scene"],
    "direction (approaching + heading_toward)": ["approaching", "heading_toward"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    a = ap.parse_args()
    dd = a.data_dir
    meta = json.load(open(os.path.join(dd, "meta.json")))
    rel2id = {l.split("\t")[0]: int(l.split("\t")[1]) for l in open(os.path.join(dd, "relation2id.txt"))}
    id2rel = {v: k for k, v in rel2id.items()}
    ent_name = {}
    for l in open(os.path.join(dd, "entity2id.txt")):
        name, i = l.rstrip("\n").split("\t")
        if name.startswith("v") and "_" in name:
            vid, rest = name[1:].split("_", 1)
            ent_name[int(i)] = (vid, int(rest[1:]) if rest.startswith("o") else None)
    facts = {sp: L.load_quads(os.path.join(dd, f"{sp}.txt")) for sp in ("train", "valid", "test")}
    hist_all = facts["train"] + L.load_quads(os.path.join(dd, "valid_context.txt")) + \
        L.load_quads(os.path.join(dd, "test_context.txt"))
    qs = L.build_queries(meta, ent_name, facts, rel2id["collides_with"], a.cutoffs)
    qs_by_split = {sp: [q for q in qs if q[0] == sp] for sp in ("train", "valid", "test")}
    print("queries:", {k: len(v) for k, v in qs_by_split.items()}, flush=True)
    out = {}
    for name, drop in GROUPS.items():
        present = [r for r in drop if r in rel2id]
        if drop and not present:
            continue
        drop_ids = {rel2id[r] for r in present}
        hist = [q for q in hist_all if q[1] not in drop_ids]
        idx = L.FactIndex(hist, ent_name, id2rel)
        rels_present = set(rel2id) - set(present)
        te, byh = L.train_eval(f"- {name}", lambda q, c: L.symbolic_features(idx, rels_present, q, c), qs_by_split)
        out[name] = {"MRR": te["MRR"], "H@1": te["H@1"], "by_horizon": byh}
    base = out["none (full)"]["MRR"]
    print(f"\n{'removed':42s} {'MRR':>7s} {'drop':>7s}   " + " ".join(f"{h:>6s}" for h in L.LABELS))
    for k, v in out.items():
        print(f"{k:42s} {v['MRR']:7.4f} {v['MRR'] - base:+7.4f}   " +
              " ".join(f"{v['by_horizon'].get(h, float('nan')):6.3f}" for h in L.LABELS))
    json.dump(out, open(os.path.join(dd, "relation_ablation.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
