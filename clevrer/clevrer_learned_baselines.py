#!/usr/bin/env python3
"""Learned linear baselines (conditional logit) for CLEVRER collision forecasting.
symbolic_lr: features from the build's facts only. physics_lr: raw-state features (vocabulary-independent).
Same protocol as clevrer_encoders.py. Results merged into <build>/baselines_test.json."""
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clevrer_to_tkg import iter_annotations, parse_video, frame_velocity, last_seen  # noqa: E402

PAIR_RELS = ["collides_with", "near", "approaching", "heading_toward"]
SPEED_RELS = ["stationary", "speed_slow", "speed_medium", "speed_fast"]
BUCKETS = [(0, 1), (2, 3), (4, 5), (6, 99)]
LABELS = ["0-1", "2-3", "4-5", "6+"]


def build_queries(meta, ent_name, facts_by_split, qrel, cuts):
    out = []
    for split, facts in facts_by_split.items():
        truth = defaultdict(set)
        for s, r, o, t in facts:
            if r == qrel:
                truth[(s, t)].add(o)
        for (s, t), objs in truth.items():
            vid, s_oid = ent_name[s]
            info = meta["videos"][vid]
            Cs = [c for c in cuts if t >= c] if split == "train" else [info["cutoff_bin"]]
            all_oids = sorted(o["oid"] for o in info["objects"].values())
            for o in objs:
                o_oid = ent_name[o][1]
                cands = [c for c in all_oids if c != s_oid and (c == o_oid or c not in {ent_name[x][1] for x in objs})]
                for C in Cs:
                    out.append((split, vid, s_oid, t, C, cands, cands.index(o_oid)))
    return out


class FactIndex:
    def __init__(self, quads, ent_name, id2rel):
        self.pair = defaultdict(lambda: defaultdict(list))
        self.self_ = defaultdict(lambda: defaultdict(list))
        self.act = defaultdict(list)
        for s, r, o, t in quads:
            rel = id2rel[r]
            if rel.startswith("has_"):
                continue
            vs = ent_name.get(s); vo = ent_name.get(o)
            if vs is None:
                continue
            vid, a = vs
            self.act[(vid, a)].append(t)
            if vo is not None and vo[0] == vid and vo[1] is not None and a is not None:
                self.pair[(vid, a, vo[1])][rel].append(t)
            elif a is not None:
                self.self_[(vid, a)][rel].append(t)
            if vo is not None and vo[1] is not None and vo[0] == vid:
                self.act[(vid, vo[1])].append(t)


def symbolic_features(idx, rels_present, q, cand):
    split, vid, s, t, C, cands, _ = q
    h = (t - C) / 8.0
    f = []
    pr = idx.pair.get((vid, s, cand), {})
    for r in PAIR_RELS:
        if r not in rels_present:
            continue
        ts = [x for x in pr.get(r, []) if x < C]
        last = float(any(x == C - 1 for x in ts))
        rec = float(any(x >= C - 3 for x in ts))
        f += [last, rec, float(len(ts) > 0), np.log1p(len(ts)), last * h, rec * h]
    sf = idx.self_.get((vid, cand), {})
    for r in SPEED_RELS:
        if r in rels_present:
            f.append(float(any(x == C - 1 for x in sf.get(r, []))))
    for r in ("enters_scene", "exits_scene"):
        ts = [x for x in sf.get(r, []) if x < C]
        f += [float(len(ts) > 0)]
    na = sum(1 for x in idx.act.get((vid, cand), []) if x < C)
    f += [np.log1p(na), np.log1p(na) * h]
    return f


_PV = {}

def physics_features(V, q, cand, bin_size):
    split, vid, s, t, C, cands, _ = q
    v = V[vid]
    f_cut = min(C * bin_size - 1, v["T"] - 1)
    i, j = v["idx"][s], v["idx"][cand]
    key = (vid, f_cut)
    if key not in _PV:
        _PV[key] = (last_seen(v["P"], f_cut), frame_velocity(v["P"], f_cut))
    pos, vel = _PV[key]
    h_frames = max(1, t * bin_size + bin_size // 2 - f_cut)
    if not (np.isfinite(pos[i]).all() and np.isfinite(pos[j]).all()):
        d, closing, miss, tca = 10.0, 0.0, 10.0, 0.0
    else:
        dvec, w = pos[j] - pos[i], vel[j] - vel[i]
        d = float(np.linalg.norm(dvec))
        closing = float(-np.dot(dvec, w) / max(d, 1e-6))
        steps = np.arange(0, h_frames + 1)[:, None]
        miss = float(np.min(np.linalg.norm(dvec[None] + steps * w[None], axis=1)))
        w2 = float(np.dot(w, w))
        tca = float(np.clip(-np.dot(dvec, w) / w2, 0, 3 * h_frames) / h_frames) if w2 > 1e-9 else 3.0
    inview = float(v["IN"][f_cut, j])
    past = float(any(f < f_cut and {a, b} == {s, cand} for f, a, b in v["cols"]))
    hz = (t - C) / 8.0
    return [np.log(d + 1e-3), closing, np.log(miss + 1e-3), np.log(miss + 1e-3) * hz, tca,
            float(np.linalg.norm(vel[j])), float(np.linalg.norm(vel[i])), inview, past]


def to_tensor(feats, qs):
    F = len(feats[0][0]); cm = max(len(x) for x in feats)
    X = np.zeros((len(feats), cm, F), np.float64); M = np.zeros((len(feats), cm), bool)
    for k, rows in enumerate(feats):
        X[k, :len(rows)] = rows; M[k, :len(rows)] = True
    return X, M, np.array([q[6] for q in qs])


def fit_clogit(X, M, y, l2, iters=600, lr=0.05):
    mu = X[M].mean(0); sd = X[M].std(0) + 1e-8
    Z = np.where(M[..., None], (X - mu) / sd, 0.0)
    w = np.zeros(X.shape[2]); m1 = np.zeros_like(w); m2 = np.zeros_like(w)
    for it in range(1, iters + 1):
        s = np.where(M, Z @ w, -1e9); s -= s.max(1, keepdims=True)
        p = np.exp(s) * M; p /= p.sum(1, keepdims=True)
        tgt = np.zeros_like(p); tgt[np.arange(len(y)), y] = 1
        g = np.einsum("qc,qcf->f", p - tgt, Z) / len(y) + l2 * w
        m1 = 0.9 * m1 + 0.1 * g; m2 = 0.999 * m2 + 0.001 * g * g
        w -= lr * (m1 / (1 - 0.9 ** it)) / (np.sqrt(m2 / (1 - 0.999 ** it)) + 1e-8)
    return lambda X2, M2: np.where(M2, np.where(M2[..., None], (X2 - mu) / sd, 0.0) @ w, -np.inf)


def evaluate(scores, M, y, qs):
    rr = []
    for k in range(len(y)):
        so = scores[k, y[k]]; oth = np.delete(scores[k][M[k]], y[k])
        lo = 1 + int((oth > so).sum()); ties = int((oth == so).sum())
        rr.append(np.arange(lo, lo + ties + 1))
    mrr = np.array([np.mean(1 / x) for x in rr]); hz = np.array([q[3] - q[4] for q in qs])
    res = {"MRR": float(mrr.mean()), "H@1": float(np.mean([np.mean(x <= 1) for x in rr])),
           "H@3": float(np.mean([np.mean(x <= 3) for x in rr])), "n": len(rr)}
    byh = {l: float(mrr[(hz >= lo) & (hz <= hi)].mean()) for l, (lo, hi) in zip(LABELS, BUCKETS)
           if ((hz >= lo) & (hz <= hi)).any()}
    res["rr"] = [float(x) for x in mrr]
    res["keys"] = [[str(q[1]), int(q[2]), int(q[5][q[6]]), int(q[3])] for q in qs]
    return res, byh


def train_eval(name, featfn, qs_by_split):
    t0 = time.time(); data = {}
    for sp, qs in qs_by_split.items():
        data[sp] = to_tensor([[featfn(q, c) for c in q[5]] for q in qs], qs)
    best = None
    for l2 in (1e-4, 1e-3, 1e-2):
        f = fit_clogit(*data["train"], l2)
        vr, _ = evaluate(f(data["valid"][0], data["valid"][1]), data["valid"][1], data["valid"][2], qs_by_split["valid"])
        if best is None or vr["MRR"] > best[0]:
            best = (vr["MRR"], l2, f)
    f = best[2]
    te, byh = evaluate(f(data["test"][0], data["test"][1]), data["test"][1], data["test"][2], qs_by_split["test"])
    print(f"  {name:12s} valid MRR {best[0]:.4f} (l2={best[1]})  TEST MRR {te['MRR']:.4f} H@1 {te['H@1']:.4f}  n={te['n']}  "
          f"by horizon " + " ".join(f"{k}:{v:.3f}" for k, v in byh.items()) + f"  [{time.time() - t0:.0f}s]", flush=True)
    te["l2"] = best[1]
    return te, byh


def fit_mlp(X, M, y, wd, epochs=60, lr=3e-3, hidden=64, seed=0):
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu = X[M].mean(0); sd = X[M].std(0) + 1e-8
    norm = lambda X2, M2: torch.tensor(np.where(M2[..., None], (X2 - mu) / sd, 0.0), dtype=torch.float32, device=dev)
    Z, Mt, yt = norm(X, M), torch.tensor(M, device=dev), torch.tensor(y, device=dev)
    nn = torch.nn
    net = nn.Sequential(nn.Linear(X.shape[2], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        perm = torch.randperm(len(yt), device=dev)
        for i in range(0, len(yt), 4096):
            idx = perm[i:i + 4096]
            s = net(Z[idx]).squeeze(-1).masked_fill(~Mt[idx], -1e9)
            loss = torch.nn.functional.cross_entropy(s, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    def score(X2, M2):
        with torch.no_grad():
            s = net(norm(X2, M2)).squeeze(-1).cpu().numpy().astype(np.float64)
        return np.where(M2, s, -np.inf)
    return score


def train_eval_mlp(name, featfn, qs_by_split):
    t0 = time.time(); data = {}
    for sp, qs in qs_by_split.items():
        data[sp] = to_tensor([[featfn(q, c) for c in q[5]] for q in qs], qs)
    best = None
    for wd in (0.0, 1e-4, 1e-3):
        f = fit_mlp(*data["train"], wd)
        vr, _ = evaluate(f(data["valid"][0], data["valid"][1]), data["valid"][1], data["valid"][2], qs_by_split["valid"])
        if best is None or vr["MRR"] > best[0]:
            best = (vr["MRR"], wd, f)
    f = best[2]
    te, byh = evaluate(f(data["test"][0], data["test"][1]), data["test"][1], data["test"][2], qs_by_split["test"])
    print(f"  {name:12s} valid MRR {best[0]:.4f} (wd={best[1]})  TEST MRR {te['MRR']:.4f} H@1 {te['H@1']:.4f}  n={te['n']}  "
          f"by horizon " + " ".join(f"{k}:{v:.3f}" for k, v in byh.items()) + f"  [{time.time() - t0:.0f}s]", flush=True)
    te["wd"] = best[1]
    return te, byh


def load_quads(path):
    return [tuple(int(x) for x in l.split("\t")[:4]) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--data-dirs", nargs="+", required=True)
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    ap.add_argument("--mlp", action="store_true", help="also train physics_mlp (nonlinear raw-state reference)")
    a = ap.parse_args()
    t0 = time.time(); V = {}
    for d in iter_annotations(a.ann_dir):
        v = parse_video(d); V[str(v["vid"])] = v
    print(f"loaded {len(V)} annotated videos ({time.time() - t0:.0f}s)", flush=True)
    phys_result = None
    phys_mlp = None
    for dd in a.data_dirs:
        print(f"\n=== {dd}", flush=True)
        meta = json.load(open(os.path.join(dd, "meta.json")))
        rel2id = {l.split("\t")[0]: int(l.split("\t")[1]) for l in open(os.path.join(dd, "relation2id.txt"))}
        id2rel = {v: k for k, v in rel2id.items()}
        ent_name = {}
        for l in open(os.path.join(dd, "entity2id.txt")):
            name, i = l.rstrip("\n").split("\t")
            if name.startswith("v") and "_" in name:
                vid, rest = name[1:].split("_", 1)
                ent_name[int(i)] = (vid, int(rest[1:]) if rest.startswith("o") else None)
        facts = {sp: load_quads(os.path.join(dd, f"{sp}.txt")) for sp in ("train", "valid", "test")}
        hist = facts["train"] + load_quads(os.path.join(dd, "valid_context.txt")) + load_quads(os.path.join(dd, "test_context.txt"))
        qs = build_queries(meta, ent_name, facts, rel2id["collides_with"], a.cutoffs)
        qs_by_split = {sp: [q for q in qs if q[0] == sp] for sp in ("train", "valid", "test")}
        print("  queries:", {k: len(v) for k, v in qs_by_split.items()}, flush=True)
        idx = FactIndex(hist, ent_name, id2rel); rels_present = set(rel2id)
        sym, sym_h = train_eval("symbolic_lr", lambda q, c: symbolic_features(idx, rels_present, q, c), qs_by_split)
        if phys_result is None:
            phys_result = train_eval("physics_lr", lambda q, c: physics_features(V, q, c, meta["bin_size"]), qs_by_split)
        else:
            print(f"  physics_lr   (vocabulary-independent, reused) TEST MRR {phys_result[0]['MRR']:.4f}")
        if a.mlp and phys_mlp is None:
            phys_mlp = train_eval_mlp("physics_mlp", lambda q, c: physics_features(V, q, c, meta["bin_size"]), qs_by_split)
        bf = os.path.join(dd, "baselines_test.json")
        B = json.load(open(bf)) if os.path.exists(bf) else {}
        B.setdefault("_by_horizon", {})
        B["symbolic_lr"] = sym; B["_by_horizon"]["symbolic_lr"] = sym_h
        B["physics_lr"] = phys_result[0]; B["_by_horizon"]["physics_lr"] = phys_result[1]
        if phys_mlp is not None:
            B["physics_mlp"] = phys_mlp[0]; B["_by_horizon"]["physics_mlp"] = phys_mlp[1]
        json.dump(B, open(bf, "w"), indent=2)
        print("  merged into", bf)


if __name__ == "__main__":
    main()
