#!/usr/bin/env python3
"""
AAE on PhysT-KG collision forecasting, evaluated exactly like baselines.py.
  * History index = train + valid_context + test_context (test objects see their observed history).
  * Trains on collides_with queries only; all facts remain history.
  * Candidates = other objects in the same video (filtered); softmax CE over them.
  * neighbor_embeddings frozen at random init (random-ID features); --train-nb to train them.
  * Score = <s*r, inductive(c)> + <s*r, LN(id(c))>   (--score ind|nb|both)
"""
import argparse
import copy
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def load(path):
    rows = [l.split("\t") for l in open(path) if l.strip()]
    return np.array([[int(x) for x in r[:4]] for r in rows], dtype=np.int64).reshape(-1, 4)


def build_history_index(quads):
    num_ent = int(quads[:, [0, 2]].max()) + 1
    ents = np.concatenate([quads[:, 0], quads[:, 2]])
    ts = np.concatenate([quads[:, 3], quads[:, 3]])
    rs = np.concatenate([quads[:, 1], quads[:, 1]])
    nbs = np.concatenate([quads[:, 2], quads[:, 0]])
    order = np.lexsort((ts, ents))
    ents, ts, rs, nbs = ents[order], ts[order], rs[order], nbs[order]
    ent_start = np.zeros(num_ent + 2, dtype=np.int64)
    np.add.at(ent_start[1:], ents, 1)
    np.cumsum(ent_start, out=ent_start)
    return ts, rs, nbs, ent_start


def histories(ents, qts, H, t_range, L):
    ts, rs, nbs, ent_start = H
    N = len(ents)
    R = np.zeros((N, L), np.int64); NB = np.zeros((N, L), np.int64)
    TT = np.zeros((N, L), np.float32); LEN = np.zeros(N, np.int64)
    for i in range(N):
        e, t = int(ents[i]), int(qts[i])
        if e < 0 or e + 1 >= len(ent_start):
            continue
        lo, hi = int(ent_start[e]), int(ent_start[e + 1])
        if hi <= lo:
            continue
        v = np.searchsorted(ts[lo:hi], t, side="left")
        if v == 0:
            continue
        st = max(0, v - L); n = v - st
        R[i, :n] = rs[lo + st:lo + v]
        NB[i, :n] = nbs[lo + st:lo + v]
        TT[i, :n] = ts[lo + st:lo + v].astype(np.float32) / t_range
        LEN[i] = n
    return R, NB, TT, LEN


def build_split(queries, ent2vid, vobj, cmax, H, t_range, L, meta, device, cuts=None):
    """History is read strictly before HT. valid/test: HT = video cutoff (as in baselines).
    train: HT = each cutoff in `cuts` with t >= HT (forecasting, like test); cuts=[] -> HT = t."""
    truth = defaultdict(set)
    for s, _, o, t in queries:
        truth[(s, t)].add(o)
    S, Rq, T, HT, C, M, TGT, HZ = [], [], [], [], [], [], [], []
    for s, r, o, t in queries:
        vid = ent2vid.get(int(s))
        if vid is None:
            continue
        info = meta["videos"][vid]
        if info["split"] != "train":
            hts = [info["cutoff_bin"]]
        elif cuts:
            hts = [c for c in cuts if t >= c]
        else:
            hts = [t]
        cands = [c for c in vobj[vid] if c != s and (c == o or c not in truth[(s, t)])]
        row = cands + [-1] * (cmax - len(cands))
        for ht in hts:
            S.append(s); Rq.append(r); T.append(t); HT.append(ht); C.append(row)
            M.append([c >= 0 for c in row]); TGT.append(cands.index(o)); HZ.append(t - ht)
    S, Rq, T, HT = np.array(S), np.array(Rq), np.array(T), np.array(HT)
    C, M = np.array(C), np.array(M)
    sh = histories(S, HT, H, t_range, L)
    ch = histories(C.reshape(-1), np.repeat(HT, cmax), H, t_range, L)
    Q = len(S)
    ch = (ch[0].reshape(Q, cmax, L), ch[1].reshape(Q, cmax, L), ch[2].reshape(Q, cmax, L), ch[3].reshape(Q, cmax))
    tt = lambda x, dt=torch.long: torch.as_tensor(x, dtype=dt, device=device)
    return dict(S=tt(S), R=tt(Rq), tau=tt(T / t_range, torch.float), C=tt(np.maximum(C, 0)), M=tt(M, torch.bool),
                TGT=tt(TGT), HZ=np.array(HZ),
                sh=[tt(sh[0]), tt(sh[1]), tt(sh[2], torch.float), tt(sh[3])],
                ch=[tt(ch[0]), tt(ch[1]), tt(ch[2], torch.float), tt(ch[3])], n=Q)


class HistEnc(torch.nn.Module):
    """Generic history encoders for comparison, on the same event inputs as the anchor encoder:
    'gru' = recurrent pass over the events in time order; 'mean' = bag of events (no attention, no anchors)."""
    def __init__(self, d, kind):
        super().__init__()
        nn = torch.nn
        self.kind = kind
        self.rel_proj, self.nbr_proj, self.time_proj = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(1, d)
        if kind == "gru":
            self.rnn = nn.GRU(3 * d, d, batch_first=True)
        else:
            self.mlp = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.cold = nn.Parameter(torch.randn(d) * 0.02)
        self.norm = nn.LayerNorm(d)

    def forward(self, model, rel_ids, nbr_ids, timestamps, lengths, tau):
        B, L = rel_ids.shape
        x = torch.cat([torch.relu(self.rel_proj(model.relation_embeddings[rel_ids])),
                       torch.relu(self.nbr_proj(model.neighbor_embeddings[nbr_ids])),
                       torch.tanh(self.time_proj((tau.unsqueeze(1) - timestamps).unsqueeze(-1)))], -1)
        mask = (torch.arange(L, device=rel_ids.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        if self.kind == "gru":
            packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.clamp(min=1).cpu(), batch_first=True,
                                                             enforce_sorted=False)
            _, h = self.rnn(packed)
            out = h[-1]
        else:
            out = self.mlp((x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1))
        has = (lengths > 0).float().unsqueeze(-1)
        out = has * out + (1 - has) * self.cold
        return model.layer_norm(model.dropout(self.norm(out)))


def score(model, D, idx, mode):
    B = len(idx); C = D["C"].shape[1]; L = D["sh"][0].shape[1]
    if mode == "distmult":   # transductive KG embedding: one learned vector per entity, no history
        s_vec = model.layer_norm(model.neighbor_embeddings[D["S"][idx]])
        r_emb = model.relation_embeddings[D["R"][idx]]
        sc = ((s_vec * r_emb).unsqueeze(1) * model.layer_norm(model.neighbor_embeddings[D["C"][idx]])).sum(-1)
        return sc.masked_fill(~D["M"][idx], float("-inf"))
    tau = D["tau"][idx]
    s_emb = model.entity_embed(*[x[idx] for x in D["sh"]], tau)
    r_emb = model.relation_embeddings[D["R"][idx]]
    q = (s_emb * r_emb).unsqueeze(1)
    sc = 0
    if mode in ("ind", "both"):
        cr, cn, ct, cl = [x[idx] for x in D["ch"]]
        c_emb = model.entity_embed(cr.view(B * C, L), cn.view(B * C, L), ct.view(B * C, L),
                                   cl.view(B * C), tau.repeat_interleave(C)).view(B, C, -1)
        sc = sc + (q * c_emb).sum(-1)
    if mode in ("nb", "both"):
        sc = sc + (q * model.layer_norm(model.neighbor_embeddings[D["C"][idx]])).sum(-1)
    return sc.masked_fill(~D["M"][idx], float("-inf"))


def shuffle_hist(hist, seed):
    """Randomly permute the valid history slots of every row (events keep their own timestamps)."""
    R, NB, TT, LEN = hist
    g = torch.Generator().manual_seed(seed)
    keys = torch.rand(tuple(R.shape), generator=g).to(R.device)
    pos = torch.arange(R.shape[-1], device=R.device).expand(R.shape)
    order = keys.masked_fill(pos >= LEN.unsqueeze(-1), 2.0).argsort(-1)
    return [R.gather(-1, order), NB.gather(-1, order), TT.gather(-1, order), LEN]


def shuffled_copy(D, seed):
    E = dict(D); E["sh"] = shuffle_hist(D["sh"], seed); E["ch"] = shuffle_hist(D["ch"], seed + 1)
    return E


@torch.no_grad()
def evaluate(model, D, mode, bs=1024):
    model.eval()
    rr = []
    for i in range(0, D["n"], bs):
        idx = torch.arange(i, min(i + bs, D["n"]), device=D["S"].device)
        sc = score(model, D, idx, mode).cpu().numpy()
        m = D["M"][idx].cpu().numpy(); tg = D["TGT"][idx].cpu().numpy()
        for k in range(len(idx)):
            so = sc[k, tg[k]]
            oth = sc[k][m[k]]; oth = np.delete(oth, np.where(np.where(m[k])[0] == tg[k])[0])
            lo = 1 + int((oth > so).sum()); ties = int((oth == so).sum())
            rr.append(np.arange(lo, lo + ties + 1))
    mrr = np.array([np.mean(1 / x) for x in rr])
    res = {"MRR": float(mrr.mean()),
           "H@1": float(np.mean([np.mean(x <= 1) for x in rr])),
           "H@3": float(np.mean([np.mean(x <= 3) for x in rr])), "n": len(rr)}
    return res, mrr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aae"))
    ap.add_argument("--query-rel", default="collides_with")
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--anchors", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max-history", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--score", default="both", choices=["ind", "nb", "both", "distmult"])
    ap.add_argument("--encoder", default="anchor", choices=["anchor", "gru", "mean"])
    ap.add_argument("--no-ids", action="store_true", help="zero the entity identifiers (identifier-free ablation)")
    ap.add_argument("--shuffle-hist", action="store_true", help="shuffle event order in every history (train/valid/test)")
    ap.add_argument("--train-nb", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    ap.add_argument("--train-cutoffs", type=int, nargs="*", default=[4, 6, 8, 10, 12],
                    help="history cutoffs (time steps) for training queries; pass with no values for old behaviour")
    a = ap.parse_args()

    sys.path.insert(0, a.model_dir)
    from model import AAE

    np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    meta = json.load(open(os.path.join(a.data_dir, "meta.json")))
    rel2id = {l.split("\t")[0]: int(l.split("\t")[1]) for l in open(os.path.join(a.data_dir, "relation2id.txt"))}
    n_ent = sum(1 for _ in open(os.path.join(a.data_dir, "entity2id.txt")))
    qr = rel2id[a.query_rel]
    ent2vid, vobj = {}, {}
    for vid, info in meta["videos"].items():
        vobj[vid] = [int(e) for e in info["objects"]]
        for e in vobj[vid]:
            ent2vid[e] = vid
    cmax = max(len(v) for v in vobj.values()) - 1

    train = load(os.path.join(a.data_dir, "train.txt"))
    hist_quads = np.concatenate([train] + [load(os.path.join(a.data_dir, f"{s}_context.txt")) for s in ("valid", "test")])
    t_range = float(max(hist_quads[:, 3].max(), 1))
    for s in ("valid", "test"):
        t_range = max(t_range, float(load(os.path.join(a.data_dir, f"{s}.txt"))[:, 3].max()))
    H = build_history_index(hist_quads)

    data = {}
    for split, q in [("train", train[train[:, 1] == qr]),
                     ("valid", load(os.path.join(a.data_dir, "valid.txt"))),
                     ("test", load(os.path.join(a.data_dir, "test.txt")))]:
        q = q[q[:, 1] == qr]
        data[split] = build_split(q, ent2vid, vobj, cmax, H, t_range, a.max_history, meta, dev, a.train_cutoffs)
        print(f"  {split}: {data[split]['n']} queries")
    if a.shuffle_hist:
        for k in list(data):
            data[k] = shuffled_copy(data[k], 7 + 101 * len(k))
    print(f"entities {n_ent}  relations {len(rel2id)}  max candidates {cmax}  prep {time.time() - t0:.0f}s  device {dev}")

    model = AAE(n_ent, len(rel2id), a.dim, a.anchors, sigma=a.sigma, dropout=a.dropout,
                  score_func="distmult", max_history=a.max_history).to(dev)
    if a.no_ids:
        model.neighbor_embeddings.data.zero_()
        model.neighbor_embeddings.requires_grad_(False)
    if a.encoder != "anchor":
        model.hist_enc = HistEnc(a.dim, a.encoder).to(dev)
        model.entity_embed = lambda *x, _m=model: _m.hist_enc(_m, *x)
    if not a.train_nb and a.score != "distmult":
        model.neighbor_embeddings.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=a.wd)

    best, best_state, bad = -1.0, None, 0
    Dtr = data["train"]
    for ep in range(1, a.epochs + 1):
        model.train(); t1 = time.time(); tot = 0.0
        perm = torch.randperm(Dtr["n"], device=dev)
        for i in range(0, Dtr["n"], a.batch):
            idx = perm[i:i + a.batch]
            loss = F.cross_entropy(score(model, Dtr, idx, a.score), Dtr["TGT"][idx])
            if model.K > 1:
                loss = loss + 1e-3 * torch.clamp(1.0 - torch.pdist(model.anchor_times.unsqueeze(1)).mean(), min=0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += loss.item() * len(idx)
        vr, _ = evaluate(model, data["valid"], a.score)
        tag = ""
        if vr["MRR"] > best:
            best, best_state, bad, tag = vr["MRR"], copy.deepcopy(model.state_dict()), 0, " *"
        else:
            bad += 1
        print(f"epoch {ep:3d} loss {tot / Dtr['n']:.4f}  val MRR {vr['MRR']:.4f} H@1 {vr['H@1']:.4f}  {time.time() - t1:.0f}s{tag}")
        if bad >= a.patience:
            print("early stop"); break

    if a.shuffle_hist:
        print("  (histories shuffled in all splits)")
    model.load_state_dict(best_state)
    te, mrr = evaluate(model, data["test"], a.score)
    print(f"\nTEST  MRR {te['MRR']:.4f}  H@1 {te['H@1']:.4f}  H@3 {te['H@3']:.4f}  (n={te['n']})")
    hz = data["test"]["HZ"]
    buckets = [(0, 1), (2, 3), (4, 5), (6, 99)]
    labels = [f"{lo}-{hi}" if hi < 99 else f"{lo}+" for lo, hi in buckets]
    by_h = {l: float(mrr[(hz >= lo) & (hz <= hi)].mean()) for l, (lo, hi) in zip(labels, buckets)
            if ((hz >= lo) & (hz <= hi)).any()}
    print("MRR by horizon: " + "  ".join(f"{k}: {v:.4f}" for k, v in by_h.items()))
    te["_by_horizon"] = by_h; te["best_val_MRR"] = best; te["args"] = vars(a)
    Dte = data["test"]; ar = torch.arange(Dte["n"], device=Dte["S"].device)
    te["rr"] = [float(x) for x in mrr]
    te["keys"] = [[int(s), int(o), int(round(float(tau) * t_range))] for s, o, tau in
                  zip(Dte["S"].tolist(), Dte["C"][ar, Dte["TGT"]].tolist(), Dte["tau"].tolist())]
    far = hz >= 6
    _, rr_sh = evaluate(model, shuffled_copy(Dte, 1234), a.score)
    te["MRR_test_shuffled"] = float(rr_sh.mean()); te["MRR6_test_shuffled"] = float(rr_sh[far].mean())
    nb0 = model.neighbor_embeddings.data.clone()
    pm = []
    for k in range(3):
        g = torch.Generator().manual_seed(100 + k)
        model.neighbor_embeddings.data = nb0[torch.randperm(nb0.shape[0], generator=g).to(nb0.device)]
        pm.append(float(evaluate(model, Dte, a.score)[1].mean()))
    model.neighbor_embeddings.data = nb0
    te["MRR_test_permids"] = pm
    print(f"ORDER/ID TESTS  test MRR {te['MRR']:.4f} | shuffled-history test {te['MRR_test_shuffled']:.4f} "
          f"(6+: {float(mrr[far].mean()):.4f} -> {te['MRR6_test_shuffled']:.4f}) | permuted IDs {' '.join(f'{x:.4f}' for x in pm)}")
    out = os.path.join(a.data_dir, f"aae_results{('_' + a.tag) if a.tag else ''}.json")
    json.dump(te, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
