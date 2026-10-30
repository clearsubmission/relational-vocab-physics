#!/usr/bin/env python3
"""
AAE for Physion OCP (binary): score(target, will_contact, zone) from their
observed-window histories. Same model code as clevrer_encoders.py; frozen random-ID
neighbor embeddings by default. Early stopping on valid AUROC; accuracy threshold
picked on valid. Writes aae_results_<tag>.json in the data dir.

Usage: python physion_encoders.py --data-dir tkg/holdout_Dominoes --gpu 0 --seed 42 --tag s42
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def load_facts(path):
    a = np.loadtxt(path, dtype=np.int64, ndmin=2) if os.path.getsize(path) else np.zeros((0, 4), np.int64)
    return a.reshape(-1, 4)


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
        R[i, :n] = rs[lo + st:lo + v]; NB[i, :n] = nbs[lo + st:lo + v]
        TT[i, :n] = ts[lo + st:lo + v].astype(np.float32) / t_range
        LEN[i] = n
    return R, NB, TT, LEN


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


class HistEnc(torch.nn.Module):
    """Generic history encoders on the same event inputs as the anchor encoder:
    'gru' = recurrent pass over events in time order; 'mean' = bag of events."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aae"))
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--anchors", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max-history", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--train-nb", action="store_true")
    ap.add_argument("--encoder", default="anchor", choices=["anchor", "gru", "mean"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    sys.path.insert(0, a.model_dir)
    from model import AAE
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    d = a.data_dir
    rel2id = {l.split("\t")[0]: int(l.split("\t")[1]) for l in open(os.path.join(d, "relation2id.txt"))}
    n_ent = sum(1 for _ in open(os.path.join(d, "entity2id.txt")))
    facts = np.concatenate([load_facts(os.path.join(d, f"facts_{s}.txt")) for s in ("train", "valid", "test")])
    H = build_history_index(facts)

    data = {}
    t_range = 1.0
    raw = {}
    for s in ("train", "valid", "test"):
        q = [l.rstrip("\n").split("\t") for l in open(os.path.join(d, f"queries_{s}.tsv"))]
        raw[s] = np.array([[int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4])] for x in q], np.int64)
        t_range = max(t_range, float(raw[s][:, 3].max()))
    tt = lambda x, dt=torch.long: torch.as_tensor(x, dtype=dt, device=dev)
    for s, q in raw.items():
        sh = histories(q[:, 0], q[:, 3], H, t_range, a.max_history)
        oh = histories(q[:, 2], q[:, 3], H, t_range, a.max_history)
        data[s] = dict(R=tt(q[:, 1]), O=tt(q[:, 2]), tau=tt(q[:, 3] / t_range, torch.float),
                       Y=tt(q[:, 4], torch.float), y=q[:, 4],
                       sh=[tt(sh[0]), tt(sh[1]), tt(sh[2], torch.float), tt(sh[3])],
                       oh=[tt(oh[0]), tt(oh[1]), tt(oh[2], torch.float), tt(oh[3])], n=len(q))
        print(f"  {s}: {len(q)} queries  pos {q[:, 4].mean():.2f}")

    model = AAE(n_ent, len(rel2id), a.dim, a.anchors, sigma=a.sigma, dropout=a.dropout,
                  score_func="distmult", max_history=a.max_history).to(dev)
    bias = torch.nn.Parameter(torch.zeros(1, device=dev))
    if a.encoder != "anchor":
        model.hist_enc = HistEnc(a.dim, a.encoder).to(dev)
        model.entity_embed = lambda *x, _m=model: _m.hist_enc(_m, *x)
    if not a.train_nb:
        model.neighbor_embeddings.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad] + [bias]
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=a.wd)

    def logits(D, idx):
        tau = D["tau"][idx]
        s_emb = model.entity_embed(*[x[idx] for x in D["sh"]], tau)
        o_emb = model.entity_embed(*[x[idx] for x in D["oh"]], tau)
        q = s_emb * model.relation_embeddings[D["R"][idx]]
        return (q * o_emb).sum(-1) + (q * model.layer_norm(model.neighbor_embeddings[D["O"][idx]])).sum(-1) + bias

    @torch.no_grad()
    def predict(D):
        model.eval()
        out = [logits(D, torch.arange(i, min(i + 1024, D["n"]), device=dev)).cpu().numpy()
               for i in range(0, D["n"], 1024)]
        return np.concatenate(out) if out else np.zeros(0)

    Dtr = data["train"]
    pw = torch.tensor((1 - Dtr["y"].mean()) / max(Dtr["y"].mean(), 1e-6), device=dev)
    best, best_state, bad = -1.0, None, 0
    for ep in range(1, a.epochs + 1):
        model.train(); t1 = time.time(); tot = 0.0
        perm = torch.randperm(Dtr["n"], device=dev)
        for i in range(0, Dtr["n"], a.batch):
            idx = perm[i:i + a.batch]
            loss = F.binary_cross_entropy_with_logits(logits(Dtr, idx), Dtr["Y"][idx], pos_weight=pw)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += loss.item() * len(idx)
        va = auroc(data["valid"]["y"], predict(data["valid"]))
        tag = ""
        if va > best:
            best, best_state, bad, tag = va, (copy.deepcopy(model.state_dict()), bias.detach().clone()), 0, " *"
        else:
            bad += 1
        print(f"epoch {ep:3d} loss {tot / Dtr['n']:.4f}  val AUROC {va:.4f}  {time.time() - t1:.0f}s{tag}", flush=True)
        if bad >= a.patience:
            print("early stop"); break

    model.load_state_dict(best_state[0]); bias.data = best_state[1]
    sv, st_ = predict(data["valid"]), predict(data["test"])
    cand = np.unique(sv)
    thr = float(cand[int(np.argmax([np.mean((sv >= c) == data["valid"]["y"]) for c in cand]))])
    yt = data["test"]["y"]
    res = {"acc": float(np.mean((st_ >= thr) == yt)), "auroc": auroc(yt, st_), "acc_at_0": float(np.mean((st_ >= 0) == yt)),
           "best_val_auroc": best, "n_test": int(len(yt)), "args": vars(a),
           "scores": st_.tolist(), "labels": yt.tolist()}
    print(f"\nTEST  acc {res['acc']:.4f}  AUROC {res['auroc']:.4f}  (n={len(yt)}, pos {yt.mean():.2f})")
    out = os.path.join(d, f"aae_results{('_' + a.tag) if a.tag else ''}.json")
    json.dump(res, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
