import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


def load_quadruples(file_path):
    quads = []
    if not os.path.exists(file_path):
        return np.array([], dtype=np.int64).reshape(0, 4)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    s, r, o, t = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                    quads.append([s, r, o, t])
                except ValueError:
                    continue  # skip non-integer lines
    return np.array(quads, dtype=np.int64)


def build_history_index(quads):
    """
    Returns sorted arrays for fast history lookup:
      hist_ent:  entity id for each history event
      hist_t:    timestamp
      hist_r:    relation
      hist_nb:   neighbor
      ent_start: start index in hist arrays for each entity (size num_ent+1)
    """
    num_ent = int(quads[:, [0,2]].max()) + 1

    # Build flat list: each quad contributes 2 events (s->o and o->s)
    ents = np.concatenate([quads[:,0], quads[:,2]])
    ts   = np.concatenate([quads[:,3], quads[:,3]])
    rs   = np.concatenate([quads[:,1], quads[:,1]])
    nbs  = np.concatenate([quads[:,2], quads[:,0]])

    # Sort by entity then timestamp
    order = np.lexsort((ts, ents))
    ents  = ents[order]
    ts    = ts[order]
    rs    = rs[order]
    nbs   = nbs[order]

    # Build start index per entity
    ent_start = np.zeros(num_ent + 2, dtype=np.int64)
    np.add.at(ent_start[1:], ents, 1)
    np.cumsum(ent_start, out=ent_start)

    return ts, rs, nbs, ent_start, num_ent


def precompute_histories_fast(quads, ts, rs, nbs, ent_start,
                               t_min, t_range, max_history=20):
    """
    For each quad, fetch last max_history events before query_t
    for both subject and object. Pure numpy — fast.
    """
    N = len(quads)
    L = max_history

    s_rels  = np.zeros((N, L), dtype=np.int64)
    s_nbrs  = np.zeros((N, L), dtype=np.int64)
    s_times = np.zeros((N, L), dtype=np.float32)
    s_lens  = np.zeros(N,      dtype=np.int64)

    o_rels  = np.zeros((N, L), dtype=np.int64)
    o_nbrs  = np.zeros((N, L), dtype=np.int64)
    o_times = np.zeros((N, L), dtype=np.float32)
    o_lens  = np.zeros(N,      dtype=np.int64)

    for i in range(N):
        s, r, o, t = int(quads[i,0]), int(quads[i,1]), int(quads[i,2]), int(quads[i,3])

        # Subject history
        lo, hi = (int(ent_start[s]), int(ent_start[s+1])) if s < len(ent_start)-1 else (0, 0)
        if hi > lo:
            ent_ts = ts[lo:hi]
            valid  = np.searchsorted(ent_ts, t, side='left')  # events before t
            if valid > 0:
                start  = max(0, valid - L)
                n      = valid - start
                s_rels[i,  :n] = rs[lo+start:lo+valid]
                s_nbrs[i,  :n] = nbs[lo+start:lo+valid]
                s_times[i, :n] = (ts[lo+start:lo+valid].astype(np.float32) - t_min) / t_range
                s_lens[i]      = n

        # Object history
        lo, hi = (int(ent_start[o]), int(ent_start[o+1])) if o+1 < len(ent_start) else (0, 0)
        if hi > lo:
            ent_ts = ts[lo:hi]
            valid  = np.searchsorted(ent_ts, t, side='left')
            if valid > 0:
                start  = max(0, valid - L)
                n      = valid - start
                o_rels[i,  :n] = rs[lo+start:lo+valid]
                o_nbrs[i,  :n] = nbs[lo+start:lo+valid]
                o_times[i, :n] = (ts[lo+start:lo+valid].astype(np.float32) - t_min) / t_range
                o_lens[i]      = n

    return s_rels, s_nbrs, s_times, s_lens, o_rels, o_nbrs, o_times, o_lens


class TKGDataset:
    def __init__(self, data_dir, max_history=20):
        self.data_dir    = data_dir
        self.max_history = max_history

        self.train = load_quadruples(os.path.join(data_dir, 'train.txt'))
        self.valid = load_quadruples(os.path.join(data_dir, 'valid.txt'))
        self.test  = load_quadruples(os.path.join(data_dir, 'test.txt'))

        if len(self.valid) == 0:
            n   = len(self.train)
            cut = int(n * 0.9)
            idx = np.random.permutation(n)
            self.valid = self.train[idx[cut:]]
            self.train = self.train[idx[:cut]]
            print(f"  No valid.txt — created split: train={len(self.train)}, valid={len(self.valid)}")

        all_quads = np.concatenate([self.train, self.valid, self.test], axis=0)
        self.num_entities  = int(all_quads[:, [0,2]].max()) + 1
        self.num_relations = int(all_quads[:, 1].max()) + 1

        all_times    = all_quads[:, 3].astype(np.float32)
        self.t_min   = float(all_times.min())
        self.t_max   = float(all_times.max())
        self.t_range = max(self.t_max - self.t_min, 1.0)

        # Inductive split stats
        train_ents           = set(self.train[:,0].tolist() + self.train[:,2].tolist())
        test_ents            = set(self.test[:,0].tolist()  + self.test[:,2].tolist())
        self.unseen_entities = test_ents - train_ents
        self.seen_entities   = test_ents & train_ents

        print(f"  Entities: {self.num_entities}, Relations: {self.num_relations}")
        print(f"  Timestamps: {self.t_min:.0f} to {self.t_max:.0f} "
              f"({int(len(np.unique(all_times)))} unique)")
        print(f"  Train: {len(self.train)}, Valid: {len(self.valid)}, Test: {len(self.test)}")
        print(f"  Seen test entities: {len(self.seen_entities)}, "
              f"Unseen (inductive): {len(self.unseen_entities)}")

        # Build sorted index from train only (no leakage)
        print("  Building history index...")
        self.ts, self.rs, self.nbs, self.ent_start, _ = build_history_index(self.train)

        print("  Precomputing train histories...")
        self.train_hist = precompute_histories_fast(
            self.train, self.ts, self.rs, self.nbs, self.ent_start,
            self.t_min, self.t_range, max_history)

        print("  Precomputing valid histories...")
        self.valid_hist = precompute_histories_fast(
            self.valid, self.ts, self.rs, self.nbs, self.ent_start,
            self.t_min, self.t_range, max_history)

        print("  Precomputing test histories...")
        self.test_hist  = precompute_histories_fast(
            self.test, self.ts, self.rs, self.nbs, self.ent_start,
            self.t_min, self.t_range, max_history)
        print("  Done.")


class QuadrupleDataset(Dataset):
    def __init__(self, quads, t_min, t_range, hist):
        self.quads = quads
        self.taus  = (quads[:,3].astype(np.float32) - t_min) / t_range
        (self.s_rels, self.s_nbrs, self.s_times, self.s_lens,
         self.o_rels, self.o_nbrs, self.o_times, self.o_lens) = hist

    def __len__(self):
        return len(self.quads)

    def __getitem__(self, idx):
        s, r, o, _ = self.quads[idx]
        return (int(s), int(r), int(o), float(self.taus[idx]),
                self.s_rels[idx], self.s_nbrs[idx],
                self.s_times[idx], int(self.s_lens[idx]),
                self.o_rels[idx], self.o_nbrs[idx],
                self.o_times[idx], int(self.o_lens[idx]))


def collate_fn(batch):
    (s_ids, r_ids, o_ids, taus,
     s_rels, s_nbrs, s_times, s_lens,
     o_rels, o_nbrs, o_times, o_lens) = zip(*batch)
    return (
        torch.tensor(s_ids,  dtype=torch.long),
        torch.tensor(r_ids,  dtype=torch.long),
        torch.tensor(o_ids,  dtype=torch.long),
        torch.tensor(taus,   dtype=torch.float),
        torch.tensor(np.stack(s_rels),  dtype=torch.long),
        torch.tensor(np.stack(s_nbrs),  dtype=torch.long),
        torch.tensor(np.stack(s_times), dtype=torch.float),
        torch.tensor(s_lens, dtype=torch.long),
        torch.tensor(np.stack(o_rels),  dtype=torch.long),
        torch.tensor(np.stack(o_nbrs),  dtype=torch.long),
        torch.tensor(np.stack(o_times), dtype=torch.float),
        torch.tensor(o_lens, dtype=torch.long),
    )


def get_dataloaders(tkgdata, batch_size=512, num_workers=0):
    def mk(quads, hist):
        return QuadrupleDataset(quads, tkgdata.t_min, tkgdata.t_range, hist)

    lkw = dict(batch_size=batch_size, num_workers=num_workers,
               collate_fn=collate_fn, pin_memory=False)
    return (DataLoader(mk(tkgdata.train, tkgdata.train_hist), shuffle=True,  **lkw),
            DataLoader(mk(tkgdata.valid, tkgdata.valid_hist), shuffle=False, **lkw),
            DataLoader(mk(tkgdata.test,  tkgdata.test_hist),  shuffle=False, **lkw))


def build_filter_dict_normalized(all_quads, t_min, t_range):
    filt = defaultdict(set)
    for s, r, o, t in all_quads:
        tau = round((float(t) - t_min) / t_range, 6)
        filt[(int(s), int(r), tau)].add(int(o))
    return filt
