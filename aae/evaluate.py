import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, data_loader, filter_dict, num_entities,
             device, split='test', unseen_entities=None):
    model.eval()
    ranks, ranks_seen, ranks_unseen = [], [], []

    # AAE uses neighbor_embeddings (no entity_table)
    all_ent_embs = model.layer_norm(
        model.neighbor_embeddings).detach()               # (N, d)

    for batch in tqdm(data_loader, desc=f'[{split}]', leave=False):
        (s_ids, r_ids, o_ids, taus,
         s_rels, s_nbrs, s_times, s_lens,
         o_rels, o_nbrs, o_times, o_lens) = [x.to(device) for x in batch]

        B = r_ids.size(0)

        s_embs = model.entity_embed(
            s_rels, s_nbrs, s_times, s_lens, taus)        # no entity_ids
        r_embs = model.relation_embeddings[r_ids]
        sr     = s_embs * r_embs                          # (B, d)

        scores_all = sr @ all_ent_embs.t()                # (B, N)

        # Unseen: use inductive encoder for true object
        o_ind        = model.inductive_embed(
            o_rels, o_nbrs, o_times, o_lens, taus)
        o_ind        = model.layer_norm(o_ind)
        o_ind_scores = (sr * o_ind).sum(dim=-1)           # (B,)

        for i in range(B):
            s   = s_ids[i].item()
            r   = r_ids[i].item()
            o   = o_ids[i].item()
            tau = round(taus[i].item(), 6)
            is_unseen = (unseen_entities is not None and o in unseen_entities)

            scores = scores_all[i].clone()
            if is_unseen:
                scores[o] = o_ind_scores[i].item()

            true_objs = filter_dict.get((s, r, tau), set())
            mask_ids  = list(true_objs - {o})
            if mask_ids:
                scores[torch.tensor(mask_ids, device=device)] = float('-inf')

            rank = int((scores >= scores[o]).sum().item())
            ranks.append(rank)
            if unseen_entities is not None:
                (ranks_unseen if is_unseen else ranks_seen).append(rank)

    def metrics(r):
        if not r:
            return {'mrr': 0.0, 'hits1': 0.0, 'hits3': 0.0, 'hits10': 0.0}
        r = np.array(r, dtype=np.float32)
        return {'mrr':    float(np.mean(1.0 / r)),
                'hits1':  float(np.mean(r <= 1)),
                'hits3':  float(np.mean(r <= 3)),
                'hits10': float(np.mean(r <= 10))}

    results = metrics(ranks)
    if unseen_entities is not None:
        results['seen']   = metrics(ranks_seen)
        results['unseen'] = metrics(ranks_unseen)
    return results


def print_results(results, split='Test'):
    print(f"\n{'='*50}")
    print(f"  {split} Results")
    print(f"{'='*50}")
    print(f"  MRR:     {results['mrr']:.4f}")
    print(f"  Hits@1:  {results['hits1']:.4f}")
    print(f"  Hits@3:  {results['hits3']:.4f}")
    print(f"  Hits@10: {results['hits10']:.4f}")
    if 'seen' in results:
        s = results['seen']
        u = results['unseen']
        print(f"\n  --- Seen entities ---")
        print(f"  MRR: {s['mrr']:.4f}  H@1: {s['hits1']:.4f}  H@10: {s['hits10']:.4f}")
        print(f"\n  --- Unseen entities (inductive) ---")
        print(f"  MRR: {u['mrr']:.4f}  H@1: {u['hits1']:.4f}  H@10: {u['hits10']:.4f}")
    print(f"{'='*50}\n")
