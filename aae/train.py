import os, argparse, json, time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import AAE
from data_loader import TKGDataset, get_dataloaders, build_filter_dict_normalized
from evaluate import evaluate, print_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',       type=str,   required=True)
    p.add_argument('--data_dir',      type=str,   default='<DATA_DIR>')
    p.add_argument('--save_dir',      type=str,   default='<DATA_DIR>')
    p.add_argument('--embedding_dim', type=int,   default=200)
    p.add_argument('--num_anchors',   type=int,   default=16)
    p.add_argument('--sigma',         type=float, default=0.5)
    p.add_argument('--score_func',    type=str,   default='distmult')
    p.add_argument('--dropout',       type=float, default=0.2)
    p.add_argument('--lambda2',       type=float, default=0.001)
    p.add_argument('--num_neg',       type=int,   default=64)
    p.add_argument('--epochs',        type=int,   default=500)
    p.add_argument('--batch_size',    type=int,   default=512)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=1e-5)
    p.add_argument('--patience',      type=int,   default=20)
    p.add_argument('--eval_every',    type=int,   default=5)
    p.add_argument('--num_workers',   type=int,   default=0)
    p.add_argument('--max_history',   type=int,   default=20)
    p.add_argument('--gpu',           type=int,   default=0)
    p.add_argument('--seed',          type=int,   default=42)
    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"  AAE (Inductive)  |  Dataset: {args.dataset}  |  Device: {device}")
    print(f"{'='*60}")

    data_path = os.path.join(args.data_dir, args.dataset)
    tkgdata   = TKGDataset(data_path)

    train_loader, valid_loader, test_loader = get_dataloaders(
        tkgdata, batch_size=args.batch_size, num_workers=args.num_workers)

    all_quads   = np.concatenate([tkgdata.train, tkgdata.valid, tkgdata.test], axis=0)
    filter_dict = build_filter_dict_normalized(
        all_quads, tkgdata.t_min, tkgdata.t_range)

    model = AAE(
        num_entities  = tkgdata.num_entities,
        num_relations = tkgdata.num_relations,
        embedding_dim = args.embedding_dim,
        num_anchors   = args.num_anchors,
        sigma         = args.sigma,
        dropout       = args.dropout,
        score_func    = args.score_func,
        max_history   = args.max_history,
    ).to(device)

    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path    = os.path.join(args.save_dir, f'aaebase_{args.dataset}_best.pt')
    results_path = os.path.join(args.save_dir, f'aaebase_{args.dataset}_results.json')

    best_mrr, patience_counter = 0.0, 0
    N = tkgdata.num_entities

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, t0 = 0.0, time.time()

        for batch in train_loader:
            (s_ids, r_ids, o_ids, taus,
             s_rels, s_nbrs, s_times, s_lens,
             o_rels, o_nbrs, o_times, o_lens) = [x.to(device) for x in batch]

            optimizer.zero_grad()
            B = r_ids.size(0)

            # Pure inductive embeddings — no entity_table
            s_embs = model.entity_embed(s_rels, s_nbrs, s_times, s_lens, taus)
            o_embs = model.entity_embed(o_rels, o_nbrs, o_times, o_lens, taus)
            r_embs = model.relation_embeddings[r_ids]

            pos_scores = (s_embs * r_embs * o_embs).sum(dim=-1)

            # Negative: random neighbor embeddings
            neg_ids    = torch.randint(0, N, (B, args.num_neg), device=device)
            neg_embs   = model.layer_norm(model.neighbor_embeddings[neg_ids])
            sr_exp     = (s_embs * r_embs).unsqueeze(1)
            neg_scores = (sr_exp * neg_embs).sum(dim=-1)

            # Positive via neighbor_embeddings
            pos_nb    = model.layer_norm(model.neighbor_embeddings[o_ids])
            pos_nb_sc = (s_embs * r_embs * pos_nb).sum(dim=-1)

            # BCE loss
            loss_pos = torch.nn.functional.binary_cross_entropy_with_logits(
                pos_nb_sc, torch.ones(B, device=device))
            loss_neg = torch.nn.functional.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros(B, args.num_neg, device=device))
            loss = loss_pos + loss_neg

            # Anchor spread regularization
            if model.K > 1:
                spread = torch.pdist(
                    model.anchor_times.unsqueeze(1), p=2).mean()
                loss = loss + args.lambda2 * torch.clamp(1.0 - spread, min=0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        print(f"Epoch {epoch:4d}/{args.epochs} | Loss: {avg_loss:.4f} | {elapsed:.1f}s")

        if epoch % args.eval_every == 0:
            vr = evaluate(model, valid_loader, filter_dict,
                          tkgdata.num_entities, device, 'valid')
            scheduler.step(vr['mrr'])
            tag = ''
            if vr['mrr'] > best_mrr:
                best_mrr, patience_counter = vr['mrr'], 0
                torch.save({'epoch': epoch,
                            'model_state': model.state_dict(),
                            'args': vars(args),
                            'valid_results': vr}, ckpt_path)
                tag = '  *** saved ***'
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            print(f"  Val MRR:{vr['mrr']:.4f} H@1:{vr['hits1']:.4f} "
                  f"H@3:{vr['hits3']:.4f} H@10:{vr['hits10']:.4f}{tag}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    test_results = evaluate(
        model, test_loader, filter_dict,
        tkgdata.num_entities, device, 'test',
        unseen_entities=tkgdata.unseen_entities,
    )
    print_results(test_results, split=f'Test [{args.dataset}] AAE')

    with open(results_path, 'w') as f:
        json.dump({
            'dataset':      args.dataset,
            'model':        'AAE',
            'best_epoch':   ckpt['epoch'],
            'valid':        ckpt['valid_results'],
            'test':         test_results,
            'args':         vars(args),
            'unseen_count': len(tkgdata.unseen_entities),
            'seen_count':   len(tkgdata.seen_entities),
        }, f, indent=2)
    print(f"Saved: {results_path}")


if __name__ == '__main__':
    main()
