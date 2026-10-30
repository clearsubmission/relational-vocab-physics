"""
AAE: Knowledge Anchors for Robust Memory-Augmented Temporal Representation Learning
model.py - Base inductive version (no transductive component)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalNeighborhoodEncoder(nn.Module):
    def __init__(self, embedding_dim, num_anchors):
        super().__init__()
        self.d = embedding_dim
        self.K = num_anchors

        self.rel_proj  = nn.Linear(embedding_dim, embedding_dim)
        self.time_proj = nn.Linear(1, embedding_dim)
        self.nbr_proj  = nn.Linear(embedding_dim, embedding_dim)
        self.attn_q    = nn.Linear(embedding_dim, embedding_dim)
        self.attn_k    = nn.Linear(embedding_dim * 3, embedding_dim)
        self.attn_v    = nn.Linear(embedding_dim * 3, embedding_dim)
        self.out_proj  = nn.Linear(embedding_dim, num_anchors * embedding_dim)
        self.norm      = nn.LayerNorm(embedding_dim)

    def forward(self, rel_embs, nbr_embs, time_deltas, mask):
        B, L, d = rel_embs.shape
        t_enc  = torch.tanh(self.time_proj(time_deltas))
        r_enc  = torch.relu(self.rel_proj(rel_embs))
        n_enc  = torch.relu(self.nbr_proj(nbr_embs))
        events = torch.cat([r_enc, n_enc, t_enc], dim=-1)

        denom  = mask.sum(dim=1, keepdim=True).clamp(min=1)
        query  = (events * mask.unsqueeze(-1)).sum(dim=1) / denom
        Q = self.attn_q(query[:, :d]).unsqueeze(1)
        K = self.attn_k(events)
        V = self.attn_v(events)

        attn = (Q * K).sum(dim=-1) / (d ** 0.5)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = (attn.unsqueeze(-1) * V).sum(dim=1)
        out = self.norm(out)
        return self.out_proj(out).view(B, self.K, d)


class AAE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, num_anchors,
                 sigma=1.0, dropout=0.2, score_func='distmult', max_history=20):
        super().__init__()
        self.num_entities  = num_entities
        self.num_relations = num_relations
        self.d             = embedding_dim
        self.K             = num_anchors
        self.sigma         = sigma
        self.score_func    = score_func
        self.max_history   = max_history

        self.anchor_times = nn.Parameter(torch.randn(num_anchors))

        self.relation_embeddings = nn.Parameter(
            torch.empty(num_relations, embedding_dim))
        nn.init.xavier_uniform_(self.relation_embeddings)

        self.neighbor_embeddings = nn.Parameter(
            torch.empty(num_entities, embedding_dim))
        nn.init.xavier_uniform_(self.neighbor_embeddings)

        self.encoder = TemporalNeighborhoodEncoder(embedding_dim, num_anchors)

        self.cold_start = nn.Parameter(
            torch.empty(num_anchors, embedding_dim))
        nn.init.xavier_uniform_(self.cold_start)

        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def anchor_attention(self, tau):
        diff  = tau.unsqueeze(1) - self.anchor_times.unsqueeze(0)
        return F.softmax(-diff.pow(2) / (self.sigma + 1e-8), dim=-1)

    def inductive_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau):
        B, L = rel_ids.shape
        dev  = rel_ids.device

        rel_embs    = self.relation_embeddings[rel_ids]
        nbr_embs    = self.neighbor_embeddings[nbr_ids]
        time_deltas = (tau.unsqueeze(1) - timestamps).unsqueeze(-1)

        mask = (torch.arange(L, device=dev).unsqueeze(0)
                < lengths.unsqueeze(1)).float()

        has_history = (lengths > 0).float().unsqueeze(-1)

        anchor_reps = self.encoder(rel_embs, nbr_embs, time_deltas, mask)
        alpha       = self.anchor_attention(tau)
        encoded     = (alpha.unsqueeze(-1) * anchor_reps).sum(dim=1)

        cold = (self.cold_start.unsqueeze(0) *
                alpha.unsqueeze(-1)).sum(dim=1)

        return has_history * encoded + (1.0 - has_history) * cold

    def entity_embed(self, rel_ids, nbr_ids, timestamps, lengths, tau,
                     entity_ids=None):
        ind_emb = self.inductive_embed(
            rel_ids, nbr_ids, timestamps, lengths, tau)
        return self.layer_norm(self.dropout(ind_emb))
