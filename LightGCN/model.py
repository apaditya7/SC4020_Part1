from typing import Tuple
import torch
from torch import nn


class LightGCN(nn.Module):
    """
    Minimal LightGCN with optional MF-like behavior when K == 0.
    - n_users, n_items: counts
    - graph: normalized sparse adjacency (torch.sparse.FloatTensor) of size (n_users+n_items, n_users+n_items)
    - embed_dim: embedding dimension
    - K: number of propagation steps (K=0 → MF-like)
    - node_dropout_p, edge_dropout_p: optional regularization
    """

    def __init__(self,
                 n_users: int,
                 n_items: int,
                 graph: torch.Tensor,
                 embed_dim: int = 64,
                 K: int = 3,
                 node_dropout_p: float = 0.0,
                 edge_dropout_p: float = 0.0):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.K = int(K)
        self.node_dropout_p = float(node_dropout_p)
        self.edge_dropout_p = float(edge_dropout_p)

        self.embedding_user = nn.Embedding(n_users, embed_dim)
        self.embedding_item = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.Graph = graph  # expected normalized adjacency
        self._sigmoid = nn.Sigmoid()

    def _edge_dropout(self, g: torch.Tensor, p: float) -> torch.Tensor:
        if p <= 0.0 or (not self.training):
            return g
        idx = g.indices().t()
        val = g.values()
        keep = (torch.rand(val.shape, device=val.device) >= p)
        if keep.sum() == 0:
            return g
        idx = idx[keep]
        val = val[keep] / (1.0 - p)
        return torch.sparse_coo_tensor(idx.t(), val, size=g.shape, device=g.device).coalesce()

    def _node_dropout(self, x: torch.Tensor, p: float) -> torch.Tensor:
        if p <= 0.0 or (not self.training):
            return x
        mask = (torch.rand_like(x[:, :1]) >= p).float()  # [N,1]
        return x * mask

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        users_ego = self.embedding_user.weight
        items_ego = self.embedding_item.weight
        all_ego = torch.cat([users_ego, items_ego], dim=0)  # [N, d]

        if self.K == 0:
            # MF-like: no propagation, just return ego embeddings
            return users_ego, items_ego

        g = self._edge_dropout(self.Graph, self.edge_dropout_p)
        embs = [all_ego]
        h = all_ego
        for _ in range(self.K):
            h = torch.sparse.mm(g, h)
            h = self._node_dropout(h, self.node_dropout_p)
            embs.append(h)
        out = torch.mean(torch.stack(embs, dim=1), dim=1)
        users, items = torch.split(out, [self.n_users, self.n_items], dim=0)
        return users, items

    def getUsersRating(self, users: torch.Tensor) -> torch.Tensor:
        all_users, all_items = self.propagate()
        users_emb = all_users[users.long()]  # [B, d]
        items_emb = all_items  # [I, d]
        scores = torch.matmul(users_emb, items_emb.t())  # [B, I]
        return self._sigmoid(scores)

    def getEmbedding(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor):
        all_users, all_items = self.propagate()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_ego = self.embedding_user(users)
        pos_ego = self.embedding_item(pos_items)
        neg_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_ego, pos_ego, neg_ego

    def bpr_loss(self, users: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, weight_decay: float = 0.0):
        users = users.long(); pos = pos.long(); neg = neg.long()
        u, i, j, u0, i0, j0 = self.getEmbedding(users, pos, neg)
        pos_scores = (u * i).sum(dim=1)
        neg_scores = (u * j).sum(dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        reg = 0.5 * (u0.norm(2).pow(2) + i0.norm(2).pow(2) + j0.norm(2).pow(2)) / float(len(users))
        return loss + weight_decay * reg, reg.detach()

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        all_users, all_items = self.propagate()
        u = all_users[users.long()]
        v = all_items[items.long()]
        return (u * v).sum(dim=1)

