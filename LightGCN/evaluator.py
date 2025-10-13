from typing import Dict, Iterable, Tuple
import numpy as np
import torch
from metrics import recall_at_k, ndcg_at_k, hit_rate_at_k


@torch.no_grad()
def rank_and_metrics(model,
                     test_dict: Dict[int, Iterable[int]],
                     train_dict: Dict[int, Iterable[int]],
                     n_items: int,
                     ks: Tuple[int, ...] = (10, 20),
                     batch_size: int = 1024,
                     device: str = "cpu"):
    """
    Computes top-K ranking per user and returns metrics at each K.

    - Masks training interactions so they are not recommended.
    - test_dict/train_dict map user -> iterable of item ids.
    """
    model_device = next(model.parameters()).device
    all_users = list(test_dict.keys())
    B = len(all_users)

    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    hits = {k: [] for k in ks}

    for start in range(0, B, batch_size):
        u_batch = all_users[start:start + batch_size]
        users_t = torch.tensor(u_batch, dtype=torch.long, device=model_device)

        # Scores for all items
        scores = model.getUsersRating(users_t)  # [B, n_items]

        # Mask seen items (training interactions)
        for i, u in enumerate(u_batch):
            seen = train_dict.get(u)
            if seen is not None:
                if isinstance(seen, np.ndarray):
                    seen_idx = torch.tensor(seen, device=model_device)
                else:
                    seen_idx = torch.tensor(list(seen), device=model_device)
                scores[i, seen_idx] = -1e9

        # Top maxK
        maxK = max(ks)
        topk_scores, topk_items = torch.topk(scores, k=maxK, dim=1)
        topk_items_np = topk_items.detach().cpu().numpy()  # [B, maxK]

        # Build binary hit matrix against ground truth
        gt_batch = [list(test_dict[u]) for u in u_batch]
        hits_mat = np.zeros_like(topk_items_np, dtype=np.float32)
        for i in range(topk_items_np.shape[0]):
            gt_set = set(gt_batch[i])
            hits_mat[i] = np.isin(topk_items_np[i], list(gt_set)).astype(np.float32)

        # Aggregate metrics per K for this batch
        for k in ks:
            recalls[k].append(recall_at_k(hits_mat, gt_batch, k))
            ndcgs[k].append(ndcg_at_k(hits_mat, gt_batch, k))
            hits[k].append(hit_rate_at_k(hits_mat, k))

    # Average across batches
    out = {}
    for k in ks:
        out[f"Recall@{k}"] = float(np.mean(recalls[k])) if len(recalls[k]) else 0.0
        out[f"NDCG@{k}"] = float(np.mean(ndcgs[k])) if len(ndcgs[k]) else 0.0
        out[f"Hit@{k}"] = float(np.mean(hits[k])) if len(hits[k]) else 0.0
    return out

