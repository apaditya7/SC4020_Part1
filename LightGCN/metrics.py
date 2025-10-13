import numpy as np

def recall_at_k(ranked_hits: np.ndarray, ground_truth: list, k: int) -> float:
    # ranked_hits: shape [B, K] binary hits for top-K predictions
    # ground_truth: list of arrays/lists of true items per user
    topk = ranked_hits[:, :k]
    gains = topk.sum(axis=1)
    denom = np.array([max(1, len(gt)) for gt in ground_truth])
    return float(np.mean(gains / denom))

def hit_rate_at_k(ranked_hits: np.ndarray, k: int) -> float:
    topk = ranked_hits[:, :k]
    return float(np.mean(topk.sum(axis=1) > 0))

def ndcg_at_k(ranked_hits: np.ndarray, ground_truth: list, k: int) -> float:
    # ranked_hits: shape [B, K] with 1/0 hits along ranking positions
    topk = ranked_hits[:, :k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # [k]
    dcg = (topk * discounts).sum(axis=1)
    # Ideal DCG per user
    ideal_lengths = np.array([min(len(gt), k) for gt in ground_truth])
    ideal = np.zeros_like(dcg)
    # precompute prefix sums of discounts to get IDCG efficiently
    prefix = np.cumsum(discounts)
    ideal = np.where(ideal_lengths > 0, prefix[ideal_lengths - 1], 0.0)
    ideal = np.where(ideal == 0.0, 1.0, ideal)  # avoid div by zero
    ndcg = dcg / ideal
    return float(np.mean(ndcg))

