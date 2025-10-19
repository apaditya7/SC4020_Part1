from typing import Dict, Tuple, Iterable
import os
from collections import defaultdict
import numpy as np
import torch
import scipy.sparse as sp


def _read_interactions(path: str) -> Dict[int, np.ndarray]:
    """
    Reads LightGCN-format interaction file where each line is:
    user item1 item2 ...
    Returns dict user -> np.array(items, dtype=int)
    """
    user_items = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u = int(parts[0])
            items = np.array([int(x) for x in parts[1:]], dtype=np.int64)
            user_items[u] = items
    return user_items


def load_dataset(name: str, data_dir: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], int, int]:
    """
    Loads dataset with LightGCN text format.
    - train.txt and test.txt are required.
    - Returns (train_dict, valid_dict, test_dict, n_users, n_items)
      where valid_dict may be empty if not provided.
    """
    train_p = os.path.join(data_dir, "train.txt")
    test_p = os.path.join(data_dir, "test.txt")
    valid_p = os.path.join(data_dir, "valid.txt")
    if not os.path.exists(train_p) or not os.path.exists(test_p):
        raise FileNotFoundError(f"Expected train.txt and test.txt in {data_dir}")

    train = _read_interactions(train_p)
    test = _read_interactions(test_p)
    valid = _read_interactions(valid_p) if os.path.exists(valid_p) else {}

    # infer n_users, n_items
    all_users = set(train.keys()) | set(test.keys()) | set(valid.keys())
    max_user = max(all_users) if all_users else -1
    max_item = -1
    for d in (train, test, valid):
        for items in d.values():
            if len(items) > 0:
                max_item = max(max_item, int(np.max(items)))
    n_users = max_user + 1
    n_items = max_item + 1
    return train, valid, test, n_users, n_items


def build_graph(train: Dict[int, Iterable[int]], n_users: int, n_items: int) -> torch.Tensor:
    """
    Builds a symmetric normalized adjacency matrix (LightGCN) as torch.sparse.FloatTensor.
    A = [[0, R], [R^T, 0]] with symmetric normalization D^-1/2 A D^-1/2
    """
    rows = []
    cols = []
    data = []
    for u, items in train.items():
        if len(items) == 0:
            continue
        items = np.asarray(items, dtype=np.int64)
        rows.extend([u] * len(items))
        cols.extend(list(items + n_users))
        data.extend([1.0] * len(items))
        # symmetric
        rows.extend(list(items + n_users))
        cols.extend([u] * len(items))
        data.extend([1.0] * len(items))

    N = n_users + n_items
    if len(rows) == 0:
        idx = torch.empty((2, 0), dtype=torch.long)
        val = torch.empty((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, size=(N, N))

    coo = sp.coo_matrix((np.array(data, dtype=np.float32), (np.array(rows), np.array(cols))), shape=(N, N))
    # symmetric normalization
    deg = np.asarray(coo.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    inv_sqrt = np.power(deg, -0.5)
    D_inv_sqrt = sp.diags(inv_sqrt)
    norm = D_inv_sqrt @ coo @ D_inv_sqrt
    norm = norm.tocoo()
    idx = torch.tensor([norm.row, norm.col], dtype=torch.long)
    val = torch.tensor(norm.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, size=(N, N)).coalesce()


def sample_batch(train: Dict[int, np.ndarray],
                 n_items: int,
                 batch_size: int,
                 negatives_per_pos: int = 1,
                 rng: np.random.Generator = None,
                 _cache={}) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Optimized BPR sampling with vectorized negative sampling.
    """
    if rng is None:
        rng = np.random.default_rng()

    train_id = id(train)
    if train_id not in _cache:
        _cache.clear()
        _cache[train_id] = {
            'users_list': np.array(list(train.keys())),
            'pos_sets': {u: set(items.tolist() if isinstance(items, np.ndarray) else list(items)) for u, items in train.items()},
            'pos_arrays': {u: np.array(list(items)) for u, items in train.items()}
        }

    cached = _cache[train_id]
    users_arr = cached['users_list']
    pos_sets = cached['pos_sets']
    pos_arrays = cached['pos_arrays']

    total_samples = batch_size * negatives_per_pos
    batch_users = np.zeros(total_samples, dtype=np.int64)
    batch_pos = np.zeros(total_samples, dtype=np.int64)
    batch_neg = np.zeros(total_samples, dtype=np.int64)

    idx = 0
    while idx < total_samples:
        u = int(rng.choice(users_arr))
        user_pos_items = pos_arrays.get(u)
        if user_pos_items is None or len(user_pos_items) == 0:
            continue

        i = int(rng.choice(user_pos_items))
        user_pos_set = pos_sets[u]

        samples_needed = min(negatives_per_pos, total_samples - idx)
        for _ in range(samples_needed):
            j = int(rng.integers(0, n_items))
            while j in user_pos_set:
                j = int(rng.integers(0, n_items))

            batch_users[idx] = u
            batch_pos[idx] = i
            batch_neg[idx] = j
            idx += 1

    return (torch.from_numpy(batch_users[:batch_size]),
            torch.from_numpy(batch_pos[:batch_size]),
            torch.from_numpy(batch_neg[:batch_size]))

