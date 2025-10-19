import argparse
import json
import os
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess a generic explicit-feedback CSV into SASRec format (supports next-item and next-basket)")

    # IO
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV in the repo root (or anywhere)")
    parser.add_argument("--output_dir", type=str, default="SasRec/datasets/processed", help="Output directory under SasRec/datasets")

    # Columns
    parser.add_argument("--user-col", type=str, default="user_id", help="User column name if CSV has header; ignored if --has-header is false and we supply names")
    parser.add_argument("--item-col", type=str, default="item_id", help="Item column name if CSV has header")
    parser.add_argument("--rating-col", type=str, default="rating", help="Rating column name if CSV has header")
    parser.add_argument("--time-col", type=str, default="timestamp", help="Timestamp column name if CSV has header")
    parser.add_argument("--has-header", action="store_true", help="Set if the CSV includes a header row")

    # Implicit conversion
    parser.add_argument("--positive-threshold", type=float, default=4.0, help="Keep ratings >= threshold as positives")

    # Quality filters
    parser.add_argument("--min-user-interactions", type=int, default=5, help="Drop users with fewer than m positives")
    parser.add_argument("--min-item-interactions", type=int, default=5, help="Drop items with fewer than n positives")
    parser.add_argument("--top-n-users", type=int, default=None, help="Optional: cap to top-N users by activity after filtering")
    parser.add_argument("--top-n-items", type=int, default=None, help="Optional: cap to top-N items by popularity after filtering")

    # Sequence + split
    parser.add_argument("--target", type=str, choices=["next-item", "next-basket"], default="next-item", help="Evaluation target type")
    parser.add_argument("--max-seq-len", type=int, default=50, help="Truncate each user's sequence to at most max_seq_len+1 from tail before split")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic subsampling/caps")

    return parser.parse_args()


def read_csv(input_csv: str, has_header: bool, user_col: str, item_col: str, rating_col: str, time_col: str) -> pd.DataFrame:
    # Support headerless CSV with 4 columns in fixed order
    if has_header:
        usecols = [user_col, item_col, rating_col, time_col]
        df = pd.read_csv(input_csv, usecols=usecols)
        df = df.rename(columns={user_col: "user_id", item_col: "item_id", rating_col: "rating", time_col: "timestamp"})
    else:
        # Expect 4 columns: user_id, item_id, rating, timestamp
        df = pd.read_csv(input_csv, header=None, names=["user_id", "item_id", "rating", "timestamp"], usecols=[0, 1, 2, 3])

    # Normalize dtypes
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)

    # Parse/normalize timestamp to epoch seconds (int)
    if np.issubdtype(df["timestamp"].dtype, np.number):
        df["timestamp"] = df["timestamp"].astype(np.int64)
    else:
        # parse string to datetime then to int seconds
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.loc[~ts.isna()].copy()
        df["timestamp"] = (pd.to_datetime(df["timestamp"]).astype("int64") // 10**9).astype(np.int64)

    # Rating to float
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])  # Drop rows with non-parsable rating

    # Deduplicate exact duplicates (user, item, timestamp)
    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "item_id", "timestamp"]).reset_index(drop=True)
    after = len(df)
    print(f"Deduplicated exact triplets: {before} -> {after} rows")

    return df


def to_implicit(df: pd.DataFrame, positive_threshold: float) -> pd.DataFrame:
    before = len(df)
    df = df.loc[df["rating"] >= positive_threshold, ["user_id", "item_id", "timestamp"]].copy()
    after = len(df)
    print(f"Converted to implicit with threshold >= {positive_threshold}: {before} -> {after} positives")
    return df


def apply_quality_filters(df: pd.DataFrame, m: int, n: int, top_n_users: Optional[int], top_n_items: Optional[int], seed: int) -> pd.DataFrame:
    # Filter by min interactions iteratively until stable
    print(f"Applying cold-start filters: min_user_interactions={m}, min_item_interactions={n}")
    prev_shape = None
    iteration = 0
    while prev_shape != df.shape:
        iteration += 1
        prev_shape = df.shape
        # Users
        u_counts = df["user_id"].value_counts()
        keep_users = set(u_counts[u_counts >= m].index)
        df = df.loc[df["user_id"].isin(keep_users)]
        # Items
        i_counts = df["item_id"].value_counts()
        keep_items = set(i_counts[i_counts >= n].index)
        df = df.loc[df["item_id"].isin(keep_items)]
        print(f"  Iter {iteration}: {prev_shape} -> {df.shape}")

    # Optional caps by popularity/activity after filtering
    if top_n_items is not None:
        i_counts = df["item_id"].value_counts()
        # Deterministic tie-break by item_id
        items_sorted = sorted(i_counts.items(), key=lambda x: (-x[1], x[0]))
        keep = set([iid for iid, _ in items_sorted[:top_n_items]])
        df = df.loc[df["item_id"].isin(keep)]
        print(f"Capped to top-{top_n_items} items: now {df['item_id'].nunique()} items, {len(df)} interactions")

    if top_n_users is not None:
        u_counts = df["user_id"].value_counts()
        users_sorted = sorted(u_counts.items(), key=lambda x: (-x[1], x[0]))
        keep = set([uid for uid, _ in users_sorted[:top_n_users]])
        df = df.loc[df["user_id"].isin(keep)]
        print(f"Capped to top-{top_n_users} users: now {df['user_id'].nunique()} users, {len(df)} interactions")

    return df


def build_sequences_next_item(df: pd.DataFrame, max_seq_len: int) -> Dict[str, List[str]]:
    # Sort by user, timestamp
    df = df.sort_values(["user_id", "timestamp", "item_id"])  # stable order with item_id tie-break
    sequences: Dict[str, List[str]] = defaultdict(list)
    # Build sequence per user
    for (uid), grp in df.groupby("user_id", sort=False):
        items = grp["item_id"].tolist()
        # tail truncate to max_seq_len + 1 (to allow at least 1 target; we will still require >=3 for split)
        if len(items) > max_seq_len + 1:
            items = items[-(max_seq_len + 1):]
        sequences[uid] = items
    return sequences


def build_sequences_next_basket(df: pd.DataFrame, max_seq_len: int) -> Tuple[Dict[str, List[str]], Dict[str, List[List[str]]]]:
    # Also need baskets per user for targets
    df = df.sort_values(["user_id", "timestamp", "item_id"])  # determinism
    flat_sequences: Dict[str, List[str]] = {}
    baskets_by_user: Dict[str, List[List[str]]] = {}

    for uid, grp in df.groupby("user_id", sort=False):
        # Group by timestamp -> basket of items. Use unique within basket and stable order by item_id
        baskets: List[List[str]] = []
        for ts, ts_grp in grp.groupby("timestamp", sort=False):
            basket_items = sorted(ts_grp["item_id"].unique().tolist())
            baskets.append(basket_items)

        # Flatten baskets for inputs (targets remain basket lists)
        flat = [it for b in baskets for it in b]
        if len(flat) > max_seq_len + 1:
            flat = flat[-(max_seq_len + 1):]

        flat_sequences[uid] = flat
        baskets_by_user[uid] = baskets

    return flat_sequences, baskets_by_user


def remap_items(sequences: Dict[str, List[str]]) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    # Gather unique items
    all_items = set()
    for items in sequences.values():
        all_items.update(items)
    # Dense mapping [1..N]
    item2id = {item: idx + 1 for idx, item in enumerate(sorted(all_items))}
    remapped: Dict[str, List[int]] = {}
    for uid, items in sequences.items():
        remapped[uid] = [item2id[x] for x in items]
    return remapped, item2id


def leave_last_split_next_item(seqs: Dict[str, List[int]]):
    train_in, val_in, val_out, test_in, test_out = {}, {}, {}, {}, {}
    for uid, s in seqs.items():
        if len(s) < 3:
            continue
        test_in[uid] = s[:-1]
        test_out[uid] = s[-1]
        val_in[uid] = s[:-2]
        val_out[uid] = s[-2]
        train_in[uid] = s[:-2]
    return train_in, val_in, val_out, test_in, test_out


def leave_last_split_next_basket(
    flat_seqs: Dict[str, List[int]],
    baskets_by_user_items: Dict[str, List[List[int]]]
):
    train_in, val_in, val_out, test_in, test_out = {}, {}, {}, {}, {}

    for uid, flat in flat_seqs.items():
        baskets = baskets_by_user_items.get(uid, [])
        if len(baskets) < 2:
            continue

        # Build flattened prefixes excluding the last basket for test, and excluding last two baskets for val/train
        # Determine indices for flattening boundaries
        # Compute lengths per basket in remapped ids
        basket_lengths = [len(b) for b in baskets]
        cum = np.cumsum([0] + basket_lengths)  # positions in flattened list

        # test prefix excludes last basket
        test_prefix_len = cum[-2]
        # val prefix excludes last two baskets
        val_prefix_len = cum[-3] if len(cum) >= 3 else 0

        # Inputs are truncated flattened sequences accordingly
        test_in_seq = flat[:test_prefix_len]
        val_in_seq = flat[:val_prefix_len]

        # Targets are entire baskets (last and second-last)
        test_target = baskets[-1]
        val_target = baskets[-2]

        # Keep only users with non-empty inputs and targets
        if len(test_target) == 0 or len(val_target) == 0:
            continue

        train_in[uid] = val_in_seq
        val_in[uid] = val_in_seq
        val_out[uid] = val_target
        test_in[uid] = test_in_seq
        test_out[uid] = test_target

    return train_in, val_in, val_out, test_in, test_out


def save_sasrec_layout(output_dir: str,
                       train_in: Dict[str, List[int]],
                       val_in: Dict[str, List[int]], val_out: Dict[str, Union[List[int], int]],
                       test_in: Dict[str, List[int]], test_out: Dict[str, Union[List[int], int]],
                       num_items: int,
                       num_users: int,
                       num_interactions: int,
                       target: str,
                       max_seq_len: int,
                       mappings: Dict[str, Dict]):
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    # Deterministic user ordering
    def sorted_users(d):
        return sorted(d.keys())

    with open(os.path.join(output_dir, "train", "input.txt"), "w") as f:
        for uid in sorted_users(train_in):
            f.write(" ".join(map(str, train_in[uid])) + "\n")

    with open(os.path.join(output_dir, "val", "input.txt"), "w") as f:
        for uid in sorted_users(val_in):
            f.write(" ".join(map(str, val_in[uid])) + "\n")

    with open(os.path.join(output_dir, "val", "output.txt"), "w") as f:
        for uid in sorted_users(val_out):
            out = val_out[uid]
            if isinstance(out, list):
                f.write(" ".join(map(str, out)) + "\n")
            else:
                f.write(str(out) + "\n")

    with open(os.path.join(output_dir, "test", "input.txt"), "w") as f:
        for uid in sorted_users(test_in):
            f.write(" ".join(map(str, test_in[uid])) + "\n")

    with open(os.path.join(output_dir, "test", "output.txt"), "w") as f:
        for uid in sorted_users(test_out):
            out = test_out[uid]
            if isinstance(out, list):
                f.write(" ".join(map(str, out)) + "\n")
            else:
                f.write(str(out) + "\n")

    stats = {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": num_interactions,
        "target": target,
        "max_seq_len": max_seq_len,
        "pad_token_id": num_items + 1,
    }
    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(output_dir, "mappings.json"), "w") as f:
        json.dump(mappings, f, indent=2)

    print(f"Saved dataset to: {output_dir}")
    print(f"Stats: {stats}")


def sanity_checks(output_dir: str, num_items: int, target: str):
    # Check ranges and non-empty targets
    def check_file(path: str, allow_list_targets: bool = False):
        if not os.path.exists(path):
            print(f"Warning: missing file {path}")
            return
        with open(path, "r") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line == "":
                    continue
                ids = [int(x) for x in line.split()]
                if len(ids) == 0:
                    print(f"Empty line at {path}:{i}")
                if not all(1 <= x <= num_items for x in ids):
                    print(f"ID out of range at {path}:{i} -> {ids}")

    check_file(os.path.join(output_dir, "train", "input.txt"))
    check_file(os.path.join(output_dir, "val", "input.txt"))
    check_file(os.path.join(output_dir, "val", "output.txt"), allow_list_targets=(target == "next-basket"))
    check_file(os.path.join(output_dir, "test", "input.txt"))
    check_file(os.path.join(output_dir, "test", "output.txt"), allow_list_targets=(target == "next-basket"))

    # Quick length stats
    def lengths(path: str) -> List[int]:
        vals = []
        if not os.path.exists(path):
            return vals
        with open(path, "r") as f:
            for line in f:
                vals.append(0 if line.strip() == "" else len(line.strip().split()))
        return vals

    lens = lengths(os.path.join(output_dir, "train", "input.txt"))
    if lens:
        arr = np.array(lens)
        print(
            "Input length percentiles (train)",
            {
                "p10": float(np.percentile(arr, 10)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "mean": float(np.mean(arr)),
            },
        )


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Derive default output_dir under SasRec/datasets from CSV basename
    if args.output_dir == "SasRec/datasets/processed":
        base = os.path.basename(args.input_csv)
        base = os.path.splitext(base)[0]
        safe_base = base.replace(" ", "_").replace("(", "").replace(")", "")
        args.output_dir = os.path.join("SasRec", "datasets", safe_base)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Configuration:")
    print({
        "input_csv": args.input_csv,
        "output_dir": args.output_dir,
        "positive_threshold": args.positive_threshold,
        "min_user_interactions": args.min_user_interactions,
        "min_item_interactions": args.min_item_interactions,
        "top_n_users": args.top_n_users,
        "top_n_items": args.top_n_items,
        "target": args.target,
        "max_seq_len": args.max_seq_len,
        "has_header": args.has_header,
        "seed": args.seed,
    })

    # Load and preprocess
    df = read_csv(
        args.input_csv,
        has_header=args.has_header,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
        time_col=args.time_col,
    )

    df = to_implicit(df, positive_threshold=args.positive_threshold)
    df = apply_quality_filters(
        df,
        m=args.min_user_interactions,
        n=args.min_item_interactions,
        top_n_users=args.top_n_users,
        top_n_items=args.top_n_items,
        seed=args.seed,
    )

    num_interactions = len(df)
    num_users = df["user_id"].nunique()
    num_items_raw = df["item_id"].nunique()

    # Build sequences and split
    if args.target == "next-item":
        sequences_str = build_sequences_next_item(df, max_seq_len=args.max_seq_len)
        # Remove users with too short sequences to support split
        sequences_str = {u: s for u, s in sequences_str.items() if len(s) >= 3}
        # Remap items
        sequences_int, item2id = remap_items(sequences_str)
        # Split
        train_in, val_in, val_out, test_in, test_out = leave_last_split_next_item(sequences_int)
        # Build mappings.json (include users for reference)
        user2id = {u: i for i, u in enumerate(sorted(sequences_int.keys()))}
        # Save
        save_sasrec_layout(
            args.output_dir,
            train_in, val_in, val_out, test_in, test_out,
            num_items=len(item2id),
            num_users=len(user2id),
            num_interactions=num_interactions,
            target=args.target,
            max_seq_len=args.max_seq_len,
            mappings={"item2id": item2id, "user2id": user2id},
        )

    else:  # next-basket
        flat_sequences_str, baskets_by_user_str = build_sequences_next_basket(df, max_seq_len=args.max_seq_len)

        # Only keep users with at least two baskets
        baskets_by_user_str = {u: b for u, b in baskets_by_user_str.items() if len(b) >= 2}
        flat_sequences_str = {u: flat_sequences_str[u] for u in baskets_by_user_str.keys()}

        # Remap items on flattened sequences
        flat_sequences_int, item2id = remap_items(flat_sequences_str)

        # Remap baskets as well
        baskets_by_user_int = {u: [[item2id[x] for x in b] for b in baskets] for u, baskets in baskets_by_user_str.items()}

        # Split respecting full-basket targets
        train_in, val_in, val_out, test_in, test_out = leave_last_split_next_basket(flat_sequences_int, baskets_by_user_int)

        # Remove users that ended up empty after split
        user_keys = sorted(set(train_in) & set(val_in) & set(val_out) & set(test_in) & set(test_out))
        train_in = {u: train_in[u] for u in user_keys}
        val_in = {u: val_in[u] for u in user_keys}
        val_out = {u: val_out[u] for u in user_keys}
        test_in = {u: test_in[u] for u in user_keys}
        test_out = {u: test_out[u] for u in user_keys}

        user2id = {u: i for i, u in enumerate(user_keys)}

        save_sasrec_layout(
            args.output_dir,
            train_in, val_in, val_out, test_in, test_out,
            num_items=len(item2id),
            num_users=len(user2id),
            num_interactions=num_interactions,
            target=args.target,
            max_seq_len=args.max_seq_len,
            mappings={"item2id": item2id, "user2id": user2id},
        )

    sanity_checks(args.output_dir, num_items=len(item2id), target=args.target)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
