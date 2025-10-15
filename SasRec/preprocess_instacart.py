"""
Preprocess Instacart dataset for SASRec training.

Expected input files (download from Kaggle Instacart Market Basket Analysis):
- instacart_raw/orders.csv
- instacart_raw/order_products__prior.csv
- instacart_raw/order_products__train.csv

This script will:
1. Subsample to top K users and products
2. Create user sequences sorted by order time
3. Split into train/val/test sets
4. Save in the format expected by SASRec
"""

import pandas as pd
import json
import os
from collections import defaultdict
import argparse


def load_instacart_data(raw_dir):
    """Load Instacart CSV files"""
    print("Loading Instacart data...")
    orders = pd.read_csv(f"{raw_dir}/orders.csv")

    # Combine prior and train order products
    order_products_prior = pd.read_csv(f"{raw_dir}/order_products__prior.csv")
    order_products_train = pd.read_csv(f"{raw_dir}/order_products__train.csv")
    order_products = pd.concat([order_products_prior, order_products_train])

    print(f"Loaded {len(orders)} orders and {len(order_products)} order-product pairs")
    return orders, order_products


def iterative_coldstart_filter(data, min_interactions=5, max_iterations=10):
    """
    Iteratively filter out users and items with too few interactions.
    This ensures data quality by removing cold-start users/items.
    """
    print(f"Starting iterative cold-start filtering (min_interactions={min_interactions})...")

    for iteration in range(max_iterations):
        initial_users = data['user_id'].nunique()
        initial_items = data['product_id'].nunique()
        initial_interactions = len(data)

        # Filter users with too few interactions
        user_counts = data['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        data = data[data['user_id'].isin(valid_users)]

        # Filter items with too few interactions
        item_counts = data['product_id'].value_counts()
        valid_items = item_counts[item_counts >= min_interactions].index
        data = data[data['product_id'].isin(valid_items)]

        final_users = data['user_id'].nunique()
        final_items = data['product_id'].nunique()
        final_interactions = len(data)

        print(f"  Iteration {iteration + 1}: "
              f"Users {initial_users} → {final_users}, "
              f"Items {initial_items} → {final_items}, "
              f"Interactions {initial_interactions} → {final_interactions}")

        # Stop if no changes
        if initial_users == final_users and initial_items == final_items:
            print(f"Converged after {iteration + 1} iterations")
            break

    return data


def preprocess_sequences(orders, order_products, top_k_users=10000, top_k_products=5000,
                        min_sequence_length=5, max_orders_per_user=None, use_coldstart_filter=True,
                        min_interactions=5):
    """
    Create user sequences from Instacart data.

    Args:
        orders: DataFrame with order information
        order_products: DataFrame with order-product pairs
        top_k_users: Number of top users to include (by sequence length)
        top_k_products: Number of top products to include (by popularity)
        min_sequence_length: Minimum sequence length to include a user
        max_orders_per_user: If set, limit each user to their most recent N orders (reduces sequence length)
        use_coldstart_filter: If True, apply iterative cold-start filtering
        min_interactions: Minimum interactions for cold-start filtering
    """

    # Merge order_products with orders to get user_id and order_number
    data = order_products.merge(orders[['order_id', 'user_id', 'order_number']], on='order_id')

    # Optional: Apply iterative cold-start filtering first
    if use_coldstart_filter:
        data = iterative_coldstart_filter(data, min_interactions=min_interactions)

    # Filter to most popular products (after cold-start filtering)
    product_counts = data['product_id'].value_counts()
    top_products = set(product_counts.head(top_k_products).index)
    print(f"Selected top {len(top_products)} products")

    # Filter to only include top products
    data = data[data['product_id'].isin(top_products)]

    # Group by user and order, aggregate products
    user_orders = data.groupby(['user_id', 'order_number'])['product_id'].apply(list).reset_index()

    # Create sequences: sort by order_number for each user
    user_sequences = defaultdict(list)
    for _, row in user_orders.iterrows():
        user_sequences[row['user_id']].append((row['order_number'], row['product_id']))

    # Sort each user's orders and flatten into item sequence
    sequences = {}
    for user_id, orders_list in user_sequences.items():
        orders_list.sort(key=lambda x: x[0])  # Sort by order_number

        # Optional: Limit to most recent N orders
        if max_orders_per_user is not None and len(orders_list) > max_orders_per_user:
            orders_list = orders_list[-max_orders_per_user:]
            print(f"User {user_id}: Limited to last {max_orders_per_user} orders")

        # Flatten: each order's products become sequential items
        sequence = [item for _, products in orders_list for item in products]
        if len(sequence) >= min_sequence_length:
            sequences[user_id] = sequence

    print(f"Created {len(sequences)} user sequences (min length {min_sequence_length})")

    # Filter to top K users by sequence length (more interactions = better signal)
    sequence_lengths = [(user_id, len(seq)) for user_id, seq in sequences.items()]
    sequence_lengths.sort(key=lambda x: x[1], reverse=True)
    top_users = [user_id for user_id, _ in sequence_lengths[:top_k_users]]

    sequences = {user_id: sequences[user_id] for user_id in top_users}
    print(f"Selected top {len(sequences)} users by sequence length")

    # Print sequence length statistics
    lengths = [len(seq) for seq in sequences.values()]
    print(f"Sequence length stats: min={min(lengths)}, max={max(lengths)}, "
          f"mean={sum(lengths)/len(lengths):.1f}, median={sorted(lengths)[len(lengths)//2]}")

    return sequences


def remap_ids(sequences):
    """Remap user and product IDs to contiguous integers starting from 1"""
    # Get all unique products
    all_products = set()
    for seq in sequences.values():
        all_products.update(seq)

    # Create mappings
    product_to_idx = {prod: idx + 1 for idx, prod in enumerate(sorted(all_products))}
    user_to_idx = {user: idx for idx, user in enumerate(sorted(sequences.keys()))}

    # Remap sequences
    remapped = {}
    for user_id, seq in sequences.items():
        new_user_id = user_to_idx[user_id]
        new_seq = [product_to_idx[prod] for prod in seq]
        remapped[new_user_id] = new_seq

    print(f"Remapped {len(all_products)} products and {len(sequences)} users")
    return remapped, len(all_products)


def split_sequences(sequences, val_ratio=0.1, test_ratio=0.1):
    """Split sequences into train/val/test using leave-one-out for val and test"""
    train_data = {}
    val_input = {}
    val_output = {}
    test_input = {}
    test_output = {}

    for user_id, seq in sequences.items():
        if len(seq) < 3:
            continue  # Need at least 3 items for train/val/test split

        # Leave last item for test, second-to-last for val
        test_input[user_id] = seq[:-1]
        test_output[user_id] = seq[-1]

        val_input[user_id] = seq[:-2]
        val_output[user_id] = seq[-2]

        train_data[user_id] = seq[:-2]

    print(f"Split: {len(train_data)} train, {len(val_input)} val, {len(test_input)} test sequences")
    return train_data, val_input, val_output, test_input, test_output


def save_dataset(output_dir, train_data, val_input, val_output, test_input, test_output, num_items):
    """Save dataset in SASRec format"""
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/val", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)

    # Save train data (full sequences for training)
    with open(f"{output_dir}/train/input.txt", 'w') as f:
        for user_id in sorted(train_data.keys()):
            f.write(' '.join(map(str, train_data[user_id])) + '\n')

    # Save validation data
    with open(f"{output_dir}/val/input.txt", 'w') as f:
        for user_id in sorted(val_input.keys()):
            f.write(' '.join(map(str, val_input[user_id])) + '\n')

    with open(f"{output_dir}/val/output.txt", 'w') as f:
        for user_id in sorted(val_output.keys()):
            f.write(str(val_output[user_id]) + '\n')

    # Save test data
    with open(f"{output_dir}/test/input.txt", 'w') as f:
        for user_id in sorted(test_input.keys()):
            f.write(' '.join(map(str, test_input[user_id])) + '\n')

    with open(f"{output_dir}/test/output.txt", 'w') as f:
        for user_id in sorted(test_output.keys()):
            f.write(str(test_output[user_id]) + '\n')

    # Save dataset stats
    stats = {
        'num_users': len(train_data),
        'num_items': num_items,
        'num_interactions': sum(len(seq) for seq in train_data.values())
    }

    with open(f"{output_dir}/dataset_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Dataset saved to {output_dir}")
    print(f"Stats: {stats}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Instacart dataset')
    parser.add_argument('--raw_dir', type=str, default='instacart_data',
                       help='Directory containing raw Instacart CSV files')
    parser.add_argument('--output_dir', type=str, default='datasets/instacart',
                       help='Output directory for preprocessed data')
    parser.add_argument('--top_k_users', type=int, default=10000,
                       help='Number of top users to include')
    parser.add_argument('--top_k_products', type=int, default=5000,
                       help='Number of top products to include')
    parser.add_argument('--min_sequence_length', type=int, default=5,
                       help='Minimum sequence length to include')
    parser.add_argument('--max_orders_per_user', type=int, default=None,
                       help='Limit each user to their most recent N orders (optional, controls sequence length)')
    parser.add_argument('--use_coldstart_filter', action='store_true', default=True,
                       help='Apply iterative cold-start filtering (default: True)')
    parser.add_argument('--no_coldstart_filter', action='store_false', dest='use_coldstart_filter',
                       help='Disable cold-start filtering')
    parser.add_argument('--min_interactions', type=int, default=5,
                       help='Minimum interactions for cold-start filtering')

    args = parser.parse_args()

    # Load data
    orders, order_products = load_instacart_data(args.raw_dir)

    # Create sequences
    sequences = preprocess_sequences(
        orders, order_products,
        top_k_users=args.top_k_users,
        top_k_products=args.top_k_products,
        min_sequence_length=args.min_sequence_length,
        max_orders_per_user=args.max_orders_per_user,
        use_coldstart_filter=args.use_coldstart_filter,
        min_interactions=args.min_interactions
    )

    # Remap IDs
    sequences, num_items = remap_ids(sequences)

    # Split into train/val/test
    train_data, val_input, val_output, test_input, test_output = split_sequences(sequences)

    # Save dataset
    save_dataset(args.output_dir, train_data, val_input, val_output,
                test_input, test_output, num_items)

    print("Preprocessing complete!")


if __name__ == '__main__':
    main()
