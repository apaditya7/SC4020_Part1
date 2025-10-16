import json
import torch
from torch.utils.data import Dataset, DataLoader


def _parse_output_line(line):
    """
    Parse one line from val/test output.txt.
    Supports both:
      - multi-label: 'i1 i2 i3'
      - single-label: 'i'
    Returns: list[int]
    """
    toks = line.strip().split()
    if not toks:
        return []
    if len(toks) == 1:
        # backward-compat: single item -> one-element list
        return [int(toks[0])]
    return [int(t) for t in toks]


class SequenceDataset(Dataset):
    def __init__(self, input_file, padding_value, output_file=None, max_length=200):
        # raw input sequences (flattened items)
        with open(input_file, 'r') as f:
            self.inputs = [list(map(int, line.strip().split())) for line in f]

        # outputs:
        # - train: None
        # - val/test: list[int] (multi-label next-basket)
        if output_file:
            with open(output_file, 'r') as f:
                self.outputs = [_parse_output_line(line) for line in f]
        else:
            self.outputs = None

        self.max_length = max_length
        self.padding_value = padding_value

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # keep 'rated' BEFORE padding (set of seen items)
        inp_raw = self.inputs[idx]
        rated = set(inp_raw)

        # pad/clip left so last events are visible
        if len(inp_raw) > self.max_length:
            inp = inp_raw[-self.max_length:]
        else:
            pad = [self.padding_value] * (self.max_length - len(inp_raw))
            inp = pad + inp_raw

        inp_tensor = torch.tensor(inp, dtype=torch.long)

        if self.outputs is not None:
            # multi-label targets; keep as Python list[int] (ragged)
            tgt_list = self.outputs[idx]
            return inp_tensor, rated, tgt_list

        # training set returns only inputs; collate adds negatives
        return (inp_tensor,)


def collate_with_random_negatives(input_batch, pad_value, num_negatives):
    """
    Train collate:
      - Stacks inputs to [B, S]
      - Samples negatives uniformly in [1, pad_value) (where pad_value = num_items+1)
      - Returns [inputs, negatives] with shapes [B, S], [B, S, K]
    """
    batch_inputs = torch.stack([sample[0] for sample in input_batch], dim=0)  # [B, S]
    negatives = torch.randint(
        low=1, high=pad_value, size=(batch_inputs.size(0), batch_inputs.size(1), num_negatives)
    )  # [B, S, K]
    return [batch_inputs, negatives]


def collate_val_test(input_batch):
    """
    Val/Test collate:
      - Stacks inputs to [B, S]
      - Collects 'rated' as list[set[int]] (length B)
      - Keeps targets as list[list[int]] (ragged, multi-label)
    """
    inputs = torch.stack([sample[0] for sample in input_batch], dim=0)       # [B, S]
    rated = [sample[1] for sample in input_batch]                            # list[set[int]]
    targets = [sample[2] for sample in input_batch]                          # list[list[int]]
    return [inputs, rated, targets]


def get_num_items(dataset):
    with open(f"datasets/{dataset}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    return stats['num_items']


def get_padding_value(dataset_dir):
    with open(f"{dataset_dir}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    # IDs are 1..num_items, so padding uses num_items+1
    return stats['num_items'] + 1


def get_train_dataloader(dataset_name, batch_size=32, max_length=200, train_neg_per_positive=256):
    dataset_dir = f"datasets/{dataset_name}"
    padding_value = get_padding_value(dataset_dir)
    # train uses max_length+1 so we can shift for next-token (labels = positives[:, 1:])
    train_dataset = SequenceDataset(
        f"{dataset_dir}/train/input.txt",
        padding_value=padding_value,
        output_file=None,
        max_length=max_length + 1
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_with_random_negatives(x, padding_value, train_neg_per_positive)
    )
    return train_loader


def get_val_or_test_dataloader(dataset_name, part='val', batch_size=32, max_length=200):
    dataset_dir = f"datasets/{dataset_name}"
    padding_value = get_padding_value(dataset_dir)
    dataset = SequenceDataset(
        f"{dataset_dir}/{part}/input.txt",
        padding_value=padding_value,
        output_file=f"{dataset_dir}/{part}/output.txt",
        max_length=max_length
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_val_test
    )
    return dataloader


def get_val_dataloader(dataset_name, batch_size=32, max_length=200):
    return get_val_or_test_dataloader(dataset_name, 'val', batch_size, max_length)


def get_test_dataloader(dataset_name, batch_size=32, max_length=200):
    return get_val_or_test_dataloader(dataset_name, 'test', batch_size, max_length)
