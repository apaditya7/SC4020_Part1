import torch
from model import SASRec
from data_utils import get_num_items


def build_model(config):
    num_items = get_num_items(config.dataset_name)
    model = SASRec(
        num_items,
        sequence_length=config.sequence_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        dropout_rate=config.dropout_rate,
        reuse_item_embeddings=config.reuse_item_embeddings
    )
    return model


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"
