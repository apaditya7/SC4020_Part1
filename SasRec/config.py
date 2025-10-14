from ir_measures import nDCG, R, P


class ExperimentConfig:
    def __init__(
        self,
        dataset_name,
        sequence_length=200,
        embedding_dim=256,
        train_batch_size=128,
        num_heads=4,
        num_blocks=3,
        dropout_rate=0.0,
        negs_per_pos=256,
        max_epochs=10000,
        max_batches_per_epoch=100,
        metrics=[nDCG@10, R@1, R@10, P@10],
        val_metric=nDCG@10,
        early_stopping_patience=200,
        gbce_t=0.75,
        filter_rated=True,
        eval_batch_size=512,
        recommendation_limit=10,
        reuse_item_embeddings=False
    ):
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.negs_per_pos = negs_per_pos
        self.max_epochs = max_epochs
        self.max_batches_per_epoch = max_batches_per_epoch
        self.val_metric = val_metric
        self.metrics = metrics
        self.early_stopping_patience = early_stopping_patience
        self.gbce_t = gbce_t
        self.filter_rated = filter_rated
        self.recommendation_limit = recommendation_limit
        self.eval_batch_size = eval_batch_size
        self.reuse_item_embeddings = reuse_item_embeddings


def get_sasrec_config(dataset_name):
    """Vanilla SASRec configuration"""
    return ExperimentConfig(
        dataset_name=dataset_name,
        sequence_length=200,
        embedding_dim=128,
        num_heads=1,
        max_batches_per_epoch=100,
        num_blocks=2,
        dropout_rate=0.5,
        negs_per_pos=1,
        gbce_t=0.0,
        reuse_item_embeddings=True
    )


def get_gsasrec_config(dataset_name):
    """gSASRec configuration"""
    return ExperimentConfig(
        dataset_name=dataset_name,
        sequence_length=200,
        embedding_dim=128,
        num_heads=1,
        max_batches_per_epoch=100,
        num_blocks=2,
        dropout_rate=0.5,
        negs_per_pos=256,
        gbce_t=0.75,
        reuse_item_embeddings=False
    )


def get_instacart_gsasrec_config():
    """gSASRec configuration optimized for Instacart dataset"""
    return ExperimentConfig(
        dataset_name='instacart',
        sequence_length=800,  # Long sequences due to item-level (not basket-level) representation
        embedding_dim=64,  # Reduced to handle longer sequences
        num_heads=1,
        max_batches_per_epoch=100,
        num_blocks=2,
        dropout_rate=0.3,  # Reduced dropout for sparse data
        negs_per_pos=128,  # Reduced negatives to speed up training
        gbce_t=0.5,  # Less aggressive calibration
        max_epochs=100,
        early_stopping_patience=15,  # More aggressive early stopping
        reuse_item_embeddings=False
    )
