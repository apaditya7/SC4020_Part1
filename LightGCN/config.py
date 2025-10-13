class LightGCNConfig:
    """Configuration for LightGCN experiments"""
    def __init__(
        self,
        dataset='yelp2018',
        data_dir='data/yelp2018',
        seed=42,
        device='cpu',
        # Training parameters
        epochs=100,
        batch_size=4096,
        lr=1e-3,
        weight_decay=1e-4,
        # Model parameters
        embed_dim=64,
        K=3,  # Number of GCN layers (K=0 for MF baseline)
        node_dropout_p=0.0,
        edge_dropout_p=0.0,
        # Sampling
        negatives_per_pos=1,
        # Evaluation
        eval_ks=(10, 20),
        eval_batch_size=2048,
        # Early stopping
        early_stopping_patience=20,
        val_metric='NDCG@10'
    ):
        self.dataset = dataset
        self.data_dir = data_dir
        self.seed = seed
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.embed_dim = embed_dim
        self.K = K
        self.node_dropout_p = node_dropout_p
        self.edge_dropout_p = edge_dropout_p
        self.negatives_per_pos = negatives_per_pos
        self.eval_ks = eval_ks
        self.eval_batch_size = eval_batch_size
        self.early_stopping_patience = early_stopping_patience
        self.val_metric = val_metric


def get_lightgcn_config(dataset='yelp2018', device='cpu'):
    """Standard LightGCN configuration"""
    return LightGCNConfig(
        dataset=dataset,
        data_dir=f'data/{dataset}',
        device=device,
        epochs=100,
        batch_size=4096,
        lr=1e-3,
        weight_decay=1e-4,
        embed_dim=64,
        K=3,
        node_dropout_p=0.0,
        edge_dropout_p=0.0,
        negatives_per_pos=1,
        eval_ks=(10, 20)
    )


def get_mf_config(dataset='yelp2018', device='cpu'):
    """Matrix Factorization baseline (K=0)"""
    config = get_lightgcn_config(dataset, device)
    config.K = 0
    return config


def get_optimized_lightgcn_config(dataset='yelp2018', device='cpu'):
    """Optimized LightGCN configuration based on ablation studies"""
    return LightGCNConfig(
        dataset=dataset,
        data_dir=f'data/{dataset}',
        device=device,
        epochs=100,
        batch_size=4096,
        lr=1e-3,
        weight_decay=1e-4,
        embed_dim=128,  # Larger embeddings
        K=3,
        node_dropout_p=0.1,
        edge_dropout_p=0.1,
        negatives_per_pos=4,
        eval_ks=(10, 20)
    )
