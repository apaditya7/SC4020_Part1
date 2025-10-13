import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_rate=0.5):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.val_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, queries, keys, causality=False):
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.val_proj(keys)

        Q_ = torch.cat(Q.chunk(self.num_heads, dim=2), dim=0)
        K_ = torch.cat(K.chunk(self.num_heads, dim=2), dim=0)
        V_ = torch.cat(V.chunk(self.num_heads, dim=2), dim=0)

        outputs = torch.matmul(Q_, K_.transpose(1, 2))
        outputs = outputs / (K_.size(-1) ** 0.5)

        # Key masking
        key_masks = torch.sign(torch.sum(torch.abs(keys), dim=-1))
        key_masks = key_masks.repeat(self.num_heads, 1)
        key_masks = key_masks.unsqueeze(1).repeat(1, queries.size(1), 1)
        outputs = outputs.masked_fill(key_masks == 0, float('-inf'))

        # Causal masking for autoregressive prediction
        if causality:
            diag_vals = torch.ones_like(outputs[0])
            tril = torch.tril(diag_vals)
            masks = tril[None, :, :].repeat(outputs.size(0), 1, 1)
            outputs = outputs.masked_fill(masks == 0, float('-inf'))

        outputs = F.softmax(outputs, dim=-1)
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)

        # Query masking
        query_masks = torch.sign(torch.sum(torch.abs(queries), dim=-1))
        query_masks = query_masks.repeat(self.num_heads, 1)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, keys.size(1))
        outputs *= query_masks

        attention_chunks = outputs.chunk(self.num_heads, dim=0)
        attention_weights = torch.stack(attention_chunks, dim=1)

        outputs = self.dropout(outputs)
        outputs = torch.matmul(outputs, V_)
        outputs = torch.cat(outputs.chunk(self.num_heads, dim=0), dim=2)

        return outputs, attention_weights


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, dropout_rate=0.5, causality=True):
        super(TransformerBlock, self).__init__()

        self.first_norm = nn.LayerNorm(dim)
        self.second_norm = nn.LayerNorm(dim)
        self.multihead_attention = MultiHeadAttention(dim, num_heads, dropout_rate)
        self.dense1 = nn.Linear(dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.causality = causality

    def forward(self, seq, mask=None):
        x = self.first_norm(seq)
        queries = x
        keys = seq
        x, attentions = self.multihead_attention(queries, keys, self.causality)

        x = x + queries
        x = self.second_norm(x)

        residual = x
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = x + residual

        if mask is not None:
            x *= mask

        return x, attentions


class SASRec(torch.nn.Module):
    """
    Self-Attentive Sequential Recommendation

    Supports both vanilla SASRec and gSASRec variants:
    - SASRec: negs_per_pos=1, gbce_t=0.0
    - gSASRec: negs_per_pos=256, gbce_t=0.75
    """
    def __init__(
        self,
        num_items,
        sequence_length=200,
        embedding_dim=256,
        num_heads=4,
        num_blocks=3,
        dropout_rate=0.5,
        reuse_item_embeddings=False
    ):
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.embeddings_dropout = torch.nn.Dropout(dropout_rate)
        self.num_heads = num_heads

        self.item_embedding = torch.nn.Embedding(self.num_items + 2, self.embedding_dim)
        self.position_embedding = torch.nn.Embedding(self.sequence_length, self.embedding_dim)

        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(self.embedding_dim, self.num_heads, self.embedding_dim, dropout_rate)
            for _ in range(num_blocks)
        ])
        self.seq_norm = torch.nn.LayerNorm(self.embedding_dim)

        self.reuse_item_embeddings = reuse_item_embeddings
        if not self.reuse_item_embeddings:
            self.output_embedding = torch.nn.Embedding(self.num_items + 2, self.embedding_dim)

    def get_output_embeddings(self):
        if self.reuse_item_embeddings:
            return self.item_embedding
        else:
            return self.output_embedding

    def forward(self, input):
        seq = self.item_embedding(input.long())
        mask = (input != self.num_items + 1).float().unsqueeze(-1)

        bs = seq.size(0)
        positions = torch.arange(seq.shape[1]).unsqueeze(0).repeat(bs, 1).to(input.device)
        pos_embeddings = self.position_embedding(positions)[:input.size(0)]
        seq = seq + pos_embeddings
        seq = self.embeddings_dropout(seq)
        seq *= mask

        attentions = []
        for i, block in enumerate(self.transformer_blocks):
            seq, attention = block(seq, mask)
            attentions.append(attention)

        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions

    def get_predictions(self, input, limit, rated=None):
        with torch.no_grad():
            model_out, _ = self.forward(input)
            seq_emb = model_out[:,-1,:]
            output_embeddings = self.get_output_embeddings()
            scores = torch.einsum('bd,nd->bn', seq_emb, output_embeddings.weight)

            scores[:,0] = float("-inf")
            scores[:,self.num_items+1:] = float("-inf")

            if rated is not None:
                for i in range(len(input)):
                    for j in rated[i]:
                        scores[i, j] = float("-inf")

            result = torch.topk(scores, limit, dim=1)
            return result.indices, result.values


GSASRec = SASRec
