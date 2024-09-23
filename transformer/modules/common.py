import torch
import torch.nn as nn
from einops import rearrange


class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input_ids) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings