import torch
import torch.nn as nn
from einops import rearrange
from transformer.modules.common import Embeddings
from transformer.modules.attention import Attention
from transformer.modules.feedforward import FeedForward
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
    
  
class TransformerEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(TransformerEncoder, self).__init__()
        self.attention = Attention(config)
        self.ffn = FeedForward(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        """Pre Norm No Dropout"""
        # att_output = self.attention(hidden_states, attention_mask)
        # residual = att_output
        # att_output = self.LayerNorm(att_output)
        # ffn_output = self.ffn(att_output)
        # output = residual + ffn_output
        """Post Norm"""
        att_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(att_output)
        ffn_output = self.dropout(ffn_output)
        output = self.LayerNorm(att_output + ffn_output)
        return output
    
    
class TransformerEncoders(nn.Module):
    def __init__(self, config):
        super(TransformerEncoders, self).__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerEncoder(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states
    
    
# tradition bert
class BertModel(nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.encoders = TransformerEncoders(config)
    
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        embeddings = self.embeddings(input_ids)
        output = self.encoders(embeddings, attention_mask)
        return output
    
    
class BertForMLM(nn.Module):
    def __init__(self, config):
        super(BertForMLM, self).__init__()
        self.config = config
        self.bert = BertModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels):
        output = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores