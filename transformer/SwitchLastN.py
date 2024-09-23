import torch
import torch.nn as nn
from einops import rearrange
from transformer.modules.common import Embeddings
from transformer.modules.attention import Attention
from transformer.modules.feedforward import FeedForward, SwitchFeedForward
from transformer.BERT import TransformerEncoder
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class SwitchEncoder(nn.Module):
    def __init__(self, config):
        super(SwitchEncoder, self).__init__()
        self.attention = Attention(config)
        self.ffn = SwitchFeedForward(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask):
        """Pre Norm"""
        # att_output = self.attention(hidden_states, attention_mask)
        # residual = att_output
        # att_output = self.LayerNorm(att_output)
        # ffn_output = self.ffn(att_output)
        # # ffn_output = self.dropout(ffn_output)
        
        # # output = self.LayerNorm(att_output + ffn_output)
        # output = residual + ffn_output
        """Post Norm"""
        att_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(att_output)
        ffn_output = self.dropout(ffn_output)
        output = self.LayerNorm(att_output + ffn_output)        
        
        return output
    
    
class SwitchLayers(nn.Module):
    def __init__(self, config):
        super(SwitchLayers, self).__init__()
        self.last_n = 6
        self.layers = nn.ModuleList(
            [TransformerEncoder(config) for _ in range(self.last_n)] + 
            [SwitchEncoder(config) for _ in range(config.num_hidden_layers - self.last_n)]
        )
        
    def forward(self, hidden_states, attention_mask):
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states
    
    
class SwitchLastNModel(nn.Module):
    def __init__(self, config):
        super(SwitchLastNModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = SwitchLayers(config)
        
    def forward(self, input_ids, attention_mask):
        embeddings = self.embeddings(input_ids)
        outputs = self.layers(embeddings, attention_mask)
        return outputs
    
    
class BertSwitchLastN(nn.Module):
    def __init__(self, config):
        super(BertSwitchLastN, self).__init__()
        self.config = config
        self.bert = SwitchLastNModel(config)
        self.head = BertOnlyMLMHead(config)
        self.criterion = nn.CrossEntropyLoss() 
        self.apply(self._init_weights)

    def _init_weights(self, module):
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
        output = self.bert(input_ids, attention_mask)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores  