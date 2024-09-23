import torch
import torch.nn as nn
from einops import rearrange
from transformer.modules.common import Embeddings
from transformer.modules.attention import Attention
from transformer.modules.feedforward import FeedForward, SwitchFeedForward
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class SwitchAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = 2
        self.experts = nn.ModuleList([Attention(config) for _ in range(self.n_experts)])
        self.switch = nn.Linear(config.hidden_size, self.n_experts)
        self.softmax = nn.Softmax(dim=-1)
        
    def routing(self, x):
        output = x.new_zeros(x.shape)        
        
        route_prob = self.softmax(self.switch(x.mean(dim=1)))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]      
        return indexes_list
        
    def forward(self, x, attention_mask):
        output = x.new_zeros(x.shape)        
        
        route_prob = self.softmax(self.switch(x.mean(dim=1)))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]      
        
        expert_output = [self.experts[i](x[indexes_list[i], :, :], attention_mask[indexes_list[i], :]) for i in range(self.n_experts)]     
        for i in range(self.n_experts):
            output[indexes_list[i], :, :] = expert_output[i]
        
        scaling_factor = (route_prob_max / route_prob_max.detach()).unsqueeze(-1).unsqueeze(-1)
        scaling_factor = scaling_factor.expand_as(x) 
        output = output * scaling_factor
        return output


class MoMoShareLayer(nn.Module):
    def __init__(self, config):
        super(MoMoShareLayer, self).__init__()
        self.config = config
        self.attention = SwitchAttention(config)
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


class BertMoMoShare(nn.Module):
    def __init__(self, config):
        super(BertMoMoShare, self).__init__()
        self.layers = nn.ModuleList([MoMoShareLayer(config) for i in range(config.num_hidden_layers)])
        
    def forward(self, hidden_states, attention_mask):
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertMoMoSwitchModel(nn.Module):
    def __init__(self, config):
        super(BertMoMoSwitchModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertMoMoShare(config)
        
    def forward(self, input_ids, attention_mask):
        embeddings = self.embeddings(input_ids)
        outputs = self.layers(embeddings, attention_mask)
        return outputs
    
    
class BertWithMoMoSwitch(nn.Module):
    def __init__(self, config):
        super(BertWithMoMoSwitch, self).__init__()
        self.config = config        
        self.bert = BertMoMoSwitchModel(config)
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