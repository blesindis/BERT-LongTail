import torch
import torch.nn as nn
from einops import rearrange
from transformer.modules.common import Embeddings
from transformer.modules.attention import Attention
from transformer.modules.feedforward import FeedForward, SwitchFeedForward
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
    
    
class ExpertAttention(nn.Module):
    def __init__(self, config, centers):
        super().__init__()
        self.config = config
        self.n_experts = 2
        self.experts = nn.ModuleList([Attention(config) for i in range(self.n_experts)])
        self.centers = centers
        
    def routing(self, hidden_states):
        cluster_list = [[] for _ in range(self.n_experts)]
        
        h = hidden_states.mean(dim=1)
        
        dist = torch.cdist(h.double(), self.centers.double().detach())
        _, min_indices = torch.min(dist, dim=1)
        cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]      
        # for i, cluster_index in enumerate(min_indices):
        #     cluster_list[cluster_index.item()].append(i)
            
        return cluster_list
    
    def forward(self, hidden_states, attention_mask, routing_states):
        output = hidden_states.new_zeros(hidden_states.shape)
        cluster_list = self.routing(routing_states)
        for i in range(self.n_experts):                        
            output[cluster_list[i], :, :] = self.experts[i](hidden_states[cluster_list[i], :, :], attention_mask[cluster_list[i], :])
        return output
    
    
class MoMoShareLayer(nn.Module):
    def __init__(self, config, centers):
        super(MoMoShareLayer, self).__init__()
        self.config = config
        self.attention = ExpertAttention(config, centers)
        self.ffn = SwitchFeedForward(config)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask, routing_states):
        """Pre Norm"""
        # att_output = self.attention(hidden_states, attention_mask)
        # residual = att_output
        # att_output = self.LayerNorm(att_output)
        # ffn_output = self.ffn(att_output)
        # # ffn_output = self.dropout(ffn_output)
        
        # # output = self.LayerNorm(att_output + ffn_output)
        # output = residual + ffn_output
        """Post Norm"""
        att_output = self.attention(hidden_states, attention_mask, routing_states)
        ffn_output = self.ffn(att_output)
        ffn_output = self.dropout(ffn_output)
        output = self.LayerNorm(att_output + ffn_output)        
        
        return output


class BertMoMoShare(nn.Module):
    def __init__(self, config, centers):
        super(BertMoMoShare, self).__init__()
        self.layers = nn.ModuleList([MoMoShareLayer(config, centers[i]) for i in range(config.num_hidden_layers)])
        
    def forward(self, hidden_states, attention_mask, routing_states):
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask, routing_states[i])
        return hidden_states


class BertMoMoModelRouterModel(nn.Module):
    def __init__(self, config, centers):
        super(BertMoMoModelRouterModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertMoMoShare(config, centers)
        
    def forward(self, input_ids, attention_mask, routing_states):
        embeddings = self.embeddings(input_ids)
        outputs = self.layers(embeddings, attention_mask, routing_states)
        return outputs
    
    
class BertWithMoMoModelRouter(nn.Module):
    def __init__(self, config, centers):
        super(BertWithMoMoModelRouter, self).__init__()
        self.config = config        
        self.bert = BertMoMoModelRouterModel(config, centers)
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

    def forward(self, input_ids, attention_mask, labels, routing_states):
        output = self.bert(input_ids, attention_mask, routing_states)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores