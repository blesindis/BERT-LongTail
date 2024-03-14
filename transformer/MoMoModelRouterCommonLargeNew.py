import torch
import torch.nn as nn
from einops import rearrange
from transformer.modules.common import Embeddings
from transformer.modules.attention import Attention, LoRAAttention
from transformer.modules.feedforward import FeedForward, SwitchFeedForward
from transformer.Switch import SwitchEncoder
    
    
class ExpertAttention(nn.Module):
    def __init__(self, config, centers):
        super().__init__()
        self.config = config
        self.n_experts = 2
        self.experts = nn.ModuleList([LoRAAttention(config, lora_dim=128) for i in range(self.n_experts)])
        self.common_expert = Attention(config)
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
        unique_output = hidden_states.new_zeros(hidden_states.shape)
        cluster_list = self.routing(routing_states)
        for i in range(self.n_experts):                        
            unique_output[cluster_list[i], :, :] = self.experts[i](hidden_states[cluster_list[i], :, :], attention_mask[cluster_list[i], :])
        common_output = self.common_expert(hidden_states, attention_mask)
        
        output = unique_output + common_output
        # output = 0.8 * unique_output + 0.2 * common_output
        return output
    
    
class MoMoShareLayer(nn.Module):
    def __init__(self, config, centers):
        super(MoMoShareLayer, self).__init__()
        self.config = config
        self.attention = ExpertAttention(config, centers)
        self.ffn = SwitchFeedForward(config)
        # self.ffn = FeedForward(config)

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
        # self.layers = nn.ModuleList(
        #     [SwitchEncoder(config) for _ in range(7)]
        #     + [MoMoShareLayer(config, centers[i]) for i in range(7,9)]
        #     + [SwitchEncoder(config) for _ in range(9, 12)]
        # )
        
    def forward(self, hidden_states, attention_mask, routing_states):
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask, routing_states[i])
            # if i == 7 or i == 8:
            #     hidden_states = layer(hidden_states, attention_mask, routing_states[i])
            # else:
            #     hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertMoMoModelRouterCommonAttnLargeNewModel(nn.Module):
    def __init__(self, config, centers):
        super(BertMoMoModelRouterCommonAttnLargeNewModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertMoMoShare(config, centers)
        
    def forward(self, input_ids, attention_mask, routing_states):
        embeddings = self.embeddings(input_ids)
        outputs = self.layers(embeddings, attention_mask, routing_states)
        return outputs