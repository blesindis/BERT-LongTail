import torch
import torch.nn as nn
from einops import rearrange
from transformer.modules.common import Embeddings
from transformer.modules.attention import Attention, LoRAAttention
from transformer.modules.feedforward import FeedForward, SwitchFeedForward


class ExpertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = 2
        self.lora_dim = 128
        
        self.experts = nn.ModuleList([LoRAAttention(config, lora_dim=self.lora_dim) for i in range(self.n_experts)])
        self.common_expert = Attention(config)
        
        """1"""
        # self.encoder = nn.Linear(config.hidden_size * config.seq_len, self.lora_dim)
        """2"""
        self.encoder = nn.Linear(config.hidden_size, self.lora_dim)
        """3"""
        # self.encoder = nn.Sequential(
        #     nn.Linear(config.hidden_size * config.seq_len, config.hidden_size),
        #     nn.Linear(config.hidden_size, self.lora_dim)
        # )
        self.switch = nn.Linear(self.lora_dim, self.n_experts)
        self.softmax = nn.Softmax(dim=-1)
        
    def routing(self, hidden_states):
        cluster_list = [[] for _ in range(self.n_experts)]
        
        batch_size, seq_len, d_model = hidden_states.shape
        """1"""
        # h_encoded = self.encoder(hidden_states.reshape(batch_size, seq_len * d_model))
        """2"""
        h_encoded = self.encoder(hidden_states.mean(dim=1))
        """3"""
        # h_encoded = self.encoder(hidden_states.reshape(batch_size, seq_len * d_model))
        
        route_prob = self.softmax(self.switch(h_encoded))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        
        cluster_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]      
            
        return cluster_list, route_prob_max
    
    def forward(self, hidden_states, attention_mask):
        output = hidden_states.new_zeros(hidden_states.shape)
        
        cluster_list, route_prob_max = self.routing(hidden_states)
        
        for i in range(self.n_experts):                        
            output[cluster_list[i], :, :] = self.experts[i](hidden_states[cluster_list[i], :, :], attention_mask[cluster_list[i], :])
        
        scaling_factor = (route_prob_max / route_prob_max.detach()).unsqueeze(-1).unsqueeze(-1)
        scaling_factor = scaling_factor.expand_as(hidden_states) 
        output = output * scaling_factor    
        
        common_output = self.common_expert(hidden_states, attention_mask)
        output += common_output
        return output
    

class MoMoShareLayer(nn.Module):
    def __init__(self, config):
        super(MoMoShareLayer, self).__init__()
        self.config = config
        self.attention = ExpertAttention(config)
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


class BertMoMoEncoderGatingModel(nn.Module):
    def __init__(self, config):
        super(BertMoMoEncoderGatingModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertMoMoShare(config)
        
    def forward(self, input_ids, attention_mask):
        embeddings = self.embeddings(input_ids)
        outputs = self.layers(embeddings, attention_mask)
        return outputs