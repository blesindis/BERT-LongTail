import torch
import torch.nn as nn
from einops import rearrange
from transformer.modules.common import Embeddings
from transformer.modules.attention import Attention
from transformer.modules.feedforward import FeedForward, SwitchFeedForward
from utils.sample_utils import cluster_kmeans
    
    
class ExpertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = 2
        self.experts = nn.ModuleList([Attention(config) for i in range(self.n_experts)])
        self.warmup = True
        self.step = 0
        self.prepare = 5
        self.finish_warmup = 5000
        self.centers = None
        self.data = []
        
        
    def routing(self, hidden_states):
        cluster_list = [[] for _ in range(self.n_experts)]
        
        h = hidden_states.mean(dim=1)        
        dist = torch.cdist(h.double(), self.centers.double().detach())
        _, min_indices = torch.min(dist, dim=1)
        cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]      
        # for i, cluster_index in enumerate(min_indices):
        #     cluster_list[cluster_index.item()].append(i)
            
        return cluster_list
    
    
    def update_centers(self, hidden_states, cluster_list):
        pass
    
    def forward(self, hidden_states, attention_mask):
        output = hidden_states.new_zeros(hidden_states.shape)
        
        if self.warmup:
            for i in range(self.n_experts):                        
                output += self.experts[i](hidden_states, attention_mask)
        else:
            cluster_list = self.routing(hidden_states)
            for i in range(self.n_experts):                        
                output[cluster_list[i], :, :] = self.experts[i](hidden_states[cluster_list[i], :, :], attention_mask[cluster_list[i], :])
        
        # Form Initial Centers
        if self.warmup:
            self.step += 1
            if self.step > self.finish_warmup - self.prepare:
                self.data.append(hidden_states.detach())
            if self.step > self.finish_warmup:
                self.warmup = False
                data = torch.cat(self.data, dim=0).to('cpu')
                _, centers = cluster_kmeans(data.mean(dim=1), self.n_experts)
                self.centers = torch.tensor(centers).to('cuda')
                
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


class BertMoMoDynamicCenterModel(nn.Module):
    def __init__(self, config):
        super(BertMoMoDynamicCenterModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertMoMoShare(config)
        
    def forward(self, input_ids, attention_mask):
        embeddings = self.embeddings(input_ids)
        outputs = self.layers(embeddings, attention_mask)
        return outputs