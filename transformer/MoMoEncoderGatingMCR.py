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
        
        self.gam1 = 1.0
        self.gam2 = 1.0
        self.eps = 0.01
        
        self.steps = 0
        self.finish_step = 3000
        
    def routing(self, hidden_states):
        cluster_list = [[] for _ in range(self.n_experts)]
        
        batch_size, seq_len, d_model = hidden_states.shape
        """1"""
        # h_encoded = self.encoder(hidden_states.reshape(batch_size, seq_len * d_model))
        """2"""
        mean_hidden_states = hidden_states.mean(dim=1)
        h_encoded = self.encoder(mean_hidden_states)
        """3"""
        # h_encoded = self.encoder(hidden_states.reshape(batch_size, seq_len * d_model))
        
        route_prob = self.softmax(self.switch(h_encoded))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        
        cluster_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]      
            
        return cluster_list, route_prob_max
    
    def _discrimn_loss_empirical(self, hidden_states):
        """
        Args:
            hidden_states (tensor): should only contain 2 dim
        """
        p, m = hidden_states.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * hidden_states.matmul(hidden_states.T))
        return logdet / 2.
    
    def _compress_loss_empirical(self, cluster_list, hidden_states):
        """
        Args:
            cluster_list (list): cluster_list[i] contain indexes of tensors routed to the i th expert
            hidden_states (tensor):should only contain 2 dim 
        """
        p, m = hidden_states.shape
        compress_loss = 0
        for j in range(self.n_experts):
            num = len(cluster_list[j]) + 1e-8
            scalar = p / (num * self.eps)
            I = torch.eye(len(cluster_list[j])).cuda()
            log_det = torch.logdet(I + scalar * hidden_states[cluster_list[j]].matmul(hidden_states[cluster_list[j]].T))
            compress_loss += log_det * num / m
        return compress_loss / 2
    
    def forward(self, hidden_states, attention_mask):
        mcr_loss = 0
        self.steps += 1
        unique_output = hidden_states.new_zeros(hidden_states.shape)
        
        cluster_list, route_prob_max = self.routing(hidden_states)
        
        for i in range(self.n_experts):                        
            unique_output[cluster_list[i], :, :] = self.experts[i](hidden_states[cluster_list[i], :, :], attention_mask[cluster_list[i], :])
        
        scaling_factor = (route_prob_max / route_prob_max.detach()).unsqueeze(-1).unsqueeze(-1)
        scaling_factor = scaling_factor.expand_as(hidden_states) 
        unique_output = unique_output * scaling_factor    
        common_output = self.common_expert(hidden_states, attention_mask)
        
        if self.steps > self.finish_step:
            output_for_loss = torch.cat((unique_output, common_output), dim=0).mean(dim=1)
            cluster_list.append([j+len(unique_output) for j in range(len(unique_output))])
            
            discrimn_loss_empi = self._discrimn_loss_empirical(output_for_loss)
            compress_loss_empi = self._compress_loss_empirical(cluster_list, output_for_loss)
        
            mcr_loss = self.gam2 * -discrimn_loss_empi + compress_loss_empi
        
        output = common_output + unique_output
        return output, mcr_loss
    

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
        att_output, mcr_loss = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(att_output)
        ffn_output = self.dropout(ffn_output)
        output = self.LayerNorm(att_output + ffn_output)        
        
        return output, mcr_loss


class BertMoMoShare(nn.Module):
    def __init__(self, config):
        super(BertMoMoShare, self).__init__()
        self.layers = nn.ModuleList([MoMoShareLayer(config) for i in range(config.num_hidden_layers)])
        
    def forward(self, hidden_states, attention_mask):
        mcr_loss = 0.0
        for i, layer in enumerate(self.layers):
            hidden_states, loss = layer(hidden_states, attention_mask)
            mcr_loss += loss
        return hidden_states, mcr_loss


class BertMoMoEncoderGatingMCRModel(nn.Module):
    def __init__(self, config):
        super(BertMoMoEncoderGatingMCRModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertMoMoShare(config)
        
    def forward(self, input_ids, attention_mask):
        embeddings = self.embeddings(input_ids)
        outputs, mcr_loss = self.layers(embeddings, attention_mask)
        return outputs, mcr_loss