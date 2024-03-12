import torch
import torch.nn as nn
from einops import rearrange
from transformer.modules.transformer import Embeddings, Attention, FeedForward
from transformer.modules.switch_transformer import SwitchFeedForward
from transformer.modules.lora_new import LoRAAttention
from utils.train_utils import copy_parameters
    
    
class ExpertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = 1
        self.formed_experts = 0
        self.experts = nn.ModuleList([Attention(config) for i in range(self.n_experts)])
        
        self.warmup = True
        self.finish_warmup = 3000
        self.step = 0
        self.prepare = 5
        
        self.update = False
        self.update_interval = 2000
        
        self.distributions = []
        self.data_for_update = []
        self.min_data_for_new_dist = 500
    
    def _zscore(self, data, mean, std):
        z_score = (data - mean) / std
        return z_score
    
    def _check_fit(self, distribution, data):
        zscore = self._zscore(data, distribution['mean'], distribution['std'])
        if torch.mean(abs(zscore)) < 2:
            return True
        return False
    
    def _distribution_updates(self):
        for i in range(self.formed_experts):
            dist = self.distributions[i]
            if len(dist['new_data']):            
                # for d in dist['new_data']:
                #     dist['N'] += 1
                #     delta = d - dist['mean']
                #     dist['mean'] += delta / dist['N']
                #     delta2 = d - dist['mean']
                #     M2 = dist['std']**2 * (dist['N'] - 1)
                #     M2 += delta * delta2
                #     dist['std'] = (M2 / (dist['N'] - 1)) ** 0.5
                data = torch.cat(dist['new_data'], dim=0)
                new_count = len(data)
                total_count = dist['N'] + new_count
                
                new_mean = ((dist['mean'] * dist['N']) + torch.sum(data, dim=0)) / total_count
                old_sum_of_squares = dist['std']**2 * (dist['N'] - 1)
                new_sum_of_squares = torch.sum((data - new_mean.reshape(1, -1))**2, dim=0)
                
                new_variance = (old_sum_of_squares + new_sum_of_squares) / (total_count - 1)
                new_std = torch.sqrt(new_variance)
                
                dist['mean'] = new_mean
                dist['std'] = new_std
                dist['N'] = total_count
                                   
    def _form_distribution(self):
        self.formed_experts += 1
        self.n_experts += 1
        # self.experts.append(LoRAAttention(self.config, lora_dim=128).to('cuda'))
        self.experts.append(Attention(self.config).to('cuda'))
        # if self.formed_experts > 1:
        #     copy_parameters(self.experts[0], self.experts[-1])
        
        data = torch.cat(self.data_for_update, dim=0)
        mean = data.mean(dim=0)
        std = data.std(dim=0)
        self.distributions.append({'mean':mean, 'std':std, 'N': len(self.data_for_update), 'new_data': []})
        self.data_for_update = []
                
        
    def routing(self, hidden_states):
        cluster_list = [[] for _ in range(self.n_experts)]
        
        h = hidden_states.mean(dim=1)        
        for ind_data in range(len(h)):            
            for i in range(self.formed_experts):
                if self._check_fit(self.distributions[i], h[ind_data]):
                    cluster_list[i].append(ind_data)
                    
                    if self.update:
                        self.distributions[i]['new_data'].append(h[ind_data].reshape(1,-1).detach())     
                    
                    break
            if i == self.formed_experts:
                cluster_list[-1].append(ind_data)
                self.data_for_update.append(h[ind_data].reshape(1,-1).detach())
        
        if self.update and self.step % self.update_interval == 0:
            self._distribution_updates()   
        if len(self.data_for_update) > self.min_data_for_new_dist:
            self._form_distribution()
        
        return cluster_list
        
    
    def forward(self, hidden_states, attention_mask):
        output = hidden_states.new_zeros(hidden_states.shape)
        self.step += 1
        if self.warmup:
            output = self.experts[0](hidden_states, attention_mask)
            # for i in range(self.n_experts):                        
            #     output += self.experts[i](hidden_states, attention_mask)
        else:
            cluster_list = self.routing(hidden_states)
            for i in range(self.n_experts):                        
                output[cluster_list[i], :, :] = self.experts[i](hidden_states[cluster_list[i], :, :], attention_mask[cluster_list[i], :])
        
        # Form Initial Distribution
        if self.warmup:
            if self.step > self.finish_warmup - self.prepare:
                self.data_for_update.append(hidden_states.mean(dim=1).detach())
            if self.step > self.finish_warmup:
                self.warmup = False
                self._form_distribution()                
                
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


class BertMoMoDynamicDistributionModel(nn.Module):
    def __init__(self, config):
        super(BertMoMoDynamicDistributionModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertMoMoShare(config)
        
    def forward(self, input_ids, attention_mask):
        embeddings = self.embeddings(input_ids)
        outputs = self.layers(embeddings, attention_mask)
        return outputs