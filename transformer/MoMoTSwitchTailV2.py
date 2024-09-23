import torch
import torch.nn as nn
from einops import rearrange
from transformer.modules.common import Embeddings
from transformer.modules.attention import Attention, LoRAAttention
from transformer.modules.feedforward import FeedForward, FeedForwardLoRA, SwitchFeedForward, SwitchFeedForwardLoRA, SwitchFeedForwardLoRALatent
from transformer.Switch import SwitchEncoder
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformer.BERT import TransformerEncoder


class TailAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_common_experts = config.common_att_experts
        self.common_attn = [ExpertAttentionCommon(config).to('cuda') for _ in range(self.n_common_experts)]
        self.switch_common = nn.Linear(config.hidden_size, self.n_common_experts)
        self.capacity_common = config.common_att_capacity
        
        self.n_unique_experts = config.unique_att_experts
        self.unique_attn = [ExpertAttentionUnique(config).to('cuda') for _ in range(self.n_unique_experts)]
        self.switch_unique = nn.Linear(config.hidden_size, self.n_unique_experts)

        self.softmax = nn.Softmax(dim=-1)
    
    def routing_common(self, hidden_states):
        cluster_list = [[] for _ in range(self.n_common_experts)]
        
        h_encoded = hidden_states.mean(dim=1)
        route_prob = self.softmax(self.switch_common(h_encoded))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        cluster_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_common_experts)]              
        
        # priority drop
        dropped_list = []
        capacity = int(self.capacity_common * len(h_encoded) / self.n_common_experts)
        for i in range(self.n_common_experts):
            if len(cluster_list[i]) > capacity:
                prob_i = route_prob_max[cluster_list[i]]
                _, sorted_indices = torch.sort(prob_i, descending=True)                
                cluster_list[i] = cluster_list[i][sorted_indices]
                
                dropped_list.append(cluster_list[i][capacity:])
                # cluster_list[i] = cluster_list[i][:capacity]
        if len(dropped_list):
            dropped_list = torch.cat(dropped_list)
                
        return cluster_list, route_prob_max, dropped_list
    
    def routing_unique(self, hidden_states):
        cluster_list = [[] for _ in range(self.n_unique_experts)]
        
        h_encoded = hidden_states.mean(dim=1)
        route_prob = self.softmax(self.switch_unique(h_encoded))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        cluster_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_unique_experts)]              
                
        return cluster_list, route_prob_max
    
    def _forward_common(self, hidden_states, attention_mask, cluster_list, route_prob_max):
        scaling_factor = (route_prob_max / route_prob_max.detach()).unsqueeze(-1).unsqueeze(-1).expand_as(hidden_states)      
        final_output = hidden_states.new_zeros(hidden_states.shape)
        for i in range(self.n_common_experts):
            final_output[cluster_list[i],:,:] = self.common_attn[i](hidden_states[cluster_list[i],:,:], attention_mask[cluster_list[i],:])
        final_output = final_output * scaling_factor
        return final_output
    
    def _forward_unique(self, hidden_states, attention_mask, cluster_list, route_prob_max):
        scaling_factor = (route_prob_max / route_prob_max.detach()).unsqueeze(-1).unsqueeze(-1).expand_as(hidden_states)      
        final_output = hidden_states.new_zeros(hidden_states.shape)
        for i in range(self.n_unique_experts):
            final_output[cluster_list[i],:,:] = self.unique_attn[i](hidden_states[cluster_list[i],:,:], attention_mask[cluster_list[i],:])
        final_output = final_output * scaling_factor
        return final_output
    
    def forward(self, hidden_states, attention_mask):
        final_output = hidden_states.new_zeros(hidden_states.shape)
        
        cluster_list_common, route_prob_max_common, dropped_list = self.routing_common(hidden_states)
        # if len(dropped_list):
        #     route_prob_max_common[dropped_list] = 1.0 - route_prob_max_common[dropped_list]
        common_output = self._forward_common(hidden_states, attention_mask, cluster_list_common, route_prob_max_common)
        
        if len(dropped_list):
            cluster_list_unique, route_prob_max_unique = self.routing_unique(hidden_states[dropped_list,:,:])
            unique_output = self._forward_unique(hidden_states[dropped_list,:,:], attention_mask[dropped_list,:], cluster_list_unique, route_prob_max_unique)
            final_output[dropped_list,:,:] = unique_output
            
            final_output += common_output
        else:
            final_output = common_output
        
        return final_output        
        
class TailFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_common_experts = config.common_ffn_experts
        self.common_ffn = [ExpertFFNCommon(config).to('cuda') for _ in range(self.n_common_experts)]
        self.switch_common = nn.Linear(config.hidden_size, self.n_common_experts)
        self.capacity_common = config.common_ffn_capacity
        
        self.n_unique_experts = config.unique_ffn_experts
        self.unique_ffn = [ExpertFFNUnique(config).to('cuda') for _ in range(self.n_unique_experts)]
        self.switch_unique = nn.Linear(config.hidden_size, self.n_unique_experts)

        self.softmax = nn.Softmax(dim=-1)
    
    def routing_common(self, hidden_states):
        cluster_list = [[] for _ in range(self.n_common_experts)]
        
        h_encoded = hidden_states
        route_prob = self.softmax(self.switch_common(h_encoded))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        cluster_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_common_experts)]              
        
        # priority drop
        dropped_list = []
        capacity = int(self.capacity_common * len(h_encoded) / self.n_common_experts)
        for i in range(self.n_common_experts):
            if len(cluster_list[i]) > capacity:
                prob_i = route_prob_max[cluster_list[i]]
                _, sorted_indices = torch.sort(prob_i, descending=True)                
                cluster_list[i] = cluster_list[i][sorted_indices]
                
                dropped_list.append(cluster_list[i][capacity:])
                # cluster_list[i] = cluster_list[i][:capacity]
        if len(dropped_list):
            dropped_list = torch.cat(dropped_list)
                
        return cluster_list, route_prob_max, dropped_list
    
    def routing_unique(self, hidden_states):
        cluster_list = [[] for _ in range(self.n_unique_experts)]
        
        h_encoded = hidden_states
        route_prob = self.softmax(self.switch_unique(h_encoded))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        cluster_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_unique_experts)]              
                
        return cluster_list, route_prob_max
    
    def _forward_common(self, hidden_states, cluster_list, route_prob_max):
        scaling_factor = (route_prob_max / route_prob_max.detach()).unsqueeze(-1).expand_as(hidden_states)      
        final_output = hidden_states.new_zeros(hidden_states.shape)
        for i in range(self.n_common_experts):
            final_output[cluster_list[i],:] = self.common_ffn[i](hidden_states[cluster_list[i],:])
        final_output = final_output * scaling_factor
        return final_output
    
    def _forward_unique(self, hidden_states, cluster_list, route_prob_max):
        scaling_factor = (route_prob_max / route_prob_max.detach()).unsqueeze(-1).expand_as(hidden_states)      
        final_output = hidden_states.new_zeros(hidden_states.shape)
        for i in range(self.n_unique_experts):
            final_output[cluster_list[i],:] = self.unique_ffn[i](hidden_states[cluster_list[i],:])
        final_output = final_output * scaling_factor
        return final_output
    
    def forward(self, hidden_states):
        h_ = hidden_states.reshape(-1, hidden_states.shape[-1])
        final_output = h_.new_zeros(h_.shape)
        
        cluster_list_common, route_prob_max_common, dropped_list = self.routing_common(h_)
        # if len(dropped_list):
        #     route_prob_max_common[dropped_list] = 1.0 - route_prob_max_common[dropped_list]
        common_output = self._forward_common(h_, cluster_list_common, route_prob_max_common)
        
        if len(dropped_list):
            cluster_list_unique, route_prob_max_unique = self.routing_unique(h_[dropped_list,:])
            unique_output = self._forward_unique(h_[dropped_list,:], cluster_list_unique, route_prob_max_unique)
            final_output[dropped_list,:] = unique_output
            
            final_output += common_output
        else:
            final_output = common_output
        final_output = final_output.view(hidden_states.shape)
        
        return final_output         

class ExpertAttentionCommon(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.common_att_lora:
            self.attention = LoRAAttention(config, lora_dim=config.common_att_lora_dim)
        else:
            self.attention = Attention(config)
        
    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        att_output = self.attention(hidden_states, attention_mask)
        return att_output
    
    
class ExpertAttentionUnique(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.unique_att_lora:            
            self.attention = LoRAAttention(config, lora_dim=config.unique_att_lora_dim)
        else:
            self.attention = Attention(config)
        
    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        att_output = self.attention(hidden_states, attention_mask)
        return att_output
    
    
class ExpertFFNCommon(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.common_ffn_lora:
            self.ffn = FeedForwardLoRA(config, lora_dim=config.common_ffn_lora_dim)
        else:
            self.ffn = FeedForward(config)
        
    def forward(self, hidden_states) -> torch.Tensor:
        ffn_output = self.ffn(hidden_states)
        return ffn_output
    
    
class ExpertFFNUnique(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.unique_ffn_lora:
            self.ffn = FeedForwardLoRA(config, lora_dim=config.unique_ffn_lora_dim)
        else:
            self.ffn = FeedForward(config)
        
    def forward(self, hidden_states) -> torch.Tensor:
        ffn_output = self.ffn(hidden_states)
        return ffn_output
    
    
class TailLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = TailAttention(config)
        self.ffn = TailFeedForward(config)                
        # self.ffn = SwitchFeedForward(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)        
        
    def forward(self, hidden_states, attention_mask):
        att_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(att_output)
        ffn_output = self.dropout(ffn_output)
        output = self.LayerNorm(att_output + ffn_output)
        
        return output
    

class BertMoMoShare(nn.Module):
    def __init__(self, config):
        super(BertMoMoShare, self).__init__()
        # self.layers = nn.ModuleList(
        #     [TransformerEncoder(config) for _ in range(config.num_common_layers)] +
        #     [TailLayer(config) for _ in range(config.num_hidden_layers - config.num_common_layers)]
        # )
        
        layers = []
        for i in range(config.num_hidden_layers // 2):
            layers += [TransformerEncoder(config), TailLayer(config)]
        self.layers = nn.ModuleList(layers)
        
    def forward(self, hidden_states, attention_mask):
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertMoMoTSwitchTailV2Model(nn.Module):
    def __init__(self, config):
        super(BertMoMoTSwitchTailV2Model, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertMoMoShare(config)
        
    def forward(self, input_ids, attention_mask):
        embeddings = self.embeddings(input_ids)
        outputs = self.layers(embeddings, attention_mask)
        return outputs
    
    
class BertWithMoMoTSwitchTailV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config        
        self.bert = BertMoMoTSwitchTailV2Model(config)
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