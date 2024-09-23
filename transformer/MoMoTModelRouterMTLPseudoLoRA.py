import torch
import torch.nn as nn
from einops import rearrange
from transformer.modules.common import Embeddings
from transformer.modules.attention import Attention, LoRAAttention
from transformer.modules.feedforward import FeedForward, FeedForwardLoRA, SwitchFeedForward, SwitchFeedForwardLoRA, SwitchFeedForwardLoRALatent
from transformer.Switch import SwitchEncoder
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformer.BERT import TransformerEncoder


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
        if config.common_ffn:
            if config.common_ffn_lora:
                self.ffn = FeedForwardLoRA(config, lora_dim=config.common_ffn_lora_dim)
            else:
                self.ffn = FeedForward(config)
        else:
            if config.common_moe_lora:
                self.ffn = SwitchFeedForwardLoRA(config, lora_dim=config.common_moe_lora_dim, n_experts=4, drop_tokens=config.unique_moe_drop_tokens)
            else:
                self.ffn = SwitchFeedForward(config)
        
    def forward(self, hidden_states) -> torch.Tensor:
        ffn_output = self.ffn(hidden_states)
        return ffn_output
    
    
class ExpertFFNUnique(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.unique_moe_lora:
            self.ffn = SwitchFeedForwardLoRA(config, lora_dim=config.unique_moe_lora_dim, n_experts=4, drop_tokens=config.unique_moe_drop_tokens)
        else:
            self.ffn = SwitchFeedForward(config)
        
    def forward(self, hidden_states) -> torch.Tensor:
        ffn_output = self.ffn(hidden_states)
        return ffn_output
    
    
class MoMoShareLayer(nn.Module):
    def __init__(self, config):
        super(MoMoShareLayer, self).__init__()
        self.config = config
        self.n_experts = config.unique_experts
       
        self.common_attn = ExpertAttentionCommon(config)
        self.common_ffn = ExpertFFNCommon(config)
        self.unique_attn = nn.ModuleList([ExpertAttentionUnique(config) for _ in range(self.n_experts)])
        self.unique_ffn = nn.ModuleList([ExpertFFNUnique(config) for _ in range(self.n_experts)])
        
        self.warm_up_step = config.warm_up_step
        self.step = 0
        
        self.switch_size = 128 
        self.switch_encoder = nn.Linear(config.hidden_size, self.switch_size)
        self.switch = nn.Linear(self.switch_size, self.n_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
        self.loss = 0
       
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def _warm_router(self, hidden_states, cluster_list):
        label_list = [0 for _ in range(hidden_states.shape[0])]
        for label, c in enumerate(cluster_list):
            for index in c:
                label_list[index] = label
        label_tensor = torch.tensor(label_list).to('cuda')
        
        h_encoded = hidden_states.mean(dim=1)
        h_encoded = self.switch_encoder(h_encoded)
        
        route_prob = self.softmax(self.switch(h_encoded))
        self.loss = self.criterion(route_prob, label_tensor)
        
    def routing(self, hidden_states):
        cluster_list = [[] for _ in range(self.n_experts)]
        
        h_encoded = hidden_states.mean(dim=1)
        h_encoded = self.switch_encoder(h_encoded)
        
        route_prob = self.softmax(self.switch(h_encoded))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        cluster_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]      

        return cluster_list, route_prob_max
        
    def forward(self, hidden_states, attention_mask, cluster_list):
        self.step += 1
        if self.step <= self.warm_up_step:
            
            self._warm_router(hidden_states, cluster_list)
            
            common_attn_output = self.common_attn(hidden_states, attention_mask)
            unique_attn_output = hidden_states.new_zeros(hidden_states.shape)
            for i in range(self.n_experts):
                unique_attn_output[cluster_list[i], :, :] = self.unique_attn[i](hidden_states[cluster_list[i], :, :], attention_mask[cluster_list[i], :])
                
            att_output = common_attn_output + unique_attn_output
            
            unique_ffn_output = hidden_states.new_zeros(hidden_states.shape)
            for i in range(self.n_experts):
                unique_ffn_output[cluster_list[i], :, :] = self.unique_ffn[i](att_output[cluster_list[i], :, :])
            common_ffn_output = self.common_ffn(att_output)
            ffn_output = common_ffn_output + unique_ffn_output
            
            ffn_output = self.dropout(ffn_output)
            output = self.LayerNorm(att_output + ffn_output)
        else:
            self.loss = 0
            common_attn_output = self.common_attn(hidden_states, attention_mask)
            unique_attn_output = hidden_states.new_zeros(hidden_states.shape)
            router_cluster_list, route_prob_max = self.routing(hidden_states)            
            for i in range(self.n_experts):
                unique_attn_output[router_cluster_list[i], :, :] = self.unique_attn[i](hidden_states[router_cluster_list[i], :, :], attention_mask[router_cluster_list[i], :])
            
            scaling_factor = route_prob_max.unsqueeze(-1).unsqueeze(-1)
            scaling_factor_unique = scaling_factor.expand_as(hidden_states) 
            unique_attn_output *= scaling_factor_unique
        
            att_output = common_attn_output + unique_attn_output
            
            unique_ffn_output = hidden_states.new_zeros(hidden_states.shape)
            for i in range(self.n_experts):
                unique_ffn_output[router_cluster_list[i], :, :] = self.unique_ffn[i](att_output[router_cluster_list[i], :, :])
            common_ffn_output = self.common_ffn(att_output)
            ffn_output = common_ffn_output + unique_ffn_output
            
            ffn_output = self.dropout(ffn_output)
            output = self.LayerNorm(att_output + ffn_output)
        
        return output


class BertMoMoShare(nn.Module):
    def __init__(self, config):
        super(BertMoMoShare, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoder(config) for _ in range(config.num_common_layers)] +
            [MoMoShareLayer(config) for _ in range(config.num_hidden_layers - config.num_common_layers)]
        )
        # self.layers = nn.ModuleList([MoMoShareLayer(config) for i in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, cluster_list):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, TransformerEncoder):
                hidden_states = layer(hidden_states, attention_mask)
            elif isinstance(layer, MoMoShareLayer):
                hidden_states = layer(hidden_states, attention_mask, cluster_list)
            else:
                raise ModuleNotFoundError
        return hidden_states


class BertMoMoTModelRouterMTLPseudoLoRAModel(nn.Module):
    def __init__(self, config):
        super(BertMoMoTModelRouterMTLPseudoLoRAModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertMoMoShare(config)
        
    def forward(self, input_ids, attention_mask, cluster_list):
        embeddings = self.embeddings(input_ids)
        outputs = self.layers(embeddings, attention_mask, cluster_list)
        return outputs
    
    
class BertWithMoMoTModelRouterMTLPseudoLoRA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config        
        self.bert = BertMoMoTModelRouterMTLPseudoLoRAModel(config)
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

    def forward(self, input_ids, attention_mask, labels, cluster_list):
        output = self.bert(input_ids, attention_mask, cluster_list)
        scores = self.head(output)
        mlm_loss = self.criterion(scores.view(-1, self.config.vocab_size), labels.view(-1)) # scores should be of size (num_words, vocab_size)

        return mlm_loss, scores