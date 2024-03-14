import torch
import torch.nn as nn
from einops import rearrange
from transformer.modules.common import Embeddings
from transformer.modules.attention import Attention, LoRAAttention
from transformer.modules.feedforward import FeedForward, SwitchFeedForward, SwitchFeedForwardLoRA, SwitchFeedForwardLoRALatent
from transformer.Switch import SwitchEncoder
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
    
    
class ExpertTransformerCommon(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.ffn = SwitchFeedForwardLoRALatent(config, lora_dim=128, n_experts=4)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        """Pre Norm No Dropout"""
        # att_output = self.attention(hidden_states, attention_mask)
        # residual = att_output
        # att_output = self.LayerNorm(att_output)
        # ffn_output = self.ffn(att_output)
        # output = residual + ffn_output
        """Post Norm"""
        att_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(att_output)
        ffn_output = self.dropout(ffn_output)
        output = self.LayerNorm(att_output + ffn_output)
        return output
    
    
class ExpertTransformerUnique(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = LoRAAttention(config, lora_dim=128)
        self.ffn = SwitchFeedForwardLoRALatent(config, lora_dim=128, n_experts=4)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        """Pre Norm No Dropout"""
        # att_output = self.attention(hidden_states, attention_mask)
        # residual = att_output
        # att_output = self.LayerNorm(att_output)
        # ffn_output = self.ffn(att_output)
        # output = residual + ffn_output
        """Post Norm"""
        att_output = self.attention(hidden_states, attention_mask)
        ffn_output = self.ffn(att_output)
        ffn_output = self.dropout(ffn_output)
        output = self.LayerNorm(att_output + ffn_output)
        return output
    
    
class MoMoShareLayer(nn.Module):
    def __init__(self, config):
        super(MoMoShareLayer, self).__init__()
        self.config = config
        self.lora_dim = 128
        self.n_experts = 2
        self.common_expert = ExpertTransformerCommon(config)
        self.unique_experts = nn.ModuleList([ExpertTransformerUnique(config) for _ in range(self.n_experts)])
        
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
        unique_output = hidden_states.new_zeros(hidden_states.shape)
        
        cluster_list, route_prob_max = self.routing(hidden_states)
        
        for i in range(self.n_experts):                        
            unique_output[cluster_list[i], :, :] = self.unique_experts[i](hidden_states[cluster_list[i], :, :], attention_mask[cluster_list[i], :])
        scaling_factor = (route_prob_max / route_prob_max.detach()).unsqueeze(-1).unsqueeze(-1)
        scaling_factor = scaling_factor.expand_as(hidden_states) 
        unique_output = unique_output * scaling_factor
        
        common_output = self.common_expert(hidden_states, attention_mask)
        
        output = unique_output + common_output 
        
        return output


class BertMoMoShare(nn.Module):
    def __init__(self, config):
        super(BertMoMoShare, self).__init__()
        self.layers = nn.ModuleList([MoMoShareLayer(config) for i in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertMoMoTSwitchModel(nn.Module):
    def __init__(self, config):
        super(BertMoMoTSwitchModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertMoMoShare(config)
        
    def forward(self, input_ids, attention_mask):
        embeddings = self.embeddings(input_ids)
        outputs = self.layers(embeddings, attention_mask)
        return outputs
    
    
class BertWithMoMoTSwitch(nn.Module):
    def __init__(self, config):
        super(BertWithMoMoTSwitch, self).__init__()
        self.config = config        
        self.bert = BertMoMoTSwitchModel(config)
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