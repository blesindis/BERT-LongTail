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
            self.ffn = SwitchFeedForwardLoRA(config, lora_dim=config.unique_moe_lora_dim, n_experts=2, drop_tokens=config.unique_moe_drop_tokens)
        else:
            self.ffn = SwitchFeedForward(config)
        
    def forward(self, hidden_states) -> torch.Tensor:
        ffn_output = self.ffn(hidden_states)
        return ffn_output
    
    
class TailLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = config.unique_experts + 1
        
        # Attention
        self.common_attn = ExpertAttentionCommon(config)
        self.unique_attn = [ExpertAttentionUnique(config) for _ in range(config.unique_experts)]
        self.attention = nn.ModuleList([self.common_attn] + self.unique_attn)
        
        self.switch = nn.Linear(config.hidden_size, self.n_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.first_expert_bias = nn.Parameter(torch.zeros(self.n_experts))
        self.first_expert_bias.data[0] = 0.3  # Bias the first expert
        
        
        # FFN
        self.common_ffn = ExpertFFNCommon(config)
        self.unique_ffn = [ExpertFFNUnique(config) for _ in range(config.unique_experts)]
        self.ffn = nn.ModuleList([self.common_ffn] + self.unique_ffn)
                
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)        
        
    def routing(self, hidden_states):
        cluster_list = [[] for _ in range(self.n_experts)]
        
        batch_size, seq_len, d_model = hidden_states.shape
        """1"""
        # h_encoded = self.encoder(hidden_states.reshape(batch_size, seq_len * d_model))
        """2"""
        # h_encoded = self.encoder(hidden_states.mean(dim=1))
        """3"""
        # h_encoded = self.encoder(hidden_states.reshape(batch_size, seq_len * d_model))
        
        h_encoded = hidden_states.mean(dim=1)
        
        logits = self.switch(h_encoded)
        
        logits = self.softmax(logits)
        logits = logits + self.first_expert_bias        
        
        route_prob = self.softmax(logits)
        
        # print(route_prob)
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        override_condition = route_prob_max < 0.5
        if override_condition.any():
            # Update route_prob_max and routes based on the condition
            route_prob_max = torch.where(override_condition, route_prob[:, 0], route_prob_max)
            routes = torch.where(override_condition, torch.zeros_like(routes), routes)
        
        # print(route_prob_max, routes)
        cluster_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]      
        # counts = h_encoded.new_tensor([len(cluster_list[i]) for i in range(self.n_experts)])
        # self.loss = self.loss_coef * self.load_balance_loss(counts, route_prob)
        return cluster_list, route_prob_max
    
    def forward(self, hidden_states, attention_mask):
        cluster_list, route_prob_max = self.routing(hidden_states)
        if len(cluster_list[0]):
            route_common = route_prob_max[cluster_list[0]]
            scaling_common = (route_common / route_common.detach()).unsqueeze(-1).unsqueeze(-1).expand_as(hidden_states[cluster_list[0],:,:])        
        
        # ATTN
        attn_output = hidden_states.new_zeros(hidden_states.shape)
        common_attn_output = self.attention[0](hidden_states, attention_mask)
        
        if len(cluster_list[0]):
            attn_output[cluster_list[0],:,:] = common_attn_output[cluster_list[0],:,:] * scaling_common
        
        for i in range(1, self.n_experts):
            if len(cluster_list[i]):
                unique_attn_output = self.attention[i](hidden_states[cluster_list[i],:,:], attention_mask[cluster_list[i],:])
                prob = route_prob_max[cluster_list[i]].unsqueeze(-1).unsqueeze(-1).expand_as(unique_attn_output)
                attn_output[cluster_list[i],:,:] = prob * unique_attn_output + (1 - prob) * common_attn_output[cluster_list[i],:,:]
        
        # FFN
        ffn_output = hidden_states.new_zeros(hidden_states.shape)
        common_ffn_output = self.ffn[0](attn_output)
        
        if len(cluster_list[0]):
            ffn_output[cluster_list[0],:,:] = common_ffn_output[cluster_list[0],:,:] * scaling_common
        
        for i in range(1, self.n_experts):
            if len(cluster_list[i]):
                unique_ffn_output = self.ffn[i](attn_output[cluster_list[i],:,:])
                prob = route_prob_max[cluster_list[i]].unsqueeze(-1).unsqueeze(-1).expand_as(unique_ffn_output)
                ffn_output[cluster_list[i],:,:] = prob * unique_ffn_output + (1 - prob) * common_ffn_output[cluster_list[i],:,:]        
            
        ffn_output = self.dropout(ffn_output)
        output = self.LayerNorm(attn_output + ffn_output)
        
        return output
    

class BertMoMoShare(nn.Module):
    def __init__(self, config):
        super(BertMoMoShare, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoder(config) for _ in range(config.num_common_layers)] +
            [TailLayer(config) for _ in range(config.num_hidden_layers - config.num_common_layers)]
        )
        
        # layers = []
        # for i in range(config.num_hidden_layers // 2):
        #     layers += [TransformerEncoder(config), TailLayer(config)]
        # self.layers = nn.ModuleList(layers)
    def forward(self, hidden_states, attention_mask):
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertMoMoTSwitchTailModel(nn.Module):
    def __init__(self, config):
        super(BertMoMoTSwitchTailModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertMoMoShare(config)
        
    def forward(self, input_ids, attention_mask):
        embeddings = self.embeddings(input_ids)
        outputs = self.layers(embeddings, attention_mask)
        return outputs
    
    
class BertWithMoMoTSwitchTail(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config        
        self.bert = BertMoMoTSwitchTailModel(config)
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