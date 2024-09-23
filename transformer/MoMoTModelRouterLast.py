import torch
import torch.nn as nn
from einops import rearrange
from transformer.modules.common import Embeddings
from transformer.modules.attention import Attention, LoRAAttention
from transformer.modules.feedforward import FeedForward, SwitchFeedForward, SwitchFeedForwardLoRA, SwitchFeedForwardLoRALatent
from transformer.Switch import SwitchEncoder
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformer.BERT import TransformerEncoder
    
    
class ExpertTransformerCommon(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        if config.same_common_unique:
            if config.use_lora_latent_moe:            
                self.ffn = SwitchFeedForwardLoRALatent(config, lora_dim=128, n_experts=4)
            elif config.use_lora_moe:
                self.ffn = SwitchFeedForwardLoRA(config, lora_dim=128, n_experts=4)
            else:
                self.ffn = SwitchFeedForward(config)
        else:
            self.ffn = SwitchFeedForward(config)
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
        if config.use_lora_latent_moe:            
            self.ffn = SwitchFeedForwardLoRALatent(config, lora_dim=128, n_experts=4)
        elif config.use_lora_moe:            
            self.ffn = SwitchFeedForwardLoRA(config, lora_dim=128, n_experts=4, drop_tokens=config.drop_tokens_unique)
        else:
            self.ffn = SwitchFeedForward(config)
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
        
    def forward(self, hidden_states, attention_mask, cluster_list):
        unique_output = hidden_states.new_zeros(hidden_states.shape)
        
        for i in range(self.n_experts):                        
            unique_output[cluster_list[i], :, :] = self.unique_experts[i](hidden_states[cluster_list[i], :, :], attention_mask[cluster_list[i], :])
        
        common_output = self.common_expert(hidden_states, attention_mask)
        
        output = unique_output + common_output 
        
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


class BertMoMoTModelRouterLastModel(nn.Module):
    def __init__(self, config):
        super(BertMoMoTModelRouterLastModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertMoMoShare(config)
        
    def forward(self, input_ids, attention_mask, cluster_list):
        embeddings = self.embeddings(input_ids)
        outputs = self.layers(embeddings, attention_mask, cluster_list)
        return outputs
    
    
class BertWithMoMoTModelRouterLast(nn.Module):
    def __init__(self, config):
        super(BertWithMoMoTModelRouterLast, self).__init__()
        self.config = config        
        self.bert = BertMoMoTModelRouterLastModel(config)
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