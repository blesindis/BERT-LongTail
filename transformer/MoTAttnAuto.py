import torch
import torch.nn as nn
import random
from einops import rearrange


class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, input_ids) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        embeddings = self.word_embeddings(input_ids)
        position_ids = self.position_ids[:, :seq_len]
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    
# """LoRA"""
# class MHSA(nn.Module):
#     def __init__(self, config):
#         super(MHSA, self).__init__()
#         self.config = config
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.num_attention_heads = config.num_attention_heads
        
#         self.input_dim = config.hidden_size
#         self.lora_dim = 128
        
#         """"""
#         # self.transform_layers = nn.ModuleList([
#         #     nn.Linear(self.input_dim, self.lora_dim*3, bias=False) 
#         #     for _ in range(self.num_attention_heads)
#         # ])
#         # self.apply_layers = nn.ModuleList([
#         #     nn.Linear(self.lora_dim*3, self.attention_head_size*3, bias=False) 
#         #     for _ in range(self.num_attention_heads)
#         # ])
#         """"""
#         self.transform_layer = nn.Linear(self.input_dim, self.lora_dim, bias=False) 
            
#         self.apply_layers = nn.ModuleList([
#             nn.Linear(self.lora_dim, self.attention_head_size*3, bias=False) 
#             for _ in range(self.num_attention_heads)
#         ])
        
#         self.scale_factor = self.input_dim ** -0.5  # 1/np.sqrt(dim)
#         self.softmax = nn.Softmax(dim=-1)

#         # self.head_mask = [1] * self.num_attention_heads
    
#     def forward(self, hidden_states: torch.Tensor, attention_mask):        
#         transforms = []
#         for h in range(self.num_attention_heads):
#             # Apply the first transformation
#             transformed = self.transform_layer(hidden_states)
            
#             # Apply the second transformation and split into Q, K, V
#             transformed = self.apply_layers[h](transformed)   
#             """"""
#             transforms.append(transformed)         
#             """"""
#             # transforms.append(transformed.repeat(1,1,3))
#         transforms = torch.stack(transforms)
#             # Q, K, V = torch.chunk(transformed, 3, dim=-1)

#             # scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', Q, K) * self.scale_factor
#             # scaled_dot_prod = scaled_dot_prod.masked_fill(attention_mask[:, None, None, :] == 0, -torch.inf)
#             # attention = self.softmax(scaled_dot_prod)

#             # head_output = torch.einsum('... i j , ... j d -> ... i d', attention, V)
#             # print(head_output.shape)
#             # multi_head_outputs.append(head_output)
#         q, k, v = tuple(rearrange(transforms, 'h b n (k d) -> k b h n d', k=3))

#         scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * self.scale_factor
#         scaled_dot_prod = scaled_dot_prod.masked_fill(attention_mask[:, None, None, :] == 0, -torch.inf)
#         attention = self.softmax(scaled_dot_prod)
#         self.attention = attention

#         # batch_size, num_head, seq_len, head_dim
#         result = torch.einsum('... i j , ... j d -> ... i d', attention, v)
#         result = rearrange(result, "b h n d -> b n (h d)")
#         # print(result.shape)
#         #     # multi_head_outputs.append(result)

#         # # Concatenate the outputs from all heads
#         # multi_head_output = torch.cat(multi_head_outputs, dim=-1)
#         # print(multi_head_output.shape)
#         return result
    
    
class MHSA(nn.Module):
    def __init__(self, config):
        super(MHSA, self).__init__()
        self.config = config
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        
        self.input_dim = config.hidden_size
        self.heads = nn.ModuleList([nn.Linear(self.input_dim, self.attention_head_size * 3, bias=False) for _ in range(self.num_attention_heads)])
        self.scale_factor = self.input_dim ** -0.5  # 1/np.sqrt(dim)
        self.softmax = nn.Softmax(dim=-1)

        # self.head_mask = [1] * self.num_attention_heads
    
    def forward(self, hidden_states: torch.Tensor, attention_mask):
        qkv = torch.stack([self.heads[h](hidden_states) for h in range(self.num_attention_heads)])
        # qkv = torch.stack([self.heads[h](hidden_states) * self.head_mask[h] for h in range(self.num_attention_heads)])
        # batch_size, seq_len, _ = hidden_states.shape
        # qkv = torch.stack([
        #     self.heads[h](hidden_states) if self.head_mask[h] else hidden_states.new_zeros((batch_size, seq_len, self.attention_head_size * 3))
        #     for h in range(self.num_attention_heads)
        # ])
        q, k, v = tuple(rearrange(qkv, 'h b n (k d) -> k b h n d', k=3))

        scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * self.scale_factor
        scaled_dot_prod = scaled_dot_prod.masked_fill(attention_mask[:, None, None, :] == 0, -torch.inf)
        attention = self.softmax(scaled_dot_prod)
        self.attention = attention

        # batch_size, num_head, seq_len, head_dim
        result = torch.einsum('... i j , ... j d -> ... i d', attention, v)
        result = rearrange(result, "b h n d -> b n (h d)")
        return result
    
    
class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.self = MHSA(config) # split multi-head
        # self.self = SelfAttention(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_tensor, attention_mask):        
        """Pre Norm"""
        # hidden_states = self.LayerNorm(input_tensor)
        # hidden_states = self.self(hidden_states, attention_mask)
        # hidden_states = self.dense(hidden_states)
        # hidden_states = hidden_states + input_tensor
        """Post Norm"""
        hidden_states = self.self(input_tensor, attention_mask)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.config = config
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states
    
    
class TransformerEncoder(nn.Module):
    def __init__(self, config) -> None:
        super(TransformerEncoder, self).__init__()
        self.attention = Attention(config)
        self.ffn = FeedForward(config)
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
    
    
class TransformerEncoders(nn.Module):
    def __init__(self, config):
        super(TransformerEncoders, self).__init__()
        self.config = config
        self.layers = nn.ModuleList([TransformerEncoder(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask) -> torch.Tensor:
        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class Experts(nn.Module):
    def __init__(self, config):
        super(Experts, self).__init__()
        self.config = config
        self.experts = nn.ModuleList([TransformerEncoder(config) for _ in range(config.num_experts)])
        self.centers = []
        self.step = 0
        
    def center_update(self, update_data):
        
        for i, data in enumerate(update_data):
            if len(data) > 0:
                delta = torch.mean(torch.stack(data))
                # self.centers[i] = (self.centers[i] + delta) / 2
                # self.centers[i] = delta
                self.centers[i] = self.centers[i] * 0.7 + delta * 0.3
        
    def routing(self, hidden_states, attention_mask):
        cluster_list = [[] for _ in range(self.config.num_experts)]
        center_updates = [[] for _ in range(self.config.num_experts)]
        
        experts_outputs = []
        for expert in self.experts:
            """PreNorm"""
            # h_ = expert.attention.LayerNorm(hidden_states)
            # h_ = expert.attention.self(h_, attention_mask)
            # h_ = expert.attention.dense(h_)
            """PostNorm"""
            h_ = expert.attention.self(hidden_states, attention_mask)
            h_ = expert.attention.dense(h_)
            experts_outputs.append(h_)
        for i in range(hidden_states.shape[0]):
            output_i = [experts_outputs[j][i].mean(dim=0) for j in range(self.config.num_experts)]
            dists_i_list = [torch.dist(output_i[j], self.centers[j]) for j in range(self.config.num_experts)]
            dists_i = torch.stack(dists_i_list)
            selected = torch.argmin(dists_i)
            cluster_list[selected].append(i)
            # center_updates[selected].append(output_i[selected].detach())
            
        # self.center_update(center_updates)
        return cluster_list
    
    def routing_forward(self, hidden_states, attention_mask):
        # output = hidden_states.new_zeros(hidden_states.shape[0] * hidden_states.shape[1], hidden_states.shape[2])                
        output = hidden_states.new_zeros(hidden_states.shape)        
        center_updates = [[] for _ in range(self.config.num_experts)]
        experts_outputs = []
        for expert in self.experts:
            """PreNorm"""
            # h_ = expert.attention.LayerNorm(hidden_states)
            # h_ = expert.attention.self(h_, attention_mask)
            # h_ = expert.attention.dense(h_)
            """PostNorm"""
            # h_ = expert.attention.self(hidden_states, attention_mask)
            # h_ = expert.attention.dense(h_)
            h_ = expert.attention(hidden_states, attention_mask)
            h_ = expert.ffn(h_)
            experts_outputs.append(h_)
        for i in range(hidden_states.shape[0]):
            output_i = [experts_outputs[j][i].mean(dim=0) for j in range(self.config.num_experts)]            
            dists_i_list = [torch.dist(output_i[j], self.centers[j]) for j in range(self.config.num_experts)]
            dists_i = torch.stack(dists_i_list)
            selected = torch.argmin(dists_i)
            output[i] = experts_outputs[selected][i]
            center_updates[selected].append(output_i[selected].detach())
        # for i in range(output.shape[0]):
        #     output_i = [experts_outputs[j][i] for j in range(self.config.num_experts)]            
        #     dists_i_list = [torch.dist(output_i[j], self.centers[j]) for j in range(self.config.num_experts)]
        #     dists_i = torch.stack(dists_i_list)
        #     selected = torch.argmin(dists_i)
        #     output[i] = output_i[selected]
        #     center_updates[selected].append(output_i[selected].detach())
            
        # print([len(d) for d in center_updates], torch.dist(self.centers[0], self.centers[1]))
        # self.center_update(center_updates)
        # output = output.view(hidden_states.shape)
        
        return output
    
    def forward(self, hidden_states, attention_mask):
        output = hidden_states.new_zeros(hidden_states.shape)        
        
        if self.step <= 5000:            
            for i in range(self.config.num_experts):                                 
                output += self.experts[i](hidden_states, attention_mask)
                if self.step == 5000:
                    h_ = self.experts[i].attention.self(hidden_states.detach(), attention_mask.detach())
                    h_ = self.experts[i].attention.dense(h_)   
                    self.centers.append(h_.mean(dim=1).mean(dim=0).detach())
        else:
            cluster_list = self.routing(hidden_states, attention_mask)        
            for i in range(self.config.num_experts):
                output[cluster_list[i], :, :] = self.experts[i](hidden_states[cluster_list[i], :, :], attention_mask[cluster_list[i], :])
            
        self.step += 1    
        # output = self.routing_forward(hidden_states, attention_mask)
        return output


class BertMOT(nn.Module):
    def __init__(self, config):
        super(BertMOT, self).__init__()
        self.layers = nn.ModuleList([Experts(config) for i in range(config.num_hidden_layers)])
        
    def forward(self, hidden_states, attention_mask):
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


# Bert with moe
class BertMOTAttnAutoModel(nn.Module):
    def __init__(self, config):
        super(BertMOTAttnAutoModel, self).__init__()
        self.embeddings = Embeddings(config)
        self.layers = BertMOT(config)
        
    def forward(self, input_ids, attention_mask):
        embeddings = self.embeddings(input_ids)
        outputs = self.layers(embeddings, attention_mask)
        return outputs

