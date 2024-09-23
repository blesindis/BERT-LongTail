import torch
import torch.nn as nn
from einops import rearrange


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
        # print(hidden_states.shape, attention_mask.shape)
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
        """Pre Norm No Dropout"""
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


class LoRAMHSA(nn.Module):
    def __init__(self, config, lora_dim):
        super(LoRAMHSA, self).__init__()
        self.config = config
        self.lora_dim = lora_dim
        self.attention_head_size = int(self.lora_dim / config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        
        self.input_dim = config.hidden_size
        
        """"""
        self.input_dim = config.hidden_size
        self.heads = nn.ModuleList([nn.Linear(self.input_dim, self.attention_head_size * 3, bias=False) for _ in range(self.num_attention_heads)])
        self.scale_factor = self.input_dim ** -0.5  # 1/np.sqrt(dim)
        self.softmax = nn.Softmax(dim=-1)

        # self.head_mask = [1] * self.num_attention_heads
    
    def forward(self, hidden_states: torch.Tensor, attention_mask):
        # print(hidden_states.shape, attention_mask.shape)
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
    
    
class LoRAAttention(nn.Module):
    def __init__(self, config, lora_dim):
        super(LoRAAttention, self).__init__()
        self.lora_dim = lora_dim
        self.self = LoRAMHSA(config, lora_dim) # split multi-head
        # self.self = SelfAttention(config)
        self.dense = nn.Linear(self.lora_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_tensor, attention_mask):        
        """Pre Norm No Dropout"""
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
    
    
class SmallMHSA(nn.Module):
    def __init__(self, config, in_dim):
        super(SmallMHSA, self).__init__()
        self.config = config
        self.in_dim = in_dim
        self.attention_head_size = int(self.in_dim / config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        
        """"""
        self.input_dim = self.in_dim
        self.heads = nn.ModuleList([nn.Linear(self.input_dim, self.attention_head_size * 3, bias=False) for _ in range(self.num_attention_heads)])
        self.scale_factor = self.input_dim ** -0.5  # 1/np.sqrt(dim)
        self.softmax = nn.Softmax(dim=-1)

        # self.head_mask = [1] * self.num_attention_heads
    
    def forward(self, hidden_states: torch.Tensor, attention_mask):
        # print(hidden_states.shape, attention_mask.shape)
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
    
    
class SmallAttention(nn.Module):
    def __init__(self, config, in_dim):
        super(SmallAttention, self).__init__()
        self.in_dim = in_dim
        self.self = SmallMHSA(config, in_dim) # split multi-head
        # self.self = SelfAttention(config)
        self.dense = nn.Linear(self.in_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_tensor, attention_mask):        
        """Pre Norm No Dropout"""
        # hidden_states = self.LayerNorm(input_tensor)
        # hidden_states = self.self(hidden_states, attention_mask)
        # hidden_states = self.dense(hidden_states)
        # hidden_states = hidden_states + input_tensor
        """Post Norm"""
        hidden_states = self.self(input_tensor, attention_mask)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states