import torch
import torch.nn as nn
from einops import rearrange


class LoRAMHSA(nn.Module):
    def __init__(self, config):
        super(LoRAMHSA, self).__init__()
        self.config = config
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        
        self.input_dim = config.hidden_size
        self.lora_dim = 256
        
        """"""
        self.transform_layer = nn.Linear(self.input_dim, self.lora_dim, bias=False) 
            
        self.apply_layers = nn.ModuleList([
            nn.Linear(self.lora_dim, self.attention_head_size*3, bias=False) 
            for _ in range(self.num_attention_heads)
        ])
        """"""
        # self.transform_layers = nn.ModuleList([
        #     nn.Linear(self.input_dim, self.lora_dim, bias=False) 
        #     for _ in range(self.num_attention_heads)
        # ])
        # self.apply_layers = nn.ModuleList([
        #     nn.Linear(self.lora_dim, self.attention_head_size, bias=False) 
        #     for _ in range(self.num_attention_heads)
        # ])
        
        self.scale_factor = self.input_dim ** -0.5  # 1/np.sqrt(dim)
        self.softmax = nn.Softmax(dim=-1)

        # self.head_mask = [1] * self.num_attention_heads
    
    def forward(self, hidden_states: torch.Tensor, attention_mask):        
        transforms = []
        for h in range(self.num_attention_heads):
            # Apply the first transformation
            transformed = self.transform_layer(hidden_states)
            
            # Apply the second transformation and split into Q, K, V
            transformed = self.apply_layers[h](transformed)   
            """"""
            transforms.append(transformed)         
            """"""
            # transforms.append(transformed.repeat(1,1,3))
        transforms = torch.stack(transforms)
        q, k, v = tuple(rearrange(transforms, 'h b n (k d) -> k b h n d', k=3))

        scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * self.scale_factor
        scaled_dot_prod = scaled_dot_prod.masked_fill(attention_mask[:, None, None, :] == 0, -torch.inf)
        attention = self.softmax(scaled_dot_prod)
        self.attention = attention

        # batch_size, num_head, seq_len, head_dim
        result = torch.einsum('... i j , ... j d -> ... i d', attention, v)
        result = rearrange(result, "b h n d -> b n (h d)")
        return result
    
    
class LoRAAttention(nn.Module):
    def __init__(self, config):
        super(LoRAAttention, self).__init__()
        self.self = LoRAMHSA(config) # split multi-head
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