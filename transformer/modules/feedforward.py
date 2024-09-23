import torch
import torch.nn as nn
from einops import rearrange


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
    
    
class FeedForwardLoRA(nn.Module):
    def __init__(self, config, lora_dim):
        super(FeedForwardLoRA, self).__init__()
        self.config = config
        self.dense_1 = nn.Linear(config.hidden_size, lora_dim)
        self.intermediate_act_fn = nn.GELU()
        self.dense_2 = nn.Linear(lora_dim, config.hidden_size)
    
    def forward(self, hidden_states):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states
    
     
class SwitchFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = config.num_experts
        self.capacity_factor = config.capacity_factor
        self.is_scale_prob = True
        self.drop_tokens = config.drop_tokens
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(config.num_experts)])
        self.switch = nn.Linear(config.hidden_size, self.n_experts)
        self.softmax = nn.Softmax(dim=-1)

        # self.loss = None
        # self.loss_coef = loss_coef

    # def load_balance_loss(self, counts, route_prob):
    #     total = counts.sum(dim=-1, keepdims=True)
    #     route_frac = counts / total
    #     route_prob = route_prob / total
    #     load_balancing_loss = self.n_experts * (route_frac * route_prob).sum()

    #     return load_balancing_loss

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        x = x.contiguous().view(-1, d_model)
        final_output = x.new_zeros(x.shape)

        route_prob = self.softmax(self.switch(x))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]      
        
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])
        # self.loss = self.loss_coef * self.load_balance_loss(counts, route_prob)
        
        dropped = []
        if self.drop_tokens:
            for i in range(self.n_experts):
                if len(indexes_list[i]) > capacity:
                    if self.config.priority_drop:
                        prob_i = route_prob_max[indexes_list[i]]
                        _, sorted_indices = torch.sort(prob_i, descending=True)
                        indexes_list[i] = indexes_list[i][sorted_indices]
                    else:
                        indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                    dropped.append(indexes_list[i][capacity:])
                    indexes_list[i] = indexes_list[i][:capacity]


        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]
            
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        final_output = final_output * route_prob_max.view(-1, 1) if self.is_scale_prob else final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)
        final_output = final_output.view(batch_size, seq_len, d_model)

        return final_output
    
    
class SwitchFeedForwardLoRA(nn.Module):
    def __init__(self, config, lora_dim, n_experts, drop_tokens=False):
        super().__init__()
        self.config = config
        self.n_experts = n_experts
        self.capacity_factor = config.capacity_factor
        self.is_scale_prob = True
        self.drop_tokens = drop_tokens
        self.experts = nn.ModuleList([FeedForwardLoRA(config, lora_dim) for _ in range(self.n_experts)])
        self.switch = nn.Linear(config.hidden_size, self.n_experts)
        self.softmax = nn.Softmax(dim=-1)

        # self.loss = None
        # self.loss_coef = loss_coef

    # def load_balance_loss(self, counts, route_prob):
    #     total = counts.sum(dim=-1, keepdims=True)
    #     route_frac = counts / total
    #     route_prob = route_prob / total
    #     load_balancing_loss = self.n_experts * (route_frac * route_prob).sum()

    #     return load_balancing_loss

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        x = x.contiguous().view(-1, d_model)
        final_output = x.new_zeros(x.shape)

        route_prob = self.softmax(self.switch(x))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]      
        
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])
        # self.loss = self.loss_coef * self.load_balance_loss(counts, route_prob)
        
        dropped = []
        if self.drop_tokens:
            for i in range(self.n_experts):
                if len(indexes_list[i]) > capacity:
                    if self.config.priority_drop:
                        prob_i = route_prob_max[indexes_list[i]]
                        _, sorted_indices = torch.sort(prob_i, descending=False)
                        indexes_list[i] = indexes_list[i][sorted_indices]
                    else:
                        indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                    indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                    dropped.append(indexes_list[i][capacity:])
                    indexes_list[i] = indexes_list[i][:capacity]


        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]
            
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]


        final_output = final_output * route_prob_max.view(-1, 1) if self.is_scale_prob else final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)
        final_output = final_output.view(batch_size, seq_len, d_model)

        return final_output
    

class SwitchFeedForwardLoRALatent(nn.Module):
    def __init__(self, config, lora_dim, n_experts):
        super().__init__()
        self.n_experts = n_experts
        self.lora_dim = lora_dim
        self.capacity_factor = config.capacity_factor
        self.is_scale_prob = True
        self.drop_tokens = False
        self.experts = nn.ModuleList([FeedForwardLoRA(config, lora_dim) for _ in range(self.n_experts)])
        
        self.encoder = nn.Linear(config.hidden_size, self.lora_dim)
        self.switch = nn.Linear(self.lora_dim, self.n_experts)
        self.softmax = nn.Softmax(dim=-1)
        

        # self.loss = None
        # self.loss_coef = loss_coef

    # def load_balance_loss(self, counts, route_prob):
    #     total = counts.sum(dim=-1, keepdims=True)
    #     route_frac = counts / total
    #     route_prob = route_prob / total
    #     load_balancing_loss = self.n_experts * (route_frac * route_prob).sum()

    #     return load_balancing_loss

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        x = x.contiguous().view(-1, d_model)
        final_output = x.new_zeros(x.shape)

        x_encoded = self.encoder(x)
        route_prob = self.softmax(self.switch(x_encoded))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]      
        
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])
        # self.loss = self.loss_coef * self.load_balance_loss(counts, route_prob)
        
        dropped = []
        if self.drop_tokens:
            for i in range(self.n_experts):
                if len(indexes_list[i]) > capacity:
                    indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                    dropped.append(indexes_list[i][capacity:])
                    indexes_list[i] = indexes_list[i][:capacity]


        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]
            
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]


        final_output = final_output * route_prob_max.view(-1, 1) if self.is_scale_prob else final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)
        final_output = final_output.view(batch_size, seq_len, d_model)

        return final_output
    
    
class DSFeedForward(nn.Module):
    def __init__(self, config, lora_dim, n_experts):
        super().__init__()
        self.lora_dim = lora_dim
        self.n_experts = n_experts
        self.config = config
        self.experts = SwitchFeedForwardLoRA(config, lora_dim=self.lora_dim, n_experts=n_experts)
        self.common_expert = FeedForwardLoRA(config, lora_dim=self.lora_dim)
        
    def forward(self, hidden_states):
        return self.experts(hidden_states) + self.common_expert(hidden_states)
    
    
class SwitchFeedForwardTail(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = config.num_experts + 1
        self.capacity_factor = config.capacity_factor
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.n_experts)])
        
        self.switch = nn.Linear(config.hidden_size, self.n_experts)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        x = x.contiguous().view(-1, d_model)
        final_output = x.new_zeros(x.shape)

        route_prob = self.softmax(self.switch(x))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]      
        
        common_output = self.experts[0](x)
        if len(indexes_list[0]):
            prob_common = route_prob_max[indexes_list[0]]
            scaling_common = (prob_common / prob_common.detach()).unsqueeze(-1).expand_as(x[indexes_list[0],:])
            final_output[indexes_list[0], :] = common_output[indexes_list[0], :] * scaling_common

        for i in range(1, self.n_experts):
            if len(indexes_list[i]):
                unique_output = self.experts[i](x[indexes_list[i], :])
                prob = route_prob_max[indexes_list[i]].unsqueeze(-1).expand_as(unique_output)
                final_output[indexes_list[i], :] = prob * unique_output + (1 - prob) * common_output[indexes_list[i], :]

        final_output = final_output.view(batch_size, seq_len, d_model)
        return final_output