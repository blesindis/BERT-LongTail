import torch
import random
import numpy as np


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(model, val_loader, accelerator):
    losses = []    
    for i, batch in enumerate(val_loader):    
        with torch.no_grad():
            loss, _ = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    loss = torch.mean(losses)
    
    return loss


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    layer_data = torch.tensor(np.array(layer_data)).to('cuda')
    return layer_data


def copy_parameters(source_module, target_module):
    for source_param, target_param in zip(source_module.parameters(), target_module.parameters()):
        target_param.data.copy_(source_param.data)


# output of a whole layer, same for pre-norm & post-norm
def get_layer_outputs_ffn_residual(model, data_loader):
    layer_outputs = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            h_ = model.bert.embeddings(batch['input_ids'])
            if i == 0:
                layer_outputs.append(h_.cpu())
            else:                    
                layer_outputs[0] = torch.cat([layer_outputs[0], h_.cpu()], dim=0)
            
            for l in range(12):              
                h_ = model.bert.encoders.layers[l](h_, batch['attention_mask'])  
                
                if i == 0:
                    layer_outputs.append(h_.cpu())
                else:                    
                    layer_outputs[l+1] = torch.cat([layer_outputs[l+1], h_.cpu()], dim=0)
                    
    return layer_outputs


# output of attention after residual connection, same for pre-norm & post-norm
def get_layer_outputs_attn_residual(model, data_loader):
    layer_outputs = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            h_ = model.bert.embeddings(batch['input_ids'])
            
            for l in range(12):
                output = model.bert.encoders.layers[l].attention(h_, batch['attention_mask'])
                
                h_ = model.bert.encoders.layers[l](h_, batch['attention_mask'])
                
                if i == 0:
                    layer_outputs.append(output.cpu())
                else:
                    layer_outputs[l] = torch.cat([layer_outputs[l], output.cpu()], dim=0)
                    
    return layer_outputs


# Output of ffn before residual connection of pre-norm bert
# Same for combine/split residual connection
def get_layer_outputs_ffn_pre(model, data_loader):
    layer_outputs = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            h_ = model.bert.embeddings(batch['input_ids'])
            
            for l in range(12):
                output = model.bert.encoders.layers[l].attention(h_, batch['attention_mask'])
                output = model.bert.encoders.layers[l].LayerNorm(output)
                output = model.bert.encoders.layers[l].ffn(output)
                
                h_ = model.bert.encoders.layers[l](h_, batch['attention_mask'])
                
                if i == 0:
                    layer_outputs.append(output.cpu())
                else:
                    layer_outputs[l] = torch.cat([layer_outputs[l], output.cpu()], dim=0)
                    
    return layer_outputs


# Output of ffn before residual connection of post-norm bert
# same for combine/split residual connection
def get_layer_outputs_ffn_post(model, data_loader):
    layer_outputs = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            h_ = model.bert.embeddings(batch['input_ids'])
            
            for l in range(12):
                output = model.bert.encoders.layers[l].attention(h_, batch['attention_mask'])
                output = model.bert.encoders.layers[l].ffn(output)
                
                h_ = model.bert.encoders.layers[l](h_, batch['attention_mask'])
                
                if i == 0:
                    layer_outputs.append(output.cpu())
                else:
                    layer_outputs[l] = torch.cat([layer_outputs[l], output.cpu()], dim=0)
                    
    return layer_outputs


# Output of attn before residual connection of pre-norm bert
# same for combine/split residual connection
def get_layer_outputs_attn_pre(model, data_loader):
    layer_outputs = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            h_ = model.bert.embeddings(batch['input_ids'])
            
            for l in range(12):
                output = model.bert.encoders.layers[l].attention.LayerNorm(h_)
                output = model.bert.encoders.layers[l].attention.self(output, batch['attention_mask'])
                output = model.bert.encoders.layers[l].attention.dense(output)
                
                h_ = model.bert.encoders.layers[l](h_, batch['attention_mask'])
                
                if i == 0:
                    layer_outputs.append(output.cpu())
                else:
                    layer_outputs[l] = torch.cat([layer_outputs[l], output.cpu()], dim=0)
                    
    return layer_outputs


# Output of attn before residual connection of post-norm bert
# Combine Residual Connection
def get_layer_outputs_attn_post_combine(model, data_loader):
    layer_outputs = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            h_ = model.bert.embeddings(batch['input_ids'])
            
            for l in range(12):
                output = model.bert.encoders.layers[l].attention.self(h_, batch['attention_mask'])
                output = model.bert.encoders.layers[l].attention.dense(output)
                output = model.bert.encoders.layers[l].attention.LayerNorm(output)
                
                h_ = model.bert.encoders.layers[l](h_, batch['attention_mask'])
                
                if i == 0:
                    layer_outputs.append(output.cpu())
                else:
                    layer_outputs[l] = torch.cat([layer_outputs[l], output.cpu()], dim=0)
                    
    return layer_outputs


# Output of attn before residual connection of post-norm bert
# Split Residual Connection (2 for attn & ffn)
def get_layer_outputs_attn_post_split(model, data_loader):
    layer_outputs = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            h_ = model.bert.embeddings(batch['input_ids'])
            
            for l in range(12):
                output = model.bert.encoders.layers[l].attention.self(h_, batch['attention_mask'])
                output = model.bert.encoders.layers[l].attention.dense(output)
                
                h_ = model.bert.encoders.layers[l](h_, batch['attention_mask'])
                
                if i == 0:
                    layer_outputs.append(output.cpu())
                else:
                    layer_outputs[l] = torch.cat([layer_outputs[l], output.cpu()], dim=0)
                    
    return layer_outputs