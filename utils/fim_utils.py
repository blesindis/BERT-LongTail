import torch
from torch.nn.functional import cosine_similarity


def compute_fim(model, dataloader, component='embedding'):
    model.eval()
    
    component_fim = None

    for i, batch in enumerate(dataloader):
        model.zero_grad()
        loss, _ = model(**batch)
        loss.backward() 
        
        if component == 'embedding':
            fim = torch.cat([p.grad.view(-1)**2 for p in model.bert.embeddings.parameters() if p.grad is not None])
        elif component == 'decoder':
            fim = torch.cat([p.grad.view(-1)**2 for p in model.head.parameters() if p.grad is not None])
            
        trace_fim = torch.sum(fim)
        normalized_fim = fim / trace_fim
        
        if i == 0:
            component_fim = normalized_fim
        else:
            component_fim += normalized_fim
    
    component_fim /= len(dataloader)
    
    return component_fim
            

def compute_layer_fim(model, dataloader, num_layers):    
    layer_fims = []
    
    model.eval()
    
    for i, batch in enumerate(dataloader):
        model.zero_grad()
        loss, _ = model(**batch)
        loss.backward() 
        
        for j in range(num_layers):
            gradients = torch.cat([p.grad.view(-1)**2 for p in model.bert.encoders.layers[j].parameters() if p.grad is not None])
            fim = gradients
            trace_fim = torch.sum(fim)
            normalized_fim = fim / trace_fim
            
            if i == 0:
                layer_fims.append(normalized_fim)
            else:
                layer_fims[j] += normalized_fim
            
    layer_fims = [fim / len(dataloader) for fim in layer_fims]
        
    return layer_fims


def calculate_overlap(fim1, fim2):
    # Ensure FIMs are on the same device and are 1-D tensors
    fim1 = fim1.view(-1).to('cuda')
    fim2 = fim2.view(-1).to('cuda')
    
    # Calculate cosine similarity between the FIMs
    similarity = cosine_similarity(fim1.unsqueeze(0), fim2.unsqueeze(0))
    return similarity
    