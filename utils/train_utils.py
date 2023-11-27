import torch
import numpy as np


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