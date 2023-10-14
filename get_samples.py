import torch.nn as nn
import base_models
from transformers import BertConfig
from Dataset import RestaurantForLM_small
from accelerate import Accelerator
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim

import os
import torch
import numpy as np
import random


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def layer_pca(model, dataset, load_path):
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    num_updates = 70 * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    accelerator = Accelerator()
    
    # load model checkpoint
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    accelerator.load_state(load_path)
    
    # run once
    model.eval()
    
    all_layer_outputs = [[] for i in range(12)]
    all_layer_inputs = []
    all_layer_labels = []
    all_layer_attns = []
    all_decoder_outputs = []
    with torch.no_grad():
        for i, batch in enumerate(train_loader):   
            print(i)  
            if i < 100:                  
                _, scores, layer_outputs = model(**batch)
                input = batch['input_ids']
                label = batch['labels']
                attention_mask = batch['attention_mask']                    
                
                # move to cpu to release cuda memory
                batch = {key: tensor.to('cpu') for key, tensor in batch.items()}
                layer_outputs = [output.to('cpu') for output in layer_outputs]
                scores = scores.to('cpu')
                
                # save in my variable
                for j, layer_output in enumerate(layer_outputs):  
                    all_layer_outputs[j].append(layer_output)
                all_layer_inputs.append(input)
                all_layer_labels.append(label)
                all_layer_attns.append(attention_mask)
                all_decoder_outputs.append(scores)
                
    accelerator.print(f'Number of Samples batches: {len(all_layer_outputs[0])}')
    
    # calculate pca
    layer_outputs = {}
    layer_inputs = {}
    layer_labels = {}
    layer_attns = {}
    decoder_outputs = {}
    # save layer outputs
    for i, layer in enumerate(all_layer_outputs):
        layer_np = [single.numpy() for single in layer]
        layer = np.vstack(layer_np)
        layer = torch.from_numpy(layer)        

        layer_outputs['layer ' + str(i+1) ] = layer
        print(layer.size())
        
    # save layer inputs, labels, attns
    for i, layer in enumerate(all_layer_inputs):
        layer_inputs['layer' + str(i+1) ] = all_layer_inputs[i]
        layer_labels['layer' + str(i+1) ] = all_layer_labels[i]
        layer_attns['layer' + str(i+1)] = all_layer_attns[i]
        decoder_outputs['layer' + str(i+1)] = all_decoder_outputs[i]
    
    # save to files
    torch.save(layer_outputs, os.path.join(load_path, 'layer_outputs.pth'))
    torch.save(layer_inputs, os.path.join(load_path, 'layer_inputs.pth'))
    torch.save(layer_labels, os.path.join(load_path, 'layer_labels.pth'))
    torch.save(layer_attns, os.path.join(load_path, 'layer_attns.pth'))
    torch.save(decoder_outputs, os.path.join(load_path, 'decoder_outputs.pth'))
    

if __name__ == "__main__":
    set_seed(45)
    
    config = BertConfig.from_json_file('config/bert.json')
    dataset = RestaurantForLM_small(config=config)
    
    model = base_models.BertWithSavers(config=config)
    # model = nn.DataParallel(model)
    
    load_path = "./output-0-savedecoder"
    layer_pca(model=model, dataset=dataset, load_path=load_path)
