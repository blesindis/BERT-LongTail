import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup

# Local imports
import base_models
from Dataset import MixedData, ACLForLM, RestaurantForLM, Wikitext, Wikitext103
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
    load_layer_data,
    copy_parameters,
)

NEED_CENTER = True
SAMPLE_BATCHES = 20

# folder paths
CENTER_MODEL_PATH = "outputs/1229-mot-warmup/checkpoint-5000"
CENTER_PATH = os.path.join(CENTER_MODEL_PATH, 'centers-2-MoTWarmup-ffn.pth')
STORE_FOLDER = "1229-MoT-2experts-copyparams-samekqv-r128-ffnoutput"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CONFIG_PATH = 'config/bert_a.json'

# training parameters
num_epochs = 3
lr = 1.5e-4
weight_decay = 0.01
decay = 0.8


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    # dataset = RestaurantForLM(config=config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    dataset = Wikitext103(config=config)
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    
    if NEED_CENTER:
        center_model = base_models.BertMOTWarmup(config)
        checkpoint = torch.load(os.path.join(CENTER_MODEL_PATH, 'pytorch_model.bin'))
        center_model.load_state_dict(checkpoint)
        center_model = accelerator.prepare(center_model)
        
        layer_cluster_centers = {}
        with torch.no_grad():
            layer_outputs = []
            for i, batch in enumerate(train_loader):
                if i > SAMPLE_BATCHES:
                    break                
                hidden_states = center_model.bert.embeddings(batch['input_ids'])
                for j in range(config.num_hidden_layers):
                    h_ = center_model.bert.layers.layers[j].attention(hidden_states, batch['attention_mask'])
                    h_ = center_model.bert.layers.layers[j].LayerNorm(h_)
                    h_ = center_model.bert.layers.layers[j].ffn(h_)
                    hidden_states = center_model.bert.layers.layers[j](hidden_states, batch['attention_mask'])
                    if i == 0:
                        layer_outputs.append(h_.to('cpu'))
                    else:
                        layer_outputs[j] = torch.cat([layer_outputs[j], h_.to('cpu')], dim=0)                
                    
        for j, layer_output in enumerate(layer_outputs):                 
            _, cluster_centers = cluster_kmeans(layer_output.view(-1, 768), config.num_experts)
            layer_cluster_centers['layer' + str(j)] = cluster_centers
                    
        torch.save(layer_cluster_centers, CENTER_PATH)
        del center_model
    
    centers = load_layer_data(CENTER_PATH)
    center_model = base_models.BertMOTWarmup(config)
    checkpoint = torch.load(os.path.join(CENTER_MODEL_PATH, 'pytorch_model.bin'))
    center_model.load_state_dict(checkpoint)
    
    model = base_models.BertWithMoT(config, centers)
    
    num_updates = num_epochs * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(os.path.join('log', STORE_FOLDER))
        
    """Initialize model params by copying warmup model(center_model)"""
    copy_parameters(center_model.bert.embeddings, model.bert.embeddings)
    copy_parameters(center_model.head, model.head)
    for i in range(config.num_hidden_layers):
        for j in range(config.num_experts):
            copy_parameters(center_model.bert.layers.layers[i], model.bert.layers.layers[i].experts[j])
    
    # prepare model and data on device
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    step = 0
    for epoch in range(num_epochs):
        model.train()
        
        losses = []        
        loss_decay = 1
        for i, batch in enumerate(train_loader):                      
            loss, _ = model(**batch)
            # ogd_loss = 0.0
            # for l in range(config.num_hidden_layers):
            #     for j in range(config.num_experts - 1):
            #         for k in range(j+1, config.num_experts):
            #             for h in range(config.num_attention_heads):
            #                 w_j = model.bert.layers.layers[l].experts[j].attention.self.transform_layers[h].weight
            #                 w_k = model.bert.layers.layers[l].experts[k].attention.self.transform_layers[h].weight
            #                 # Compute W1^T * W2
            #                 orthogonality = torch.matmul(w_j.T, w_k)
            #                 # Compute the Frobenius norm
            #                 ogd_loss += torch.norm(orthogonality, p='fro')
                            
            #             w_j = torch.cat([model.bert.layers.layers[l].experts[j].attention.self.transform_layers[h].weight for h in range(config.num_attention_heads)], dim=-1)
            #             w_k = torch.cat([model.bert.layers.layers[l].experts[k].attention.self.transform_layers[h].weight for h in range(config.num_attention_heads)], dim=-1)
            #             # Compute W1^T * W2
            #             orthogonality = torch.matmul(w_j.T, w_k)
            #             # Compute the Frobenius norm
            #             ogd_loss += torch.norm(orthogonality, p='fro')
                        
            # l1_norm = sum(p.abs().sum() for p in model.parameters())
            # ogd_loss = ogd_loss
            
            # # print(loss, ogd_loss, l1_norm)
            # loss = loss + ogd_loss  + l1_norm *  5e-8 * loss_decay
            
            # print(loss)
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()          
            
            if step % 100 == 0:
                
                loss_train = torch.mean(torch.cat(losses)[:6400])
                loss_valid = validate(model, val_loader, accelerator)
                accelerator.print(f'Iteration:{step}, Train Loss: {loss_train}, Valid Loss: {loss_valid}')
                torch.cuda.empty_cache()
                losses = []
                if accelerator.is_local_main_process:
                    writer.add_scalar(f'perplexity_train_epoch', loss_train, step)
                    writer.add_scalar(f'perplexity_valid', loss_valid, step)
                    writer.add_scalar(f'learning_rate', optimizer.param_groups[-1]['lr'], step)
            if step % 5000 == 0:
                loss_decay *= decay
                accelerator.save_state(os.path.join(STORE_PATH, 'checkpoint-' + str(step)))
            step += 1
    
    accelerator.save_state(STORE_PATH)
    

if __name__ == "__main__":
    main()
    
    