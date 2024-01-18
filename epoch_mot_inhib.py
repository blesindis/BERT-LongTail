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
from Dataset import Wikitext103, Wikitext
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
    load_layer_data,
)

NEED_CENTER = True
SAMPLE_BATCHES = 20

# train and validation size for pretrain
TRAIN_LEN = 50000
VAL_LEN = 500

# folder paths
CENTER_MODEL_PATH = "outputs/0111-bert-wiki2(256)"
CENTER_PATH = os.path.join(CENTER_MODEL_PATH, 'centers-4-attn.pth')
STORE_FOLDER = "0112-mot-inhib-wiki2(256)"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CONFIG_PATH = 'config/bert_256.json'

# training parameters
num_epochs = 50
lr = 1.5e-4
weight_decay = 0.01


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    # dataset = RestaurantForLM(config=config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    dataset = Wikitext(config=config)
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    
    if NEED_CENTER:
        center_model = base_models.BertForMLM(config)
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
                    """PreNorm"""
                    # h_ = center_model.bert.encoders.layers[j].attention.LayerNorm(hidden_states)
                    # h_ = center_model.bert.encoders.layers[j].attention.self(h_, batch['attention_mask'])
                    # h_ = center_model.bert.encoders.layers[j].attention.dense(h_)
                    """PostNorm"""
                    h_ = center_model.bert.encoders.layers[j].attention.self(hidden_states, batch['attention_mask'])
                    h_ = center_model.bert.encoders.layers[j].attention.dense(h_)
                    hidden_states = center_model.bert.encoders.layers[j](hidden_states, batch['attention_mask'])
                    if i == 0:
                        layer_outputs.append(h_.to('cpu'))
                    else:
                        layer_outputs[j] = torch.cat([layer_outputs[j], h_.to('cpu')], dim=0)                
                    
        for j, layer_output in enumerate(layer_outputs):                 
            _, cluster_centers = cluster_kmeans(layer_output.mean(axis=1), config.num_experts)
            layer_cluster_centers['layer' + str(j)] = cluster_centers
                    
        torch.save(layer_cluster_centers, CENTER_PATH)
        del center_model
    
    centers = load_layer_data(CENTER_PATH)
    model = base_models.BertWithMoTAttn(config, centers)
    
    num_updates = num_epochs * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(os.path.join('log', STORE_FOLDER))
    
    # prepare model and data on device
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    step = 0
    print(STORE_FOLDER)
    for epoch in range(num_epochs):
        model.train()
        
        losses = []        
        for i, batch in enumerate(train_loader):                      
            loss, _ = model(**batch)
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            # ogd_loss = 0.0
            # for l in range(config.num_hidden_layers):
            #     for j in range(config.num_experts - 1):
            #         for k in range(j+1, config.num_experts):
            #             # w_j = torch.cat([model.bert.layers.layers[l].experts[j].attention.self.transform_layers[h].weight for h in range(config.num_attention_heads)], dim=-1).view(-1)
            #             # w_k = torch.cat([model.bert.layers.layers[l].experts[k].attention.self.transform_layers[h].weight for h in range(config.num_attention_heads)], dim=-1).view(-1)
            #             w_j = torch.cat([model.bert.layers.layers[l].experts[j].attention.self.heads[h].weight for h in range(config.num_attention_heads)], dim=-1).view(-1)
            #             w_k = torch.cat([model.bert.layers.layers[l].experts[k].attention.self.heads[h].weight for h in range(config.num_attention_heads)], dim=-1).view(-1)
                        
            #             orthogonality = torch.dot(w_j, w_k)
            #             ogd_loss += torch.abs(orthogonality)
                        
            # l1_loss = 0
            # l1_loss_layers = []
            # for l in range(config.num_hidden_layers):
            #     layer_loss = 0
            #     for e in range(config.num_experts):
            #         # layer_loss += sum(p.abs().mean() for p in model.bert.layers.layers[l].experts[e].attention.self.parameters())
            #         # layer_loss += sum(p.abs().mean() for p in model.bert.layers.layers[l].experts[e].ffn.parameters())
            #         layer_loss += sum(p.abs().mean() for p in model.bert.layers.layers[l].experts[e].parameters())
            #     l1_loss_layers.append(layer_loss)
            # l1_loss = sum(l1_loss_layers) / 12
            
            # loss = loss + ogd_loss + l1_loss
            
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()          
             
        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_valid = validate(model, val_loader, accelerator)
        accelerator.print(f'Iteration:{epoch}, Train Loss: {loss_train}, Valid Loss: {loss_valid}')
                
        writer.add_scalar(f'perplexity_train_epoch', loss_train, epoch)
        writer.add_scalar(f'perplexity_valid', loss_valid, epoch)
        # writer.add_scalar(f'loss_ogd', ogd_loss, epoch)
        # writer.add_scalar(f'loss_l1', l1_loss, epoch)
        writer.add_scalar(f'learning_rate', optimizer.param_groups[-1]['lr'], epoch)
            
    
    accelerator.save_state(STORE_PATH)
    

if __name__ == "__main__":
    main()
    
    