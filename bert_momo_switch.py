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
from Dataset import MixedData, ACLForLM, RestaurantForLM, Wikitext, Wikitext103, MixedPretrain
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
    load_layer_data,
)

NEED_CENTER = True
NUM_EXPERTS = 2
NUM_FFN_EXPERTS = 2
SAMPLE_BATCHES = 20

# train and validation size for pretrain
TRAIN_LEN = 50000
VAL_LEN = 500

# folder paths
STORE_FOLDER = "0220-momoshareswitch(corrected)-mix2-bs64-5epoch"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CONFIG_PATH = 'config/bert_a.json'

# training parameters
num_epochs = 5
lr = 1.5e-4
weight_decay = 0.01
decay = 0.8


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    
    # dataset = Wikitext103(config=config)
    dataset = MixedPretrain(config=config)
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    
    model = base_models.BertWithMoMoShareSwitch(config)
    
    num_updates = num_epochs * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(os.path.join('log', STORE_FOLDER))
    
    # prepare model and data on device
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    step = 0
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
            #             # for h in range(config.num_attention_heads):
            #             #     w_j = model.bert.layers.layers[l].experts[j].attention.self.transform_layers[h].weight
            #             #     w_k = model.bert.layers.layers[l].experts[k].attention.self.transform_layers[h].weight
            #             #     # Compute W1^T * W2
            #             #     orthogonality = torch.matmul(w_j.T, w_k)
            #             #     # Compute the Frobenius norm
            #             #     ogd_loss += torch.norm(orthogonality, p='fro')
                            
            #             w_j = torch.cat([model.bert.layers.layers[l].experts[j].attention.self.transform_layers[h].weight for h in range(config.num_attention_heads)], dim=-1).view(-1)
            #             w_k = torch.cat([model.bert.layers.layers[l].experts[k].attention.self.transform_layers[h].weight for h in range(config.num_attention_heads)], dim=-1).view(-1)
            #             w_j = w_j / torch.norm(w_j)
            #             w_k = w_k / torch.norm(w_k)
            #             # Compute W1^T * W2
            #             orthogonality = torch.dot(w_j, w_k)
            #             # Compute the Frobenius norm
            #             ogd_loss += torch.abs(orthogonality)
            # # ogd_loss /=12
            
            # l1_loss = 0
            # l1_loss_layers = []
            # for l in range(config.num_hidden_layers):
            #     layer_loss = 0
            #     for e in range(config.num_experts):
            #         layer_loss += sum(p.abs().mean() for p in model.bert.layers.layers[l].experts[e].attention.self.parameters())
            #         layer_loss += sum(p.abs().mean() for p in model.bert.layers.layers[l].experts[e].ffn.parameters())
            #     l1_loss_layers.append(layer_loss)
            # sum_layer_loss = sum(l1_loss_layers)
            # l1_loss = sum(l1_loss_layers[l] for l in range(config.num_hidden_layers)) / 12
            # # l1_loss /= config.num_hidden_layers
            # # l1_loss /= config.num_experts
            # loss = loss + ogd_loss  + l1_loss
            
            # print(loss)
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()          
            
            if step % 100 == 0:
                
                loss_train = torch.mean(torch.cat(losses)[:6400])
                loss_valid = validate(model, val_loader, accelerator)
                accelerator.print(f'Iteration:{step}, Train Loss: {loss_train}, Valid Loss: {loss_valid}')
                
                losses = []
                if accelerator.is_local_main_process:
                    writer.add_scalar(f'perplexity_train_epoch', loss_train, step)
                    writer.add_scalar(f'perplexity_valid', loss_valid, step)
                    writer.add_scalar(f'learning_rate', optimizer.param_groups[-1]['lr'], step)
            if step % 5000 == 0:
                accelerator.save_state(os.path.join(STORE_PATH, 'checkpoint-' + str(step)))
            step += 1
    
    accelerator.save_state(STORE_PATH)
    

if __name__ == "__main__":
    main()
    
    