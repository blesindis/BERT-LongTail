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
from Dataset import BERTPretrain
from utils.sample_utils import *
from utils.train_utils import (
    set_seed,
    load_layer_data,
)

# folder paths
STORE_FOLDER = "bert(128)300w-bs64-epoch1-lr3-momo_encoder_gating_mcr_128(mean)-test1"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CONFIG_PATH = 'config/bert_a.json'

# training parameters
num_epochs = 1
lr = 3e-4
weight_decay = 0.01
decay = 0.8


def validate(model, val_loader, accelerator):
    losses = []    
    mcr_losses = []
    for i, batch in enumerate(val_loader):    
        with torch.no_grad():
            loss, mcr_loss, _ = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
        if isinstance(mcr_loss, torch.Tensor):
            mcr_losses.append(accelerator.gather(mcr_loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    if isinstance(mcr_loss, torch.Tensor):
        mcr_losses = torch.cat(mcr_losses)[:len(val_loader.dataset)]
        mcr_loss = torch.mean(mcr_losses)
    else:
        mcr_loss = 0
    loss = torch.mean(losses)
    return loss, mcr_loss


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    
    dataset = BERTPretrain(config=config)
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    
    model = base_models.BertWithMoMoEncoderGatingMCR(config)
    
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
        mcr_losses = []   
        for i, batch in enumerate(train_loader):                      
            loss, mcr_loss, _ = model(**batch)
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            if isinstance(mcr_loss, torch.Tensor):
                mcr_losses.append(accelerator.gather(mcr_loss.repeat(config.batch_size)))
            
            if isinstance(mcr_loss, torch.Tensor):
                loss += mcr_loss
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()          
            
            if step % 100 == 0:
                
                loss_train = torch.mean(torch.cat(losses)[:6400])
                if isinstance(mcr_loss, torch.Tensor):
                    mcr_loss_train = torch.mean(torch.cat(mcr_losses)[:6400])
                else:
                    mcr_loss_train = 0
                loss_valid, mcr_loss_valid = validate(model, val_loader, accelerator)
                accelerator.print(f'Iteration:{step}, Train Loss: {loss_train}, Train mse: {mcr_loss_train}, Valid Loss: {loss_valid}, Valid mse: {mcr_loss_valid}')
                
                losses = []
                mcr_losses = []
                if accelerator.is_local_main_process:
                    writer.add_scalar(f'mcr_loss_train_step', mcr_loss_train, step)
                    writer.add_scalar(f'mcr_loss_valid_step', mcr_loss_valid, step)
                    writer.add_scalar(f'perplexity_train_epoch', loss_train, step)
                    writer.add_scalar(f'perplexity_valid', loss_valid, step)
                    writer.add_scalar(f'learning_rate', optimizer.param_groups[-1]['lr'], step)
            if step % 1000 == 0:
                accelerator.save_state(os.path.join(STORE_PATH, 'checkpoint-' + str(step)))
            step += 1
    
    accelerator.save_state(STORE_PATH)
    

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()
    
    