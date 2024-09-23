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
from transformer.Switch import BertSwitch

# Local imports
from Dataset import LePuWiki
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
    load_layer_data,
)

# train and validation size for pretrain
TRAIN_LEN = 200000
VAL_LEN = 500

# folder paths
STORE_FOLDER = "lepuwiki(128)50w-bs64-epoch1-lr3-moe"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CONFIG_PATH = 'config/bert_a.json'

# training parameters
num_epochs = 1
lr = 3e-4
weight_decay = 0.01
decay = 0.8


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    dataset = LePuWiki(config=config)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    
    model = BertSwitch(config)
    
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
                    writer.add_scalar(f'loss_train_epoch', loss_train, step)
                    writer.add_scalar(f'loss_valid', loss_valid, step)
                    writer.add_scalar(f'learning_rate', optimizer.param_groups[-1]['lr'], step)
            if step % 2500 == 0:
                model_dir = os.path.join(STORE_PATH, f'checkpoint-{step}') 
                os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(model_dir, 'pytorch_model.bin'))
            step += 1
    
    model_dir = os.path.join(STORE_PATH, f'checkpoint-{step}') 
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'pytorch_model.bin'))
    
    

if __name__ == "__main__":
    main()
    
    