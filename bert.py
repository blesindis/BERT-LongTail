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
from transformer.BERT import BertForMLM
from torch.cuda.amp import autocast, GradScaler

# Local imports
from Dataset import Wikipedia ,BookCorpus, BERTPretrain, LePuWiki
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
)

LOAD_CHECKPOINT = False

# folder paths
STORE_FOLDER = "LPW(128)(28+1+1)w-bs64-1epoch-lr1-bert-bf16"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CONFIG_PATH = 'config/bert_a.json'

# training parameters
num_epochs = 1
lr = 1e-4
weight_decay = 0.01


def main():
    set_seed(45)
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    dataset = LePuWiki(config=config)
    
    model = BertForMLM(config)
    if LOAD_CHECKPOINT:
        checkpoint = torch.load(os.path.join(STORE_PATH, 'pytorch_model.bin'))
        model.load_state_dict(checkpoint)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    val_loader_l, val_loader_p, val_loader_w = dataset.val_loader_legal, dataset.val_loader_pub, dataset.val_loader_wiki
    num_updates = num_epochs * len(train_loader)
    print(len(train_loader), len(val_loader))
    
    grad_scaler = GradScaler()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', STORE_FOLDER))
    
    # prepare model and data on device
    grad_scaler = accelerator.prepare(grad_scaler)
    model, optimizer, lr_scheduler, val_loader, train_loader = accelerator.prepare(model, optimizer, lr_scheduler, val_loader, train_loader)
    val_loader_l, val_loader_p, val_loader_w = accelerator.prepare(val_loader_l, val_loader_p, val_loader_w)
    
    if not LOAD_CHECKPOINT:
        step = 0
        for epoch in range(num_epochs):
            model.train()
            
            losses = []        
            for i, batch in enumerate(train_loader):   
                with autocast(dtype=torch.bfloat16):
                    loss, _ = model(**batch)
                    optimizer.zero_grad()                            
                
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                # loss, _ = model(**batch)
                losses.append(accelerator.gather(loss.repeat(config.batch_size)))
                
                # optimizer.zero_grad()
                # accelerator.backward(loss)
                # optimizer.step()
                lr_scheduler.step()          
                  
                if step % 100 == 0:
                    loss_train = torch.mean(torch.cat(losses)[:6400])
                    loss_valid = validate(model, val_loader, accelerator)
                    loss_valid_l = validate(model, val_loader_l, accelerator)
                    loss_valid_p = validate(model, val_loader_p, accelerator)
                    loss_valid_w = validate(model, val_loader_w, accelerator)
                    accelerator.print(f'iteration:{step} , Train Loss: {loss_train}, Valid Loss: {loss_valid}')

                    writer.add_scalar(f'loss_train_epoch', loss_train, step)
                    writer.add_scalar(f'loss_valid', loss_valid, step)
                    writer.add_scalar(f'loss_valid_legal', loss_valid_l, step)
                    writer.add_scalar(f'loss_valid_pubmed', loss_valid_p, step)
                    writer.add_scalar(f'loss_valid_wiki', loss_valid_w, step)
                    writer.add_scalar(f'learning_rate', optimizer.param_groups[-1]['lr'], step)
                    
                    losses = []   
                if step % 5000 == 0:
                    accelerator.save_state(os.path.join(STORE_PATH, 'checkpoint-' + str(step)))
                step += 1
            
        accelerator.save_state(STORE_PATH)
    

if __name__ == "__main__":
    main()
    
    