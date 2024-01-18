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
from Dataset import MixedData, BookCorpus, RestaurantForLM, Wikitext, Wikitext103
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
)

LOAD_CHECKPOINT = False

# train and validation size for pretrain
TRAIN_LEN = 200000
VAL_LEN = 500

# folder paths
STORE_FOLDER = "0111-moe4-wiki2(256)"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CONFIG_PATH = 'config/bert_256.json'

# training parameters
num_epochs = 50
lr = 1.5e-4
weight_decay = 0.01


def main():
    set_seed(45)
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    dataset = Wikitext(config=config)
    # dataset = MixedData(config=config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    
    model = base_models.BertSwitch(config)
    if LOAD_CHECKPOINT:
        checkpoint = torch.load(os.path.join(STORE_PATH, 'pytorch_model.bin'))
        model.load_state_dict(checkpoint)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    num_updates = num_epochs * len(train_loader)
    print(len(train_loader), len(val_loader))
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', STORE_FOLDER))
    
    # prepare model and data on device
    model, optimizer, lr_scheduler, val_loader, train_loader = accelerator.prepare(model, optimizer, lr_scheduler, val_loader, train_loader)

    
    if not LOAD_CHECKPOINT:
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
                
            loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
            loss_valid = validate(model, val_loader, accelerator)
            accelerator.print(f'Epoch:{epoch} , Train Loss: {loss_train}, Valid Loss: {loss_valid}')

            writer.add_scalar(f'perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar(f'perplexity_valid', loss_valid, epoch)
            writer.add_scalar(f'learning_rate', optimizer.param_groups[-1]['lr'], epoch)
                                 
            
        accelerator.save_state(STORE_PATH)
    

if __name__ == "__main__":
    main()
    
    