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
from utils.train_utils import (
    validate,
    load_layer_data,
)
from Dataset import MixedData, ACLForLM

LAMBDA = 1000

# train and validation size for pretrain
TRAIN_LEN = 10000
VAL_LEN = 500

# folder paths
LOAD_FOLDER = "1102-mixed-stage1"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
CENTER_FILE = 'centers.pth'
CENTER_PATH = os.path.join(LOAD_PATH, CENTER_FILE)
FIM_FILE = 'fim-1.pth'
FIM_PATH = os.path.join(LOAD_PATH, FIM_FILE)
CONFIG_PATH = 'config/bert.json'

STORE_FOLDER = '1121-mixed-stage2-ewc-1000(1102-mixed-stage1)'
STORE_PATH = os.path.join('outputs', STORE_FOLDER)

# training parameters
num_epochs = 50
lr = 1e-4
weight_decay = 0
        
        
def ewc(model, original_model, fim):
    loss = 0
    original_model_params = {name: param.clone() for name, param in original_model.named_parameters()}
    for name, param in model.named_parameters():
        loss += (fim[name] * (param - original_model_params[name]) ** 2).sum()        
    return LAMBDA * loss


def normalize_fim(fim):    
    for name in fim.keys():
        sum = torch.sum(fim[name])
        fim[name] /= sum
        fim[name] = 1 - fim[name]
        
    return fim

def main():
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    
    dataset = ACLForLM(config, TRAIN_LEN, VAL_LEN)
    dataset_old = MixedData(config, TRAIN_LEN, VAL_LEN)
    
    centers = load_layer_data(CENTER_PATH)
    
    fim = torch.load(FIM_PATH, map_location='cuda')    
    # fim = normalize_fim(fim)
    
    model = base_models.BertWithMOE(config, centers)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    model_origin = base_models.BertWithMOE(config, centers)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    val_loader_old = dataset_old.val_loader
    num_updates = num_epochs * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', STORE_PATH))
    
    model, optimizer, lr_scheduler, train_loader, val_loader, val_loader_old = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, val_loader_old)
    model_origin = accelerator.prepare(model_origin)
    
    # train
    for epoch in range(num_epochs):
        model.train()
        
        losses = []
        for i, batch in enumerate(train_loader):           
            loss, _ = model(**batch)
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            

                        
            optimizer.zero_grad()
            accelerator.backward(loss)
            # optimizer.step()
                        
            ewc_loss = ewc(model, model_origin, fim)
            # optimizer.zero_grad()
            ewc_loss.backward()
            optimizer.step()
                        
            lr_scheduler.step()  

        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        # loss_train = 0
        loss_valid = validate(model, val_loader, accelerator)
        loss_valid_old = validate(model, val_loader_old, accelerator)
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Valid Loss Old: {loss_valid_old}')
        

        writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
        writer.add_scalar('perplexity_valid', loss_valid, epoch)
        writer.add_scalar('perplexity_valid_old', loss_valid_old, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    accelerator.save_state(os.path.join('outputs', STORE_PATH))
    

if __name__ == '__main__':
    main()