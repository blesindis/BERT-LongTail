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
from utils import *
from Dataset import MixedData, ACLForLM

# replay
REPLAY = True

# train and validation size for pretrain
TRAIN_LEN = 10000
VAL_LEN = 500

# folder paths
LOAD_FOLDER = "1102-mixed-stage1"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
CENTER_FILE = 'centers.pth'
CENTER_PATH = os.path.join(LOAD_PATH, CENTER_FILE)
REPLAY_FILE = 'replay_single_batch.pth'
REPLAY_PATH = os.path.join(LOAD_PATH, REPLAY_FILE)
CONFIG_PATH = 'config/bert.json'

STORE_FOLDER = '1114-mixed-stage2-replay-path-single_batch(1102-mixed-stage1)'
STORE_PATH = os.path.join('outputs', STORE_FOLDER)

# training parameters
num_epochs = 50
lr = 1e-4
weight_decay = 0


def validate(model, val_loader, accelerator):
    losses = []
    print(len(val_loader))
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
        


def main():
    config = BertConfig.from_json_file(CONFIG_PATH)
    
    dataset = ACLForLM(config, TRAIN_LEN, VAL_LEN)
    dataset_old = MixedData(config, TRAIN_LEN, VAL_LEN)

    centers = load_layer_data(CENTER_PATH)
    
    
    if REPLAY:
        replay_batch = torch.load(REPLAY_PATH, map_location='cuda') # {'input_ids: ', 'labels': , 'attention_mask': }
           
    model = base_models.BertWithMOE(config, centers)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    val_loader_old = dataset_old.val_loader
    num_updates = num_epochs * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', STORE_FOLDER))
    
    model, optimizer, lr_scheduler, train_loader, val_loader, val_loader_old = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader, val_loader_old)
    
    # train
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
                        
            # replay layerwise by path
            if REPLAY:                
                replay_batch = {key: tensor.to('cuda') for key, tensor in replay_batch.items()}
                loss, _ = model(**replay_batch)
                
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_valid = validate(model, val_loader, accelerator)
        loss_valid_old = validate(model, val_loader_old, accelerator)
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, Valid Loss Old: {loss_valid_old}')
        
        writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
        writer.add_scalar('perplexity_valid', loss_valid, epoch)
        writer.add_scalar('perplexity_valid_old', loss_valid_old, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    accelerator.save_state(STORE_PATH)
    

if __name__ == '__main__':
    main()