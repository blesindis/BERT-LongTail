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
from transformer.MoMoTSwitchMTL import BertWithMoMoTSwitchMTL
from transformer.BERT import BertForMLM

# Local imports
from Dataset import BERTPretrain, LePuWiki, Wikipedia, WikiSST
from utils.sample_utils import *
from utils.train_utils import (
    set_seed,
    load_layer_data_last,
    validate
)

NEED_CENTER = False
NUM_EXPERTS = 6
SAMPLE_BATCHES = 20

# train and validation size for pretrain
TRAIN_LEN = 50000
VAL_LEN = 500

# folder paths
STORE_FOLDER = "wikisst(128)300w-bs64-1epoch-lr3-momot_switch_mtl(same-param, common-lora)_layer(6)_router5000"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CONFIG_PATH = 'config/mtl.json'

# training parameters
num_epochs = 1
lr = 3e-4
weight_decay = 0.01


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    
    dataset = WikiSST(config=config)
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    # val_loader_l, val_loader_p, val_loader_w = dataset.val_loader_legal, dataset.val_loader_pub, dataset.val_loader_wiki
    
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    # val_loader_l, val_loader_p, val_loader_w = accelerator.prepare(val_loader_l, val_loader_p, val_loader_w)
    
    model = BertWithMoMoTSwitchMTL(config)
    
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
            
            # for l in range(config.num_common_layers, config.num_hidden_layers):
            #     w_unique = torch.cat([model.bert.layers.layers[l].unique_attn[e].attention.self.weight for e in range(config.unique_experts)], dim=-1).view(-1)
            #     w_common = model.bert.layers.layers[l].unique_attn.attention.self.weight.view(-1)
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()          
            
            if step % 100 == 0:
                
                loss_train = torch.mean(torch.cat(losses)[:6400])
                loss_valid = validate(model, val_loader, accelerator)
                # loss_valid_l = validate(model, val_loader_l, accelerator, center_model, centers)
                # loss_valid_p = validate(model, val_loader_p, accelerator, center_model, centers)
                # loss_valid_w = validate(model, val_loader_w, accelerator, center_model, centers)
                accelerator.print(f'Iteration:{step}, Train Loss: {loss_train}, Valid Loss: {loss_valid}')
                
                writer.add_scalar(f'loss_train_epoch', loss_train, step)
                writer.add_scalar(f'loss_valid', loss_valid, step)
                # writer.add_scalar(f'loss_valid_legal', loss_valid_l, step)
                # writer.add_scalar(f'loss_valid_pubmed', loss_valid_p, step)
                # writer.add_scalar(f'loss_valid_wiki', loss_valid_w, step)
                writer.add_scalar(f'learning_rate', optimizer.param_groups[-1]['lr'], step)
                
                losses = []
            if step % 5000 == 0:
                config.save_pretrained(os.path.join(STORE_PATH, 'checkpoint-' + str(step)))
                model_dir = os.path.join(STORE_PATH, f'checkpoint-{step}') 
                # os.makedirs(model_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(model_dir, 'pytorch_model.bin'))
            step += 1
    
    config.save_pretrained(os.path.join(STORE_PATH, 'checkpoint-' + str(step)))
    model_dir = os.path.join(STORE_PATH, f'checkpoint-{step}') 
    # os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'pytorch_model.bin'))
    

if __name__ == "__main__":
    main()
    
    