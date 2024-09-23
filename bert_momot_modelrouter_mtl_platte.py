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
from transformer.MoMoTModelRouterMTLPlatte import BertWithMoMoTModelRouterMTLPlatte
from transformer.BERT import BertForMLM

# Local imports
from Dataset import BERTPretrain, LePuWiki
from utils.sample_utils import *
from utils.train_utils import (
    set_seed,
    load_layer_data_last,
)

NEED_CENTER = True
NUM_EXPERTS = 6
SAMPLE_BATCHES = 20

# train and validation size for pretrain
TRAIN_LEN = 50000
VAL_LEN = 500

# folder paths
CENTER_MODEL_PATH = "outputs/LPW(128)(280+10+10)w-bs64-1epoch-lr1-bert/checkpoint-5000"
CENTER_PATH = os.path.join(CENTER_MODEL_PATH, f'centers-{NUM_EXPERTS}-momoe-transformer-lastlayer.pth')
STORE_FOLDER = "LPW(128)(280+10+10)w-bs64-1epoch-lr3-momot_model_router_mtl_platte(same-param, common-full)_layer(6)_router5000"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CONFIG_PATH = 'config/mtl_platte.json'

# training parameters
num_epochs = 1
lr = 3e-4
weight_decay = 0.01


def validate(model, val_loader, accelerator, center_model, centers):
    
    losses = []    
    for i, batch in enumerate(val_loader):    
        with torch.no_grad():
            cluster_list = [[] for _ in range(NUM_EXPERTS)]
                
            hidden_states = center_model.bert(batch['input_ids'], batch['attention_mask'])
            
            h = hidden_states.mean(dim=1)
            
            dist = torch.cdist(h.double(), centers.double())
            _, min_indices = torch.min(dist, dim=1)
            cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(NUM_EXPERTS)]   
            
            loss, _ = model(**batch, cluster_list=cluster_list)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    loss = torch.mean(losses)
    
    return loss


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    
    dataset = LePuWiki(config=config)
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    val_loader_l, val_loader_p, val_loader_w = dataset.val_loader_legal, dataset.val_loader_pub, dataset.val_loader_wiki
    
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    val_loader_l, val_loader_p, val_loader_w = accelerator.prepare(val_loader_l, val_loader_p, val_loader_w)
    
    if NEED_CENTER:
        center_model = BertForMLM(config)
        checkpoint = torch.load(os.path.join(CENTER_MODEL_PATH, 'pytorch_model.bin'))
        center_model.load_state_dict(checkpoint)
        center_model = accelerator.prepare(center_model)
        
        layer_cluster_centers = {}
        with torch.no_grad():
            layer_outputs = []
            last_layer_outputs = None
            for i, batch in enumerate(train_loader):
                if i > SAMPLE_BATCHES:
                    break                
                hidden_states = center_model.bert(batch['input_ids'], batch['attention_mask'])                
                
                if i == 0:
                    last_layer_outputs = hidden_states.to('cpu')
                else:
                    last_layer_outputs = torch.cat([last_layer_outputs, hidden_states.to('cpu')], dim=0)  
                    
        cluster_indexes, cluster_centers = cluster_kmeans(last_layer_outputs.mean(dim=1), NUM_EXPERTS)
        layer_cluster_centers['layer_last'] = cluster_centers        
                    
        torch.save(layer_cluster_centers, CENTER_PATH)
        del center_model
    
    center_model = BertForMLM(config)
    checkpoint = torch.load(os.path.join(CENTER_MODEL_PATH, 'pytorch_model.bin'))
    center_model.load_state_dict(checkpoint)
    center_model = accelerator.prepare(center_model)
    
    centers = load_layer_data_last(CENTER_PATH)
    model = BertWithMoMoTModelRouterMTLPlatte(config)
    
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
            with torch.no_grad():          
                cluster_list = [[] for _ in range(NUM_EXPERTS)]
                
                hidden_states = center_model.bert(batch['input_ids'], batch['attention_mask'])
                
                h = hidden_states.mean(dim=1)
                
                dist = torch.cdist(h.double(), centers.double())
                _, min_indices = torch.min(dist, dim=1)
                cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(NUM_EXPERTS)]      
                        
            loss, _ = model(**batch, cluster_list=cluster_list)
            losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()          
            
            if step % 100 == 0:
                
                loss_train = torch.mean(torch.cat(losses)[:6400])
                loss_valid = validate(model, val_loader, accelerator, center_model, centers)
                loss_valid_l = validate(model, val_loader_l, accelerator, center_model, centers)
                loss_valid_p = validate(model, val_loader_p, accelerator, center_model, centers)
                loss_valid_w = validate(model, val_loader_w, accelerator, center_model, centers)
                accelerator.print(f'Iteration:{step}, Train Loss: {loss_train}, Valid Loss: {loss_valid}')
                
                writer.add_scalar(f'loss_train_epoch', loss_train, step)
                writer.add_scalar(f'loss_valid', loss_valid, step)
                writer.add_scalar(f'loss_valid_legal', loss_valid_l, step)
                writer.add_scalar(f'loss_valid_pubmed', loss_valid_p, step)
                writer.add_scalar(f'loss_valid_wiki', loss_valid_w, step)
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
    
    