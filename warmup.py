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
from Dataset import MixedData, ACLForLM
from utils import *


SEED = 42

LOAD_CHECKPOINT = True

# cluster config
SAMPLE_BATCHES = 50
NUM_CLUSTERS = 4

# train and validation size for pretrain
TRAIN_LEN = 2000
VAL_LEN = 500

# folder paths
STORE_FOLDER = "1027-mixed-warmup"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CENTER_FILE = '4expert-centers.pth'
CENTER_PATH = os.path.join(STORE_PATH, CENTER_FILE)
CONFIG_PATH = 'config/bert.json'

# training parameters
num_epochs = 50
lr = 1e-4
weight_decay = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(model, val_loader, accelerator):
    losses = []
    for i, batch in enumerate(val_loader):        
        with torch.no_grad():
            loss, _ = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    loss = torch.mean(losses)
    
    return loss
    

def main():
    set_seed(SEED)
       
    config = BertConfig.from_json_file(CONFIG_PATH)
    dataset = MixedData(config=config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    
    # 1. Train a tradition Bert using data from each dataset
    model = base_models.BertForMLM(config)
    if LOAD_CHECKPOINT:
        checkpoint = torch.load(os.path.join(STORE_PATH, 'pytorch_model.bin'))
        model.load_state_dict(checkpoint)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    num_updates = num_epochs * len(train_loader)
    
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
            accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}')

            writer.add_scalar(f'perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar(f'perplexity_valid', loss_valid, epoch)
            writer.add_scalar(f'learning_rate', optimizer.param_groups[-1]['lr'], epoch)
            
        accelerator.print("Warm-up training stage finished.")
        accelerator.save_state(os.path.join(STORE_PATH))
    
    # 2. get cluster centers from each layer outputs
    
    layer_cluster_centers = {}

    with torch.no_grad():
        layer_outputs = []
        for i, batch in enumerate(train_loader):
            if i < SAMPLE_BATCHES:
                hidden_states = model.bert.embeddings(batch['input_ids'])
                for j in range(config.num_hidden_layers):
                    hidden_states = model.bert.encoders.layers[j](hidden_states, batch['attention_mask'])
                    if i == 0:
                        layer_outputs.append(hidden_states.to('cpu'))
                    else:
                        layer_outputs[j] = torch.cat([layer_outputs[j], hidden_states.to('cpu')], dim=0)
                batch = {key: tensor.to('cpu') for key, tensor in batch.items()}
                
    for j, layer_output in enumerate(layer_outputs):                 
        _, cluster_centers = cluster_kmeans(layer_output.mean(dim=1), NUM_CLUSTERS)
        layer_cluster_centers['layer' + str(j)] = cluster_centers
        
            
    torch.save(layer_cluster_centers, CENTER_PATH)
    

if __name__ == "__main__":
    main()