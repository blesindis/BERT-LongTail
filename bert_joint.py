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
from Dataset import ACLForLM, RestaurantForLM
from utils.sample_utils import *
from utils.train_utils import (
    validate,
)
from utils.fim_utils import (
    compute_fim,
    compute_layer_fim, 
    calculate_overlap,
)

LOAD_CHECKPOINT = False

# train and validation size for pretrain
TRAIN_LEN = 10000
VAL_LEN = 500

# folder paths
STORE_FOLDER = "1127-bert-joint-lr1.5-100epoch"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CONFIG_PATH = 'config/bert_a.json'

# training parameters
num_epochs = 100
lr = 1.5e-4
weight_decay = 0


def main():
    config = BertConfig.from_json_file(CONFIG_PATH)
    dataset1 = RestaurantForLM(config, TRAIN_LEN, VAL_LEN)
    dataset2 = ACLForLM(config, TRAIN_LEN, VAL_LEN)
    
    model = base_models.BertForMLM(config)
    if LOAD_CHECKPOINT:
        checkpoint = torch.load(os.path.join(STORE_PATH, 'pytorch_model.bin'))
        model.load_state_dict(checkpoint)
    
    train_loader1, val_loader1 = dataset1.train_loader, dataset1.val_loader
    train_loader2, val_loader2 = dataset2.train_loader, dataset2.val_loader
    num_updates = num_epochs * len(train_loader1)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', STORE_FOLDER))
    
    # prepare model and data on device
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    train_loader1, val_loader1, train_loader2, val_loader2 = accelerator.prepare(train_loader1, val_loader1, train_loader2, val_loader2)

    
    if not LOAD_CHECKPOINT:
        for epoch in range(num_epochs):
            model.train()
            
            losses1 = []        
            losses2 = []
            for i, (batch1, batch2) in enumerate(zip(train_loader1, train_loader2)):    
                # dataset 1
                loss1, _ = model(**batch1)
                losses1.append(accelerator.gather(loss1.repeat(config.batch_size)))
                
                optimizer.zero_grad()
                accelerator.backward(loss1)
                optimizer.step()
                
                # dataset 2
                loss2, _ = model(**batch2)
                losses2.append(accelerator.gather(loss2.repeat(config.batch_size)))
                
                optimizer.zero_grad()
                accelerator.backward(loss2)
                optimizer.step()
                
                
                lr_scheduler.step()          
                                      
                            
            loss_train1 = torch.mean(torch.cat(losses1)[:len(train_loader1.dataset)])
            loss_train2 = torch.mean(torch.cat(losses2)[:len(train_loader2.dataset)])
            loss_train = (loss_train1 + loss_train2) / 2
            
            loss_valid1 = validate(model, val_loader1, accelerator)
            loss_valid2 = validate(model, val_loader2, accelerator)
            accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss1: {loss_valid1}, , Valid Loss2: {loss_valid2}')
            
            layer_fims1 = compute_layer_fim(model, val_loader1, config.num_hidden_layers)
            layer_fims2 = compute_layer_fim(model, val_loader2, config.num_hidden_layers)
            
            embed_fim1, embed_fim2 = compute_fim(model, val_loader1, 'embedding'), compute_fim(model, val_loader2, 'embedding')
            decoder_fim1, decoder_fim2 = compute_fim(model, val_loader1, 'decoder'), compute_fim(model, val_loader2, 'decoder')
            
            layer_param_similarity = [calculate_overlap(fim1, fim2) for fim1, fim2 in zip(layer_fims1, layer_fims2)]
            embed_similarity = calculate_overlap(embed_fim1, embed_fim2)
            decoder_similarity = calculate_overlap(decoder_fim1, decoder_fim2)

            writer.add_scalar(f'perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar(f'perplexity_valid1', loss_valid1, epoch)
            writer.add_scalar(f'perplexity_valid2', loss_valid2, epoch)
            writer.add_scalar(f'similarity of embedding', embed_similarity, epoch)
            for l, sim in enumerate(layer_param_similarity):
                writer.add_scalar(f'similarity of layer {l}', sim, epoch)
            writer.add_scalar(f'similarity of decoder', decoder_similarity, epoch)
            writer.add_scalar(f'learning_rate', optimizer.param_groups[-1]['lr'], epoch)
            
            if epoch % 10 == 0:
                accelerator.save_state(os.path.join(STORE_PATH, 'checkpoint-' + str(epoch)))
            
        accelerator.save_state(STORE_PATH)
    

if __name__ == "__main__":
    main()