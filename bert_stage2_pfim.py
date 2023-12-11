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

# train and validation size for pretrain
TRAIN_LEN = 10000
VAL_LEN = 500

# folder paths
LOAD_FOLDER = "1127-bert-stage1-8heads"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
STORE_FOLDER = "1127-bert-stage2-8heads"
STORE_PATH = os.path.join('outputs', STORE_FOLDER)
CONFIG_PATH = 'config/bert_a.json'

# training parameters
num_epochs = 50
lr = 1e-4
weight_decay = 0


def main():
    config = BertConfig.from_json_file(CONFIG_PATH)
    dataset = ACLForLM(config, TRAIN_LEN, VAL_LEN)
    stage1_dataset = RestaurantForLM(config, TRAIN_LEN, VAL_LEN)
    
    model = base_models.BertForMLM(config)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    stage1_val_loader = stage1_dataset.val_loader
    num_updates = num_epochs * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', STORE_FOLDER))
    
    # prepare model and data on device
    model, optimizer, lr_scheduler, val_loader, train_loader = accelerator.prepare(model, optimizer, lr_scheduler, val_loader, train_loader)
    stage1_val_loader = accelerator.prepare(stage1_val_loader)
    
    
    # prepare for param choosing
    def calculate_overlap_percentage(fim1, fim2, threshold):
        important_params1 = fim1 > threshold
        important_params2 = fim2 > threshold
        overlap = (important_params1 & important_params2).float().mean()
        return overlap
    
    def find_threshold(fim1, fim2, cosine_similarity_value, tolerance=0.01, max_iter=100):
        low, high = 0, max(fim1.max(), fim2.max())
        for _ in range(max_iter):
            threshold = (low + high) / 2
            overlap_percentage = calculate_overlap_percentage(fim1, fim2, threshold)
            
            if abs(overlap_percentage - cosine_similarity_value) < tolerance:
                return threshold
            elif overlap_percentage > cosine_similarity_value:
                high = threshold
            else:
                low = threshold
        print(f'threshold: {threshold}, overlap percentage{overlap_percentage}, similarity{cosine_similarity_value}')
        return threshold
    
    # Calculate fim
    layer_fims1 = compute_layer_fim(model, val_loader, config.num_hidden_layers)
    layer_fims2 = compute_layer_fim(model, stage1_val_loader, config.num_hidden_layers)
    
    embed_fim1, embed_fim2 = compute_fim(model, val_loader, 'embedding'), compute_fim(model, stage1_val_loader, 'embedding')
    decoder_fim1, decoder_fim2 = compute_fim(model, val_loader, 'decoder'), compute_fim(model, stage1_val_loader, 'decoder')
        
    # calculate similarity
    layer_param_similarity = [calculate_overlap(fim1, fim2) for fim1, fim2 in zip(layer_fims1, layer_fims2)]
    embed_similarity = calculate_overlap(embed_fim1, embed_fim2)
    decoder_similarity = calculate_overlap(decoder_fim1, decoder_fim2)
    
    # calculate threshold
    layer_threshold = [find_threshold(layer_fims1[l], layer_fims2[l], layer_param_similarity[l]) for l in range(config.num_hidden_layers)]
    embed_threshold = find_threshold(embed_fim1, embed_fim2, embed_similarity)
    decoder_threshold = find_threshold(decoder_fim1, decoder_fim2, decoder_similarity)


    # for epoch in range(num_epochs):
    #     model.train()
        
    #     losses = []        
    #     for i, batch in enumerate(train_loader):                      
    #         loss, _ = model(**batch)
    #         losses.append(accelerator.gather(loss.repeat(config.batch_size)))
            
    #         optimizer.zero_grad()
    #         accelerator.backward(loss)
    #         optimizer.step()
    #         lr_scheduler.step()                
                        
    #     loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
    #     loss_valid = validate(model, val_loader, accelerator)
    #     loss_valid_stage1 = validate(model, stage1_val_loader, accelerator)
    #     accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}, stage1 valid loss: {loss_valid_stage1}')
        
    #     layer_fims1 = compute_layer_fim(model, val_loader, config.num_hidden_layers)
    #     layer_fims2 = compute_layer_fim(model, stage1_val_loader, config.num_hidden_layers)
        
    #     embed_fim1, embed_fim2 = compute_fim(model, val_loader, 'embedding'), compute_fim(model, stage1_val_loader, 'embedding')
    #     decoder_fim1, decoder_fim2 = compute_fim(model, val_loader, 'decoder'), compute_fim(model, stage1_val_loader, 'decoder')
        
    #     layer_param_similarity = [calculate_overlap(fim1, fim2) for fim1, fim2 in zip(layer_fims1, layer_fims2)]
    #     embed_similarity = calculate_overlap(embed_fim1, embed_fim2)
    #     decoder_similarity = calculate_overlap(decoder_fim1, decoder_fim2)

    #     writer.add_scalar(f'perplexity_train_epoch', loss_train, epoch)
    #     writer.add_scalar(f'perplexity_valid', loss_valid, epoch)
    #     writer.add_scalar(f'perplexity_valid_stage1', loss_valid_stage1, epoch)
    #     writer.add_scalar(f'similarity of embedding', embed_similarity, epoch)
    #     for l, sim in enumerate(layer_param_similarity):
    #         writer.add_scalar(f'similarity of layer {l}', sim, epoch)
    #     writer.add_scalar(f'similarity of decoder', decoder_similarity, epoch)
    #     writer.add_scalar(f'learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    # accelerator.save_state(os.path.join(STORE_PATH))
    

if __name__ == "__main__":
    main()