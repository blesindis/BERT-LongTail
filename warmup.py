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
from Dataset import MixedData
from utils import *


SEED = 42

WARMUP_SIZE = 2000

LOAD_CHECKPOINT = True
DIM_EXPERT = True

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Training script for BERT model.")
    
    # Add arguments
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--store_path', type=str, default='1027-mixed-warmup', help='Path to store output model.')
    parser.add_argument('--config_path', type=str, default='config/bert.json', help='Path to BERT config file.')

    return parser.parse_args()


def validate(model, val_loader, accelerator):
    losses = []
    for i, batch in enumerate(val_loader):        
        with torch.no_grad():
            loss, _ = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    
    return perplexity
    

def main():
    set_seed(SEED)
    
    args = parse_arguments()
    
    config = BertConfig.from_json_file(args.config_path)
    dataset = MixedData(config=config, dataset_len=WARMUP_SIZE)
    
    num_epochs = args.num_epochs
    store_path = args.store_path
    
    # 1. Train a tradition Bert using data from each dataset
    model = base_models.BertForMLM(config)
    if LOAD_CHECKPOINT:
        checkpoint = torch.load(os.path.join('outputs', store_path, 'pytorch_model.bin'))
        model.load_state_dict(checkpoint)
    
    train_loader, val_loader, val_loader_set = dataset.train_loader, dataset.val_loader, dataset.val_loader_set
    num_updates = num_epochs * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', store_path))
    
    # prepare model and data on device
    model, optimizer, lr_scheduler, val_loader, train_loader = accelerator.prepare(model, optimizer, lr_scheduler, val_loader, train_loader)
    for i, set in enumerate(val_loader_set):
        val_loader_set[i] = accelerator.prepare(set)

    
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
            
            for index, val_set in enumerate(val_loader_set):
                loss_valid_per_set = validate(model, val_set, accelerator)
                accelerator.print(f'Dataset: {index}, Valid Loss: {loss_valid_per_set}')
                writer.add_scalar(f'Dataset: {index}', loss_valid_per_set, epoch)

            writer.add_scalar(f'perplexity_train_epoch', loss_train, epoch)
            writer.add_scalar(f'perplexity_valid', loss_valid, epoch)
            writer.add_scalar(f'learning_rate', optimizer.param_groups[-1]['lr'], epoch)
            
        accelerator.print("Warm-up training stage finished.")
        accelerator.save_state(os.path.join('outputs', store_path))
    
    # 2. get cluster centers from each layer outputs
    
    layer_cluster_centers = {}
    layer_pca_components = {}
    layer_unique_centers = {}
    
    with torch.no_grad():
        layer_outputs = []
        for i, batch in enumerate(train_loader):
            if i < 100:
                hidden_states = model.bert.embeddings(batch['input_ids'])
                for j in range(config.num_hidden_layers):
                    hidden_states = model.bert.encoders.layers[j](hidden_states, batch['attention_mask'])
                    if i == 0:
                        layer_outputs.append(hidden_states.to('cpu'))
                    else:
                        layer_outputs[j] = torch.cat([layer_outputs[j], hidden_states.to('cpu')], dim=0)
                batch = {key: tensor.to('cpu') for key, tensor in batch.items()}
                
    for j, layer_output in enumerate(layer_outputs):               
        if DIM_EXPERT:            
            components = pca_all(layer_output)
            layer_pca_components['layer' + str(j)] = components
            
            n_components = 16
            layer_unique = torch.matmul(layer_output.mean(dim=1), components[:n_components].T)
            _, unique_centers = cluster_kmeans(layer_unique, config.num_experts)
            
            layer_unique_centers['layer' + str(j)] = unique_centers
            
        else:
            layer_pca = pca(layer_output, n_components=16)
            _, cluster_centers = cluster_kmeans(layer_pca, config.num_experts)
            layer_cluster_centers['layer' + str(j)] = cluster_centers
        
            
    if DIM_EXPERT:
        torch.save(layer_pca_components, os.path.join('outputs', store_path, 'pca_components.pth'))
        torch.save(layer_unique_centers, os.path.join('outputs', store_path, 'unique_centers.pth'))
    else:
        torch.save(layer_cluster_centers, os.path.join('outputs', store_path, 'centers.pth'))
    

if __name__ == "__main__":
    main()