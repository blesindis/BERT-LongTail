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


DATASET_SIZE = 10000


def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Training script for BERT model.")
    
    # Add arguments
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--load_path', type=str, default="1027-mixed-warmup", help='Path to load model parameters.')
    parser.add_argument('--store_path', type=str, default="1030-dim-moe", help='Path to store model parameters.')
    parser.add_argument('--config_path', type=str, default='config/bert.json', help='Path to BERT config file.')
    
    return parser.parse_args()


def validate(model, val_loader, accelerator):
    losses = []
    print(len(val_loader))
    for i, batch in enumerate(val_loader):    
        with torch.no_grad():
            loss, _ = model(**batch)
        losses.append(accelerator.gather(loss.repeat(len(batch))))
    
    losses = torch.cat(losses)[:len(val_loader.dataset)]
    perplexity = torch.mean(losses)
    # perplexity = torch.exp(perplexity)
    
    return perplexity


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    layer_data = torch.tensor(np.array(layer_data))
    layer_data = layer_data.to('cuda')
    return layer_data


def copy_parameters(source_module, target_module):
    for source_param, target_param in zip(source_module.parameters(), target_module.parameters()):
        target_param.data.copy_(source_param.data)


def main():
    args = parse_arguments()
    
    num_epochs = args.num_epochs
    config_path = args.config_path
    load_path = args.load_path
    store_path = args.store_path
    
    config = BertConfig.from_json_file(config_path)
    
    dataset = MixedData(config, dataset_len=DATASET_SIZE)
    unique_centers = load_layer_data(os.path.join('outputs', load_path, 'unique_centers.pth'))
    # pca_components = load_layer_data(os.path.join('outputs', load_path, 'pca_components.pth'))
    layer_data_dict = torch.load(os.path.join('outputs', load_path, 'pca_components.pth'), map_location='cuda')
    layer_data = list(layer_data_dict.values())
    pca_components = torch.cat(layer_data, dim=0).view(len(layer_data), *layer_data[0].shape)
    
    model = base_models.BertWithDimMOE(config, unique_centers, pca_components)
    
    train_loader, val_loader, val_loader_set = dataset.train_loader, dataset.val_loader, dataset.val_loader_set
    num_updates = num_epochs * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0., betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.06, num_training_steps=num_updates)
    accelerator = Accelerator()
    writer = SummaryWriter(os.path.join('log', store_path))
    
    # """initialize model parameters by copy bert"""
    # model_warmup = base_models.BertForMLM(config)
    # checkpoint = torch.load(os.path.join('outputs', load_path, 'pytorch_model.bin'))
    # model_warmup.load_state_dict(checkpoint)
    # # Common: Embedding & Decoder
    # copy_parameters(model_warmup.bert.embeddings, model.bert.embeddings)
    # copy_parameters(model_warmup.head, model.head)
        
    # # Experts: 
    # for i in range(config.num_hidden_layers):
    #     for j in range(config.num_experts):
    #         copy_parameters(model_warmup.bert.encoders.layers[i], model.bert.layers.layers[i].experts[j])
    
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    
    for i, set in enumerate(val_loader_set):
        val_loader_set[i] = accelerator.prepare(set)
    
    
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

        loss_train = torch.mean(torch.cat(losses)[:len(train_loader.dataset)])
        loss_valid = validate(model, val_loader, accelerator)
        accelerator.print(f'Epoch:{epoch} ({i} Updates), Train Loss: {loss_train}, Valid Loss: {loss_valid}')
        
        for index, val_set in enumerate(val_loader_set):
            loss_valid_per_set = validate(model, val_set, accelerator)
            accelerator.print(f'Dataset: {index}, Valid Loss: {loss_valid_per_set}')

        writer.add_scalar('perplexity_train_epoch', loss_train, epoch)
        writer.add_scalar('perplexity_valid', loss_valid, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], epoch)
        
    accelerator.save_state(os.path.join('outputs', store_path))
    

if __name__ == '__main__':
    main()