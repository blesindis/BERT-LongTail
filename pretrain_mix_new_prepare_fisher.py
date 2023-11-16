import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers import BertConfig

# Local imports
import base_models
from utils import *
from Dataset import MixedData


DATASET_SIZE = 1008

EWC = True


def parse_arguments():
    """Parse arguments from command line."""
    parser = argparse.ArgumentParser(description="Training script for BERT model.")
    
    # Add arguments
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--load_path', type=str, default="1027-mixed-warmup", help='Path to load model parameters.')
    parser.add_argument('--store_path', type=str, default="1027-mixed", help='Path to store model parameters.')
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
    # perplexity = torch.exp(perplexity)
    
    return perplexity


def load_layer_data(path):
    layer_data_dict = torch.load(path, map_location='cuda')
    layer_data = list(layer_data_dict.values())
    layer_data = torch.tensor(np.array(layer_data)).to('cuda')
    return layer_data


def copy_parameters(source_module, target_module):
    for source_param, target_param in zip(source_module.parameters(), target_module.parameters()):
        target_param.data.copy_(source_param.data)


def compute_fisher_information(model, inputs, outputs_std, criterion, num_layer, num_expert):
    model.eval()
    fisher_information = {}
    for name, param in model.bert.layers.layers[num_layer].experts[num_expert].named_parameters():
        fisher_information[name] = torch.zeros_like(param)
    
    
    model.zero_grad()
    output = model.bert.layers.layers[num_layer].experts[num_expert](inputs, torch.ones(inputs.shape[0], inputs.shape[1]).to('cuda'))
    loss = criterion(outputs_std, output)
    loss.backward()
        
    with torch.no_grad():
        for name, param in model.bert.layers.layers[num_layer].experts[num_expert].named_parameters():
            fisher_information[name] += param.grad ** 2 / len(inputs)
                
    return fisher_information


def main():
    args = parse_arguments()
    
    config_path = args.config_path
    load_path = args.load_path
    store_path = args.store_path
    
    config = BertConfig.from_json_file(config_path)
    
    dataset = MixedData(config, dataset_len=DATASET_SIZE)
    centers = load_layer_data(os.path.join('outputs', load_path, 'centers.pth'))
    
    model = base_models.BertWithMOE(config, centers)
    checkpoint = torch.load(os.path.join('outputs', store_path, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    
    accelerator = Accelerator()
    
    model, train_loader, val_loader = accelerator.prepare(model, train_loader, val_loader)
    
    # get data from each cluster: inputs and outputs are correspondinged; I may save index instead ?    
    layer_cluster_index = [] # len = config.num_hidden_layers
    layer_outputs = [] # len = config.num_hidden_layers + 1
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            print(i)
            if i >= 4: 
                break
            
            h_ = model.bert.embeddings(batch['input_ids'])
            cluster_list = model.bert.layers.layers[0].routing(h_)  
            cluster_list = [np.array(l) + config.batch_size * i for l in cluster_list]
            print(cluster_list)
            if i == 0:
                layer_outputs.append(h_.to('cpu'))            
                layer_cluster_index.append(cluster_list)
            else:
                layer_outputs[0] = torch.cat([layer_outputs[0], h_.to('cpu')], dim=0)       
                layer_cluster_index[0] = [np.concatenate((layer_cluster_index[0][k], cluster_list[k])) for k in range(config.num_experts)]                
            
            for j in range(config.num_hidden_layers):
                h_ = model.bert.layers.layers[j](h_, batch['attention_mask'])
                if j + 1 < config.num_hidden_layers:
                    cluster_list = model.bert.layers.layers[j+1].routing(h_)                                        
                    cluster_list = [np.array(l) + config.batch_size * i for l in cluster_list]
                if i == 0:
                    layer_outputs.append(h_.to('cpu'))
                    layer_cluster_index.append(cluster_list)
                else:
                    layer_outputs[j+1] = torch.cat([layer_outputs[j+1], h_.to('cpu')], dim=0)                    
                    layer_cluster_index[j] = [np.concatenate((layer_cluster_index[j][k], cluster_list[k])) for k in range(config.num_experts)]                
                    
            batch = {key: tensor.to('cpu') for key, tensor in batch.items()}
    
    layer_cluster_FIM = {}
    
    for j in range(config.num_hidden_layers):        
        cluster_FIM = []
        for k in range(config.num_experts):
            cluster_data = layer_outputs[j][layer_cluster_index[j][k]]
            data_outputs = layer_outputs[j+1][layer_cluster_index[j][k]]
            print(cluster_data.shape)
            FIM = compute_fisher_information(model, cluster_data.to('cuda'), data_outputs.to('cuda'), nn.MSELoss(), j, k)
            cluster_FIM.append(FIM)
        
        layer_cluster_FIM['layer' + str(j)] = cluster_FIM        
    
    torch.save(layer_cluster_FIM, os.path.join('outputs', store_path, 'FIM.pth'))
    
if __name__ == '__main__':
    main()