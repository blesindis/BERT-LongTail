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
from Dataset import MixedData


DATASET_SIZE = 40000


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
    centers = load_layer_data(os.path.join('outputs', load_path, 'centers.pth'))
    
    model = base_models.BertWithMOE(config, centers=centers)
    checkpoint = torch.load(os.path.join('outputs', store_path, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    train_loader = dataset.train_loader
    
    accelerator = Accelerator()
    
    model, train_loader = accelerator.prepare(model, train_loader)
    
    # get data from each cluster: inputs and outputs are correspondinged; I may save index instead ?
    layer_outputs_clustered = [[] for _ in range(config.num_hidden_layers)]
    layer_cluster_index = [] # len = config.num_hidden_layers
    layer_outputs = [] # len = config.num_hidden_layers + 1
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            print(i)
            if i >= 100: 
                break
            
            h_ = model.bert.embeddings(batch['input_ids'])
            cluster_list = model.bert.layers.layers[0].routing(h_)  
            cluster_list = [np.array(l) + config.batch_size * i for l in cluster_list]
            
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
    
    layer_cluster_centers = {}        
    layer_sample_inputs = {}
    layer_sample_outputs = {}
    
    
    for j in range(config.num_hidden_layers):
        cluster_centers = []        
        sample_inputs = []
        sample_outputs = []
        for k in range(config.num_experts):
            cluster_data = layer_outputs[j][layer_cluster_index[j][k]]
            data_outputs = layer_outputs[j+1][layer_cluster_index[j][k]]
            
            cluster_pca = pca(cluster_data, n_components=16)
            _, cluster_center = cluster_kmeans(cluster_pca, 1)
            cluster_centers.append(cluster_center)
            
            sample_indexes = sample_by_cluster(cluster_pca, 4, 4)
            sample_inputs.append(cluster_data[sample_indexes])
            sample_outputs.append(data_outputs[sample_indexes])
        
        layer_cluster_centers['layer' + str(j)] = cluster_centers
        layer_sample_inputs['layer' + str(j)] = sample_inputs
        layer_sample_outputs['layer' + str(j)] = sample_outputs
    
    torch.save(layer_cluster_centers, os.path.join('outputs', store_path, 'new_centers.pth'))
    torch.save(layer_sample_inputs, os.path.join('outputs', store_path, 'sample_inputs.pth'))
    torch.save(layer_sample_outputs, os.path.join('outputs', store_path, 'sample_outputs.pth'))
    
if __name__ == '__main__':
    main()