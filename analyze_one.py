import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup

# Local imports
import base_models
from Dataset import MixedData, ACLForLM, RestaurantForLM, Wikitext103
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
    load_layer_data,
)

NEED_CENTER = False
SAMPLE_BATCHES = 10

# train and validation size for pretrain
TRAIN_LEN = 50000
VAL_LEN = 500

# folder paths
CENTER_MODEL_PATH = "outputs/1227-bert-pre-save%10-wiki103/checkpoint-5000"
CENTER_PATH = os.path.join(CENTER_MODEL_PATH, 'centers-2.pth')
LOAD_FOLDER = "0103-mot2-vanilla-lora/checkpoint-20000"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
CONFIG_PATH = 'config/bert_a.json'

# training parameters
num_epochs = 50
lr = 5e-4
weight_decay = 0.01
decay = 0.8


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    # dataset = RestaurantForLM(config=config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    dataset = Wikitext103(config=config)
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    
    centers = load_layer_data(CENTER_PATH)
    model = base_models.BertWithMOE(config, centers)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    # prepare model and data on device
    model = accelerator.prepare(model)
        

    with torch.no_grad():
        layer_outputs = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break
            if i < SAMPLE_BATCHES:
                print(i)
                hidden_states = model.bert.embeddings(batch['input_ids'])
                for j in range(config.num_hidden_layers):
                    hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'])
                    if i == 0:
                        layer_outputs.append(hidden_states.to('cpu'))
                    else:
                        layer_outputs[j] = torch.cat([layer_outputs[j], hidden_states.to('cpu')], dim=0)
                batch = {key: tensor.to('cpu') for key, tensor in batch.items()}
                
                
    # Sparsity    
          
    for j, layer_output in enumerate(layer_outputs):    
        output = layer_output.mean(dim=1).mean(dim=0)
        normalized_output = (output - output.min()) / (output.max() - output.min())
        # Sort the normalized tensor in descending order
        sorted_tensor, sorted_indices = torch.sort(normalized_output, descending=True)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(768), sorted_tensor.numpy(), color='r')
        plt.title("Normalized and Sorted Tensor Elements")
        plt.xlabel("Element Index")
        plt.ylabel("Normalized Value")
        plt.grid(True)
        plt.savefig('split-res' + ' layer' + str(j) + '.png')
        
    # Orthogonal
    layer_ogds = []
    for j, layer_output in enumerate(layer_outputs):
        output = layer_output.mean(dim=1)
        norms = torch.norm(output, p=2, dim=1, keepdim=True)
        normalized_output = (output - output.min()) / (output.max() - output.min())
        dot_product_matrix = torch.matmul(normalized_output, normalized_output.T)
        upper_triangle = torch.triu(dot_product_matrix, diagonal=1)

        # Calculate the mean of the upper triangle
        mean_upper_triangle = torch.mean(upper_triangle[upper_triangle != 0])
        layer_ogds.append(mean_upper_triangle.item())
        
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(12), layer_ogds, color='b')
    plt.title("Sorted Tensor Elements")
    plt.xlabel("Element Index")
    plt.ylabel("Normalized Value")
    plt.grid(True)
    plt.savefig('ogds.png') 
    

if __name__ == "__main__":
    main()
    
    