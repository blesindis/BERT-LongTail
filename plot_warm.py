import os
import math
import umap
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
from transformer.FT_BERT import BertForSequenceClassification
from transformer.BERT import BertForMLM
from transformer.MoMoTModelRouterMTL import BertWithMoMoTModelRouterMTL
import matplotlib.pyplot as plt

# Local imports
from Dataset_ft_pure import SST2_pure
from Dataset_ft import SST2
from Dataset import Wikipedia, WikiSST
from utils.sample_utils import *
from utils.train_utils import (
    validate,
    set_seed,
    copy_parameters, 
    load_layer_data_last
)
NUM_EXPERTS = 4
SAMPLE_BATCHES = 1600
# folder paths
model_name = 'wiki-mtl'
CENTER_MODEL_PATH = "outputs/wiki(128)300w-bs64-1epoch-lr1-bert/checkpoint-5000"
CENTER_PATH = os.path.join(CENTER_MODEL_PATH, f'centers-{NUM_EXPERTS}-momoe-transformer-lastlayer.pth')
LOAD_FOLDER = "wiki(128)300w-bs64-1epoch-lr3-momot_model_router_mtl(lora-full-384, 4-4, att-ogd)_layer(full)_router5000/checkpoint-46875"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
LOAD_FOLDER_FT = "ft-sst-wiki(128)300w-bs64-1epoch-lr1-bert/checkpoint-5000"
LOAD_PATH_FT = os.path.join('outputs', LOAD_FOLDER_FT)
CONFIG_PATH = 'config/bert_bs1.json'

    
def plot_cluster(outputs, lengths):
    colors = ['b', 'r', 'green', 'y', 'grey']
    
    outputs_list = [a for sub_list in outputs for a in sub_list]
    outputs_all = torch.cat(outputs_list, dim=0)
    
    data_reducer = umap.UMAP(random_state=42)
    outputs_transformed = data_reducer.fit_transform(outputs_all.mean(dim=1))
    print(outputs_transformed.shape)
    plt.figure()
    for e in range(NUM_EXPERTS):
        if e == 0:
            start = 0
            end = lengths[0]
            # continue
            cluster_outputs = outputs_transformed[:lengths[0]]
        else:
            start = lengths[e-1]
            end = lengths[e]
            cluster_outputs = outputs_transformed[lengths[e-1]:lengths[e]]
        print(start, end)
        plt.scatter(cluster_outputs[:,0], cluster_outputs[:,1], color=colors[e], alpha=0.5, s=20, label=f'Expert {e}')
            
    plt.title("Cluster Distribution")
    plt.legend()
    plt.savefig(f'Cluster Distribution of {model_name}')


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    config_path = os.path.join(LOAD_PATH, 'config.json')
    config = BertConfig.from_json_file(config_path)
    
    
    """GET LOSSES"""
    center_model = BertForMLM(config)
    checkpoint = torch.load(os.path.join(CENTER_MODEL_PATH, 'pytorch_model.bin'))
    center_model.load_state_dict(checkpoint)
    center_model = accelerator.prepare(center_model)
    
    dataset = Wikipedia(config=config)
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    # val_loader_l, val_loader_p, val_loader_w = dataset.val_loader_legal, dataset.val_loader_pub, dataset.val_loader_wiki
    
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    
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
    # print(cluster_indexes[i] for i in range(NUM_EXPERTS))
    # print(cluster_indexes)
    outputs = [[] for _ in range(NUM_EXPERTS)]
    lengths = []
    sum_len = 0
    for i, indexes in enumerate(cluster_indexes):
        sum_len += len(indexes)
        for c in indexes:
            print(c)
            outputs[i].append(last_layer_outputs[c,:,:].unsqueeze(0))
        lengths.append(sum_len)
            
        # outputs[i] = [layer_outputs[c] for c in indexes]
    # print(cluster_indexes[i])
    # outputs = [layer_outputs[cluster_indexes[i], :, :] for i in range(NUM_EXPERTS)]
    # print(cluster_indexes)
    # outputs = [[] for _ in range(NUM_EXPERTS)]
    # for i, index in enumerate(cluster_indexes):
    #     outputs[index].append(last_layer_outputs[i])
    plot_cluster(outputs, lengths)
    

if __name__ == "__main__":
    main()
    
    