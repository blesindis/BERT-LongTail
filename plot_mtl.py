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
NUM_SAMPLES = 5000
# folder paths
model_name = 'wiki-mtl'
CENTER_MODEL_PATH = "outputs/wiki(128)300w-bs64-1epoch-lr1-bert/checkpoint-5000"
CENTER_PATH = os.path.join(CENTER_MODEL_PATH, f'centers-{NUM_EXPERTS}-momoe-transformer-lastlayer.pth')
LOAD_FOLDER = "wiki(128)300w-bs64-1epoch-lr3-momot_model_router_mtl(lora-full-384, 4-4, att-ogd)_layer(full)_router5000/checkpoint-46875"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
LOAD_FOLDER_FT = "ft-sst-wiki(128)300w-bs64-1epoch-lr1-bert/checkpoint-5000"
LOAD_PATH_FT = os.path.join('outputs', LOAD_FOLDER_FT)
CONFIG_PATH = 'config/bert_bs1.json'

# training parameters
num_epochs = 1
lr = 1e-4
weight_decay = 0.01


def compute_accuracy(logits, labels):
    # Convert logits to probabilities and then to predicted class indexes
    probs = torch.softmax(logits, dim=1)
    predicted_labels = torch.argmax(probs, dim=1)

    # Compare with true labels to find how many predictions were correct
    correct_predictions = (predicted_labels == labels).sum().item()

    # Calculate accuracy
    accuracy = correct_predictions / labels.size(0)  # labels.size(0) gives the batch size

    return accuracy


def get_losses(accelerator, model, data_loader, center_model, centers):
    losses = []
    h_common, h_tail = [], []
    h_tail_indexes = []
    model, data_loader = accelerator.prepare(model, data_loader)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i > NUM_SAMPLES:
                break
            cluster_list = [[] for _ in range(NUM_EXPERTS)]
                
            hidden_states = center_model.bert(batch['input_ids'], batch['attention_mask'])
            
            h = hidden_states.mean(dim=1)
            
            dist = torch.cdist(h.double(), centers.double())
            min_dist, min_indices = torch.min(dist, dim=1)
            print(min_dist)
            cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(NUM_EXPERTS)]  
            
            loss, _ = model(**batch, cluster_list=cluster_list)
            h_ = model.bert(batch['input_ids'], batch['attention_mask'], cluster_list=cluster_list)
            if loss.item() > 4.67:
                h_tail.append(h_.cpu())
                h_tail_indexes.append(i)
            else:
                h_common.append(h_.cpu())
            # print(i, loss.item())
            if math.isnan(loss.item()):
                continue
            losses.append(loss.item())
        print(h_tail_indexes)
    return losses, h_common, h_tail


def focus_tail(accelerator, model, data_loader, center_model, centers):
    losses = []
    h_common, h_tail = [], []
    dist_common, dist_tail = [], []
    h_tail_indexes = []
    model, data_loader = accelerator.prepare(model, data_loader)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i > NUM_SAMPLES:
                break
            cluster_list = [[] for _ in range(NUM_EXPERTS)]
                
            hidden_states = center_model.bert(batch['input_ids'], batch['attention_mask'])
            
            h = hidden_states.mean(dim=1)
            
            dist = torch.cdist(h.double(), centers.double())
            min_dist, min_indices = torch.min(dist, dim=1)
            cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(NUM_EXPERTS)]  
            
            loss, _ = model(**batch, cluster_list=cluster_list)
            h_ = model.bert(batch['input_ids'], batch['attention_mask'], cluster_list=cluster_list)
            if loss.item() > 4.67:
                h_tail.append(h_.cpu())
                dist_tail.append(min_dist.cpu().item())
                h_tail_indexes.append(i)
            else:
                h_common.append(h_.cpu())
                dist_common.append(min_dist.cpu().item())
            # print(i, loss.item())
            if math.isnan(loss.item()):
                continue
            losses.append(loss.item())
        
    return losses, h_common, h_tail, dist_common, dist_tail


def loss_dist(accelerator, model, data_loader, center_model, centers):
    losses = []
    h_common, h_tail = [], []
    dist_common, dist_tail = [], []
    h_tail_indexes = []
    model, data_loader = accelerator.prepare(model, data_loader)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i > NUM_SAMPLES:
                break
            cluster_list = [[] for _ in range(NUM_EXPERTS)]
                
            hidden_states = center_model.bert(batch['input_ids'], batch['attention_mask'])
            
            h = hidden_states.mean(dim=1)
            
            dist = torch.cdist(h.double(), centers.double())
            min_dist, min_indices = torch.min(dist, dim=1)
            cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(NUM_EXPERTS)]  
            
            loss, _ = model(**batch, cluster_list=cluster_list)
            h_ = model.bert(batch['input_ids'], batch['attention_mask'], cluster_list=cluster_list)
            if loss.item() > 4.67:
                h_tail.append(h_.cpu())
                dist_tail.append(min_dist.cpu().item())
                h_tail_indexes.append(i)
            else:
                h_common.append(h_.cpu())
                dist_common.append(min_dist.cpu().item())
            # print(i, loss.item())
            if math.isnan(loss.item()):
                continue
            losses.append(loss.item())
        
    return losses, h_common, h_tail, dist_common, dist_tail



def cluster_dist(accelerator, model, data_loader, center_model, centers):
    layer_cluster_list = []
    model, data_loader = accelerator.prepare(model, data_loader)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i > NUM_SAMPLES: 
                break
            cluster_list = [[] for _ in range(NUM_EXPERTS)]
                
            hidden_states = center_model.bert(batch['input_ids'], batch['attention_mask'])
            
            h = hidden_states.mean(dim=1)
            
            dist = torch.cdist(h.double(), centers.double())
            _, min_indices = torch.min(dist, dim=1)
            cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(NUM_EXPERTS)]   
            
            h_ = model.bert.embeddings(batch['input_ids'])
            # for l in range(6):
            #     h_ = model.bert.layers.layers[l](h_, batch['attention_mask'])
            for l in range(12):
                # c_, _ = model.bert.layers.layers[l].routing(h_)
                c_ = cluster_list
                counts = [len(c) for c in c_]
                if i == 0:
                    layer_cluster_list.append(counts)
                else:
                    layer_cluster_list[l] = [counts[c]+layer_cluster_list[l][c] for c in range(NUM_EXPERTS)]
                h_ = model.bert.layers.layers[l](h_, batch['attention_mask'], cluster_list=cluster_list)
    return layer_cluster_list


def cluster_last(accelerator, model, data_loader, center_model, centers):
    outputs = [[] for _ in range(NUM_EXPERTS)]
    losses = [[] for _ in range(NUM_EXPERTS)]
    model, data_loader = accelerator.prepare(model, data_loader)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i > NUM_SAMPLES: 
                break
            cluster_list = [[] for _ in range(NUM_EXPERTS)]
                
            hidden_states = center_model.bert(batch['input_ids'], batch['attention_mask'])
            
            h = hidden_states.mean(dim=1)
            
            dist = torch.cdist(h.double(), centers.double())
            _, min_indices = torch.min(dist, dim=1)
            cluster_list = [torch.eq(min_indices, i).nonzero(as_tuple=True)[0] for i in range(NUM_EXPERTS)]   
            
            h_ = model.bert(batch['input_ids'], batch['attention_mask'], cluster_list=cluster_list)
            loss, _ = model(**batch, cluster_list=cluster_list)
            # print(cluster_list)

            for c_index, c in enumerate(cluster_list):
                if len(c):
                    outputs[c_index].append(h_.to('cpu'))
                    losses[c_index].append(loss.item())
    return outputs, losses


def plot_tail(h_common, h_tail):
    h_all = h_common + h_tail
    h_all_tensor = torch.concat(h_all, dim=0)
    
    # _ = pca(torch.cat(h_common, dim=0))
    # _ = pca(torch.cat(h_tail, dim=0))
    # _ = pca(h_all_tensor)
    
    data_reducer = umap.UMAP(random_state=42)
    h_transformed = data_reducer.fit_transform(h_all_tensor.mean(dim=1))
    # h_transformed = pca(h_all_tensor, n_components=2)
    h_common_transformed = h_transformed[:len(h_common)]
    h_tail_transformed = h_transformed[len(h_common):]
    
    plt.figure()
    plt.scatter(h_common_transformed[:,0], h_common_transformed[:,1], color='b', alpha=0.5, s=1)
    plt.scatter(h_tail_transformed[:,0], h_tail_transformed[:,1], color='r', alpha=0.5, s=1)
    
    plt.title("Distribution of Common & Tail")
    plt.legend()
    plt.savefig('PCA Distribution of Common & Tail MTL.png')
    
    
def plot_cluster(outputs):
    colors = ['b', 'r', 'green', 'y', 'grey']
    
    outputs_list = [a for sub_list in outputs for a in sub_list]
    outputs_all = torch.cat(outputs_list, dim=0)
    
    data_reducer = umap.UMAP(random_state=42)
    outputs_transformed = data_reducer.fit_transform(outputs_all.mean(dim=1))
    # outputs_transformed = pca(outputs_all, n_components=2)
    
    plt.figure()
    for e in range(NUM_EXPERTS):
        if e == 0:
            cluster_outputs = outputs_transformed[:len(outputs[0])]
        else:
            cluster_outputs = outputs_transformed[len(outputs[e-1]):len(outputs[e])]
        plt.scatter(cluster_outputs[:,0], cluster_outputs[:,1], color=colors[e], alpha=0.5, s=1, label=f'Expert {e}')
            
    plt.title("Cluster Distribution")
    plt.legend()
    plt.savefig(f'PCA Cluster Distribution of {model_name}')


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
    
    centers = load_layer_data_last(CENTER_PATH)
    
    dataset = Wikipedia(config=config)
    base_model = BertWithMoMoTModelRouterMTL(config)
    checkpoint = torch.load(os.path.join(LOAD_PATH, 'pytorch_model.bin'))
    base_model.load_state_dict(checkpoint)
    
    # FOCUS TAIL
    print(NUM_SAMPLES)
    losses, h_common, h_tail, dist_common, dist_tail = focus_tail(accelerator, base_model, dataset.train_loader, center_model, centers)
    print(len(h_common), len(h_tail))    
    print(sum(dist_common)/len(dist_common), sum(dist_tail)/len(dist_tail))
    plot_tail(h_common, h_tail)
    
    # # PLOT TAIL
    # print(NUM_SAMPLES)
    # losses, h_common, h_tail = get_losses(accelerator, base_model, dataset.train_loader, center_model, centers)
    # print(len(h_common), len(h_tail))    
    # plot_tail(h_common, h_tail)
    
    # # PLOT CLUSTER
    # outputs, losses = cluster_last(accelerator, base_model, dataset.val_loader, center_model, centers)
    # mean_cluster_loss = [sum(l)/len(l) for l in losses]
    # print([len(l) for l in losses])
    # print(mean_cluster_loss)
    # plot_cluster(outputs)
    

if __name__ == "__main__":
    main()
    
    