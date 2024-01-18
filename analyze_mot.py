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
LOAD_FOLDER = "0109-mot2-useattn-normalattn(corrected)-l1+ogd/checkpoint-10000"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
LOAD_FOLDER1= "0109-mot2-useattn-normalattn(corrected)-mask+ogd/checkpoint-10000"
LOAD_PATH1 = os.path.join('outputs', LOAD_FOLDER1)
CONFIG_PATH = 'config/bert_a.json'


def get_model(load_path, config, accelerator):
    centers = load_layer_data(CENTER_PATH)
    model = base_models.BertWithMoTAttn(config, centers)
    checkpoint = torch.load(os.path.join(load_path, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)
    
    # prepare model and data on device
    model = accelerator.prepare(model) 
    return model


def get_layer_outputs(model, train_loader, config):
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
    return layer_outputs


def get_layer_outputs_moe(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        cluster_lists = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break
            if i < SAMPLE_BATCHES:
                print(i)
                hidden_states = model.bert.embeddings(batch['input_ids'])
                for j in range(config.num_hidden_layers):
                    cluster = model.bert.layers.layers[j].routing(hidden_states)
                    cluster[0] = [d + 64 * i for d in cluster[0]]
                    cluster[1] = [d + 64 * i for d in cluster[1]]
                    hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'])
                    if i == 0:
                        layer_outputs.append(hidden_states.to('cpu'))
                        cluster_lists.append(cluster)
                    else:
                        layer_outputs[j] = torch.cat([layer_outputs[j], hidden_states.to('cpu')], dim=0)
                        cluster_lists[j][0] += cluster[0]
                        cluster_lists[j][1] += cluster[1]
                batch = {key: tensor.to('cpu') for key, tensor in batch.items()}
    return layer_outputs, cluster_lists


def get_layer_attn_outputs_moe(model, train_loader, config):
    with torch.no_grad():
        layer_outputs = []
        cluster_lists = []
        for i, batch in enumerate(train_loader):
            if i >= SAMPLE_BATCHES:
                break
            if i < SAMPLE_BATCHES:
                print(i)
                hidden_states = model.bert.embeddings(batch['input_ids'])
                for j in range(config.num_hidden_layers):
                    cluster = model.bert.layers.layers[j].routing(hidden_states, batch['attention_mask'])
                    print(len(cluster[0]), len(cluster[1]))
                    att_outputs = hidden_states.new_zeros(hidden_states.shape)
                    norm_outputs_0 = model.bert.layers.layers[j].experts[0].attention.LayerNorm(hidden_states[cluster[0],:,:])
                    norm_outputs_1 = model.bert.layers.layers[j].experts[1].attention.LayerNorm(hidden_states[cluster[1],:,:])  
                    h_0 = model.bert.layers.layers[j].experts[0].attention.self(norm_outputs_0, batch['attention_mask'][cluster[0],:]) 
                    h_1 = model.bert.layers.layers[j].experts[1].attention.self(norm_outputs_1, batch['attention_mask'][cluster[1],:]) 
                    att_outputs[cluster[0],:,:] = model.bert.layers.layers[j].experts[0].attention.dense(h_0)
                    att_outputs[cluster[1],:,:] = model.bert.layers.layers[j].experts[1].attention.dense(h_1)
                    cluster[0] = [d + 64 * i for d in cluster[0]]
                    cluster[1] = [d + 64 * i for d in cluster[1]]
                    hidden_states = model.bert.layers.layers[j](hidden_states, batch['attention_mask'])
                    
                    if i == 0:
                        layer_outputs.append(att_outputs.to('cpu'))
                        cluster_lists.append(cluster)
                    else:
                        layer_outputs[j] = torch.cat([layer_outputs[j], att_outputs.to('cpu')], dim=0)
                        cluster_lists[j][0] += cluster[0]
                        cluster_lists[j][1] += cluster[1]
                batch = {key: tensor.to('cpu') for key, tensor in batch.items()}
    return layer_outputs, cluster_lists


def get_layer_attn_params_moe(model, config):
    layer_params = []
    for l in range(config.num_hidden_layers):
        expert_params = []
        for j in range(config.num_experts):
            w_j = torch.cat([model.bert.layers.layers[l].experts[j].attention.self.transform_layers[h].weight for h in range(config.num_attention_heads)], dim=-1).view(-1)
            expert_params.append(w_j.detach().cpu())
        layer_params.append(expert_params)
    return layer_params
                       

def get_layer_ogd(layer_outputs, cluster_lists):
    layer_ogds = []
    for j, layer_output in enumerate(layer_outputs):
        output = layer_output.mean(dim=1)
        if len(cluster_lists[j][0]) == 0 or len(cluster_lists[j][1]) == 0:
            ogd = 1
            layer_ogds.append(ogd)
        else:
            output1, output2 = output[cluster_lists[j][0]], output[cluster_lists[j][1]]
            output1, output2 = output1.mean(dim=0), output2.mean(dim=0)
            normalized_output1 = output1 / torch.norm(output1)
            normalized_output2 = output2 / torch.norm(output2)
            ogd = torch.dot(normalized_output1, normalized_output2)
            layer_ogds.append(ogd.item())
    return layer_ogds


def get_sorted_tensor(output):
    output = output.mean(dim=1).mean(dim=0)
    output = torch.abs(output)
    # sorted_tensor, sorted_indices = torch.sort(output, descending=True) 
    normalized_output = (output - output.min()) / (output.max() - output.min())
    sorted_tensor, sorted_indices = torch.sort(normalized_output, descending=True) 
    return sorted_tensor


def get_sorted_tensor_param(output):
    output = torch.abs(output)
    sorted_tensor, sorted_indices = torch.sort(output, descending=True) 
    normalized_output = (output - output.min()) / (output.max() - output.min())
    sorted_tensor, sorted_indices = torch.sort(normalized_output, descending=True) 
    return sorted_tensor


def get_layer_param_ogd_moe(layer_params):
    layer_ogds = []
    for param in layer_params:
        w1 = param[0]
        w2 = param[1]
        w1 = w1 / torch.norm(w1)
        w2 = w2 / torch.norm(w2)
        ogd = torch.dot(w1, w2)
        layer_ogds.append(torch.abs(ogd))
    return layer_ogds


def pic_representation(config, accelerator, train_loader):
    model = get_model(LOAD_PATH, config, accelerator)
    layer_outputs, cluster_lists = get_layer_attn_outputs_moe(model, train_loader, config)
                
    model1 = get_model(LOAD_PATH1, config, accelerator)               
    layer_outputs1, cluster_lists1 = get_layer_attn_outputs_moe(model1, train_loader, config)
                
                
    # Sparsity    
          
    for j in range(config.num_hidden_layers):  
        sorted_tensor = get_sorted_tensor(layer_outputs[j])           
        sorted_tensor1 = get_sorted_tensor(layer_outputs1[j])
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(768), sorted_tensor.numpy(), color='r', label='with l1 loss')
        plt.plot(np.arange(768), sorted_tensor1.numpy(), color='b', label='vanilla')
        plt.title("Sorted Tensor Elements")
        plt.xlabel("Element Index")
        plt.ylabel("Normalized Value")
        plt.grid(True)
        plt.legend()
        plt.savefig('layer-sparsity(un-normalized)' + str(j) + '.png')
        
    # Orthogonal
    layer_ogds = get_layer_ogd(layer_outputs, cluster_lists)
    layer_ogds1 = get_layer_ogd(layer_outputs1, cluster_lists1)
        
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(12), layer_ogds, color='r', label='with orthogonal loss')
    plt.plot(np.arange(12), layer_ogds1, color='b', label='vanilla')
    plt.title("Layer Orthogonality")
    plt.xlabel("Layer Index")
    plt.ylabel("Orthonality Among Experts")
    plt.grid(True)
    plt.legend()
    plt.savefig('layer-orthogonality.png') 
    
    
def pic_param(config, accelerator):
    model = get_model(LOAD_PATH, config, accelerator)
    layer_outputs = get_layer_attn_params_moe(model, config)
                
    model1 = get_model(LOAD_PATH1, config, accelerator)               
    layer_outputs1 = get_layer_attn_params_moe(model1, config)
                
                
    # Sparsity    
          
    for j in range(config.num_hidden_layers):  
        sorted_tensor = get_sorted_tensor_param(layer_outputs[j][0] + layer_outputs[j][1])           
        sorted_tensor1 = get_sorted_tensor_param(layer_outputs1[j][0] + layer_outputs1[j][1])
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(786432), sorted_tensor.numpy(), color='r', label='with l1 loss')
        plt.plot(np.arange(786432), sorted_tensor1.numpy(), color='b', label='vanilla')
        plt.title("Sorted Tensor Elements")
        plt.xlabel("Element Index")
        plt.ylabel("Normalized Value")
        plt.grid(True)
        plt.legend()
        plt.savefig('layer-sparsity(un-normalized)(param)' + str(j) + '.png')
        
    # Orthogonal
    layer_ogds = get_layer_param_ogd_moe(layer_outputs)
    layer_ogds1 = get_layer_param_ogd_moe(layer_outputs1)
        
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(12), layer_ogds, color='r', label='with orthogonal loss')
    plt.plot(np.arange(12), layer_ogds1, color='b', label='vanilla')
    plt.title("Layer Orthogonality")
    plt.xlabel("Layer Index")
    plt.ylabel("Orthonality Among Experts")
    plt.grid(True)
    plt.legend()
    plt.savefig('layer-orthogonality(param).png') 


def main():
    set_seed(45)
    
    accelerator = Accelerator()
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    # dataset = RestaurantForLM(config=config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    dataset = Wikitext103(config=config)
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    
    pic_representation(config, accelerator, train_loader)
    # pic_param(config, accelerator)
    

if __name__ == "__main__":
    main()
    
    