import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from transformers import BertConfig
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from torch.nn.functional import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from itertools import combinations

# Local imports
import base_models
from Dataset import MixedData, ACLForLM, RestaurantForLM
from utils.train_utils import (    
    get_layer_outputs_attn_residual,    
    get_layer_outputs_attn_pre,
    get_layer_outputs_attn_post_combine,
    get_layer_outputs_attn_post_split,
    get_layer_outputs_ffn_residual,
    get_layer_outputs_ffn_pre,
    get_layer_outputs_ffn_post,
)
from utils.pic_utils import pca_scatter


MODEL_NAMES = (
    "1211-bert-no-dropout-pre-norm",
    "1211-bert-no-dropout",
    "1211-bert-combine-residual-no-dropout-pre-norm",
    "1211-bert-combine-residual-no-dropout"
)


# train and validation size for pretrain
TRAIN_LEN = 10000
VAL_LEN = 500

# folder paths
LOAD_FOLDER = "1211-bert-no-dropout-pre-norm"
LOAD_PATH = os.path.join('outputs', LOAD_FOLDER)
CONFIG_PATH = 'config/bert_a.json'

# training parameters
num_epochs = 50
lr = 1.5e-4
weight_decay = 0

def avg_coding_rate(data):
    eps = 0.01
    W = data.T.cuda() # W is of size (hidden_size, n_samples)
    p, m = W.shape
    I = torch.eye(p).cuda()
    scalar = p / (m * eps)
    logdet = torch.logdet(I + scalar * W.matmul(W.T))
    return logdet / 2
    

def greedy_mcr(data):
    """do cluster to minimize average mcr
    input -> data(tensor): of size [n, 128, 768]
    """
    # Determine clusters
    X = data.mean(axis=1).T
    tensor_list = list(torch.unbind(X, dim=1))
    tensor_list = [tensor.view(-1,1) for tensor in tensor_list]
    
    while len(tensor_list) > 1:
        print(len(tensor_list))
        min_loss = float('inf')
        selected_pair_indices = None
        
        for i, j in combinations(range(len(tensor_list)), 2):
            concatenated = torch.cat((tensor_list[i], tensor_list[j]), dim=1)
            loss = avg_coding_rate(concatenated) - avg_coding_rate(tensor_list[i]) - avg_coding_rate(tensor_list[j])
            if loss < min_loss:
                min_loss = loss
                selected_pair_indices = (i, j)
        
        if min_loss >= 0 or selected_pair_indices is None:
            break
        
        i, j = selected_pair_indices
        new_tensor = torch.cat((tensor_list[i], tensor_list[j]), dim=1)
        tensor_list = [tensor_list[k] for k in range(len(tensor_list)) if k != i and k != j] + [new_tensor]
    
    # calculate cluster centers
    cluster_centers = [tensor.mean(axis=1).view(1, -1) for tensor in tensor_list] # center size = [1, 768]
    
    # calculate average coding rate
    sum_acr = sum([avg_coding_rate(tensor).item() * tensor.shape[1] for tensor in tensor_list])
    aacr = sum_acr / X.shape[1]
    
    return cluster_centers, aacr
    
    
    
def main():
    config = BertConfig.from_json_file(CONFIG_PATH)
    dataset = MixedData(config=config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    val_loader = dataset.val_loader
    
    accelerator = Accelerator()
    
    # prepare model and data on device
    val_loader = accelerator.prepare(val_loader)

    model_name = MODEL_NAMES[1]
    
    load_path = os.path.join('outputs', model_name)
    if 'combine' in model_name:
        if 'pre' in model_name:
            model = base_models.BertForMLMCombineResidualPreNorm(config)
        else:
            model = base_models.BertForMLMCombineResidualPostNorm(config)
    else:        
        if 'pre' in model_name:            
            model = base_models.BertForMLMPreNorm(config)
        else:
            model = base_models.BertForMLM(config)
    checkpoint = torch.load(os.path.join(load_path, 'pytorch_model.bin'), map_location='cuda')
    model.load_state_dict(checkpoint)
    
    model = accelerator.prepare(model)
    
    outputs = {'attention_output': None, 'attention_residual_output': None, 'ffn_output': None}
    # get outputs of each layer
    # layer_outputs = get_layer_outputs_attn_residual(model, val_loader) 
    if 'pre' in model_name:
        outputs['attention_output'] = get_layer_outputs_attn_pre(model, val_loader)     
        outputs['ffn_output'] = get_layer_outputs_ffn_pre(model, val_loader)
    else:
        outputs['ffn_output'] = get_layer_outputs_ffn_post(model, val_loader)
        if 'combine' in model_name:
            outputs['attention_output'] = get_layer_outputs_attn_post_combine(model, val_loader)
        else:
            outputs['attention_output'] = get_layer_outputs_attn_post_split(model, val_loader)        
    
    # get dim-wise variance
    for k in outputs.keys(): 
        # if outputs[k] is not None:
        #     output = outputs[k] # output is a list of tensors, len(list) = 12, representing the output of each layer
        #     mean_cos_sims = []
        #     acrs = []
        #     num_clusters = []
        #     print(len(output))
        #     for o in output:
                
        #         centers, acr = greedy_mcr(o)
        #         num_center = len(centers)
        #         similarity_matrix = torch.zeros(num_center, num_center)
        #         for i, j in combinations(range(num_center), 2):
        #             similarity_matrix[i,j] = cosine_similarity(centers[i], centers[j]) if i != j else 0
        #         mean_cos_sim = torch.sum(similarity_matrix) / (num_center*(num_center - 1))
                
        #         mean_cos_sims.append(mean_cos_sim)
        #         acrs.append(acr)
        #         num_clusters.append(num_center)
                
        #     outputs[k] = {'cos_sim': mean_cos_sims, 'acr': acrs, 'num_clusters': num_clusters}
        
        if outputs[k] is not None:
            output = outputs[k] # output is a list of tensors, len(list) = 12, representing the output of each layer
            ginis = []
            for o in output:
                o = torch.abs(o.mean(dim=1))
                n, _ = o.shape
                mean = torch.mean(o, dim=1, keepdim=True)
                diff_matrix = torch.abs(o.unsqueeze(2) - o.unsqueeze(1))
                sum_diff = torch.sum(diff_matrix, dim=[1, 2])
                gini = sum_diff / (2 * n * n * mean)
                gini = gini.mean().item()
                ginis.append(gini)
            print(ginis)
            outputs[k] = {'sparsity(ginis)': ginis}
        
    
    # plot att & att-res
    for k, v in outputs.items():
        if v is not None:
            for name, value in v.items():
                plt.figure()
                plt.plot(np.arange(12), value)
                plt.title('LayerWise ' + k + ' ' + name)        
                plt.xlabel('Layer Index')
                plt.ylabel(name)        
                # Set y-axis limits
                # plt.ylim(0, 1)            
                plt.savefig('LayerWise ' + k + ' ' + name + '.jpg')
            

if __name__ == "__main__":
    main()