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

# Local imports
import base_models
from Dataset import MixedData, ACLForLM, RestaurantForLM
from utils.train_utils import (    
    get_layer_outputs_attn_residual,    
    get_layer_outputs_attn_pre,
    get_layer_outputs_attn_post_combine,
    get_layer_outputs_attn_post_split,
    get_layer_outputs_ffn_residual,
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


def main():
    config = BertConfig.from_json_file(CONFIG_PATH)
    dataset = MixedData(config=config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    val_loader = dataset.val_loader
    
    accelerator = Accelerator()
    
    # prepare model and data on device
    val_loader = accelerator.prepare(val_loader)

    model_name = MODEL_NAMES[3]
    
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
    
    outputs = {'attention_output': None, 'attention_residual_output': None}
    # get outputs of each layer
    # layer_outputs = get_layer_outputs_attn_residual(model, val_loader) 
    outputs['attention_residual_output'] = get_layer_outputs_ffn_residual(model, val_loader)
    if 'pre' in model_name:
        outputs['attention_output'] = get_layer_outputs_attn_pre(model, val_loader)     
    else:
        if 'combine' in model_name:
            outputs['attention_output'] = get_layer_outputs_attn_post_combine(model, val_loader)
        else:
            outputs['attention_output'] = get_layer_outputs_attn_post_split(model, val_loader)        
    
    # get dim-wise variance
    for k in outputs.keys(): 
        output = outputs[k] # output is a list of tensors, len(list) = 12
        normalized_output_var = []
        for o in output:
            """PCA-like Boolean"""
            # o_var = torch.var(o.mean(dim=1), dim=0, unbiased=True)
            # # Step 1: Sort the tensor in descending order and keep track of the original indices
            # sorted_variances, indices = torch.sort(o_var, descending=True)

            # # Step 2: Compute the cumulative sum of the sorted variances
            # cumulative_variances = torch.cumsum(sorted_variances, dim=0)

            # # Step 3: Find the number of top elements that make up at least 80% of the total variance
            # total_variance = torch.sum(o_var)
            # threshold = 0.8 * total_variance
            # k_index = torch.argmax((cumulative_variances >= threshold).int()) + 1  # +1 because argmax returns the index

            # # Step 4: Create a new tensor where elements contributing to 80% variance are 1, others are 0
            # top_k_mask = torch.zeros_like(o_var)
            # top_k_indices = indices[:k_index]
            # top_k_mask[top_k_indices] = 1            
            # normalized_output_var.append(top_k_mask)
            """Min-Max Normalization"""
            # o_var = torch.var(o.mean(dim=1), dim=0, unbiased=True)            
            # min_v = torch.min(o_var)
            # max_v = torch.max(o_var)
            # normalized_output_var.append((o_var - min_v) / (max_v - min_v))
            
            """Gaussian Normalization"""
            o_var = torch.mean(o.mean(dim=1), dim=0)       
            
            mean = torch.mean(o_var)
            std = torch.std(o_var)
            normalized_output_var.append((o_var - mean) / std)
            
        outputs[k] = normalized_output_var
    
    # plot layer-output
    similarity_matrix = torch.zeros(12, 12)
    for i in range(12):
        for j in range(12):
            similarity_matrix[i, j] = cosine_similarity(outputs['attention_residual_output'][i+1].unsqueeze(0), outputs['attention_residual_output'][j+1].unsqueeze(0))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    # Adding a title and labels
    plt.title("Cosine Similarity of Layer-wise Output")
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    plt.colorbar()

    # Saving the figure
    plt.savefig('cosine_similarity_matrix(layer-output).png')
    plt.close()
    
    # plot att
    similarity_matrix = torch.zeros(12, 12)
    for i in range(12):
        for j in range(12):
            similarity_matrix[i, j] = cosine_similarity(outputs['attention_output'][i].unsqueeze(0), outputs['attention_output'][j].unsqueeze(0))
    
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
    # Adding a title and labels
    plt.title("Cosine Similarity of Layer-wise Attention Output")
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    plt.colorbar()

    # Saving the figure
    plt.savefig('cosine_similarity_matrix(attn).png')
    plt.close()
    
    # plot att & att-res
    similarity = torch.zeros(12)
    for i in range(12):
        similarity[i] = cosine_similarity(outputs['attention_output'][i].unsqueeze(0), outputs['attention_residual_output'][i].unsqueeze(0))
    plt.figure()    
    plt.plot(np.arange(12), similarity.cpu().numpy())
    plt.title('Cosine Similarity of Layer-wise Attn-output & Attn-input')        
    plt.xlabel('Layer Index')
    plt.ylabel('Cosine Similarity')        
    # Set y-axis limits
    plt.ylim(0, 1)
    # plt.legend()
    plt.savefig('cosine_similarity(attn-out & attn-in).jpg')
    
    
    # colors = ['r', 'b', 'g', 'k', 'y', 'c', 'm', 'w', 'grey', 'orange', 'purple', 'pink']
    # for output_name, output_value in outputs.items():
    #     for i, layer_value in enumerate(output_value):
    #         plt.figure()
    #         # plt.plot(np.arange(768), layer_value.cpu().numpy(), label='layer'+str(i), color=colors[i])
    #         plt.plot(np.arange(768), layer_value.cpu().numpy())
    #         plt.title('Layer' + str(i) + ' dim variance of ' + output_name)        
    #         plt.xlabel('Dim Index')
    #         plt.ylabel('Dim Variance')        
    #         # plt.legend()
    #         plt.savefig('Layer' + str(i) + ' dim variance of ' + output_name + ".jpg")
        
        
    
    

if __name__ == "__main__":
    main()