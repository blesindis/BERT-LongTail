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
VAL_LEN = 100

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


def compute_compress_acr(W, Pi):
    eps = 0.01
    p, m = W.shape
    k, _, _ = Pi.shape
    I = torch.eye(p).cuda()
    compress_loss = 0.
    for j in range(k):
        trPi = torch.trace(Pi[j]) + 1e-8
        trPi = trPi.cuda()
        scalar = p / (trPi * eps)
        log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
        compress_loss += log_det * trPi / m
    return compress_loss / 2


def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot


def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.
    Parameters:
        targets (np.ndarray): matrix with one hot labels
    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)
    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi
    
    
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
    
    outputs = {'attention_output': None, 'attention_residual_output': None, 'ffn_output': None, 'ffn_residual_output': None}
    # get outputs of each layer
    # layer_outputs = get_layer_outputs_attn_residual(model, val_loader) 
    outputs['ffn_residual_output'] = get_layer_outputs_ffn_residual(model, val_loader)[:12]
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
    for key in outputs.keys(): 
        if outputs[key] is not None:
            output = outputs[key] # output is a list of tensors, len(list) = 12, representing the output of each layer
            acr_loss = []
            num_clusters = []
            print(len(output))
            for o in output:
                o = o.mean(dim=1)
                min_loss = float('inf')
                opt_k = 0
                for k in range(5,30):
                    kmeans = KMeans(k)
                    clusters = kmeans.fit_predict(o)
                    Pi = label_to_membership(clusters, len(clusters))
                    Pi = torch.tensor(Pi, dtype=torch.float32).cuda()
                    loss = compute_compress_acr(o.T.cuda(), Pi).item()                    
                    if loss < min_loss:
                        min_loss = loss
                        opt_k = k
                print(min_loss)
                acr_loss.append(min_loss)
                num_clusters.append(opt_k)
                
            outputs[key] = {'acr_loss': acr_loss, 'num_clusters': num_clusters}

        
    
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