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
    validate,
    get_layer_outputs,
)
from utils.pic_utils import pca_scatter


MODEL_NAMES = (
    "1211-bert-no-dropout-pre-norm",
    "1211-bert-no-dropout",
    "1211-bert-combine-residual-no-dropout-pre-norm",
    "1211-bert-combine-residual-no-dropout"
    # "1211-bert-combine-residual-no-dropout-7epoch",
    # "1211-bert-combine-residual-no-dropout-8epoch",
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


def pca_auto(i, data, threshold=0.80):
    
    pca = PCA(n_components=768)
    data_pca = pca.fit_transform(data)
    
    explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
    n_components = np.argmax(explained_variance_ratio >= threshold) + 1
    
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    components = pca.components_[:10]
        
    print(f'layer {i} Need components {n_components} to pass information threshold {threshold}')
    
    return torch.tensor(data_pca), n_components, components


def main():
    config = BertConfig.from_json_file(CONFIG_PATH)
    dataset = MixedData(config=config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    
    
    val_loader1, val_loader2 = dataset.val_loader_set[0], dataset.val_loader_set[1]
    
    accelerator = Accelerator()
    
    # prepare model and data on device
    val_loader1, val_loader2 = accelerator.prepare(val_loader1, val_loader2)

    
    model_metrics_center = []
    model_metrics_mean = []
    for model_name in MODEL_NAMES:
        load_path = os.path.join('outputs', model_name)
        if 'combine' in model_name:
            model = base_models.BertForMLMCombineResidual(config)
        else:
            model = base_models.BertForMLM(config)
        checkpoint = torch.load(os.path.join(load_path, 'pytorch_model.bin'), map_location='cuda')
        model.load_state_dict(checkpoint)
        
        model = accelerator.prepare(model)
            
        del model
        # for i in range(12):
        #     print(f'layer {i} dbi: {center_dists[i]}')
        #     print(f'layer {i} var: {mean_dist[i]}')
    
    data1 = {
            "BERT+combined residual connection+Post LayerNorm": model_metrics_center[3],
            "BERT+combined residual connection+Pre LayerNorm": model_metrics_center[2],
            "BERT+Post LayerNorm": model_metrics_center[1],
            "BERT+Pre LayerNorm": model_metrics_center[0]
        }
    data2 = {
            "BERT+combined residual connection+Post LayerNorm": model_metrics_mean[3],
            "BERT+combined residual connection+Pre LayerNorm": model_metrics_mean[2],
            "BERT+Post LayerNorm": model_metrics_mean[1],
            "BERT+Pre LayerNorm": model_metrics_mean[0]
        }
        
    # Defining a list of colors to use for each line
    colors = ['blue', 'green', 'red', 'purple']
    
    plt.figure(1)
    for i, (model_name, values) in enumerate(data1.items()):
        # Using a different color for each line
        plt.plot(values, label=model_name, color=colors[i])

    plt.title("Center Distance of Two Datasets Across Layers")
    # plt.title("Cosine Similarity of Representation Between Two Datasets")
    plt.xlabel("Layer Index")
    plt.ylabel("Center Distance")
    # plt.ylabel("Cosine Similarity")
    plt.xticks(range(12))  # Assuming layer indices are from 0 to 11
    plt.legend()
    plt.savefig("Center Distance of Two Datasets Across Layers.jpg")
    
    plt.figure(2)
    for i, (model_name, values) in enumerate(data2.items()):
        # Using a different color for each line
        plt.plot(values, label=model_name, color=colors[i])

    plt.title("Average Distance from Points to Cluster Center Across Layers")
    # plt.title("Number of PCA Components to pass information threshold 80% Across Layers")
    plt.xlabel("Layer Index")
    plt.ylabel("Average Distance from Points to Cluster Center") 
    # plt.ylabel("Number of PCA Components")
    plt.xticks(range(12))  # Assuming layer indices are from 0 to 11
    plt.legend()
    plt.savefig("Average Distance from Points to Cluster Center Across Layers.jpg")

    # pic
    # for i, (d1, d2) in enumerate(zip(layer_outputs1, layer_outputs2)):
    #     pca_scatter(d1, d2, 'bert-combine-residual-8epoch-layer' + str(i))

if __name__ == "__main__":
    main()