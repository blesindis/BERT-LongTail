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
    get_layer_outputs_ffn_residual,
    get_layer_outputs_attn_residual,
    get_layer_outputs_ffn_pre,
    get_layer_outputs_ffn_post,
    get_layer_outputs_attn_pre,
    get_layer_outputs_attn_post,
)
from utils.pic_utils import pca_scatter


MODEL_NAMES = (
    "1211-bert-no-dropout-pre-norm",
    "1211-bert-no-dropout",
    "1211-bert-combine-residual-no-dropout-pre-norm",
    "1211-bert-combine-residual-no-dropout"
)


DBI = {
    'title': "Davies Bouldin Score (DBI) Across Layers",
    'xlabel': "Layer Index",
    'ylabel': "DBI"
}
NUMBER_OF_CLUSTERS = {
    'title': "Number of Clusters Across Layers",
    'xlabel': "Layer Index",
    'ylabel': "Number of Clusters"
}
INTER_CLUSTER_DISTANCE = {
    'title': "Inter Cluster Distance Across Layers",
    'xlabel': "Layer Index",
    'ylabel': "Inter Cluster Distance"
}
INTRA_CLUSTER_DISTANCE = {
    'title': "Intra Cluster Distance Across Layers",
    'xlabel': "Layer Index",
    'ylabel': "Intra Cluster Distance"
}
PCA_N_COMPONENTS = {
    'title': "Number of PCA Components to pass information threshold 80% Across Layers",
    'xlabel': "Layer Index",
    'ylabel': "Number of PCA Components"
}
COSINE_SIMILARITY = {
    'title': "Cosine Similarity of First PCA Dimension",
    'xlabel': "Layer Index",
    'ylabel': "Cosine Similarity"
}


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


def k_for_min_dbi(data):
    min_dbi = 9999999
    optimal_k = 0
    for k in range(2, 20):
        kmeans = KMeans(n_clusters=k)
        clusters = kmeans.fit_predict(data)
        dbi = davies_bouldin_score(data, clusters)
        if dbi < min_dbi:
            min_dbi = dbi
            optimal_k = k
            
    return optimal_k, min_dbi


def main():
    config = BertConfig.from_json_file(CONFIG_PATH)
    dataset = MixedData(config=config, train_len=TRAIN_LEN, val_len=VAL_LEN)
    
    
    val_loader = dataset.val_loader
    
    accelerator = Accelerator()
    
    # prepare model and data on device
    val_loader = accelerator.prepare(val_loader)

    
    model_dbis = []
    model_num_clusters = []
    for model_name in MODEL_NAMES:
        load_path = os.path.join('outputs', model_name)
        if 'combine' in model_name:
            model = base_models.BertForMLMCombineResidual(config)
        else:
            model = base_models.BertForMLM(config)
        checkpoint = torch.load(os.path.join(load_path, 'pytorch_model.bin'), map_location='cuda')
        model.load_state_dict(checkpoint)
        
        model = accelerator.prepare(model)
            
        # get outputs of each layer
        # layer_outputs = get_layer_outputs_attn_residual(model, val_loader) 
        if 'pre' in model_name:
            layer_outputs = get_layer_outputs_ffn_pre(model, val_loader)
        else:
            layer_outputs = get_layer_outputs_ffn_post(model, val_loader)                       
        
        # dbi
        dbis = []
        num_clusters = []
        for o in layer_outputs:
            k, dbi = k_for_min_dbi(o.mean(axis=1))
            dbis.append(dbi)
            num_clusters.append(k)
        # for i in range(len(layer_outputs)):
        #     # outputs = torch.cat((layer_outputs1[i].mean(axis=1), layer_outputs2[i].mean(axis=1)), dim=0)
        #     # dbscan = DBSCAN(eps=0.5, min_samples=5)
        #     # clusters = dbscan.fit_predict(outputs)
        #     center1 = layer_outputs1[i].mean(axis=1).mean(axis=0).view(1,-1)
        #     center2 = layer_outputs2[i].mean(axis=1).mean(axis=0).view(1,-1)
            
        #     # sim = cosine_similarity(center1, center2).item()
        #     center_dist = torch.dist(center1, center2, p=2)
            
        #     output1 = layer_outputs1[i].mean(axis=1)
        #     output2 = layer_outputs2[i].mean(axis=1)
            
        #     # _, pca_n, _ = pca_auto(i, torch.cat((layer_outputs1[i], layer_outputs2[i]), dim=0).mean(axis=1))
            
        #     var1 = torch.dist(output1, center1, p=2).mean(axis=0)
        #     var2 = torch.dist(output2, center2, p=2).mean(axis=0)
        #     var = (var1 + var2) / 2
        #     mean_dist.append(var)
            
        #     # clusters = kmeans.fit_predict(outputs)
        #     # db_index = davies_bouldin_score(outputs, clusters)
        #     center_dists.append(center_dist)
            
        # model_metrics_center.append(center_dists)
        # model_metrics_mean.append(mean_dist)
        print(dbis)
        print(num_clusters)
        model_dbis.append(dbis)
        model_num_clusters.append(num_clusters)
        del model
        # for i in range(12):
        #     print(f'layer {i} dbi: {center_dists[i]}')
        #     print(f'layer {i} var: {mean_dist[i]}')
    
    data1 = {
            "BERT+combined residual connection+Post LayerNorm": model_dbis[3],
            "BERT+combined residual connection+Pre LayerNorm": model_dbis[2],
            "BERT+Post LayerNorm": model_dbis[1],
            "BERT+Pre LayerNorm": model_dbis[0]
        }
    data2 = {
            "BERT+combined residual connection+Post LayerNorm": model_num_clusters[3],
            "BERT+combined residual connection+Pre LayerNorm": model_num_clusters[2],
            "BERT+Post LayerNorm": model_num_clusters[1],
            "BERT+Pre LayerNorm": model_num_clusters[0]
        }
        
    # Defining a list of colors to use for each line
    colors = ['blue', 'green', 'red', 'purple']
    
    plt.figure(1)
    for i, (model_name, values) in enumerate(data1.items()):
        # Using a different color for each line
        plt.plot(values, label=model_name, color=colors[i])

    plt.title(DBI["title"])
    # plt.title("Cosine Similarity of Representation Between Two Datasets")
    plt.xlabel(DBI["xlabel"])
    plt.ylabel(DBI["ylabel"])
    # plt.ylabel("Cosine Similarity")
    plt.xticks(range(12))  # Assuming layer indices are from 0 to 11
    plt.legend()
    plt.savefig(DBI["title"] + ".jpg")
    
    plt.figure(2)
    for i, (model_name, values) in enumerate(data2.items()):
        # Using a different color for each line
        plt.plot(values, label=model_name, color=colors[i])

    plt.title(NUMBER_OF_CLUSTERS["title"])
    # plt.title("Number of PCA Components to pass information threshold 80% Across Layers")
    plt.xlabel(NUMBER_OF_CLUSTERS["xlabel"])
    plt.ylabel(NUMBER_OF_CLUSTERS["ylabel"]) 
    # plt.ylabel("Number of PCA Components")
    plt.xticks(range(12))  # Assuming layer indices are from 0 to 11
    plt.legend()
    plt.savefig(NUMBER_OF_CLUSTERS["title"] + ".jpg")

    # pic
    # for i, (d1, d2) in enumerate(zip(layer_outputs1, layer_outputs2)):
    #     pca_scatter(d1, d2, 'bert-combine-residual-8epoch-layer' + str(i))

if __name__ == "__main__":
    main()