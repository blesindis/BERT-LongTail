import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from transformers import BertConfig, get_cosine_schedule_with_warmup
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import distance

import base_models
from Dataset import RestaurantForLM_small


# Constants and Configurations
SEED = 45
LOAD_PATH = "./output-0-saveall-1016"
CONFIG_PATH = 'config/bert.json'


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def pca(input):
    # X = X.view(-1, X.shape[-1])
    X = input.mean(axis=1)
    print(X.size())
    X = X.cpu().numpy()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    pca = PCA(n_components=X.shape[1])
    X_pca = pca.fit_transform(X_std)
    
    explained_variance_ratio = pca.explained_variance_ratio_.cumsum()

    """Set threshold and Find the number of components for a threshold"""
    num_components = np.argmax(explained_variance_ratio >= 0.80) + 1

    # Now, you can recompute PCA with this number of components
    pca = PCA(n_components=num_components)
    X_pca_efficient = pca.fit_transform(X_std)
    
    X_pca_efficient = torch.tensor(X_pca_efficient) 
    print(X_pca_efficient.size())
    
    return X_pca_efficient


def find_largest_eps(X, min_eps=0.01, max_eps=10.0, min_samples=5, tolerance=1e-4):
    low = min_eps
    high = max_eps
    best_eps = min_eps
    
    while high - low > tolerance:
        mid = (low + high) / 2
        dbscan = DBSCAN(eps=mid, min_samples=min_samples)
        clusters = dbscan.fit_predict(X)
        
        # Number of unique clusters excluding noise
        unique_labels = np.unique(clusters)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if n_clusters >= 2:
            best_eps = mid
            low = mid
        else:
            high = mid
            
    return best_eps


def cluster_DBSCAN(X):
    # eps = find_largest_eps(X)
    # print(eps)
    dbscan = DBSCAN(eps=10, min_samples=5)
    clusters = dbscan.fit_predict(X)
    # Get unique cluster labels
    unique_labels = np.unique(clusters)

    # Count of clusters excluding noise
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    print(f"Number of clusters: {n_clusters}")


def find_elbow(X, K_range):
    distortions = []
    for k in K_range:
        kmeans = KMeans(n_clusters=k).fit(X)
        distortions.append(kmeans.inertia_)

    # Calculate the differences between consecutive distortions
    diffs = np.diff(distortions)

    # Calculate the differences of the differences
    diff_diffs = np.diff(diffs)

    # Find the "elbow" point
    optimal_k = K_range[np.argmin(diff_diffs) + 1] # +1 because we are using the difference of differences
    return optimal_k


def cluster_kmeans(X_pca, X, X_pre, labels, attns, n_samples):
    K_range = range(1, 10)
    optimal_k = find_elbow(X_pca, K_range)
    
    kmeans = KMeans(n_clusters=optimal_k)
    y_kmeans = kmeans.fit_predict(X_pca)
        
    # Extract cluster centers
    centers = kmeans.cluster_centers_

    nearest_points_per_center = []
    nearest_pre_points_per_center = []
    points_lables = []
    points_attns = []

    print(len(centers))
    for center in centers:
        # Compute distances from the center to each point
        distances = [distance.euclidean(center, point) for point in X_pca]
        
        # Get indices of 64 nearest samples
        nearest_indices = np.argsort(distances)[:n_samples]
        
        # Append nearest samples to the result list
        nearest_points_per_center.append(X[nearest_indices])
        nearest_pre_points_per_center.append(X_pre[nearest_indices])
        points_lables.append(labels[nearest_indices])
        points_attns.append(attns[nearest_indices])
    
    output_points = torch.cat(nearest_points_per_center, dim=0)
    input_points = torch.cat(nearest_pre_points_per_center, dim=0)
    label_points = torch.cat(points_lables, dim=0)
    attn_points = torch.cat(points_attns, dim=0)

    return output_points, input_points, label_points, attn_points


def layerwise_pca(model, dataset, load_path):
    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    num_updates = 70 * len(train_loader)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_updates * 0.1, num_training_steps=num_updates)
    accelerator = Accelerator()
    
    # load model checkpoint
    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, lr_scheduler, train_loader, val_loader)
    accelerator.load_state(load_path)
    
    # run once
    model.eval()
    
    all_layer_outputs = [[] for i in range(13)]
    all_layer_inputs = []
    all_layer_labels = []
    all_layer_attns = []
    all_decoder_outputs = []
    with torch.no_grad():
        for i, batch in enumerate(train_loader):   
            print(i)  
            if i < 108:                  
                _, scores, layer_outputs = model(**batch)
                input = batch['input_ids']
                label = batch['labels']
                attention_mask = batch['attention_mask']
                
                # move to cpu to release cuda memory
                batch = {key: tensor.to('cpu') for key, tensor in batch.items()}
                layer_outputs = [output.to('cpu') for output in layer_outputs]
                scores = scores.to('cpu')
                
                # save in my variable
                for j, layer_output in enumerate(layer_outputs):  
                    all_layer_outputs[j].append(layer_output)
                all_layer_inputs.append(input)
                all_layer_labels.append(label)
                all_layer_attns.append(attention_mask)
                all_decoder_outputs.append(scores)
            else:
                break
                
    accelerator.print(f'Number of Samples batches: {len(all_layer_outputs[0])}')
    
    layer_inputs = {}
    layer_outputs = {}
    layer_labels = {}
    layer_attns = {} 
    
    for i in range(1, 13):
        per_layer_outputs, per_layer_inputs = all_layer_outputs[i], all_layer_outputs[i-1]
        layer_outputs_tensor, layer_inputs_tensor, layer_labels_tensor, layer_attns_tensor = torch.cat(per_layer_outputs, dim=0), torch.cat(per_layer_inputs, dim=0), torch.cat(all_layer_labels, dim=0), torch.cat(all_layer_attns, dim=0)
        layer_pca = pca(layer_outputs_tensor)
        sampled_inputs, sampled_outputs, sampled_labels, sampled_attns = cluster_kmeans(layer_pca, layer_outputs_tensor, layer_inputs_tensor, layer_labels_tensor, layer_attns_tensor, 8)
        layer_inputs['layer' + str(i)] = sampled_inputs
        layer_outputs['layer' + str(i)] = sampled_outputs
        layer_labels['layer' + str(i)] = sampled_labels
        layer_attns['layer' + str(i)] = sampled_attns
         
    torch.save(layer_outputs, os.path.join(load_path, 'layer_outputs.pth'))
    torch.save(layer_inputs, os.path.join(load_path, 'layer_inputs.pth'))
    torch.save(layer_labels, os.path.join(load_path, 'layer_labels.pth'))
    torch.save(layer_attns, os.path.join(load_path, 'layer_attns.pth'))
    

def main():
    set_seed(SEED)
    
    config = BertConfig.from_json_file(CONFIG_PATH)
    dataset = RestaurantForLM_small(config=config)
    
    model = base_models.BertWithSavers(config=config)
    # model = nn.DataParallel(model) # Uncomment if using DataParallel
    
    layerwise_pca(model=model, dataset=dataset, load_path=LOAD_PATH)


if __name__ == "__main__":
    main()
