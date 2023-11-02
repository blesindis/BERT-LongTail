import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import distance


def pca_all(input):
    X = input.mean(axis=1)
    X = X.cpu().numpy()
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X) 
    pca = PCA(n_components=X.shape[1])
    pca.fit_transform(X_std)
    
    components = pca.components_
    components = torch.tensor(components)
    
    return components


def pca(input, n_components=-1, threshold=0.80):
    X = input.mean(axis=1)
    X = X.cpu().numpy()
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    if n_components == -1:
        pca = PCA(n_components=X.shape[1])
        
        explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
        n_components = np.argmax(explained_variance_ratio >= threshold) + 1

    pca = PCA(n_components=n_components)
    X_pca_efficient = pca.fit_transform(X_std)    
    X_pca_efficient = torch.tensor(X_pca_efficient) 
    
    return X_pca_efficient


def pca_components(input, n_components=-1, threshold=0.80):
    X = input.mean(axis=1)
    X = X.cpu().numpy()
    
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    
    if n_components == -1:
        pca = PCA(n_components=X.shape[1])
        
        explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
        n_components = np.argmax(explained_variance_ratio >= threshold) + 1

    pca = PCA(n_components=n_components)
    X_pca_efficient = pca.fit_transform(X_std)    
    X_pca_efficient = torch.tensor(X_pca_efficient) 
    
    components = pca.components_
    components = torch.tensor(components)
    
    explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
    print(explained_variance_ratio)
    
    return X_pca_efficient, components


def cluster_kmeans(X_pca, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit_predict(X_pca)
    
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    cluster_indexes = [torch.eq(torch.tensor(labels), i).nonzero(as_tuple=True)[0] for i in range(k)]

    return cluster_indexes, centers


def sample_by_cluster(X_pca, k, n_samples):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit_predict(X_pca)
    
    centers = kmeans.cluster_centers_
    
    sample_indexes = []
    
    for c in centers:
        dist = [distance.euclidean(c, point) for point in X_pca]
        nearest_indices = np.argsort(dist)[:n_samples]
        
        sample_indexes = np.concatenate((sample_indexes, nearest_indices))
        
    return sample_indexes