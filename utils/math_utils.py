import torch
import numpy as np
from sklearn.decomposition import PCA


def pca_fix(data, n_components=10):    
    data = data.mean(dim=1)
    data = data.cpu().numpy()
    
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    
    explained_variance_ratio = pca.explained_variance_ratio_.cumsum()    
    print(f'with components {n_components}, information kept is {explained_variance_ratio[n_components - 1]}')
    
    return torch.tensor(data_pca)


def pca_auto(data, threshold=0.2):
    data = data.mean(dim=1)
    data = data.cpu().numpy()
    
    pca = PCA(n_components=data.shape[1])
    data_pca = pca.fit_transform(data)
    
    explained_variance_ratio = pca.explained_variance_ratio_.cumsum()
    n_components = np.argmax(explained_variance_ratio >= threshold) + 1
    
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
        
    print(f'Need components {n_components} to pass information threshold {threshold}')
    
    return torch.tensor(data_pca)